/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <condition_variable>
#include <mutex>
#include "utils/ms_exception.h"
#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/stub_tensor.h"
#include "pybind_api/gil_scoped_long_running.h"

namespace mindspore {
namespace stub {
namespace {
std::condition_variable stub_cond_var_;
std::mutex stub_mutex_;

static std::string MakeId() {
  // Use atomic to make id generator thread safe.
  static std::atomic<uint64_t> last_id{1};
  return "ST" + std::to_string(last_id.fetch_add(1, std::memory_order_relaxed));
}

StubNodePtr MakeStubNode(const TypePtr &type) {
  if (type->isa<Tuple>() || type->isa<List>()) {
    TypePtrList elements;
    if (type->isa<Tuple>()) {
      auto tuple_type = type->cast<TuplePtr>();
      elements = tuple_type->elements();
    } else {
      auto list_type = type->cast<ListPtr>();
      elements = list_type->elements();
    }
    auto node = std::make_shared<SequenceNode>(elements.size());
    for (size_t i = 0; i < elements.size(); ++i) {
      auto elem = MakeStubNode(elements[i]);
      node->SetElement(i, elem);
    }
    return node;
  } else if (type == kTypeAny) {
    return std::make_shared<AnyTypeNode>();
  } else if (type == kTypeNone) {
    return std::make_shared<NoneTypeNode>();
  } else {
    if (!type->isa<TensorType>()) {
      MS_LOG(WARNING) << "stub tensor is create for type: " << type->ToString();
    }
    return std::make_shared<TensorNode>();
  }
  return nullptr;
}

py::object MakeOutput(const StubNodePtr &node) {
  if (node->isa<TensorNode>()) {
    auto tensor = node->cast<std::shared_ptr<TensorNode>>();
    return py::cast(tensor);
  } else if (node->isa<SequenceNode>()) {
    auto seq = node->cast<std::shared_ptr<SequenceNode>>();
    MS_EXCEPTION_IF_NULL(seq);
    auto &elements = seq->Elements();
    if (elements.empty()) {
      return py::cast(seq);
    }
    py::tuple out(elements.size());
    for (size_t i = 0; i < elements.size(); ++i) {
      out[i] = MakeOutput(elements[i]);
    }
    return out;
  } else if (node->isa<AnyTypeNode>()) {
    auto tensor = node->cast<std::shared_ptr<AnyTypeNode>>();
    return py::cast(tensor);
  } else {
    auto tensor = node->cast<std::shared_ptr<NoneTypeNode>>();
    return py::cast(tensor);
  }
}
}  // namespace

StubNode::StubNode() : id_(MakeId()) {}

bool StubNode::SetAbstract(const AbstractBasePtr &abs) {
  abstract_ = abs;
  if (wait_flag_.load()) {
    std::unique_lock<std::mutex> lock(stub_mutex_);
    stub_cond_var_.notify_all();
  }
  return true;
}

void StubNode::SetValue(const ValuePtr &val) {
  value_ = val;
  if (wait_flag_.load()) {
    std::unique_lock<std::mutex> lock(stub_mutex_);
    stub_cond_var_.notify_all();
  }
}

void StubNode::SetException(const std::exception_ptr &e_ptr) {
  e_ptr_ = e_ptr;
  if (wait_flag_.load()) {
    std::unique_lock<std::mutex> lock(stub_mutex_);
    stub_cond_var_.notify_all();
  }
}

AbstractBasePtr StubNode::WaitAbstract() {
  GilReleaseWithCheck gil_release;
  if (abstract_.get() == nullptr) {
    wait_flag_.store(true);
    std::unique_lock<std::mutex> lock(stub_mutex_);
    stub_cond_var_.wait(lock, [this] { return abstract_.get() != nullptr || e_ptr_ != nullptr; });
    wait_flag_.store(false);
    if (e_ptr_ != nullptr) {
      // Need to clear exception in the instance.
      MsException::Instance().CheckException();
      std::rethrow_exception(e_ptr_);
    }
  }
  return abstract_;
}

ValuePtr StubNode::WaitValue() {
  GilReleaseWithCheck gil_release;
  if (value_.get() == nullptr) {
    wait_flag_.store(true);
    std::unique_lock<std::mutex> lock(stub_mutex_);
    stub_cond_var_.wait(lock, [this] { return value_.get() != nullptr || e_ptr_ != nullptr; });
    wait_flag_.store(false);
    if (e_ptr_ != nullptr) {
      // Need to clear exception in the instance.
      MsException::Instance().CheckException();
      std::rethrow_exception(e_ptr_);
    }
  }
  return value_;
}

py::object TensorNode::GetValue() {
  auto val = WaitValue();
  return ValueToPyData(val);
}

py::object TensorNode::GetShape() {
  auto abs = WaitAbstract();
  auto base = abs->BuildShape();
  auto shape = base->cast<abstract::ShapePtr>();
  ShapeVector shape_vector;
  if (shape && !shape->IsDynamic()) {
    shape_vector = shape->shape();
  } else {
    auto val = WaitValue();
    auto tensor = val->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    shape_vector = tensor->shape();
  }
  auto ret = py::tuple(shape_vector.size());
  for (size_t i = 0; i < shape_vector.size(); ++i) {
    ret[i] = shape_vector[i];
  }
  return ret;
}

py::object TensorNode::GetDtype() {
  auto abs = WaitAbstract();
  auto base = abs->BuildType();
  if (base->isa<TensorType>()) {
    base = base->cast<TensorTypePtr>()->element();
  }
  return py::cast(base);
}

bool TensorNode::SetAbstract(const AbstractBasePtr &abs) {
  if (!abs->isa<abstract::AbstractTensor>() && !abs->isa<abstract::AbstractMapTensor>()) {
    if (!abs->isa<abstract::AbstractScalar>() || abs->BuildValue() != kValueAny) {
      return false;
    }
  }
  return StubNode::SetAbstract(abs);
}

void TensorNode::SetValue(const ValuePtr &val) {
  MS_EXCEPTION_IF_NULL(val);
  if (!val->isa<tensor::Tensor>()) {
    MS_LOG(EXCEPTION) << "SetValue failed, val=" << val->ToString() << " is not a tensor";
  }
  auto t = val->cast<tensor::TensorPtr>();
  MS_LOG(DEBUG) << "Update tensor id from " << t->id() << " to " << id_;
  t->set_id(id_);

  StubNode::SetValue(val);
}

py::object SequenceNode::GetElements() {
  if (!is_elements_build_.load()) {
    (void)WaitAbstract();
  }
  py::tuple out(elements_.size());
  for (size_t i = 0; i < elements_.size(); ++i) {
    out[i] = MakeOutput(elements_[i]);
  }
  return out;
}

bool SequenceNode::SetAbstract(const AbstractBasePtr &abs) {
  auto seq_abs = abs->cast<abstract::AbstractSequencePtr>();
  if (seq_abs == nullptr) {
    return false;
  }
  auto children = seq_abs->elements();
  if (!is_elements_build_.load()) {
    for (auto child : children) {
      elements_.emplace_back(MakeStubNode(child->BuildType()));
    }
  }
  is_elements_build_ = true;
  if (elements_.size() != children.size()) {
    return false;
  }
  for (size_t i = 0; i < elements_.size(); ++i) {
    if (!elements_[i]->SetAbstract(children[i])) {
      return false;
    }
  }
  return StubNode::SetAbstract(abs);
}

void SequenceNode::SetValue(const ValuePtr &val) {
  auto seq_value = val->cast<ValueSequencePtr>();
  auto children = seq_value->value();
  for (size_t i = 0; i < children.size(); ++i) {
    elements_[i]->SetValue(children[i]);
  }
  StubNode::SetValue(val);
}

void SequenceNode::SetException(const std::exception_ptr &e_ptr) {
  for (auto &element : elements_) {
    element->SetException(e_ptr);
  }
  StubNode::SetException(e_ptr);
}

bool AnyTypeNode::SetAbstract(const AbstractBasePtr &abs) {
  real_node_ = MakeStubNode(abs->BuildType());
  auto flag = real_node_->SetAbstract(abs);
  (void)StubNode::SetAbstract(abs);
  return flag;
}

void AnyTypeNode::SetValue(const ValuePtr &val) {
  real_node_->SetValue(val);
  StubNode::SetValue(val);
}

py::object AnyTypeNode::GetRealNode() {
  (void)WaitAbstract();
  return py::cast(real_node_);
}

py::object NoneTypeNode::GetRealValue() {
  auto val = WaitValue();
  return ValueToPyData(val);
}

void AnyTypeNode::SetException(const std::exception_ptr &e_ptr) {
  StubNode::SetException(e_ptr);
  if (real_node_ != nullptr) {
    real_node_->SetException(e_ptr);
  }
}

std::pair<py::object, StubNodePtr> MakeTopNode(const TypePtr &type) {
  auto top = MakeStubNode(type);
  auto ret = MakeOutput(top);
  return std::make_pair(ret, top);
}

void RegStubNodes(const py::module *m) {
  (void)py::class_<StubNode, std::shared_ptr<StubNode>>(*m, "StubNode");
  (void)py::class_<TensorNode, StubNode, std::shared_ptr<TensorNode>>(*m, "TensorNode")
    .def("get_value", &TensorNode::GetValue, "get output value of async stub.")
    .def("get_shape", &TensorNode::GetShape, "get output shape of async stub.")
    .def("get_dtype", &TensorNode::GetDtype, "get output dtype of async stub.");
  (void)py::class_<SequenceNode, StubNode, std::shared_ptr<SequenceNode>>(*m, "SequenceNode")
    .def("get_elements", &SequenceNode::GetElements, "get the elements of async stub_seq.");
  (void)py::class_<AnyTypeNode, StubNode, std::shared_ptr<AnyTypeNode>>(*m, "AnyTypeNode")
    .def("get_real_node", &AnyTypeNode::GetRealNode, "get the real StubNode");
  (void)py::class_<NoneTypeNode, StubNode, std::shared_ptr<NoneTypeNode>>(*m, "NoneTypeNode")
    .def("get_real_value", &NoneTypeNode::GetRealValue, "get the real value");
}
}  // namespace stub
}  // namespace mindspore
