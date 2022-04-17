/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/kernel/mkldnn/reduction_cpu_kernel.h"
#include <string>
#include <unordered_map>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "utils/ms_utils.h"
#include "mindspore/core/ops/lp_norm.h"

namespace mindspore {
namespace kernel {
namespace {
struct ReductionDescParam {
  dnnl::algorithm algorithm{dnnl::algorithm::undef};
  float p_{2.0f};
  float eps_{0.0f};
};
}  // namespace

void ReductionCpuKernelMod::MergeContinuousShape() {
  // The axis may be not sorted in order, std::set will order it.
  std::set<int64_t> axis_set(axis_.begin(), axis_.end());
  std::vector<int64_t> merge_input_shape;
  std::vector<int64_t> merge_output_shape;
  std::vector<int64_t> merge_axis_shape;

  auto is_merge = [](const std::set<int64_t> &axis_set, int64_t axis, int64_t limit_rank) {
    if (axis >= limit_rank - 1) {
      return false;
    }
    if (axis_set.count(axis) && axis_set.count(axis + 1)) {
      return true;
    }
    return !axis_set.count(axis) && !axis_set.count(axis + 1);
  };

  for (size_t i = 0; i < input_shape_.size(); ++i) {
    int64_t current_element = input_shape_.at(i);
    bool is_merge_axis = false;
    if (axis_set.count(SizeToLong(i))) {
      is_merge_axis = true;
    }
    while (is_merge(axis_set, SizeToLong(i), SizeToLong(input_shape_.size()))) {
      current_element *= input_shape_.at(i);
      ++i;
    }
    if (is_merge_axis) {
      merge_axis_shape.emplace_back(current_element);
    } else {
      merge_output_shape.emplace_back(current_element);
    }
    merge_input_shape.emplace_back(current_element);
  }
  input_shape_ = merge_input_shape;
  output_shape_ = merge_output_shape;
}

void ReductionCpuKernelMod::GetReductionAttr(const BaseOperatorPtr &base_operator) {
  if (!kernel_ptr_->HasAttr(AXIS)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' has no kernel attribute: " << AXIS;
  }
  axis_ = GetValue<std::vector<int64_t>>(kernel_ptr_->GetAttr(AXIS));
  const std::string p = "p";
  if (!kernel_ptr_->HasAttr(p)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' has no kernel attribute: " << p;
  }
  p_ = LongToFloat(GetValue<int64_t>(kernel_ptr_->GetAttr(p)));
  const std::string eps = "epsilon";
  if (!kernel_ptr_->HasAttr(eps)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "' has no kernel attribute: " << eps;
  }
  eps_ = GetValue<float>(kernel_ptr_->GetAttr(eps));
}

dnnl::reduction::desc ReductionCpuKernelMod::GetReductionDesc(const dnnl::memory::desc &src_desc,
                                                              const dnnl::memory::desc &dst_desc) {
  static const std::unordered_map<std::string, ReductionDescParam> reduction_op_desc_map{
    {prim::kPrimLrn->name(), ReductionDescParam{dnnl::algorithm::reduction_norm_lp_sum, p_, eps_}}};
  const auto desc_pair = reduction_op_desc_map.find(kernel_name_);
  if (desc_pair == reduction_op_desc_map.end()) {
    MS_LOG(EXCEPTION) << "ReductionCpuKernelMod does not support " << kernel_name_;
  }
  auto desc = CreateDesc<dnnl::reduction::desc>(desc_pair->second.algorithm, src_desc, dst_desc, desc_pair->second.p_,
                                                desc_pair->second.eps_);
  return desc;
}

bool ReductionCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  if (kernel_type_ == prim::kPrimLrn->name()) {
    kernel_ptr_ = std::dynamic_pointer_cast<ops::LpNorm>(base_operator);
  }
  kernel_name_ = kernel_ptr_->name();
  GetReductionAttr(base_operator);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << kernel_name_ << " does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  input_shape_ = inputs.at(kIndex0)->GetShapeVector();
  output_shape_ = outputs.at(kIndex0)->GetShapeVector();
  // This is a pre optimization for discontinuous shape. We may get better performance by merging continuous shapes
  MergeContinuousShape();
  std::vector<size_t> input_shape;
  std::vector<size_t> output_shape;
  std::transform(input_shape_.begin(), input_shape_.end(), input_shape.end(),
                 [](int64_t shape) { return static_cast<size_t>(shape); });
  std::transform(output_shape_.begin(), output_shape_.end(), output_shape.end(),
                 [](int64_t shape) { return static_cast<size_t>(shape); });
  dnnl::memory::desc src_desc = GetDefaultMemDesc(input_shape);
  dnnl::memory::desc dst_desc = GetDefaultMemDesc(output_shape);
  auto desc = GetReductionDesc(src_desc, dst_desc);
  auto prim_desc = CreateDesc<dnnl::reduction::primitive_desc>(desc, engine_);
  primitive_ = CreatePrimitive<dnnl::reduction>(prim_desc);
  AddArgument(DNNL_ARG_SRC, src_desc);
  AddArgument(DNNL_ARG_DST, dst_desc);
  return true;
}

template <typename T>
bool ReductionCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  auto input = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  auto output = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
  SetArgumentHandle(DNNL_ARG_SRC, input);
  SetArgumentHandle(DNNL_ARG_DST, output);
  ExecutePrimitive();
  return true;
}

std::vector<std::pair<KernelAttr, ReductionCpuKernelMod::ReductionFunc>> ReductionCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &ReductionCpuKernelMod::LaunchKernel<float>}};

std::vector<KernelAttr> ReductionCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, ReductionCpuKernelMod> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, LpNorm,
                                 []() { return std::make_shared<ReductionCpuKernelMod>(prim::kPrimLrn->name()); });
}  // namespace kernel
}  // namespace mindspore
