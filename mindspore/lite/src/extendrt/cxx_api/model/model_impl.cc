/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include <algorithm>
#include <shared_mutex>
#include "extendrt/cxx_api/model/model_impl.h"
#include "extendrt/cxx_api/dlutils.h"
#include "extendrt/cxx_api/file_utils.h"
#include "extendrt/utils/tensor_utils.h"
#include "mindspore/core/utils/ms_context.h"
#include "extendrt/mindir_loader/mindir_model/mindir_model_util.h"
#include "src/extendrt/convert/runtime_convert.h"
#include "src/common/config_file.h"
#include "src/extendrt/utils/serialization.h"
#include "mindapi/ir/func_graph.h"
#include "mindapi/base/base.h"
#include "src/extendrt/delegate/graph_executor/litert/func_graph_reuse_manager.h"
#include "mindspore/core/load_mindir/load_model.h"
#include "src/common/common.h"
#include "src/extendrt/delegate/plugin/tensorrt_executor_plugin.h"
#include "src/extendrt/kernel/ascend/plugin/ascend_kernel_plugin.h"

namespace mindspore {
namespace {
const char *const kExecutionPlan = "execution_plan";
constexpr size_t kMaxSectionNum = 100;
constexpr size_t kMaxConfigNumPerSection = 1000;
std::shared_mutex g_model_converter_lock;

std::map<std::string, tensor::TensorPtr> GetParams(const FuncGraphPtr &anf_graph) {
  MS_EXCEPTION_IF_NULL(anf_graph);
  std::map<std::string, tensor::TensorPtr> res;
  for (auto &anf_node : anf_graph->parameters()) {
    MS_EXCEPTION_IF_NULL(anf_node);
    auto para = anf_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(para);
    if (para->has_default()) {
      auto value = para->default_param();
      MS_EXCEPTION_IF_NULL(value);
      auto tensor = value->cast<std::shared_ptr<tensor::Tensor>>();
      res.emplace(para->name(), tensor);
      MS_LOG(INFO) << "Parameter " << para->name() << " has default value.";
    }
  }
  return res;
}
}  // namespace

void ModelImpl::SetMsContext() {
  if (MsContext::GetInstance() != nullptr) {
    auto back_policy_env = std::getenv("ASCEND_BACK_POLICY");
    if (back_policy_env != nullptr) {
      MsContext::GetInstance()->set_backend_policy(std::string(back_policy_env));
    }
  }
}

std::mutex ConverterPlugin::mutex_;
ConverterPlugin::ConverterPlugin() = default;

ConverterPlugin::~ConverterPlugin() {
#ifndef _WIN32
  if (handle_ != nullptr) {
    (void)dlclose(handle_);
    handle_ = nullptr;
  }
#endif
}

ConverterPlugin::ConverterFunc ConverterPlugin::GetConverterFunc() {
  std::lock_guard<std::mutex> lock(mutex_);
  static ConverterPlugin instance;
  return instance.GetConverterFuncInner();
}

ConverterPlugin::ConverterFunc ConverterPlugin::GetConverterFuncInner() {
#ifndef _WIN32
  if (converter_func_ == nullptr) {
    std::string plugin_path;
    auto ret = DLSoPath({"libmindspore-lite.so", "_c_lite"}, "libruntime_convert_plugin.so", &plugin_path);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Get path of libruntime_convert_plugin.so failed. error: " << ret;
      return nullptr;
    }
    void *function = nullptr;
    ret = DLSoOpen(plugin_path, "RuntimeConvert", &handle_, &function, true);
    if (ret != kSuccess) {
      MS_LOG(WARNING) << "DLSoOpen RuntimeConvert failed, so path: " << plugin_path;
      return nullptr;
    }
    converter_func_ = reinterpret_cast<ConverterPlugin::ConverterFunc>(function);
  }
  return converter_func_;
#else
  MS_LOG(ERROR) << "Not support libruntime_convert_plugin.so in Windows";
  return nullptr;
#endif
}

FuncGraphPtr ModelImpl::LoadGraphByBufferImpl(const void *model_buff, size_t model_size, ModelType model_type,
                                              const std::shared_ptr<Context> &model_context,
                                              const std::string &model_path) {
  if (model_buff == nullptr) {
    MS_LOG(ERROR) << "The input model buffer is nullptr.";
    return nullptr;
  }
  if (model_size == 0) {
    MS_LOG(ERROR) << "The input model buffer size is 0.";
    return nullptr;
  }
  if (model_type != kMindIR) {
    MS_LOG(ERROR) << "Invalid model type";
    return nullptr;
  }
  if (model_context == nullptr) {
    MS_LOG(ERROR) << "Invalid context pointers.";
    return nullptr;
  }
  auto mindir_path = GetConfig(lite::kConfigModelFileSection, lite::kConfigMindIRPathKey);
  std::string weight_path = "./";
  std::string base_path = "";
  if (!mindir_path.empty()) {
    base_path = mindir_path;
  } else {
    // user does not set mindir_path, convert from model_path
    base_path = model_path;
  }
  if (base_path.find("/") != std::string::npos) {
    weight_path = base_path.substr(0, base_path.rfind("/"));
  }
  auto dump_path = GetConfig(lite::kAscendContextSection, lite::kDumpPathKey);
  if (!dump_path.empty()) {
    auto dir_pos = model_path.find_last_of('/');
    auto mindir_name = dir_pos != std::string::npos ? model_path.substr(dir_pos + 1) : model_path;
    auto dot_pos = mindir_name.find_last_of('.');
    auto model_name = mindir_name.substr(0, dot_pos);
    (void)UpdateConfig(lite::kAscendContextSection,
                       std::pair<std::string, std::string>(lite::kDumpModelNameKey, model_name));
  }
  MindIRLoader mindir_loader(true, nullptr, 0, kDecModeAesGcm, false);
  auto func_graph = mindir_loader.LoadMindIR(model_buff, model_size, weight_path);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Failed to load MindIR model, please check the validity of the model: " << weight_path;
    return nullptr;
  }
  return func_graph;
}

Status ModelImpl::BuildByBufferImpl(const void *model_buff, size_t model_size, ModelType model_type,
                                    const std::shared_ptr<Context> &model_context, const std::string &model_path) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_) {
    MS_LOG(ERROR) << "Model has been called Build";
    return kLiteError;
  }
  SetMsContext();
  auto thread_num = model_context->GetThreadNum();
  if (thread_num < 0) {
    MS_LOG(ERROR) << "Invalid thread num " << thread_num;
    return kLiteError;
  }
  session_ = InferSession::CreateSession(model_context, config_info_);
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Create session failed.";
    return kLiteError;
  }
  auto ret = session_->Init(model_context);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Init session failed.";
    return ret;
  }

  // for model pool
  FuncGraphPtr func_graph = FuncGraphReuseManager::GetInstance()->GetSharedFuncGraph(config_info_);
  if (func_graph != nullptr) {
    MS_LOG(INFO) << "the model buffer is the same as the last time. we can directly use the cached function graph.";
    return session_->CompileGraph(func_graph, nullptr, 0, &graph_id_);
  }
  func_graph = LoadGraphByBufferImpl(model_buff, model_size, model_type, model_context, model_path);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "Failed to load MindIR model, please check the validity of the model: " << model_path;
    return kLiteError;
  }
  // convert and optimize func graph to infer
  ret = ConvertGraphOnline(func_graph, model_context);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "convert graph failed.";
    return ret;
  }
  ret = session_->CompileGraph(func_graph, nullptr, 0, &graph_id_);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "compile graph failed.";
    return ret;
  }
  std::shared_lock<std::shared_mutex> build_lock(g_model_converter_lock);
  return FuncGraphReuseManager::GetInstance()->StoreFuncGraph(func_graph, config_info_);
}

Status ModelImpl::Build(const std::vector<std::shared_ptr<ModelImpl>> &model_impls,
                        const std::vector<std::string> &model_paths, ModelType model_type,
                        const std::shared_ptr<Context> &model_context) {
  if (model_impls.empty()) {
    MS_LOG(ERROR) << "Model impls size is 0";
    return kLiteError;
  }
  if (model_impls.size() != model_paths.size()) {
    MS_LOG(ERROR) << "Model impls size " << model_impls.size() << " != model path size " << model_paths.size();
    return kLiteError;
  }
  for (size_t i = 0; i < model_impls.size(); i++) {
    if (model_impls[i] == nullptr || model_paths[i].empty()) {
      MS_LOG(ERROR) << "Model " << i << " is invalid, model impl is nullptr or model path is empty";
      return kLiteError;
    }
  }
  if (model_impls.size() == 1) {
    return model_impls[0]->Build(model_paths[0], model_type, model_context);
  }
  SetMsContext();
  auto thread_num = model_context->GetThreadNum();
  if (thread_num < 0) {
    MS_LOG(ERROR) << "Invalid thread num " << thread_num;
    return kLiteError;
  }
  auto session = InferSession::CreateSession(model_context, model_impls[0]->config_info_);
  if (session == nullptr) {
    MS_LOG(ERROR) << "Create session failed.";
    return kLiteError;
  }
  auto ret = session->Init(model_context);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Init session failed.";
    return ret;
  }

  std::map<std::string, tensor::TensorPtr> total_params;
  for (size_t i = 0; i < model_impls.size(); i++) {
    auto &impl = model_impls[i];
    auto &model_path = model_paths[i];
    std::lock_guard<std::recursive_mutex> lock(impl->mutex_);
    if (impl->session_) {
      MS_LOG(ERROR) << "Model " << i << " has been called Build";
      return kLiteError;
    }
    impl->session_ = session;
    auto buffer = ReadFile(model_path);
    if (buffer.DataSize() == 0) {
      MS_LOG(ERROR) << "Failed to read buffer from model file: " << model_path;
      return kLiteError;
    }
    auto func_graph =
      impl->LoadGraphByBufferImpl(buffer.Data(), buffer.DataSize(), model_type, model_context, model_path);
    if (func_graph == nullptr) {
      MS_LOG(ERROR) << "Failed to load graph";
      return kCoreFailed;
    }
    std::map<std::string, tensor::TensorPtr> params = GetParams(func_graph);
    for (auto &param : params) {
      if (total_params.find(param.first) != total_params.end()) {
        param.second->set_init_flag(true);
      } else {
        total_params.emplace(param);
      }
    }
    std::shared_lock<std::shared_mutex> build_lock(g_model_converter_lock);
    ret = session->CompileGraph(func_graph, buffer.Data(), buffer.DataSize(), &impl->graph_id_);
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Failed to compile graph " << model_path;
      return ret;
    }
  }
  // warm up
  MS_LOG(INFO) << "Start to warm-up to compile graph, model paths: " << model_paths;
  for (auto &model : model_impls) {
    ret = model->Warmup();
    if (ret != kSuccess) {
      MS_LOG(ERROR) << "Failed to compile graph, model paths: " << model_paths;
      return ret;
    }
  }
  MS_LOG(INFO) << "Compile graph success, model paths: " << model_paths;
  return kSuccess;
}

Status ModelImpl::Build(const void *model_data, size_t data_size, ModelType model_type,
                        const std::shared_ptr<Context> &model_context) {
  return BuildByBufferImpl(model_data, data_size, model_type, model_context);
}

Status ModelImpl::Build(const std::string &model_path, ModelType model_type,
                        const std::shared_ptr<Context> &model_context) {
  if (model_path.empty()) {
    MS_LOG(ERROR) << "Model path cannot be empty";
    return kLiteError;
  }
  auto buffer = ReadFile(model_path);
  if (buffer.DataSize() == 0) {
    MS_LOG(ERROR) << "Failed to read buffer from model file: " << model_path;
    return kLiteError;
  }
  return BuildByBufferImpl(buffer.Data(), buffer.DataSize(), model_type, model_context, model_path);
}

Status ModelImpl::ConvertGraphOnline(const FuncGraphPtr &func_graph, const std::shared_ptr<Context> &model_context) {
  MS_ASSERT(func_graph != nullptr);
  auto device_list = model_context->MutableDeviceInfo();
  for (const auto &device_info : device_list) {
    if (device_info == nullptr) {
      continue;
    }
    if (device_info->GetDeviceType() == DeviceType::kAscend && device_info->GetProvider() == "ge") {
      return kSuccess;
    }
  }
  auto value = func_graph->get_attr(lite::kIsOptimized);
  if (value != nullptr) {
    if (GetValue<bool>(value)) {
      // it does not need to convert, if funcgraph is optimized.
      return kSuccess;
    }
  }

  auto convert = ConverterPlugin::GetConverterFunc();
  if (convert == nullptr) {
    MS_LOG(ERROR) << "get Converter func failed";
    return kLiteError;
  }
  auto api_graph = mindspore::api::MakeShared<mindspore::api::FuncGraph>(func_graph);
  std::unique_lock<std::shared_mutex> build_lock(g_model_converter_lock);
  auto status = convert(api_graph, model_context, config_info_);
  if (status != kSuccess) {
    MS_LOG(ERROR) << "Failed to converter graph";
    return kLiteError;
  }

  return kSuccess;
}  // namespace mindspore

Status ModelImpl::Resize(const std::vector<MSTensor> &inputs, const std::vector<std::vector<int64_t>> &dims) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed";
    return kLiteError;
  }
  if (inputs.empty()) {
    MS_LOG(ERROR) << "Inputs is null.";
    return kLiteInputParamInvalid;
  }
  if (dims.empty()) {
    MS_LOG(ERROR) << "Dims is null.";
    return kLiteInputParamInvalid;
  }
  for (size_t j = 0; j < dims.size(); j++) {
    auto dims_v = dims[j];
    for (size_t i = 0; i < dims_v.size(); i++) {
      auto dim = dims_v[i];
      if (dim <= 0 || dim > INT_MAX) {
        MS_LOG(ERROR) << "Invalid shape! dim: " << dim;
        return kLiteInputParamInvalid;
      }
    }
  }
  if (inputs.size() != dims.size()) {
    MS_LOG(ERROR) << "The size of inputs does not match the size of dims.";
    return kLiteInputParamInvalid;
  }
  auto model_inputs = session_->GetInputs(graph_id_);
  if (model_inputs.empty()) {
    MS_LOG(ERROR) << "The inputs of model is null.";
    return kLiteParamInvalid;
  }
  if (inputs.size() != model_inputs.size()) {
    MS_LOG(ERROR) << "The size of inputs is incorrect.";
    return kLiteInputParamInvalid;
  }
  std::vector<mindspore::tensor::Tensor> resize_inputs = TensorUtils::MSTensorToTensor(inputs);
  return session_->Resize(graph_id_, resize_inputs, dims);
}

std::vector<MSTensor> ModelImpl::GetInputs() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed";
    return {};
  }
  auto graph_inputs = session_->GetInputs(graph_id_);
  std::vector<MSTensor> inputs;
  std::transform(graph_inputs.begin(), graph_inputs.end(), std::back_inserter(inputs),
                 [](auto &impl) { return MSTensor(impl); });
  return inputs;
}

std::vector<MSTensor> ModelImpl::GetOutputs() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed";
    return {};
  }
  auto graph_outputs = session_->GetOutputs(graph_id_);
  std::vector<MSTensor> outputs;
  std::transform(graph_outputs.begin(), graph_outputs.end(), std::back_inserter(outputs),
                 [](auto &impl) { return MSTensor(impl); });
  return outputs;
}

MSTensor ModelImpl::GetInputByTensorName(const std::string &name) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed";
    return MSTensor(nullptr);
  }
  auto tensor_impl = session_->GetInputByTensorName(graph_id_, name);
  if (tensor_impl == nullptr) {
    MS_LOG(ERROR) << "Model does not contains tensor " << name << " .";
    return MSTensor(nullptr);
  }
  return MSTensor(tensor_impl);
}

std::vector<std::string> ModelImpl::GetOutputTensorNames() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed";
    return {};
  }
  return session_->GetOutputNames(graph_id_);
}

MSTensor ModelImpl::GetOutputByTensorName(const std::string &name) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed";
    return MSTensor(nullptr);
  }
  auto tensor_impl = session_->GetOutputByTensorName(graph_id_, name);
  if (tensor_impl == nullptr) {
    MS_LOG(ERROR) << "Model does not contains tensor " << name << " .";
    return MSTensor(nullptr);
  }
  return MSTensor(tensor_impl);
}

Status ModelImpl::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs,
                          const MSKernelCallBack &before, const MSKernelCallBack &after) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed";
    return kLiteError;
  }
  MS_EXCEPTION_IF_NULL(outputs);
  std::vector<mindspore::tensor::Tensor> graph_inputs = TensorUtils::MSTensorToTensor(inputs);
  std::vector<mindspore::tensor::Tensor> graph_outputs;
  std::vector<mindspore::tensor::Tensor> org_graph_outputs;
  if (!outputs->empty()) {
    graph_outputs = TensorUtils::MSTensorToTensor(*outputs);
    org_graph_outputs = graph_outputs;
  }
  auto ret = session_->RunGraph(graph_id_, graph_inputs, &graph_outputs, before, after);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "ModelImpl::Predict RunGraph failed with " << ret;
    return ret;
  }
  bool output_remain = false;
  if (!org_graph_outputs.empty() && org_graph_outputs.size() == graph_outputs.size()) {
    output_remain = true;
    for (size_t i = 0; i < org_graph_outputs.size(); i++) {
      if (org_graph_outputs[i].data_ptr() != graph_outputs[i].data_ptr() ||
          org_graph_outputs[i].device_address() != graph_outputs[i].device_address()) {
        output_remain = false;
        break;
      }
    }
  }
  if (!output_remain) {
    auto session_outputs = session_->GetOutputNames(graph_id_);
    if (session_outputs.empty() || session_outputs.size() != graph_outputs.size()) {
      MS_LOG(ERROR) << "output name is wrong.";
      return kLiteError;
    }
    *outputs = TensorUtils::TensorToMSTensor(graph_outputs, session_outputs);
  }
  auto session_outputs = session_->GetOutputs(graph_id_);
  if (graph_outputs.size() != session_outputs.size()) {
    MS_LOG(ERROR) << "Outputs count get from session " << session_outputs.size() << " != outputs count of RunGraph "
                  << graph_outputs.size();
    return kCoreFailed;
  }
  for (size_t i = 0; i < session_outputs.size(); i++) {
    MSTensor session_output(session_outputs[i]);
    auto &execute_output = outputs->at(i);
    session_output.SetShape(execute_output.Shape());
    if (session_output.Data().get() != execute_output.Data().get()) {
      session_output.SetData(execute_output.MutableData(), false);
    }
    if (session_output.GetDeviceData() != execute_output.GetDeviceData()) {
      session_output.SetDeviceData(execute_output.GetDeviceData());
    }
  }
  return kSuccess;
}

Status ModelImpl::Predict(const std::vector<MSTensor> &inputs, std::vector<MSTensor> *outputs) {
  return Predict(inputs, outputs, nullptr, nullptr);
}

Status ModelImpl::Predict() {
  auto inputs = GetInputs();
  auto outputs = GetOutputs();
  return Predict(inputs, &outputs);
}

Status ModelImpl::Warmup() {
  auto inputs = GetInputs();
  for (auto &input : inputs) {
    auto shape = input.Shape();
    if (std::any_of(shape.begin(), shape.end(), [](auto dim) { return dim < 0; })) {
      MS_LOG(WARNING) << "Failed to warm-up because of dynamic inputs";
      return kSuccess;
    }
  }
  auto outputs = GetOutputs();
  return Predict(inputs, &outputs);
}

bool ModelImpl::HasPreprocess() {
  if (!graph_ || !graph_->graph_data_) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed";
    return false;
  }
  return graph_->graph_data_->GetPreprocess().empty() ? false : true;
}

Status ModelImpl::Preprocess(const std::vector<std::vector<MSTensor>> &inputs, std::vector<MSTensor> *outputs) {
#if !defined(_WIN32) && !defined(_WIN64)
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed";
    return kLiteError;
  }
  // Config preprocessor, temporary way to let mindspore.so depends on _c_dataengine
  std::string dataengine_so_path;
  Status dlret = DLSoPath({"libmindspore.so"}, "_c_dataengine", &dataengine_so_path);
  CHECK_FAIL_AND_RELEASE(dlret, nullptr, "Parse dataengine_so failed: " + dlret.GetErrDescription());

  // Run preprocess
  if (!HasPreprocess()) {
    MS_LOG(ERROR) << "Attempt to predict with data preprocessor, but no preprocessor is defined in MindIR.";
    return Status(kMEFailed, "Attempt to predict with data preprocessor, but no preprocessor is defined in MindIR.");
  }

  void *handle = nullptr;
  void *function = nullptr;
  dlret = DLSoOpen(dataengine_so_path, "ExecuteRun_C", &handle, &function);
  CHECK_FAIL_AND_RELEASE(dlret, handle, "Parse ExecuteRun_C failed: " + dlret.GetErrDescription());
  auto ExecuteRun =
    (void (*)(const std::vector<std::shared_ptr<dataset::Execute>> &, const std::vector<mindspore::MSTensor> &,
              std::vector<mindspore::MSTensor> *, Status *))(function);

  // perform preprocess on each tensor separately
  std::vector<std::shared_ptr<dataset::Execute>> preprocessor = graph_->graph_data_->GetPreprocess();
  std::vector<std::vector<MSTensor>> output_unbatch;
  std::vector<MSTensor> output_batched;
  for (auto tensor : inputs) {
    std::vector<MSTensor> temp;
    ExecuteRun(preprocessor, tensor, &temp, &dlret);
    CHECK_FAIL_AND_RELEASE(dlret, handle, "Run preprocess failed: " + dlret.GetErrDescription());
    output_unbatch.push_back(temp);
  }

  // Construct a tensor with batch dim
  output_batched.resize(output_unbatch[0].size());
  for (size_t i = 0; i < output_batched.size(); i++) {
    std::vector<int64_t> ori_shape = output_unbatch[0][i].Shape();
    ori_shape.insert(ori_shape.begin(), output_unbatch.size());
    output_batched[i] = mindspore::MSTensor("outputs", output_unbatch[0][i].DataType(), ori_shape, nullptr,
                                            output_unbatch[0][i].DataSize() * output_unbatch.size());
  }

  // Copy unbatch data into tensor
  for (size_t i = 0; i < output_unbatch[0].size(); i++) {
    size_t offset = 0;
    for (size_t j = 0; j < output_unbatch.size(); j++) {
      auto ret =
        memcpy_s(reinterpret_cast<unsigned uint8_t *>(output_batched[i].MutableData()) + offset,
                 output_unbatch[j][i].DataSize(), output_unbatch[j][i].MutableData(), output_unbatch[j][i].DataSize());
      if (ret) {
        MS_LOG(ERROR) << "Memory copy failed to construct High-Dim Tensor.";
        return Status(kMEFailed, "Memory copy failed to construct High-Dim Tensor.");
      }
      offset += output_unbatch[j][i].DataSize();
    }
  }
  *outputs = output_batched;
  DLSoClose(handle);
  return kSuccess;
#else
  MS_LOG(ERROR) << "Data preprocess is not supported on Windows yet.";
  return Status(kMEFailed, "Data preprocess is not supported on Windows yet.");
#endif
}

Status ModelImpl::PredictWithPreprocess(const std::vector<std::vector<MSTensor>> &inputs,
                                        std::vector<MSTensor> *outputs) {
#if !defined(_WIN32) && !defined(_WIN64)
  if (session_ == nullptr) {
    MS_LOG(ERROR) << "Model has not been called Build, or Model Build has failed";
    return kLiteError;
  }
  // Run preprocess
  std::vector<MSTensor> preprocess_outputs;
  Status ret = Preprocess(inputs, &preprocess_outputs);
  if (ret != kSuccess) {
    return ret;
  }

  // Run prediction
  ret = Predict(preprocess_outputs, outputs);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "Run predict failed: " << ret.GetErrDescription();
    return ret;
  }
  return kSuccess;
#else
  MS_LOG(ERROR) << "Predict with data preprocess is not supported on Windows yet.";
  return Status(kMEFailed, "Predict with data preprocess is not supported on Windows yet.");
#endif
}

Status ModelImpl::LoadConfig(const std::string &config_path) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_) {
    MS_LOG(ERROR) << "Model has been called Build, please call LoadConfig before Build.";
    return kLiteError;
  }
  ConfigInfos all_config_info;
  int ret = lite::GetAllSectionInfoFromConfigFile(config_path, &all_config_info);
  if (ret != kSuccess) {
    MS_LOG(ERROR) << "GetAllSectionInfoFromConfigFile fail!ret: " << ret;
    return kLiteFileError;
  }
  config_info_ = all_config_info;
  return kSuccess;
}

Status ModelImpl::UpdateConfig(const std::string &section, const std::pair<std::string, std::string> &config) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (session_) {
    MS_LOG(ERROR) << "Model has been called Build, please call UpdateConfig before Build.";
    return kLiteError;
  }
  auto iter = config_info_.find(section);
  if (iter == config_info_.end()) {
    if (config_info_.size() >= kMaxSectionNum) {
      MS_LOG(ERROR) << "config too many sections!";
      return kLiteError;
    }
    config_info_[section][config.first] = config.second;
    return kSuccess;
  }
  if (iter->second.size() >= kMaxConfigNumPerSection) {
    MS_LOG(ERROR) << "config too many items!";
    return kLiteError;
  }
  iter->second[config.first] = config.second;
  return kSuccess;
}

std::string ModelImpl::GetConfig(const std::string &section, const std::string &key) {
  auto iter = config_info_.find(section);
  if (iter == config_info_.end()) {
    return "";
  }
  auto elem_iter = iter->second.find(key);
  if (elem_iter == iter->second.end()) {
    return "";
  }
  return elem_iter->second;
}

ModelImpl::~ModelImpl() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  FuncGraphReuseManager::GetInstance()->ReleaseSharedFuncGraph(config_info_);
  session_ = nullptr;
}

bool ModelImpl::CheckModelSupport(DeviceType device_type, ModelType model_type) {
  if (model_type != kMindIR) {
    return false;
  }
  if (device_type == kCPU) {
    return true;
  }
  if (device_type == kGPU) {
    return lite::TensorRTExecutorPlugin::GetInstance().TryRegister().IsOk();
  }
  if (device_type == kAscend) {
    return kernel::AscendKernelPlugin::TryRegister().IsOk();
  }
  return false;
}
}  // namespace mindspore
