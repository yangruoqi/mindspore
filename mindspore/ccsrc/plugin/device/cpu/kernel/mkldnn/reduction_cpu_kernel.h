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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_REDUCTION_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_REDUCTION_CPU_KERNEL_H_
#include <memory>
#include <vector>
#include <map>
#include <string>
#include "plugin/device/cpu/kernel/mkldnn/mkl_cpu_kernel.h"

namespace mindspore {
namespace kernel {
class ReductionCpuKernelMod : public MKLCpuKernelMod {
 public:
  ReductionCpuKernelMod() = default;
  explicit ReductionCpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~ReductionCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  // TO be Deprecated API.
  void InitKernel(const CNodePtr &kernel_node) override {}

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  using ReductionFunc = std::function<bool(ReductionCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                           const std::vector<kernel::AddressPtr> &)>;
  void MergeContinuousShape();
  void GetReductionAttr(const BaseOperatorPtr &base_operator);
  dnnl::reduction::desc GetReductionDesc(const dnnl::memory::desc &src_desc, const dnnl::memory::desc &dst_desc);

  ReductionFunc kernel_func_;
  float p_{2.0};
  float eps_{1e-12};
  std::vector<int64_t> axis_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  BaseOperatorPtr kernel_ptr_;
  static std::vector<std::pair<KernelAttr, ReductionFunc>> func_list_;
  std::string kernel_type_{};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_REDUCTION_CPU_KERNEL_H_
