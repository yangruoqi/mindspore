/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <iostream>
#include "src/common/log_adapter.h"
#include "common/common_test.h"
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel_exec.h"
#include "src/litert/tensor_category.h"
#include "src/litert/kernel/cpu/nnacl/nnacl_manager.h"

namespace mindspore {
class TestMatMulFp32 : public mindspore::CommonTest {
 public:
  TestMatMulFp32() {}
};

TEST_F(TestMatMulFp32, Row2Col8Test1) {
  float in[] = {0.21, 0.38, 0.81, 0.98, 0.09, 0.68, 0.02, 0.33, 0.85, 0.67, 0.81, 0.57, 0.70, 0.27, 0.90,
                0.07, 0.13, 0.03, 0.53, 0.97, 0.92, 0.35, 0.74, 0.78, 0.87, 0.23, 0.34, 0.09, 0.50, 0.39,
                0.09, 0.93, 0.91, 0.20, 0.97, 0.61, 0.43, 0.14, 0.67, 0.10, 0.73, 0.37, 0.24, 0.93, 0.31,
                0.35, 0.52, 0.02, 0.33, 0.99, 0.49, 0.67, 0.75, 0.66, 0.04, 0.10, 0.18, 0.92, 0.46, 0.08,
                0.04, 0.24, 0.52, 0.43, 0.14, 0.67, 0.10, 0.73, 0.37, 0.24, 0.93, 0.31, 0.35, 0.52, 0.02,
                0.33, 0.99, 0.49, 0.67, 0.75, 0.66, 0.04, 0.10, 0.18, 0.92, 0.46, 0.08, 0.04, 0.24, 0.52};
  float co[] = {0.21, 0.67, 0.53, 0.09, 0.43, 0.35, 0.04, 0.43, 0.38, 0.81, 0.97, 0.50, 0.14, 0.52, 0.10, 0.14,
                0.81, 0.57, 0.92, 0.39, 0.67, 0.02, 0.18, 0.67, 0.98, 0.70, 0.35, 0.09, 0.10, 0.33, 0.92, 0.10,
                0.09, 0.27, 0.74, 0.93, 0.73, 0.99, 0.46, 0.73, 0.68, 0.90, 0.78, 0.91, 0.37, 0.49, 0.08, 0.37,
                0.02, 0.07, 0.87, 0.20, 0.24, 0.67, 0.04, 0.24, 0.33, 0.13, 0.23, 0.97, 0.93, 0.75, 0.24, 0.93,
                0.85, 0.03, 0.34, 0.61, 0.31, 0.66, 0.52, 0.31, 0.35, 0.04, 0,    0,    0,    0,    0,    0,
                0.52, 0.10, 0,    0,    0,    0,    0,    0,    0.02, 0.18, 0,    0,    0,    0,    0,    0,
                0.33, 0.92, 0,    0,    0,    0,    0,    0,    0.99, 0.46, 0,    0,    0,    0,    0,    0,
                0.49, 0.08, 0,    0,    0,    0,    0,    0,    0.67, 0.04, 0,    0,    0,    0,    0,    0,
                0.75, 0.24, 0,    0,    0,    0,    0,    0,    0.66, 0.52, 0,    0,    0,    0,    0,    0};
  float out[144] = {0};
  RowMajor2Col8Major(in, out, 10, 9);
  ASSERT_EQ(0, CompareOutputData(out, co, 144, 0.0001));
}

TEST_F(TestMatMulFp32, Row2Col8Test2) {
  float in[] = {0.21, 0.38, 0.81, 0.98, 0.09, 0.68, 0.02, 0.33, 0.85, 0.67, 0.81, 0.57, 0.70, 0.27, 0.90,
                0.07, 0.13, 0.03, 0.53, 0.97, 0.92, 0.35, 0.74, 0.78, 0.87, 0.23, 0.34, 0.09, 0.50, 0.39,
                0.09, 0.93, 0.91, 0.20, 0.97, 0.61, 0.43, 0.14, 0.67, 0.10, 0.73, 0.37, 0.24, 0.93, 0.31,
                0.35, 0.52, 0.02, 0.33, 0.99, 0.49, 0.67, 0.75, 0.66, 0.04, 0.10, 0.18, 0.92, 0.46, 0.08,
                0.04, 0.24, 0.52, 0.43, 0.14, 0.67, 0.10, 0.73, 0.37, 0.24, 0.93, 0.31, 0.35, 0.52, 0.02,
                0.33, 0.99, 0.49, 0.67, 0.75, 0.66, 0.04, 0.10, 0.18, 0.92, 0.46, 0.08, 0.04, 0.24, 0.52};
  float co[] = {0.21, 0.68, 0.81, 0.07, 0.92, 0.23, 0.09, 0.61, 0.38, 0.02, 0.57, 0.13, 0.35, 0.34, 0.93,
                0.43, 0.81, 0.33, 0.70, 0.03, 0.74, 0.09, 0.91, 0.14, 0.98, 0.85, 0.27, 0.53, 0.78, 0.50,
                0.20, 0.67, 0.09, 0.67, 0.90, 0.97, 0.87, 0.39, 0.97, 0.10, 0.73, 0.35, 0.49, 0.10, 0.04,
                0.67, 0.93, 0.33, 0.37, 0.52, 0.67, 0.18, 0.24, 0.10, 0.31, 0.99, 0.24, 0.02, 0.75, 0.92,
                0.52, 0.73, 0.35, 0.49, 0.93, 0.33, 0.66, 0.46, 0.43, 0.37, 0.52, 0.67, 0.31, 0.99, 0.04,
                0.08, 0.14, 0.24, 0.02, 0.75, 0.66, 0.46, 0,    0,    0,    0,    0,    0,    0.04, 0.08,
                0,    0,    0,    0,    0,    0,    0.10, 0.04, 0,    0,    0,    0,    0,    0,    0.18,
                0.24, 0,    0,    0,    0,    0,    0,    0.92, 0.52, 0,    0,    0,    0,    0,    0};
  float out[120] = {0};
  RowMajor2Col8Major(in, out, 18, 5);
  ASSERT_EQ(0, CompareOutputData(out, co, 120, 0.0001));
}

int MMTestInit(std::vector<lite::Tensor *> *inputs_, std::vector<lite::Tensor *> *outputs_, float *a_ptr, float *b_ptr,
               const std::vector<int> &a_shape, const std::vector<int> &b_shape, const std::vector<int> &c_shape) {
  auto in_t = new lite::Tensor(kNumberTypeFloat32, a_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  in_t->MallocData();
  memcpy(in_t->MutableData(), a_ptr, sizeof(float) * in_t->ElementsNum());
  inputs_->push_back(in_t);

  auto weight_t = new lite::Tensor(kNumberTypeFloat32, b_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  weight_t->MallocData();
  memcpy(weight_t->MutableData(), b_ptr, sizeof(float) * weight_t->ElementsNum());
  inputs_->push_back(weight_t);

  auto out_t = new lite::Tensor(kNumberTypeFloat32, c_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  out_t->MallocData();
  outputs_->push_back(out_t);

  return out_t->ElementsNum();
}

int MMTestInit2(std::vector<lite::Tensor *> *inputs_, std::vector<lite::Tensor *> *outputs_, float *a_ptr, float *b_ptr,
                float *bias_ptr, const std::vector<int> &a_shape, const std::vector<int> &b_shape,
                const std::vector<int> &bias_shape, const std::vector<int> &c_shape) {
  auto in_t = new lite::Tensor(kNumberTypeFloat32, a_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  in_t->MallocData();
  memcpy(in_t->MutableData(), a_ptr, sizeof(float) * in_t->ElementsNum());
  inputs_->push_back(in_t);

  auto weight_t = new lite::Tensor(kNumberTypeFloat32, b_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  weight_t->MallocData();
  memcpy(weight_t->MutableData(), b_ptr, sizeof(float) * weight_t->ElementsNum());
  inputs_->push_back(weight_t);

  auto bias_t = new lite::Tensor(kNumberTypeFloat32, bias_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  bias_t->MallocData();
  memcpy(bias_t->MutableData(), bias_ptr, sizeof(float) * bias_t->ElementsNum());
  inputs_->push_back(bias_t);

  auto out_t = new lite::Tensor(kNumberTypeFloat32, c_shape, mindspore::NHWC, lite::Category::CONST_TENSOR);
  out_t->MallocData();
  outputs_->push_back(out_t);

  return out_t->ElementsNum();
}

TEST_F(TestMatMulFp32, simple) {
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  auto matmul_param = new MatMulParameter();
  matmul_param->a_transpose_ = false;
  matmul_param->b_transpose_ = false;
  matmul_param->has_bias_ = false;
  matmul_param->op_parameter_.thread_num_ = 1;
  matmul_param->op_parameter_.type_ = schema::PrimitiveType_MatMulFusion;
  float a[] = {-3.2366564, -4.7733846, -7.8329225, 16.146885, 5.060793,  -6.1471,  -1.7680453, -6.5721383,
               17.87506,   -5.1192183, 10.742863,  1.4536934, 19.693445, 19.45783, 5.063163,   0.5234792};
  float b[] = {-0.0024438887, 0.0006738146, -0.008169129, 0.0021510671,  -0.012470592,   -0.0053063435,
               0.006050155,   0.008656233,  0.012911413,  -0.0028635843, -0.00034080597, -0.0010622552,
               -0.012254699,  -0.01312836,  0.0025241964, -0.004706142,  0.002451482,    -0.009558459,
               0.004481974,   0.0033251503, -0.011705584, -0.001720293,  -0.0039410214,  -0.0073637343};
  std::vector<int> a_shape = {2, 8};
  std::vector<int> b_shape = {8, 3};
  std::vector<int> c_shape = {2, 3};
  int total_size = MMTestInit(&inputs_, &outputs_, a, b, a_shape, b_shape, c_shape);
  auto ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());

  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, schema::PrimitiveType_MatMulFusion};
  auto *mm = nnacl::NnaclKernelRegistry(reinterpret_cast<OpParameter *>(matmul_param), inputs_, outputs_, ctx, desc);
  mm->Prepare();
  mm->Run();
  float correct[] = {-0.1256939023733139, -0.07744802534580231,  0.07410638779401779,
                     -0.3049793541431427, -0.027687929570674896, -0.18109679222106934};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct, total_size, 0.0001));
  delete mm;
  delete ctx;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
}

TEST_F(TestMatMulFp32, simple_bias) {
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  auto matmul_param = new MatMulParameter();
  matmul_param->a_transpose_ = false;
  matmul_param->b_transpose_ = false;
  matmul_param->has_bias_ = false;
  matmul_param->op_parameter_.thread_num_ = 1;
  matmul_param->op_parameter_.type_ = schema::PrimitiveType_MatMulFusion;
  float a[] = {-3.2366564, -4.7733846, -7.8329225, 16.146885, 5.060793,  -6.1471,  -1.7680453, -6.5721383,
               17.87506,   -5.1192183, 10.742863,  1.4536934, 19.693445, 19.45783, 5.063163,   0.5234792};
  float b[] = {-0.0024438887, 0.0006738146, -0.008169129, 0.0021510671,  -0.012470592,   -0.0053063435,
               0.006050155,   0.008656233,  0.012911413,  -0.0028635843, -0.00034080597, -0.0010622552,
               -0.012254699,  -0.01312836,  0.0025241964, -0.004706142,  0.002451482,    -0.009558459,
               0.004481974,   0.0033251503, -0.011705584, -0.001720293,  -0.0039410214,  -0.0073637343};
  float bias[] = {1, 2, 3};
  std::vector<int> a_shape = {2, 8};
  std::vector<int> b_shape = {8, 3};
  std::vector<int> bias_shape = {1, 3};
  std::vector<int> c_shape = {2, 3};
  int total_size = MMTestInit2(&inputs_, &outputs_, a, b, bias, a_shape, b_shape, bias_shape, c_shape);
  auto ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, schema::PrimitiveType_MatMulFusion};
  auto *mm = nnacl::NnaclKernelRegistry(reinterpret_cast<OpParameter *>(matmul_param), inputs_, outputs_, ctx, desc);
  mm->Prepare();
  mm->Run();
  float correct[] = {-0.1256939023733139 + 1, -0.07744802534580231 + 2,  0.07410638779401779 + 3,
                     -0.3049793541431427 + 1, -0.027687929570674896 + 2, -0.18109679222106934 + 3};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct, total_size, 0.0001));
  delete mm;
  delete ctx;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
}

TEST_F(TestMatMulFp32, simple2) {
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  auto matmul_param = new MatMulParameter();
  matmul_param->a_transpose_ = false;
  matmul_param->b_transpose_ = false;
  matmul_param->has_bias_ = false;
  matmul_param->op_parameter_.thread_num_ = 1;
  matmul_param->op_parameter_.type_ = schema::PrimitiveType_MatMulFusion;
  float a[25 * 12] = {
    1, 4,  10, 2,  3,  10, 4,  6,  5,  6,  9,  5,  4,  2,  5, 7,  5,  8,  0, 5, 1, 0,  10, 3,  0,  4,  2, 3, 2,  9,
    8, 9,  5,  4,  4,  9,  7,  4,  2,  6,  10, 2,  1,  7,  2, 10, 5,  10, 1, 2, 2, 9,  8,  8,  2,  5,  6, 3, 2,  8,
    3, 3,  7,  3,  0,  4,  10, 9,  0,  5,  2,  6,  1,  10, 7, 6,  9,  6,  0, 3, 8, 0,  8,  3,  10, 4,  7, 7, 0,  5,
    6, 5,  4,  6,  5,  5,  3,  7,  1,  9,  3,  2,  8,  3,  0, 0,  6,  7,  6, 3, 6, 5,  1,  0,  4,  2,  6, 0, 7,  7,
    7, 4,  9,  8,  6,  1,  10, 10, 7,  3,  0,  6,  9,  4,  1, 4,  4,  3,  1, 6, 7, 3,  8,  6,  4,  10, 9, 8, 10, 5,
    2, 3,  8,  10, 0,  8,  2,  9,  5,  3,  3,  0,  1,  8,  1, 1,  2,  0,  1, 5, 5, 0,  1,  10, 9,  9,  3, 6, 7,  1,
    2, 3,  7,  5,  0,  8,  2,  8,  7,  8,  9,  10, 4,  2,  5, 3,  10, 1,  5, 0, 6, 2,  3,  5,  5,  1,  5, 5, 5,  1,
    8, 2,  6,  9,  10, 4,  9,  1,  10, 9,  8,  2,  5,  2,  4, 2,  3,  7,  7, 2, 9, 10, 10, 10, 5,  1,  8, 8, 10, 3,
    2, 10, 2,  6,  5,  9,  10, 6,  10, 0,  5,  5,  4,  0,  9, 4,  4,  9,  4, 6, 4, 2,  5,  2,  10, 5,  9, 8, 1,  4,
    7, 9,  6,  5,  0,  3,  6,  4,  3,  10, 6,  4,  10, 5,  8, 8,  9,  4,  5, 6, 8, 9,  2,  2,  4,  4,  8, 0, 4,  5};
  float b[12 * 36] = {
    6,  6, 7,  2,  1,  10, 3,  7,  7,  5,  5,  5,  6,  6,  9,  8,  4,  10, 9,  5,  5,  7,  2,  1, 7,  9,  10, 0, 3,
    10, 4, 2,  7,  4,  3,  10, 5,  3,  1,  3,  3,  1,  9,  6,  7,  6,  6,  6,  7,  6,  10, 8,  2, 8,  5,  2,  1, 7,
    5,  9, 10, 9,  0,  8,  10, 2,  3,  4,  0,  7,  5,  5,  0,  9,  6,  1,  6,  7,  4,  1,  0,  3, 0,  7,  3,  0, 10,
    7,  6, 4,  10, 7,  6,  5,  10, 2,  10, 9,  10, 6,  9,  10, 8,  8,  5,  3,  9,  10, 8,  3,  3, 4,  6,  2,  6, 0,
    4,  0, 3,  4,  1,  0,  3,  10, 5,  4,  0,  2,  3,  2,  4,  3,  10, 5,  4,  10, 8,  2,  0,  4, 0,  5,  8,  0, 1,
    10, 0, 3,  1,  1,  9,  4,  0,  3,  0,  1,  6,  3,  10, 0,  10, 3,  3,  0,  6,  7,  3,  2,  3, 5,  10, 6,  1, 5,
    7,  3, 3,  1,  1,  10, 5,  4,  0,  8,  4,  0,  9,  6,  2,  3,  6,  10, 10, 0,  2,  2,  1,  2, 7,  10, 9,  7, 10,
    2,  8, 5,  3,  7,  0,  4,  3,  4,  8,  3,  8,  0,  5,  5,  6,  9,  10, 0,  1,  5,  6,  6,  4, 7,  7,  6,  7, 9,
    5,  5, 6,  0,  4,  1,  2,  6,  8,  4,  10, 4,  10, 9,  8,  8,  1,  7,  1,  8,  1,  0,  10, 8, 8,  1,  8,  0, 10,
    3,  1, 7,  0,  10, 5,  0,  2,  8,  4,  1,  8,  1,  6,  7,  1,  8,  3,  4,  3,  4,  7,  0,  9, 1,  1,  4,  8, 10,
    0,  3, 3,  2,  7,  9,  3,  3,  10, 10, 9,  4,  4,  0,  7,  1,  1,  3,  5,  1,  4,  8,  5,  7, 3,  9,  10, 1, 5,
    9,  7, 4,  10, 10, 3,  4,  3,  5,  1,  10, 5,  2,  3,  3,  0,  3,  1,  2,  8,  7,  4,  2,  0, 8,  7,  6,  6, 6,
    5,  7, 5,  5,  3,  0,  4,  10, 1,  7,  8,  9,  6,  7,  0,  1,  9,  3,  1,  6,  8,  4,  9,  0, 3,  2,  4,  0, 2,
    7,  2, 2,  8,  0,  4,  1,  3,  2,  6,  8,  5,  5,  2,  3,  9,  0,  1,  7,  6,  9,  1,  10, 4, 10, 5,  10, 0, 9,
    5,  1, 6,  2,  9,  9,  8,  8,  10, 8,  1,  6,  5,  8,  8,  6,  4,  8,  10, 3,  0,  6,  2,  8, 4,  2};
  std::vector<int> a_shape = {25, 12};
  std::vector<int> b_shape = {12, 36};
  std::vector<int> c_shape = {25, 36};
  int total_size = MMTestInit(&inputs_, &outputs_, a, b, a_shape, b_shape, c_shape);
  auto ctx = new lite::InnerContext;
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, schema::PrimitiveType_MatMulFusion};
  auto *mm = nnacl::NnaclKernelRegistry(reinterpret_cast<OpParameter *>(matmul_param), inputs_, outputs_, ctx, desc);
  mm->Prepare();
  mm->Run();
  float correct[] = {
    263, 386, 184, 309, 338, 244, 359, 294, 252, 254, 273, 353, 320, 183, 412, 273, 271, 307, 329, 314, 391, 261, 400,
    280, 416, 399, 355, 427, 373, 302, 288, 349, 336, 241, 349, 393, 226, 285, 134, 209, 264, 163, 281, 212, 219, 171,
    221, 228, 227, 131, 289, 196, 204, 270, 238, 205, 303, 196, 280, 156, 311, 284, 282, 335, 243, 245, 181, 188, 280,
    142, 229, 256, 270, 310, 184, 377, 323, 187, 345, 295, 255, 262, 259, 332, 310, 222, 357, 275, 253, 301, 296, 254,
    316, 221, 323, 322, 370, 353, 281, 386, 363, 240, 245, 301, 270, 263, 275, 292, 278, 388, 199, 324, 252, 336, 385,
    300, 257, 274, 215, 243, 272, 230, 485, 335, 343, 366, 293, 272, 337, 313, 310, 305, 385, 421, 377, 398, 343, 262,
    249, 309, 258, 280, 286, 411, 268, 337, 127, 307, 244, 185, 368, 263, 178, 205, 223, 281, 288, 154, 339, 255, 295,
    250, 241, 236, 289, 240, 296, 261, 361, 333, 282, 399, 315, 202, 203, 272, 231, 229, 300, 273, 199, 253, 246, 315,
    307, 213, 257, 202, 243, 230, 163, 288, 220, 212, 361, 314, 219, 296, 300, 217, 274, 196, 285, 264, 351, 339, 312,
    289, 338, 282, 256, 274, 214, 243, 228, 302, 276, 394, 110, 224, 274, 163, 395, 296, 231, 223, 289, 311, 331, 177,
    405, 236, 294, 293, 264, 213, 314, 258, 330, 270, 403, 381, 305, 450, 382, 250, 248, 287, 278, 211, 324, 374, 306,
    350, 246, 298, 309, 305, 315, 289, 292, 256, 264, 341, 295, 218, 427, 382, 272, 359, 335, 286, 333, 263, 327, 275,
    448, 423, 380, 369, 397, 330, 260, 329, 285, 284, 333, 397, 259, 258, 146, 261, 281, 156, 248, 234, 236, 219, 220,
    207, 233, 173, 326, 316, 223, 301, 237, 145, 202, 181, 209, 236, 357, 279, 265, 332, 352, 230, 165, 219, 154, 233,
    189, 237, 246, 316, 147, 197, 247, 221, 212, 256, 201, 208, 239, 220, 231, 153, 322, 263, 237, 278, 254, 178, 215,
    164, 217, 211, 326, 295, 284, 306, 354, 247, 178, 244, 216, 199, 229, 308, 298, 409, 306, 359, 359, 273, 388, 291,
    301, 281, 239, 395, 323, 290, 505, 398, 370, 381, 365, 235, 344, 268, 340, 351, 473, 481, 445, 415, 481, 373, 354,
    365, 284, 309, 338, 469, 285, 336, 166, 244, 245, 247, 305, 304, 273, 233, 281, 260, 276, 218, 364, 241, 255, 330,
    257, 213, 296, 221, 252, 251, 325, 355, 301, 341, 319, 246, 206, 243, 295, 210, 249, 357, 328, 481, 196, 345, 276,
    338, 493, 349, 236, 299, 265, 388, 383, 224, 573, 425, 411, 354, 353, 340, 363, 385, 414, 387, 541, 528, 412, 515,
    486, 298, 320, 438, 254, 361, 454, 494, 120, 156, 151, 140, 176, 99,  231, 113, 197, 132, 113, 190, 134, 171, 264,
    169, 137, 219, 165, 92,  172, 145, 188, 186, 225, 260, 166, 216, 225, 161, 173, 134, 147, 130, 152, 218, 226, 273,
    205, 314, 331, 157, 311, 242, 289, 228, 238, 346, 285, 223, 344, 235, 194, 282, 274, 238, 358, 207, 333, 270, 345,
    345, 302, 339, 309, 273, 284, 291, 297, 219, 261, 338, 319, 396, 200, 356, 349, 311, 377, 330, 280, 280, 308, 351,
    311, 204, 421, 319, 294, 348, 328, 346, 387, 261, 403, 335, 434, 428, 333, 467, 422, 270, 254, 370, 345, 285, 381,
    378, 200, 347, 110, 195, 189, 184, 252, 242, 134, 191, 179, 205, 256, 140, 349, 219, 287, 216, 225, 155, 223, 203,
    203, 196, 295, 281, 321, 291, 292, 235, 219, 255, 177, 186, 213, 349, 286, 389, 180, 262, 306, 275, 269, 284, 257,
    239, 256, 262, 270, 189, 410, 306, 302, 297, 244, 226, 335, 213, 276, 257, 371, 351, 398, 376, 378, 289, 265, 355,
    258, 252, 286, 446, 274, 419, 214, 263, 277, 296, 317, 276, 202, 240, 214, 287, 292, 174, 454, 366, 352, 328, 342,
    247, 300, 273, 300, 232, 440, 401, 436, 374, 394, 351, 269, 317, 247, 255, 312, 416, 384, 533, 202, 336, 369, 322,
    449, 373, 291, 282, 343, 409, 416, 198, 526, 383, 405, 363, 355, 355, 478, 348, 435, 296, 544, 490, 519, 540, 449,
    390, 345, 444, 378, 307, 454, 542, 356, 394, 179, 370, 364, 152, 424, 370, 316, 291, 358, 420, 419, 267, 429, 323,
    311, 348, 320, 232, 344, 260, 344, 369, 472, 424, 339, 479, 470, 297, 298, 350, 300, 302, 340, 389, 211, 314, 186,
    248, 277, 184, 294, 217, 204, 184, 203, 311, 262, 154, 324, 221, 233, 249, 283, 241, 331, 210, 318, 191, 341, 330,
    331, 323, 278, 289, 255, 259, 294, 174, 280, 323, 295, 348, 303, 319, 321, 286, 365, 266, 310, 251, 240, 406, 302,
    265, 457, 396, 297, 366, 350, 270, 343, 271, 347, 314, 469, 476, 396, 375, 428, 351, 315, 341, 291, 296, 361, 428,
    383, 442, 232, 360, 387, 279, 391, 349, 348, 288, 334, 374, 360, 262, 485, 391, 362, 379, 296, 262, 406, 270, 346,
    346, 486, 451, 451, 490, 475, 339, 319, 409, 315, 324, 367, 493, 286, 348, 185, 240, 287, 214, 312, 265, 237, 218,
    261, 316, 279, 186, 377, 319, 279, 304, 281, 207, 261, 209, 287, 270, 415, 378, 312, 388, 423, 273, 230, 294, 239,
    243, 319, 346};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct, total_size, 0.0001));
  delete mm;
  delete ctx;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
}

TEST_F(TestMatMulFp32, simple_transb) {
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  auto matmul_param = new MatMulParameter();
  matmul_param->a_transpose_ = false;
  matmul_param->b_transpose_ = true;
  matmul_param->has_bias_ = false;
  matmul_param->op_parameter_.thread_num_ = 1;
  matmul_param->op_parameter_.type_ = schema::PrimitiveType_MatMulFusion;
  float a[] = {-3.2366564, -4.7733846, -7.8329225, 16.146885, 5.060793,  -6.1471,  -1.7680453, -6.5721383,
               17.87506,   -5.1192183, 10.742863,  1.4536934, 19.693445, 19.45783, 5.063163,   0.5234792};
  float b[] = {-0.0024438887, 0.0006738146, -0.008169129, 0.0021510671,  -0.012470592,   -0.0053063435,
               0.006050155,   0.008656233,  0.012911413,  -0.0028635843, -0.00034080597, -0.0010622552,
               -0.012254699,  -0.01312836,  0.0025241964, -0.004706142,  0.002451482,    -0.009558459,
               0.004481974,   0.0033251503, -0.011705584, -0.001720293,  -0.0039410214,  -0.0073637343};
  std::vector<int> a_shape = {1, 2, 8};
  std::vector<int> b_shape = {1, 3, 8};
  std::vector<int> c_shape = {1, 2, 3};
  int total_size = MMTestInit(&inputs_, &outputs_, a, b, a_shape, b_shape, c_shape);
  auto ctx = new lite::InnerContext;
  ctx->thread_num_ = 2;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, schema::PrimitiveType_MatMulFusion};
  auto *mm = nnacl::NnaclKernelRegistry(reinterpret_cast<OpParameter *>(matmul_param), inputs_, outputs_, ctx, desc);
  mm->Prepare();
  mm->Run();
  float correct[] = {0.00533547, 0.002545945, 0.062974121, -0.445441471, -0.246223617, -0.142070031};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct, total_size, 0.0001));
  delete mm;
  delete ctx;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
}

TEST_F(TestMatMulFp32, batch) {
  std::vector<lite::Tensor *> inputs_;
  std::vector<lite::Tensor *> outputs_;
  auto matmul_param = new MatMulParameter();
  matmul_param->a_transpose_ = false;
  matmul_param->b_transpose_ = true;
  matmul_param->has_bias_ = false;
  matmul_param->op_parameter_.thread_num_ = 1;
  matmul_param->op_parameter_.type_ = schema::PrimitiveType_MatMulFusion;
  float a[] = {-4.946672525326248,  11.154420027909701,  -7.831129637356922,  17.309845099949953,  -10.46177877610444,
               2.5412751480833897,  2.700113860276929,   -12.616715572097341, -15.513316568881574, -9.513294738065516,
               17.931148376418896,  -10.83801964632579,  -14.023733862948017, -14.50805001403956,  0.7952221556310306,
               6.619720423569035,   -19.277904230909357, -13.450479287024839, 19.914652156692625,  16.542571697048878,
               -2.9715041389268926, 4.949555349889412,   -1.9408110276290103, -15.062828261031868, 0.20012569643335,
               8.260383531209776,   3.1092344458607357,  16.742272486091487,  17.31277252415167,   -16.60303202099434,
               -8.980314693173042,  -11.735087989358268, -14.918976184088514, -11.347592686892733, 11.808756029220604,
               -18.76179414554809,  7.579758962360987,   3.13240880962163,    6.528181981442103,   -16.802624652419794,
               -14.323146919914901, -16.197579076296144, 9.738053920125779,   -12.245780062949866, 8.817905278096319,
               0.5261391331275007,  -18.26152522535471,  -2.400461208771226};
  float b[] = {
    -0.895183867395529,    -0.8146900207660068,   -0.27931593219652817,  0.783554361201179,     -0.05080215007779798,
    -0.9879631271568501,   0.07710949009001333,   -0.9562579726211344,   0.29505553318356825,   -0.26651960351085124,
    -0.12755456259718279,  -0.8221417897250098,   -0.5094334041431876,   -0.9117373380256013,   0.991501784215064,
    0.20131976450979394,   0.07889260559412059,   -0.8138407752750305,   -0.047622075866657454, -0.2778043115153188,
    -0.6269973420163957,   -0.44345812666611617,  -0.8571568605933642,   0.020192166011526735,  0.4860054298402434,
    0.41525925469513614,   -0.40270506445219967,  -0.8716538067535347,   0.5276448387223114,    0.6064500154192936,
    -0.9553204135772526,   0.3253219646257437,    -0.7237956595774822,   0.3271284879679077,    -0.534543967339336,
    -0.4076498484281894,   0.01574797075171963,   -0.37322004720586244,  0.16425071396119928,   -0.5328652244800547,
    0.7389336170615435,    -0.6552069958923377,   -0.042305872596973604, -0.6714941466767734,   -0.9281411415119043,
    -0.7748558258281224,   -0.6209799945964443,   0.02526428593887675,   -0.44984776800225856,  0.6281401952319337,
    0.9907258228680276,    0.6288646615999687,    -0.82076880150175,     0.3065944740797497,    -0.29201038744043584,
    -0.025685501802048982, -0.07273175145419652,  0.9370449239208709,    -0.8233807408078093,   -0.4195634619023012,
    0.9799555630257346,    -0.23461882935715228,  -0.8884793313829993,   -0.4760267734754635,   -0.2874539543614072,
    -0.8795685985480997,   -0.08099698251915255,  -0.1626521023321741,   -0.9337167240793414,   0.40924842916829207,
    -0.7375713045221615,   -0.0065659291539015285};
  std::vector<int> a_shape = {3, 2, 8};
  std::vector<int> b_shape = {3, 3, 8};
  std::vector<int> c_shape = {3, 2, 3};
  int total_size = MMTestInit(&inputs_, &outputs_, a, b, a_shape, b_shape, c_shape);
  auto ctx = new lite::InnerContext;
  ctx->thread_num_ = 1;
  ASSERT_EQ(lite::RET_OK, ctx->Init());
  kernel::KernelKey desc = {kernel::KERNEL_ARCH::kCPU, kNumberTypeFloat32, NHWC, schema::PrimitiveType_MatMulFusion};
  auto *mm = nnacl::NnaclKernelRegistry(reinterpret_cast<OpParameter *>(matmul_param), inputs_, outputs_, ctx, desc);
  mm->Prepare();
  mm->Run();
  float correct[] = {21.38518524169922,  -14.514888763427734, -11.040614128112793, 16.91403579711914,
                     27.07421112060547,  23.35394287109375,   -39.006141662597656, -2.021998405456543,
                     -17.63555145263672, -8.490625381469727,  5.317771911621094,   -14.561882019042969,
                     -7.251564025878906, -2.508212089538574,  5.86458683013916,    -3.466249465942383,
                     8.869029998779297,  25.034008026123047};
  ASSERT_EQ(0, CompareOutputData(reinterpret_cast<float *>(outputs_[0]->MutableData()), correct, total_size, 0.0001));
  delete mm;
  delete ctx;
  for (auto t : inputs_) delete t;
  for (auto t : outputs_) delete t;
}
}  // namespace mindspore
