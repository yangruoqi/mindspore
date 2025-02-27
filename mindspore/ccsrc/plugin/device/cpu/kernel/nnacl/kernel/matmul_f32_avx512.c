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

#ifdef ENABLE_AVX512
#include "nnacl/kernel/matmul_f32_avx512.h"
#include "nnacl/kernel/matmul_f32_base.h"
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/fp32/matmul_fp32.h"
#include "nnacl/fp32/matmul_avx512_fp32.h"
#include "nnacl/fp32/matmul_avx512_mask_fp32.h"

#define MIN_CALC_COST 24576 /* 1 x 6 x 64x 64 */

void MatmulFp32Avx512_BatchRowThreadCut(MatmulFp32Struct *matmul) {
  // BatchCut
  matmul->batch_stride_ = DOWN_DIV(matmul->batch_, matmul->base_.thread_nr_);

  // RowCut
  int row_step = MSMAX(matmul->row_ / matmul->base_.thread_nr_, matmul->row_min_unit_);
  int row_remaining = matmul->row_ - row_step * matmul->base_.thread_nr_;

  matmul->row_split_points_size_ = 0;
  int row_split_point = 0;
  while (row_split_point < matmul->row_) {
    matmul->row_split_points_[matmul->row_split_points_size_++] = row_split_point;
    row_split_point += row_step;
    if (row_remaining > 0) {
      ++row_split_point;
      --row_remaining;
    }
  }
  matmul->row_split_points_[matmul->row_split_points_size_] = matmul->row_;
  if (matmul->batch_stride_ == 0) {
    matmul->base_.thread_nr_ = matmul->row_split_points_size_;
  }
}
void MatmulFp32Avx512_BatchColThreadCut(MatmulFp32Struct *matmul) {
  // BatchCut
  matmul->batch_stride_ = DOWN_DIV(matmul->batch_, matmul->base_.thread_nr_);

  // ColCut
  int total_col_unit = UP_DIV(matmul->col_align_, matmul->col_min_unit_);
  int thread_num_tmp = MSMIN(matmul->base_.thread_nr_, total_col_unit);
  int block_col_unit = UP_DIV(total_col_unit, thread_num_tmp);
  int count = 0;
  int split_point = 0;
  while (split_point < total_col_unit) {
    matmul->col_split_points_[count++] = split_point * matmul->col_min_unit_;
    split_point += block_col_unit;
  }
  if (matmul->batch_stride_ == 0) {
    matmul->base_.thread_nr_ = count;
  }
}
void MatmulFp32Avx512_BatchColRowSliceThreadCut(MatmulFp32Struct *matmul) {
  // BatchCut
  matmul->batch_stride_ = DOWN_DIV(matmul->batch_, matmul->base_.thread_nr_);

  int row_s = 0;
  int row_e = matmul->row_;
  int col_s = 0;
  int col_e = matmul->col_;

  // ColCut
  int total_col_unit = UP_DIV(matmul->col_align_, matmul->col_min_unit_);
  matmul->block_col_unit_ = DOWN_DIV(total_col_unit, matmul->base_.thread_nr_);
  matmul->col_split_points_size_ = 1;
  matmul->col_split_points_[0] = 0;
  if (matmul->block_col_unit_ > 0) {
    int col_split_point = 0;
    for (int i = 0; i < matmul->base_.thread_nr_; i++) {
      MatmulSlice matmul_slice;
      matmul_slice.row_s_ = row_s;
      matmul_slice.row_e_ = row_e;
      matmul_slice.col_s_ = col_split_point * matmul->col_min_unit_;
      col_split_point += matmul->block_col_unit_;
      col_s = MSMIN(col_split_point * matmul->col_min_unit_, matmul->col_step_);
      matmul_slice.col_e_ = col_s;
      matmul->matmul_slice_set_[i][matmul->matmul_slice_count_[i]++] = matmul_slice;
    }
  }
  if (col_e - col_s <= 0) {
    return;
  }

  // RowColCut
  int row_thread = 0;
  int less_col_align = UP_ROUND(col_e - col_s, C16NUM);
  bool use_colrowcut_flag = ((less_col_align / C64NUM) * C64NUM) == less_col_align;
  bool use_rowcut_flag = matmul->row_ >= C6NUM * matmul->base_.thread_nr_ || col_e - col_s <= C64NUM;
  if (use_rowcut_flag && !use_colrowcut_flag) {
    int row_step = MSMAX(matmul->row_ / matmul->base_.thread_nr_, matmul->row_min_unit_);
    int row_remaining = matmul->row_ - row_step * matmul->base_.thread_nr_;
    int row_split_point = 0;

    for (row_thread = 0; row_thread < matmul->base_.thread_nr_ && row_split_point < matmul->row_; row_thread++) {
      MatmulSlice matmul_slice;
      matmul_slice.row_s_ = row_split_point;

      row_split_point += row_step;
      if (row_remaining > 0) {
        ++row_split_point;
        --row_remaining;
      }

      matmul_slice.row_e_ = row_split_point;
      matmul_slice.col_s_ = col_s;
      matmul_slice.col_e_ = col_e;
      matmul->matmul_slice_set_[row_thread][matmul->matmul_slice_count_[row_thread]++] = matmul_slice;
    }
  } else {
    int col_num = UP_DIV(col_e - col_s, C64NUM);
    int row_num = MSMIN(UP_DIV(matmul->base_.thread_nr_, col_num), (row_e - row_s));
    int tile_remaining = MSMAX(col_num * row_num - matmul->base_.thread_nr_, 0);

    NNACL_CHECK_ZERO_RETURN(row_num);
    int row_step = (row_e - row_s) / row_num;
    int row_remaining_tmp = (row_e - row_s) - row_step * row_num;

    int row_step_cut2 = (row_num == 1) ? row_step : (row_e - row_s) / (row_num - 1);
    int row_remaining_cut2_tmp = (row_e - row_s) - row_step_cut2 * (row_num - 1);

    MatmulSlice matmul_slice;
    for (int c = 0; c < col_num; c++) {
      matmul_slice.col_s_ = col_s + c * C64NUM;
      matmul_slice.col_e_ = MSMIN(col_s + (c + 1) * C64NUM, matmul->col_);
      int row_split_point = 0;
      int row_remaining = row_remaining_tmp;
      int row_remaining_cut2 = row_remaining_cut2_tmp;
      if (c < col_num - tile_remaining) {
        for (int r = 0; r < row_num; r++) {
          matmul_slice.row_s_ = row_split_point;
          row_split_point += row_step;
          if (row_remaining > 0) {
            ++row_split_point;
            --row_remaining;
          }
          matmul_slice.row_e_ = MSMIN(row_split_point, matmul->row_);
          matmul->matmul_slice_set_[row_thread][matmul->matmul_slice_count_[row_thread]++] = matmul_slice;
          row_thread++;
        }
      } else {
        for (int r = 0; r < row_num - 1; r++) {
          matmul_slice.row_s_ = row_split_point;
          row_split_point += row_step_cut2;
          if (row_remaining_cut2 > 0) {
            ++row_split_point;
            --row_remaining_cut2;
          }
          matmul_slice.row_e_ = MSMIN(row_split_point, matmul->row_);
          matmul->matmul_slice_set_[row_thread][matmul->matmul_slice_count_[row_thread]++] = matmul_slice;
          row_thread++;
        }
      }
    }
  }
  if ((matmul->batch_stride_ == 0) && (matmul->block_col_unit_ == 0)) {
    matmul->base_.thread_nr_ = row_thread;
  }
}

void MatmulFp32Avx512_GetThreadCuttingPolicy(MatmulFp32Struct *matmul) {
  size_t total_cost =
    (size_t)(matmul->batch_) * (size_t)(matmul->row_) * (size_t)(matmul->col_) * (size_t)(matmul->deep_);

  // Thread Update
  matmul->base_.thread_nr_ = MSMAX(MSMIN((int)(total_cost / MIN_CALC_COST), matmul->base_.thread_nr_), C1NUM);

  if (matmul->deep_ < C128NUM) {
    return MatmulFp32Base_GetThreadCuttingPolicy(matmul);
  }

  for (int i = 0; i < SPLIT_COUNT; i++) {
    matmul->matmul_slice_count_[i] = 0;
  }
  if (matmul->col_ == 1 && !matmul->a_const_) {
    MatmulFp32Avx512_BatchRowThreadCut(matmul);
    if (matmul->deep_ == 1) {
      matmul->gemm_not_pack_fun_ = GemmIsNotPack;
    } else {
      matmul->gemm_not_pack_fun_ = GemmIsNotPackOptimize;
    }
    matmul->parallel_run_ = matmul->parallel_run_by_gepdot_;
  } else if (matmul->row_ == 1 && !matmul->b_const_) {
    MatmulFp32Avx512_BatchColThreadCut(matmul);
    if (matmul->deep_ == 1) {
      matmul->parallel_run_ = matmul->parallel_run_by_row1_deep1_gepdot_;
      if (matmul->matrix_c_.pack_ptr_ != NULL) {
        matmul->gemm_not_pack_fun_ = Row1Deep1GemmIsNotPack;
      } else {
        matmul->gemm_not_pack_fun_ = Row1Deep1NoBiasGemmIsNotPack;
      }
      return;
    }
    matmul->parallel_run_ = matmul->parallel_run_by_gepm_;
  } else {
    MatmulFp32Avx512_BatchColRowSliceThreadCut(matmul);
    matmul->parallel_run_ = matmul->parallel_run_by_batch_col_row_gemm_;
  }
  return;
}

bool MatmulFp32Avx512_CheckThreadCuttingByRow(MatmulFp32Struct *matmul) {
  if (matmul->b_batch_ != C1NUM) {
    return false;
  }
  if (matmul->row_num_ < matmul->base_.thread_nr_) {
    return false;
  }
  if (matmul->col_ == 1) {
    matmul->row_min_unit_ = C8NUM;
    return true;
  }
  if (matmul->row_ == 1 && !matmul->b_const_ && matmul->col_ <= C128NUM) {
    return false;
  }
  matmul->row_min_unit_ = C6NUM;
  if (matmul->col_step_ < C48NUM) {
    matmul->row_min_unit_ = C12NUM;
  } else if (matmul->col_step_ < C64NUM) {
    matmul->row_min_unit_ = C8NUM;
  }
  return MSMIN(matmul->row_num_ / matmul->row_min_unit_, matmul->base_.thread_nr_) >
         MSMIN(matmul->col_step_ / matmul->col_min_unit_, matmul->base_.thread_nr_);
}
void MatmulFp32Avx512_InitGlobalVariable(MatmulFp32Struct *matmul) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  matmul->matrix_a_pack_fun_ = param->a_transpose_ ? RowMajor2ColMajorParallel : RowMajor2RowMajorParallel;
  matmul->matrix_b_pack_fun_ = param->b_transpose_ ? RowMajor2Col64MajorParallel : RowMajor2Row64MajorParallel;
  matmul->matrix_a_.need_pack_ = param->a_transpose_;
  matmul->matrix_b_.need_pack_ = true;
  matmul->row_tile_ = C1NUM;
  matmul->col_tile_ = C16NUM;
  matmul->col_min_unit_ = C64NUM;

  if (matmul->row_ == 1) {
    if (!matmul->b_const_ && matmul->col_ <= C128NUM) {
      matmul->out_need_aligned_ = true;
    }
  } else if (matmul->col_ == 1) {
    matmul->out_need_aligned_ = true;
  } else {
    matmul->out_need_aligned_ = false;
  }

  if (matmul->deep_ >= C128NUM) {
    matmul->out_need_aligned_ = false;
  }
}
int MatmulFp32Avx512_InitParameter(MatmulFp32Struct *matmul) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  if (matmul->deep_ < C128NUM) {
    return MatmulFp32Base_InitParameter(matmul);
  }
  matmul->init_global_varibale_(matmul);
  if (matmul->col_ == 1 && !matmul->a_const_) {
    matmul->out_need_aligned_ = false;
    matmul->row_tile_ = 1;
    matmul->col_tile_ = 1;
    matmul->matrix_a_pack_fun_ = param->a_transpose_ ? RowMajor2ColMajorParallel : RowMajor2RowMajorParallel;
    matmul->matrix_b_pack_fun_ = param->b_transpose_ ? RowMajor2ColMajorParallel : RowMajor2RowMajorParallel;
    matmul->matrix_a_.need_pack_ = param->a_transpose_ && matmul->row_ != 1;
    matmul->matrix_b_.need_pack_ = false;
    matmul->pack_opt_ = false;
  } else if (matmul->row_ == 1 && !matmul->b_const_) {
    matmul->out_need_aligned_ = false;
    matmul->row_tile_ = 1;
    matmul->col_tile_ = 1;
    matmul->matrix_a_pack_fun_ = param->a_transpose_ ? RowMajor2ColMajorParallel : RowMajor2RowMajorParallel;
    matmul->matrix_b_pack_fun_ = param->b_transpose_ ? RowMajor2ColMajorParallel : RowMajor2RowMajorParallel;
    matmul->matrix_a_.need_pack_ = false;
    matmul->matrix_b_.need_pack_ = param->b_transpose_;
    matmul->pack_opt_ = false;
  }
  matmul->row_align_ = UP_ROUND(matmul->row_, matmul->row_tile_);
  matmul->col_align_ = UP_ROUND(matmul->col_, matmul->col_tile_);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(matmul->a_batch_, matmul->row_align_, NNACL_ERR);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(matmul->a_batch_ * matmul->row_align_, matmul->deep_, NNACL_ERR);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(matmul->a_batch_, matmul->col_align_, NNACL_ERR);
  MS_CHECK_INT_MUL_NOT_OVERFLOW(matmul->a_batch_ * matmul->col_align_, matmul->deep_, NNACL_ERR);
  int a_pack_size = matmul->a_batch_ * matmul->row_align_ * matmul->deep_;
  int b_pack_size = matmul->b_batch_ * matmul->col_align_ * matmul->deep_;
  if ((matmul->matrix_a_.has_packed_ && matmul->matrix_a_.pack_size_ != a_pack_size) ||
      (matmul->matrix_b_.has_packed_ && matmul->matrix_b_.pack_size_ != b_pack_size)) {
    return NNACL_ERR;
  }
  matmul->matrix_a_.pack_size_ = a_pack_size;
  matmul->matrix_b_.pack_size_ = b_pack_size;
  matmul->row_align_ = UP_ROUND(matmul->row_, matmul->row_tile_);
  matmul->out_need_aligned_ = (matmul->out_need_aligned_ && ((matmul->col_ % matmul->col_tile_) != 0));
  matmul->col_step_ = matmul->out_need_aligned_ ? matmul->col_align_ : matmul->col_;
  MS_CHECK_FALSE(INT_MUL_OVERFLOW(matmul->a_batch_, matmul->row_), NNACL_ERR);
  matmul->row_num_ = matmul->a_batch_ * matmul->row_;
  return NNACL_OK;
}

int MatmulFp32Avx512_ParallelRunByRow(MatmulFp32Struct *matmul, int task_id) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  MS_CHECK_FALSE(task_id < 0 || task_id >= matmul->base_.thread_nr_, NNACL_ERR);

  int start_row = matmul->split_points_[task_id];
  int end_row = matmul->row_num_;
  if (task_id < (matmul->base_.thread_nr_ - 1)) {
    end_row = matmul->split_points_[task_id + 1];
  }
  int row_num = end_row - start_row;
  if (row_num <= 0) {
    return NNACL_OK;
  }
  const float *input = matmul->matrix_a_.pack_ptr_ + start_row * matmul->deep_;
  float *output = matmul->output_data_ + start_row * matmul->col_step_;
  if (matmul->col_ == 1) {
    float bias = 0;
    if (matmul->matrix_c_.pack_ptr_ != NULL) {
      bias = matmul->matrix_c_.pack_ptr_[0];
    }
    matmul->gemm_not_pack_fun_(input, matmul->matrix_b_.pack_ptr_, output, &bias, row_num, matmul->deep_,
                               param->act_type_);
  } else {
    if (matmul->out_need_aligned_) {
      MatMulAvx512Fp32(input, matmul->matrix_b_.pack_ptr_, output, matmul->matrix_c_.pack_ptr_, param->act_type_,
                       matmul->deep_, matmul->col_align_, matmul->col_align_, row_num);
    } else {
      MatMulMaskAvx512Fp32(input, matmul->matrix_b_.pack_ptr_, output, matmul->matrix_c_.pack_ptr_, param->act_type_,
                           matmul->deep_, matmul->col_, matmul->col_, row_num);
    }
  }
  return NNACL_OK;
}
int MatmulFp32Avx512_ParallelRunByOC(MatmulFp32Struct *matmul, int task_id) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  MS_CHECK_FALSE(task_id < 0 || task_id >= matmul->base_.thread_nr_, NNACL_ERR);

  int start_oc = matmul->split_points_[task_id];
  int end_oc = matmul->col_step_;
  if (task_id < (matmul->base_.thread_nr_ - 1)) {
    end_oc = matmul->split_points_[task_id + 1];
  }
  int compute_oc = end_oc - start_oc;
  if (compute_oc <= 0) {
    return NNACL_OK;
  }
  int func_flag = 0;
  if (matmul->row_ == 1) {
    func_flag += (!matmul->b_const_ && matmul->col_ <= C128NUM) ? C2NUM : C1NUM;
  }
  int b_stride = func_flag == C2NUM ? 1 : matmul->deep_;
  for (int i = 0; i < matmul->batch_; ++i) {
    float *a = matmul->matrix_a_.pack_ptr_ + matmul->a_offset_[i] * matmul->row_align_ * matmul->deep_;
    float *b =
      matmul->matrix_b_.pack_ptr_ + matmul->b_offset_[i] * matmul->deep_ * matmul->col_align_ + start_oc * b_stride;
    float *c = matmul->output_data_ + i * matmul->row_ * matmul->col_step_ + start_oc;
    float *bias = (matmul->matrix_c_.pack_ptr_ == NULL) ? NULL : matmul->matrix_c_.pack_ptr_ + start_oc;
    if (func_flag == 0) {
      if (matmul->out_need_aligned_) {
        MatMulAvx512Fp32(a, b, c, bias, param->act_type_, matmul->deep_, compute_oc, matmul->col_align_, matmul->row_);
      } else {
        MatMulMaskAvx512Fp32(a, b, c, bias, param->act_type_, matmul->deep_, compute_oc, matmul->col_, matmul->row_);
      }
    } else if (func_flag == C1NUM) {
      if (matmul->out_need_aligned_) {
        MatVecMulAvx512Fp32(a, b, c, bias, param->act_type_, matmul->deep_, compute_oc, matmul->col_align_);
      } else {
        MatVecMulMaskAvx512Fp32(a, b, c, bias, param->act_type_, matmul->deep_, compute_oc, matmul->col_);
      }
    } else {
      MatVecMulNoPackFp32(a, b, c, bias, param->act_type_, matmul->deep_, compute_oc, matmul->col_step_);
    }
  }
  return NNACL_OK;
}
int MatmulFp32Avx512_ParallelRunByBatch(MatmulFp32Struct *matmul, int task_id) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);

  int start_batch = task_id * matmul->batch_stride_;
  int end_batch = MSMIN(matmul->batch_, start_batch + matmul->batch_stride_);
  int func_flag = 0;
  if (matmul->row_ == 1) {
    func_flag += (!matmul->b_const_ && matmul->col_ <= C128NUM) ? C2NUM : C1NUM;
  }

  for (int index = start_batch; index < end_batch; ++index) {
    const float *a = matmul->matrix_a_.pack_ptr_ + matmul->a_offset_[index] * matmul->row_align_ * matmul->deep_;
    const float *b = matmul->matrix_b_.pack_ptr_ + matmul->b_offset_[index] * matmul->deep_ * matmul->col_align_;
    float *c = matmul->output_data_ + index * matmul->row_ * matmul->col_step_;
    float *bias = (matmul->matrix_c_.pack_ptr_ == NULL) ? NULL : matmul->matrix_c_.pack_ptr_;
    if (func_flag == 0) {
      if (matmul->out_need_aligned_) {
        MatMulAvx512Fp32(a, b, c, bias, param->act_type_, matmul->deep_, matmul->col_step_, matmul->col_align_,
                         matmul->row_);
      } else {
        MatMulMaskAvx512Fp32(a, b, c, bias, param->act_type_, matmul->deep_, matmul->col_step_, matmul->col_,
                             matmul->row_);
      }
    } else if (func_flag == C1NUM) {
      if (matmul->out_need_aligned_) {
        MatVecMulAvx512Fp32(a, b, c, bias, param->act_type_, matmul->deep_, matmul->col_step_, matmul->col_align_);
      } else {
        MatVecMulMaskAvx512Fp32(a, b, c, bias, param->act_type_, matmul->deep_, matmul->col_step_, matmul->col_);
      }
    } else {
      MatVecMulNoPackFp32(a, b, c, bias, param->act_type_, matmul->deep_, matmul->col_step_, matmul->col_step_);
    }
  }
  return NNACL_OK;
}

int MatmulFp32Avx512_ParallelRunByGEPM(MatmulFp32Struct *matmul, int task_id) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  MS_CHECK_FALSE(task_id < 0 || task_id >= matmul->base_.thread_nr_, NNACL_ERR);

  int a_plane_size = matmul->row_align_ * matmul->deep_;
  int b_plane_size = matmul->deep_ * matmul->col_align_;
  int c_plane_size = matmul->row_ * matmul->col_step_;
  // int matrix_row = matmul->row_;
  int matrix_col = matmul->col_step_;
  int matrix_deep = matmul->deep_;

  // by BatchCut
  int start_batch = task_id * matmul->batch_stride_;
  int end_batch = MSMIN(matmul->batch_, start_batch + matmul->batch_stride_);

  for (int index = start_batch; index < end_batch; ++index) {
    const float *a = matmul->matrix_a_.pack_ptr_ + matmul->a_offset_[index] * a_plane_size;
    const float *b = matmul->matrix_b_.pack_ptr_ + matmul->b_offset_[index] * b_plane_size;
    float *c = matmul->output_data_ + index * c_plane_size;

    float *bias = (matmul->matrix_c_.pack_ptr_ == NULL) ? NULL : matmul->matrix_c_.pack_ptr_;
    MatVecMulNoPackFp32(a, b, c, bias, param->act_type_, matrix_deep, matrix_col, matrix_col);
  }

  // by ColCut
  int col_split_points_size = matmul->col_split_points_size_;
  if (task_id < col_split_points_size) {
    int start_oc = matmul->col_split_points_[task_id];
    int end_oc = matrix_col;
    if (task_id < (col_split_points_size - 1)) {
      end_oc = matmul->col_split_points_[task_id + 1];
    }
    int compute_oc = end_oc - start_oc;
    if (compute_oc <= 0) {
      return NNACL_OK;
    }

    float *bias = (matmul->matrix_c_.pack_ptr_ == NULL) ? NULL : matmul->matrix_c_.pack_ptr_ + start_oc;
    for (int i = matmul->base_.thread_nr_ * matmul->batch_stride_; i < matmul->batch_; ++i) {
      float *a = matmul->matrix_a_.pack_ptr_ + matmul->a_offset_[i] * a_plane_size;
      float *b = matmul->matrix_b_.pack_ptr_ + matmul->b_offset_[i] * b_plane_size + start_oc;
      float *c = matmul->output_data_ + i * c_plane_size + start_oc;
      MatVecMulNoPackFp32(a, b, c, bias, param->act_type_, matrix_deep, compute_oc, matrix_col);
    }
  }
  return NNACL_OK;
}
int MatmulFp32Avx512_ParallelRunByGEMM(MatmulFp32Struct *matmul, int task_id) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  MS_CHECK_FALSE(task_id < 0 || task_id >= matmul->base_.thread_nr_, NNACL_ERR);

  int a_plane_size = matmul->row_align_ * matmul->deep_;
  int b_plane_size = matmul->deep_ * matmul->col_align_;
  int c_plane_size = matmul->row_ * matmul->col_step_;
  int matrix_row = matmul->row_;
  int matrix_col = matmul->col_step_;
  int matrix_deep = matmul->deep_;

  // by BatchCut
  int start_batch = task_id * matmul->batch_stride_;
  int end_batch = start_batch + matmul->batch_stride_;
  float *bias = (matmul->matrix_c_.pack_ptr_ == NULL) ? NULL : matmul->matrix_c_.pack_ptr_;
  for (int index = start_batch; index < end_batch; ++index) {
    const float *a = matmul->matrix_a_.pack_ptr_ + matmul->a_offset_[index] * a_plane_size;
    const float *b = matmul->matrix_b_.pack_ptr_ + matmul->b_offset_[index] * b_plane_size;
    float *c = matmul->output_data_ + index * c_plane_size;
    MatMulMaskAvx512Fp32(a, b, c, bias, param->act_type_, matrix_deep, matrix_col, matrix_col, matrix_row);
  }

  // by ColCut
  int col_split_points_size = matmul->col_split_points_size_;
  if (task_id < col_split_points_size) {
    int start_oc = matmul->col_split_points_[task_id];
    int end_oc = matmul->col_split_points_[task_id + 1];
    int compute_oc = end_oc - start_oc;

    bias = (matmul->matrix_c_.pack_ptr_ == NULL) ? NULL : matmul->matrix_c_.pack_ptr_ + start_oc;
    if (compute_oc > 0) {
      for (int i = matmul->base_.thread_nr_ * matmul->batch_stride_; i < matmul->batch_; ++i) {
        float *a = matmul->matrix_a_.pack_ptr_ + matmul->a_offset_[i] * a_plane_size;
        float *b = matmul->matrix_b_.pack_ptr_ + matmul->b_offset_[i] * b_plane_size + start_oc * matrix_deep;
        float *c = matmul->output_data_ + i * c_plane_size + start_oc;
        MatMulMaskAvx512Fp32(a, b, c, bias, param->act_type_, matrix_deep, compute_oc, matrix_col, matrix_row);
      }
    }
  }

  // by RowCut
  int start_oc = matmul->col_split_points_[col_split_points_size];
  int end_oc = matrix_col;
  int compute_oc = end_oc - start_oc;
  if (compute_oc <= 0) {
    return NNACL_OK;
  }

  int row_split_points_size = matmul->row_split_points_size_;
  if (task_id >= row_split_points_size) {
    return NNACL_OK;
  }
  int start_row = matmul->row_split_points_[task_id];
  int end_row = matmul->row_split_points_[task_id + 1];
  int row_num = end_row - start_row;
  if (row_num <= 0) {
    return NNACL_OK;
  }

  bias = (matmul->matrix_c_.pack_ptr_ == NULL) ? NULL : matmul->matrix_c_.pack_ptr_ + start_oc;
  for (int i = matmul->base_.thread_nr_ * matmul->batch_stride_; i < matmul->batch_; ++i) {
    float *a = matmul->matrix_a_.pack_ptr_ + matmul->a_offset_[i] * a_plane_size + start_row * matrix_deep;
    float *b = matmul->matrix_b_.pack_ptr_ + matmul->b_offset_[i] * b_plane_size + start_oc * matrix_deep;
    float *c = matmul->output_data_ + i * c_plane_size + start_row * matrix_col + start_oc;
    MatMulMaskAvx512Fp32(a, b, c, bias, param->act_type_, matrix_deep, compute_oc, matrix_col, row_num);
  }

  return NNACL_OK;
}
int MatmulFp32Avx512_ParallelRunByGEPDOT(MatmulFp32Struct *matmul, int task_id) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  MS_CHECK_FALSE(task_id < 0 || task_id >= matmul->base_.thread_nr_, NNACL_ERR);

  // by BatchCut
  int start_batch = task_id * matmul->batch_stride_;
  int end_batch = start_batch + matmul->batch_stride_;
  float bias = 0;
  if (matmul->matrix_c_.pack_ptr_ != NULL) {
    bias = matmul->matrix_c_.pack_ptr_[0];
  }
  for (int index = start_batch; index < end_batch; ++index) {
    const float *a = matmul->matrix_a_.pack_ptr_ + matmul->a_offset_[index] * matmul->row_ * matmul->deep_;
    const float *b = matmul->matrix_b_.pack_ptr_ + matmul->b_offset_[index] * matmul->deep_ * matmul->col_;
    float *c = matmul->output_data_ + index * matmul->row_ * matmul->col_;
    matmul->gemm_not_pack_fun_(a, b, c, &bias, matmul->row_, matmul->deep_, param->act_type_);
  }

  // by RowCut
  int split_points_size = matmul->row_split_points_size_;
  if (task_id >= split_points_size) {
    return NNACL_OK;
  }
  for (int index = matmul->base_.thread_nr_ * matmul->batch_stride_; index < matmul->batch_; ++index) {
    int start_row = matmul->row_split_points_[task_id];
    int end_row = matmul->row_split_points_[task_id + 1];
    int row_num = end_row - start_row;
    if (row_num <= 0) {
      continue;
    }
    const float *a =
      matmul->matrix_a_.pack_ptr_ + matmul->a_offset_[index] * matmul->row_ * matmul->deep_ + start_row * matmul->deep_;
    const float *b = matmul->matrix_b_.pack_ptr_ + matmul->b_offset_[index] * matmul->deep_ * matmul->col_;
    float *c = matmul->output_data_ + index * matmul->row_ * matmul->col_ + start_row * matmul->col_step_;
    matmul->gemm_not_pack_fun_(a, b, c, &bias, row_num, matmul->deep_, param->act_type_);
  }

  return NNACL_OK;
}
int MatmulFp32Avx512_ParallelRunByRow1Deep1GEPDOT(MatmulFp32Struct *matmul, int task_id) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  MS_CHECK_FALSE(task_id < 0 || task_id >= matmul->base_.thread_nr_, NNACL_ERR);

  int a_plane_size = matmul->row_align_ * matmul->deep_;
  int b_plane_size = matmul->deep_ * matmul->col_align_;
  int c_plane_size = matmul->row_ * matmul->col_step_;
  int matrix_col = matmul->col_step_;
  int matrix_deep = matmul->deep_;

  // by BatchCut
  int start_batch = task_id * matmul->batch_stride_;
  int end_batch = MSMIN(matmul->batch_, start_batch + matmul->batch_stride_);

  for (int index = start_batch; index < end_batch; ++index) {
    const float *a = matmul->matrix_a_.pack_ptr_ + matmul->a_offset_[index] * a_plane_size;
    const float *b = matmul->matrix_b_.pack_ptr_ + matmul->b_offset_[index] * b_plane_size;
    float *c = matmul->output_data_ + index * c_plane_size;
    float *bias = (matmul->matrix_c_.pack_ptr_ == NULL) ? NULL : matmul->matrix_c_.pack_ptr_;
    matmul->gemm_not_pack_fun_(a, b, c, bias, matrix_col, matrix_deep, param->act_type_);
  }

  // by ColCut
  int col_split_points_size = matmul->col_split_points_size_;
  if (task_id < col_split_points_size) {
    int start_oc = matmul->col_split_points_[task_id];
    int end_oc = matrix_col;
    if (task_id < (col_split_points_size - 1)) {
      end_oc = matmul->col_split_points_[task_id + 1];
    }
    int compute_oc = end_oc - start_oc;
    if (compute_oc <= 0) {
      return NNACL_OK;
    }

    float *bias = (matmul->matrix_c_.pack_ptr_ == NULL) ? NULL : matmul->matrix_c_.pack_ptr_ + start_oc;
    for (int i = matmul->base_.thread_nr_ * matmul->batch_stride_; i < matmul->batch_; ++i) {
      float *a = matmul->matrix_a_.pack_ptr_ + matmul->a_offset_[i] * a_plane_size;
      float *b = matmul->matrix_b_.pack_ptr_ + matmul->b_offset_[i] * b_plane_size + start_oc;
      float *c = matmul->output_data_ + i * c_plane_size + start_oc;
      matmul->gemm_not_pack_fun_(a, b, c, bias, compute_oc, matrix_deep, param->act_type_);
    }
  }
  return NNACL_OK;
}
int MatmulFp32Avx512_ParallelRunByBatchColRowGEMM(MatmulFp32Struct *matmul, int task_id) {
  MatMulParameter *param = (MatMulParameter *)(matmul->base_.param_);
  MS_CHECK_FALSE(task_id < 0 || task_id >= matmul->base_.thread_nr_, NNACL_ERR);

  int a_plane_size = matmul->row_align_ * matmul->deep_;
  int b_plane_size = matmul->deep_ * matmul->col_align_;
  int c_plane_size = matmul->row_ * matmul->col_step_;
  int matrix_row = matmul->row_;
  int matrix_col = matmul->col_step_;
  int matrix_deep = matmul->deep_;

  // by BatchCut
  int start_batch = task_id * matmul->batch_stride_;
  int end_batch = start_batch + matmul->batch_stride_;
  float *bias = (matmul->matrix_c_.pack_ptr_ == NULL) ? NULL : matmul->matrix_c_.pack_ptr_;
  for (int index = start_batch; index < end_batch; ++index) {
    const float *a = matmul->matrix_a_.pack_ptr_ + matmul->a_offset_[index] * a_plane_size;
    const float *b = matmul->matrix_b_.pack_ptr_ + matmul->b_offset_[index] * b_plane_size;
    float *c = matmul->output_data_ + index * c_plane_size;
    MatMulMaskAvx512Fp32(a, b, c, bias, param->act_type_, matrix_deep, matrix_col, matrix_col, matrix_row);
  }

  MatmulSlice *matmul_slices = matmul->matmul_slice_set_[task_id];
  int slice_count = matmul->matmul_slice_count_[task_id];
  for (int s = 0; s < slice_count; s++) {
    MatmulSlice matmul_slice = matmul_slices[s];

    int start_oc = matmul_slice.col_s_;
    int end_oc = matmul_slice.col_e_;
    int compute_oc = end_oc - start_oc;
    if (compute_oc <= 0) {
      return NNACL_OK;
    }

    int start_row = matmul_slice.row_s_;
    int end_row = matmul_slice.row_e_;
    int row_num = end_row - start_row;
    if (row_num <= 0) {
      return NNACL_OK;
    }

    bias = (matmul->matrix_c_.pack_ptr_ == NULL) ? NULL : matmul->matrix_c_.pack_ptr_ + start_oc;
    for (int i = matmul->base_.thread_nr_ * matmul->batch_stride_; i < matmul->batch_; ++i) {
      float *a = matmul->matrix_a_.pack_ptr_ + matmul->a_offset_[i] * a_plane_size + start_row * matrix_deep;
      float *b = matmul->matrix_b_.pack_ptr_ + matmul->b_offset_[i] * b_plane_size + start_oc * matrix_deep;
      float *c = matmul->output_data_ + i * c_plane_size + start_row * matrix_col + start_oc;
      MatMulMaskAvx512Fp32(a, b, c, bias, param->act_type_, matrix_deep, compute_oc, matrix_col, row_num);
    }
  }
  return NNACL_OK;
}

KernelBase *CreateMatmulFp32Avx512() {
  MatmulFp32Struct *matmul = (MatmulFp32Struct *)CreateMatmulFp32Base();
  matmul->matmul_type_ = kNotImplemented;
  matmul->check_thread_cutting_by_row_ = MatmulFp32Avx512_CheckThreadCuttingByRow;
  matmul->get_thread_cutting_policy_ = MatmulFp32Avx512_GetThreadCuttingPolicy;
  matmul->init_parameter_ = MatmulFp32Avx512_InitParameter;
  matmul->init_global_varibale_ = MatmulFp32Avx512_InitGlobalVariable;
  matmul->parallel_run_by_oc_ = MatmulFp32Avx512_ParallelRunByOC;
  matmul->parallel_run_by_row_ = MatmulFp32Avx512_ParallelRunByRow;
  matmul->parallel_run_by_batch_ = MatmulFp32Avx512_ParallelRunByBatch;
  matmul->parallel_run_by_gemm_ = MatmulFp32Avx512_ParallelRunByGEMM;
  matmul->parallel_run_by_gepm_ = MatmulFp32Avx512_ParallelRunByGEPM;
  matmul->parallel_run_by_gepdot_ = MatmulFp32Avx512_ParallelRunByGEPDOT;
  matmul->parallel_run_by_batch_col_row_gemm_ = MatmulFp32Avx512_ParallelRunByBatchColRowGEMM;
  matmul->parallel_run_by_row1_deep1_gepdot_ = MatmulFp32Avx512_ParallelRunByRow1Deep1GEPDOT;
  return (KernelBase *)matmul;
}
#endif
