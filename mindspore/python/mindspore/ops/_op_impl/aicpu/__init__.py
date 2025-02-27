# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""aicpu ops"""
from .adaptive_max_pool_3d_grad import _adaptive_max_pool_3d_grad_aicpu
from .adaptive_max_pool_2d_grad import _adaptive_max_pool_2d_grad_aicpu
from .adaptive_avg_pool_3d_grad import _adaptiveavgpool3d_grad_aicpu
from .adaptive_avg_pool_3d import _adaptiveavgpool3d_aicpu
from .tile import _tile_aicpu
from .tanh import _tanh_aicpu
from .less import _less_aicpu
from .add import _add_aicpu
from .sparse_matrix_transpose import _sparse_matrix_transpose_aicpu
from .sparse_matrix_nnz import _sparse_matrix_nnz_aicpu
from .sparse_dense_cwise_mul import _sparse_dense_cwise_mul_aicpu
from .sparse_dense_cwise_div import _sparse_dense_cwise_div_aicpu
from .sparse_dense_cwise_add import _sparse_dense_cwise_add_aicpu
from .sparse_concat import _sparse_concat_aicpu
from .sparse_apply_centered_rms_prop import _sparse_apply_centered_rms_prop_aicpu
from .broadcast_to import _broadcast_to_aicpu
from .blackman_window import _blackman_window_aicpu
from .bincount import _bincount_aicpu
from .asinh_grad import _asinh_grad_aicpu
from .unique import _unique_aicpu
from .add_n import _add_n_aicpu
from .add_v2 import _add_v2_aicpu
from .adjust_contrastv2 import _adjust_contrastv2_aicpu
from .adjust_hue import _adjust_hue_aicpu
from .adjust_saturation import _adjust_saturation_aicpu
from .affine_grid_grad import _affine_grid_grad_aicpu
from .angle import _angle_aicpu
from .arg_max import _arg_max_aicpu
from .argmax_with_value import _argmax_with_value_aicpu
from .arg_min import _arg_min_aicpu
from .argmin_with_value import _argmin_with_value_aicpu
from .avgpool_v1 import _avgpool_v1_aicpu
from .avgpool_grad_v1 import _avgpool_grad_v1_aicpu
from .matrix_solve import _matrix_solve_aicpu
from .betainc import _betainc_aicpu
from .bartlett_window import _bartlett_window_aicpu
from .batch_norm_grad_grad import _batch_norm_grad_grad_aicpu
from .no_repeat_ngram import _no_repeat_ngram_aicpu
from .init_data_set_queue import _init_data_set_queue_aicpu
from .embedding_lookup import _embedding_lookup_aicpu
from .padding import _padding_aicpu
from .gather import _gather_aicpu
from .gather_grad import _gather_grad_aicpu
from .gather_d_grad_v2 import _gather_d_grad_v2_aicpu
from .gather_d import _gather_d_aicpu
from .gather_nd import _gather_nd_aicpu
from .scatter import _scatter_aicpu
from .identity import _identity_aicpu
from .edit_distance import _edit_distance_aicpu
from .unique_with_pad import _unique_with_pad_aicpu
from .sub_and_filter import _sub_and_filter_aicpu
from .pad_and_shift import _pad_and_shift_aicpu
from .dropout_genmask import _dropout_genmask_aicpu
from .dropout_genmask_v3 import _dropout_genmask_v3_aicpu
from .stateless_dropout_genmask import _stateless_dropout_genmask_aicpu
from .dropout2d import _dropout2d_aicpu
from .dropout3d import _dropout3d_aicpu
from .dynamic_stitch import _dynamic_stitch_aicpu
from .get_next import _get_next_aicpu
from .print_tensor import _print_aicpu
from .topk import _top_k_aicpu
from .tensor_scatter_update import _tensor_scatter_update_aicpu
from .log1p import _log1p_aicpu
from .asin import _asin_aicpu
from .asin_grad import _asin_grad_aicpu
from .is_finite import _is_finite_aicpu
from .is_inf import _is_inf_aicpu
from .is_nan import _is_nan_aicpu
from .reshape import _reshape_aicpu
from .flatten import _flatten_aicpu
from .cosh import _cosh_aicpu
from .sign import _sign_aicpu
from .squeeze import _squeeze_aicpu
from .acos import _acos_aicpu
from .acos_grad import _acos_grad_aicpu
from .expand import _expand_aicpu
from .expand_dims import _expand_dims_aicpu
from .randperm import _randperm_aicpu
from .random_choice_with_mask import _random_choice_with_mask_aicpu
from .rsqrt import _rsqrt_aicpu
from .sqrt import _sqrt_aicpu
from .sqrt_grad import _sqrt_grad_aicpu
from .masked_fill import _masked_fill_aicpu
from .sort import _sort_aicpu
from .search_sorted import _search_sorted_aicpu
from .stack import _stack_aicpu
from .unstack import _unstack_aicpu
from .unsorted_segment_sum import _unsorted_segment_sum_aicpu
from .addcmul import _addcmul_aicpu
from .uniform_candidate_sampler import _uniform_candidate_sampler_aicpu
from .log_uniform_candidate_sampler import _log_uniform_candidate_sampler_aicpu
from .compute_accidental_hits import _compute_accidental_hits_aicpu
from .ctcloss import _ctcloss_aicpu
from .reverse_sequence import _reverse_sequence_aicpu
from .log_matrix_determinant import _log_matrix_determinant_aicpu
from .crop_and_resize import _crop_and_resize_aicpu
from .acosh import _acosh_aicpu
from .acosh_grad import _acosh_grad_aicpu
from .rnnt_loss import _rnnt_loss_aicpu
from .greater import _greater_aicpu
from .greater_equal import _greater_equal_aicpu
from .random_categorical import _random_categorical_aicpu
from .tanh_grad import _tanh_grad_aicpu
from .cast import _cast_aicpu
from .mirror_pad import _mirror_pad_aicpu
from .mirror_pad_grad import _mirror_pad_grad_aicpu
from .masked_select import _masked_select_aicpu
from .masked_select_grad import _masked_select_grad_aicpu
from .mul import _mul_aicpu
from .standard_normal import _standard_normal_aicpu
from .gamma import _gamma_aicpu
from .random_gamma import _random_gamma_aicpu
from .sub import _sub_aicpu
from .not_equal import _not_equal_aicpu
from .poisson import _poisson_aicpu
from .update_cache import _update_cache_aicpu
from .upper_bound import _upper_bound_aicpu
from .cache_swap_table import _cache_swap_table_aicpu
from .uniform_int import _uniform_int_aicpu
from .uniform_real import _uniform_real_aicpu
from .standard_laplace import _standard_laplace_aicpu
from .strided_slice import _strided_slice_aicpu
from .neg import _neg_aicpu
from .strided_slice_grad import _strided_slice_grad_aicpu
from .end_of_sequence import _end_of_sequence_aicpu
from .fused_sparse_adam import _fused_sparse_adam_aicpu
from .fused_sparse_lazy_adam import _fused_sparse_lazy_adam_aicpu
from .fused_sparse_ftrl import _fused_sparse_ftrl_aicpu
from .sparse_fill_empty_rows_grad import _sparse_fill_empty_rows_grad_aicpu
from .sparse_reshape import _sparse_reshape_aicpu
from .sparse_segment_sqrt_n_grad import _sparse_segment_sqrt_n_grad_aicpu
from .sparse_segment_sum import _sparse_segment_sum_aicpu
from .sparse_segment_sum_with_num_segments import _sparse_segment_sum_with_num_segments_aicpu
from .sparse_softmax_cross_entropy_with_logits_v2 import _sparse_softmax_cross_entropy_with_logits_v2_aicpu
from .sparsesparsemaximum import _sparsesparsemaximum_aicpu
from .split import _split_aicpu
from .transpose import _transpose_aicpu
from .tril_indices import _tril_indices_aicpu
from .triu_indices import _triu_indices_aicpu
from .unravel_index import _unravel_index_aicpu
from .xlogy import _xlogy_aicpu
from .xdivy import _xdivy_aicpu
from .fused_sparse_proximal_adagrad import _fused_sparse_proximal_adagrad_aicpu
from .meshgrid import _meshgrid_aicpu
from .div import _div_aicpu
from .trans_data import _trans_data_aicpu
from .stack_push_pop import _stack_init_aicpu
from .stack_push_pop import _stack_push_aicpu
from .stack_push_pop import _stack_pop_aicpu
from .asinh import _asinh_aicpu
from .stack_push_pop import _stack_destroy_aicpu
from .matrix_diag_v3 import _matrix_diag_v3_aicpu
from .matrix_diag_part_v3 import _matrix_diag_part_v3_aicpu
from .tan import _tan_aicpu
from .ctc_greedy_decoder import _ctc_greedy_decoder_aicpu
from .resize_bilinear import _resize_bilinear_aicpu
from .resize_bilinear_grad import _resize_bilinear_grad_aicpu
from .resize_bicubic_grad import _resize_bicubic_grad_aicpu
from .resize_nearest_neighbor_v2 import _resize_nearest_neighbor_v2_aicpu
from .resize_nearest_neighbor_v2_grad import _resize_nearest_neighbor_v2_grad_aicpu
from .scatter_elements import _scatter_elements_aicpu
from .non_max_suppression import _non_max_suppression_aicpu
from .square import _square_aicpu
from .squared_difference import _squared_difference_aicpu
from .non_zero import _non_zero_aicpu
from .zeros_like import _zeros_like_aicpu
from .ones_like import _ones_like_aicpu
from .grid_sampler_3d import _grid_sampler_3d_aicpu
from .atanh import _atanh_aicpu
from .grid_sampler_3d_grad import _grid_sampler_3d_grad_aicpu
from .environ_create import _environ_create_aicpu
from .environ_set import _environ_set_aicpu
from .environ_get import _environ_get_aicpu
from .environ_destroy_all import _environ_destroy_all_aicpu
from .cross import _cross_aicpu
from .check_numerics import _check_numerics_aicpu
from .cumsum import _cumsum_aicpu
from .round import _round_aicpu
from .stft import _stft_aicpu
from .floor_div import _floor_div_aicpu
from .priority_replay_buffer import _prb_create_op_cpu
from .priority_replay_buffer import _prb_push_op_cpu
from .conjugate_transpose import _conjugate_transpose_aicpu
from .priority_replay_buffer import _prb_sample_op_cpu
from .priority_replay_buffer import _prb_update_op_cpu
from .equal import _equal_aicpu
from .priority_replay_buffer import _prb_destroy_op_cpu
from .right_shift import _right_shift_aicpu
from .tril import _tril_aicpu
from .linspace import _lin_space_aicpu
from .triu import _triu_aicpu
from .zeta import _zeta_aicpu
from .grid_sampler_2d import _grid_sampler_2d_aicpu
from .grid_sampler_2d_grad import _grid_sampler_2d_grad_aicpu
from .sparse_segment_mean_grad import _sparse_segment_mean_grad_aicpu
from .scatter_nd import _scatter_nd_aicpu
from .scatter_nd_update import _scatter_nd_update_aicpu
from .scatter_nd_max import _scatter_nd_max_aicpu
from .conj import _conj_aicpu
from .scatter_nd_min import _scatter_nd_min_aicpu
from .compare_and_bitpack import _compare_and_bitpack_aicpu
from .addcdiv import _addcdiv_aicpu
from .unique_consecutive import _unique_consecutive_aicpu
from .sparse_tensor_to_csr_sparse_matrix import _sparse_tensor_to_csr_sparse_matrix_aicpu
from .csr_sparse_matrix_to_sparse_tensor import _csr_sparse_matrix_to_sparse_tensor_aicpu
from .linear_sum_assignment import _linear_sum_assignment_aicpu
from .random_shuffle import _random_shuffle_aicpu
from .reservoir_replay_buffer import _rrb_create_op_cpu
from .reservoir_replay_buffer import _rrb_push_op_cpu
from .reservoir_replay_buffer import _rrb_sample_op_cpu
from .reservoir_replay_buffer import _rrb_destroy_op_cpu
from .concat_offset import _concat_offset_aicpu
from .range import _range_aicpu
from .slice_grad import _slice_grad_aicpu
from .median import _median_aicpu
from .median_grad import _median_grad_aicpu
from .reduce_sum import _reduce_sum_aicpu
from .adaptive_avg_pool_2d import _adaptive_avg_pool_2d_aicpu
from .adaptive_avg_pool_2d_grad import _adaptive_avg_pool_2d_grad_aicpu
from .fill_v2 import _fill_v2_aicpu
from .data_format_vec_permute import _data_format_vec_permute_aicpu
from .multinomial import _multinomial_aicpu
from .fft_with_size import _fft_with_size_aicpu
from .histogram import _histogram_aicpu
from .matrix_determinant import _matrix_determinant_aicpu
from .matrix_set_diag_v3 import _matrix_set_diag_v3_aicpu
from .nan_to_num import _nan_to_num_aicpu
from .qr import _qr_aicpu
from .col2im import _col2im_aicpu
from .matrix_solve_ls import _matrix_solve_ls_aicpu
from .cauchy import _cauchy_aicpu
from .bucketize import _bucketize_aicpu
from .channel_shuffle import _channel_shuffle_aicpu
from .choleskygrad import _choleskygrad_aicpu
from .cholesky_inverse import _cholesky_inverse_aicpu
from .cholesky_solve import _cholesky_solve_aicpu
from .combined_non_max_suppression import _combined_non_max_suppression_aicpu
from .complex import _complex_aicpu
from .complex_abs import _complex_abs_aicpu
from .concat import _concat_aicpu
from .cos import _cos_aicpu
from .csr_sparse_matrix_to_dense import _csr_sparse_matrix_to_dense_aicpu
from .cumprod import _cumprod_aicpu
from .exp import _exp_aicpu
from .matrix_triangular_solve import _matrix_triangular_solve_aicpu
from .maximum_grad_grad import _maximum_grad_grad_aicpu
from .maxpool_v1 import _maxpool_v1_aicpu
from .minimum_grad_grad import _minimum_grad_grad_aicpu
from .mul_no_nan import _mul_no_nan_aicpu
from .multilabel_margin_loss_grad import _multilabel_margin_loss_grad_aicpu
from .nth_element import _nth_element_aicpu
from .non_max_suppression_with_overlaps import _non_max_suppression_with_overlaps_aicpu
from .one_hot import _one_hot_aicpu
from .orgqr import _orgqr_aicpu
from .parameterized_truncated_normal import _parameterized_truncated_normal_aicpu
from .polar import _polar_aicpu
from .pdist_grad import _pdist_grad_aicpu
from .ragged_range import _raggedrange_aicpu
from .ragged_tensor_to_sparse import _ragged_tensor_to_sparse_aicpu
from .ragged_tensor_to_tensor import _ragged_tensor_to_tensor_aicpu
from .reciprocal import _reciprocal_aicpu
from .reciprocal_grad import _reciprocal_grad_aicpu
from .reduce_mean import _reduce_mean_aicpu
from .reduce_prod import _reduce_prod_aicpu
from .relu_v3 import _relu_v3_aicpu
from .reversev2 import _reversev2_aicpu
from .rgb_to_hsv import _rgb_to_hsv_aicpu
from .rsqrt_grad import _rsqrt_grad_aicpu
from .sample_distorted_bounding_box_v2 import _sample_distorted_bounding_box_v2_aicpu
from .scale_and_translate_grad import _scale_and_translate_grad_aicpu
from .select import _select_aicpu
from .self_adjoint_eig import _self_adjoint_eig_aicpu
from .sin import _sin_aicpu
from .sinc import _sinc_aicpu
from .sinh import _sinh_aicpu
from .smooth_l1_loss_grad import _smooth_l1_loss_grad_aicpu
from .smooth_l1_loss import _smooth_l1_loss_aicpu
from .cumulative_logsumexp import _cumulative_logsumexp_aicpu
from .sparse_segment_sqrt_n import _sparse_segment_sqrt_n_aicpu
from .scale_and_translate import _scale_and_translate_aicpu
from .quant_dtype_cast import _quant_dtype_cast_aicpu
from .fse_decode import _fse_decode_aicpu
from .dense_to_csr_sparse_matrix import _dense_to_csr_sparse_matrix_aicpu
from .dense_to_sparse_set_operation import _dense_to_sparse_set_operation_aicpu
from .diag import _diag_aicpu
from .diagonal import _diagonal_aicpu
from .diag_part import _diag_part_aicpu
from .bias_add import _bias_add_aicpu
from .bias_add_grad import _bias_add_grad_aicpu
from .eig import _eig_aicpu
from .eye import _eye_aicpu
from .fmin import _fmin_aicpu
from .fractional_avg_pool import _fractional_avg_pool_aicpu
from .fractional_avg_pool_grad import _fractional_avg_pool_grad_aicpu
from .fractional_max_pool import _fractional_max_pool_aicpu
from .fractional_max_pool_grad import _fractional_max_pool_grad_aicpu
from .fractional_max_pool_grad_with_fixed_ksize import _fractional_max_pool_grad_with_fixed_ksize_aicpu
from .gcd import _gcd_aicpu
from .geqrf import _geqrf_aicpu
from .hard_sigmoid import _hard_sigmoid_aicpu
from .hard_sigmoid_grad import _hard_sigmoid_grad_aicpu
from .heaviside import _heaviside_aicpu
from .hypot import _hypot_aicpu
from .identity_n import _identity_n_aicpu
from .index_fill import _index_fill_aicpu
from .kldivloss import _kldiv_loss_aicpu
from .kldivlossgrad import _kldiv_loss_grad_aicpu
from .lcm import _lcm_aicpu
from .less_equal import _less_equal_aicpu
from .logical_xor import _logical_xor_aicpu
from .logit import _logit_aicpu
from .logit_grad import _logit_grad_aicpu
from .log_normal_reverse import _log_normal_reverse_aicpu
from .lower_bound import _lower_bound_aicpu
from .lu_unpack_grad import _lu_unpack_grad_aicpu
from .pad_v3_grad import _pad_v3_grad_aicpu
from .pad_v3 import _pad_v3_aicpu
from .cholesky import _cholesky_aicpu
from .hsv_to_rgb import _hsv_to_rgb_aicpu
from .im2col import _im2col_aicpu
from .lu_solve import _lu_solve_aicpu
from .relu_grad_v3 import _relu_grad_v3_aicpu
from .resize_bicubic import _resize_bicubic_aicpu
from .extract_glimpse import _extract_glimpse_aicpu
from .real_div import _real_div_aicpu
from .multinomial_with_replacement import _multinomial_with_replacement_aicpu
from .coalesce import _coalesce_aicpu
from .crop_and_resize_grad_boxes import _crop_and_resize_grad_boxes_aicpu
from .crop_and_resize_grad_image import _crop_and_resize_grad_image_aicpu
from .dense_to_dense_set_operation import _dense_to_dense_set_operation_aicpu
from .div_no_nan import _div_no_nan_aicpu
from .expm1 import _expm1_aicpu
from .fractional_max_pool3d_with_fixed_ksize import _fractional_max_pool3d_with_fixed_ksize_aicpu
from .fractional_max_pool3d_grad_with_fixed_ksize import _fractional_max_pool3d_grad_with_fixed_ksize_aicpu
from .fractional_max_pool_with_fixed_ksize import _fractional_max_pool_with_fixed_ksize_aicpu
from .hamming_window import _hamming_window_aicpu
from .igamma import _igamma_aicpu
from .igammac import _igammac_aicpu
from .igammagrada import _igammagrada_aicpu
from .imag import _imag_aicpu
from .instance_norm_v2 import _instance_norm_v2_aicpu
from .instance_norm_v2_grad import _instance_norm_v2_grad_aicpu
from .layer_norm_grad_grad import _layernorm_grad_grad_aicpu
from .list_diff import _list_diff_aicpu
from .log import _log_aicpu
from .logspace import _logspace_aicpu
from .matrix_inverse import _matrix_inverse_aicpu
from .matrix_power import _matrix_power_aicpu
from .max_pool3d_grad_with_argmax import _max_pool3d_grad_with_argmax_aicpu
from .max_pool3d_with_argmax import _max_pool3d_with_argmax_aicpu
from .maxpool_grad_v1 import _maxpool_grad_v1_aicpu
from .max_unpool2d import _max_unpool2d_aicpu
from .max_unpool2d_grad import _max_unpool2d_grad_aicpu
from .max_unpool3d import _max_unpool3d_aicpu
from .max_unpool3d_grad import _max_unpool3d_grad_aicpu
from .mvlgamma import _mvlgamma_aicpu
from .mvlgamma_grad import _mvlgamma_grad_aicpu
from .nextafter import _nextafter_aicpu
from .non_deterministic_ints import _non_deterministic_ints_aicpu
from .pow import _pow_aicpu
from .real import _real_aicpu
from .resize_area import _resize_area_aicpu
from .segment_sum import _segment_sum_aicpu
from .set_size import _set_size_aicpu
from .slice import _slice_aicpu
from .sparse_cross import _sparse_cross_aicpu
from .sparse_slice import _sparse_slice_aicpu
from .sparse_softmax import _sparse_softmax_aicpu
from .sparse_tensor_dense_add import _sparse_tensor_dense_add_aicpu
from .sparse_tensor_dense_mat_mul import _sparse_tensor_dense_mat_mul_aicpu
from .trace import _trace_aicpu
from .tracegrad import _tracegrad_aicpu
from .tridiagonal_solve import _tridiagonal_solve_aicpu
from .truncated_normal import _truncated_normal_aicpu
from .glu import _glu_aicpu
from .deformable_offsets import _deformable_offsets_aicpu
from .deformable_offsets_grad import _deformable_offsets_grad_aicpu
from .multi_margin_loss import _multi_margin_loss_aicpu
from .multi_margin_loss_grad import _multi_margin_loss_grad_aicpu
from .sparse_to_dense_v2 import _sparse_to_dense_v2_aicpu
from .bernoulli import _bernoulli_aicpu
from .glu_grad import _glu_grad_aicpu
from .sspaddmm import _sspaddmm_aicpu
from .sequence_addn import _sequence_addn_aicpu
from .sequence_concat import _sequence_concat_aicpu
from .affine_grid import _affine_grid_aicpu
