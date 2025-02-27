/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include <vector>
#include <algorithm>
#include <numeric>
#include "pipeline/pynative/grad/bprop_expander/bprop_irbuilder.h"
#include "pipeline/pynative/grad/bprop_expander/grad_ops/common_utils.h"
#include "include/common/utils/utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore::expander::bprop {
namespace {
const auto diag_max_length = 200000000;

int64_t RecorrectAxis(int64_t axis, size_t rank) {
  auto rank_i = SizeToLong(rank);
  if (rank == 0 || axis < -rank_i || axis >= rank_i) {
    MS_EXCEPTION(ValueError) << "Rank can not be 0 and 'axis' must be in range [-" << rank_i << ", " << rank_i
                             << "), but got " << axis;
  }
  return (axis < 0) ? (axis + rank_i) : axis;
}

NodePtrList GatherDropNegatives(const BpropIRBuilder *ib, const NodePtr &params, const NodePtr &ids,
                                const NodePtr &zero_clipped_indices_param = nullptr,
                                const NodePtr &is_positive_param = nullptr) {
  NodePtr zero_clipped_indices = zero_clipped_indices_param;
  if (zero_clipped_indices_param == nullptr) {
    zero_clipped_indices = ib->Maximum(ids, ib->ZerosLike(ids));
  }
  auto gathered = ib->Gather(params, zero_clipped_indices, 0);

  NodePtr is_positive = is_positive_param;
  if (is_positive_param == nullptr) {
    is_positive = ib->GreaterEqual(ids, ib->Tensor(0, ib->GetDtype(ids)));
    auto broadcastable_shape = ib->GetShape(is_positive);
    auto gathered_shape = ib->GetShape(gathered);
    if (IsDynamic(broadcastable_shape) || IsDynamic(gathered_shape)) {
      auto shape_func = [](const ShapeArray &inputs) -> ShapeArray {
        auto is_pos = inputs.at(1);
        auto gather_rank = inputs.at(0).size();
        auto is_pos_rank = is_pos.size();

        std::vector<int64_t> res_shape(is_pos.begin(), is_pos.end());
        if (gather_rank > is_pos_rank) {
          auto expand_len = gather_rank - is_pos_rank;
          for (size_t i = 0; i < expand_len; ++i) {
            res_shape.push_back(1);
          }
        }
        return {res_shape};
      };
      auto infer_func = [](const ShapeArray &inputs, const std::unordered_set<size_t> &invalid_indices) -> ShapeVector {
        auto gather = inputs.at(0);
        auto is_pos = inputs.at(1);
        if (!invalid_indices.empty() || IsDynamicRank(gather) || IsDynamicRank(is_pos)) {
          return {-1};
        }
        auto gather_rank = gather.size();
        auto is_pos_rank = is_pos.size();
        return {SizeToLong(std::max(gather_rank, is_pos_rank))};
      };

      auto is_positive_shape = ib->ShapeCalc({gathered, is_positive}, shape_func, infer_func, {})[0];
      is_positive = ib->Reshape(is_positive, is_positive_shape);
      auto shape_gather = ib->Shape(gathered, true);
      is_positive = ib->LogicalAnd(is_positive, ib->Fill(1.0, shape_gather, TypeId::kNumberTypeBool));
    } else {
      auto back_size = ib->GetShape(gathered).size() - ib->GetShape(is_positive).size();
      for (size_t i = 0; i < back_size; ++i) {
        broadcastable_shape.push_back(1);
      }
      is_positive = ib->Reshape(is_positive, broadcastable_shape);
      auto ones = ib->Fill(1.0, gathered_shape, TypeId::kNumberTypeBool);
      is_positive = ib->LogicalAnd(is_positive, ones);
    }
  }
  auto zero_slice = ib->ZerosLike(gathered);
  return {ib->Select(is_positive, gathered, zero_slice), zero_clipped_indices, is_positive};
}

NodePtrList UnsortedSegmentMinOrMaxGrad(const BpropIRBuilder *ib, const NodePtr &x, const NodePtr &segment_ids,
                                        const NodePtr &num_segments, const NodePtr &out, const NodePtr &dout) {
  auto temp_outs = GatherDropNegatives(ib, out, segment_ids, nullptr, nullptr);
  constexpr size_t out_size = 3;
  MS_EXCEPTION_IF_CHECK_FAIL(temp_outs.size() == out_size, "Outputs' size should be 3.");
  auto gathered_outputs = temp_outs[0];
  auto zero_clipped_indices = temp_outs[1];
  auto is_positive = temp_outs[2];

  auto tmp = ib->Equal(x, gathered_outputs);
  auto is_selected = ib->LogicalAnd(tmp, is_positive);
  auto num_selected =
    ib->Emit("UnsortedSegmentSum", {ib->Cast(is_selected, ib->GetDtype(dout)), segment_ids, num_segments});
  auto weighted_grads = ib->RealDiv(dout, num_selected);
  auto temp_outs_2 = GatherDropNegatives(ib, weighted_grads, nullptr, zero_clipped_indices, is_positive);
  MS_EXCEPTION_IF_CHECK_FAIL(temp_outs.size() > 0, "Outputs should not be empty.");
  auto gathered_grads = temp_outs_2[0];
  auto zeros = ib->ZerosLike(gathered_grads);
  return {ib->Select(is_selected, gathered_grads, zeros), ib->ZerosLike(segment_ids), ib->ZerosLike(num_segments)};
}

NodePtrList SegmentMinOrMaxGrad(const BpropIRBuilder *ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto segment_ids = ib->GetInput(kIndex1);
  auto output = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex3);
  auto input_x_type = ib->GetDtype(input_x);
  if (input_x_type->type_id() != kNumberTypeFloat32) {
    input_x = ib->Cast(input_x, kFloat32);
    output = ib->Cast(output, kFloat32);
    dout = ib->Cast(dout, kFloat32);
  }
  auto zero_value = ib->Value<int64_t>(0);
  auto gathered_outputs = ib->Gather(output, segment_ids, zero_value);
  auto is_selected = ib->Equal(input_x, gathered_outputs);
  const int64_t max_len = 1000000;
  auto num_selected =
    ib->Emit("SegmentSum", {ib->Cast(is_selected, kFloat32), segment_ids}, {{"max_length", MakeValue(max_len)}});
  auto weighted_grads = ib->Div(dout, num_selected);
  auto gathered_grads = ib->Gather(weighted_grads, segment_ids, zero_value);
  auto dx = ib->Select(is_selected, gathered_grads, ib->ZerosLike(input_x));
  if (input_x_type->type_id() != kNumberTypeFloat32) {
    dx = ib->Cast(dx, input_x_type);
  }
  return {dx, ib->ZerosLike(segment_ids)};
}

NodePtrList TensorScatterPossibleReplacement(const BpropIRBuilder *ib) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto updates = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  auto x_indicators = ib->Cast(ib->Equal(x, out), kInt32);
  auto possibly_updated = ib->Emit("GatherNd", {out, indices});
  auto out_indicators = ib->Cast(ib->Equal(updates, possibly_updated), kInt32);
  auto input_shape = ib->Shape(x, true);
  auto scattered_out_indicators = ib->Emit("ScatterNd", {indices, out_indicators, input_shape});
  auto indicators = ib->Add(x_indicators, scattered_out_indicators);
  auto dx = ib->RealDiv((ib->Mul(dout, (ib->Cast(x_indicators, ib->GetDtype(dout))))),
                        (ib->Cast(indicators, ib->GetDtype(dout))));
  auto dupdates =
    ib->Mul((ib->Emit("GatherNd", {ib->RealDiv(dout, (ib->Cast(indicators, ib->GetDtype(dout)))), indices})),
            (ib->Cast(out_indicators, ib->GetDtype(dout))));
  return {ib->Cast(dx, ib->GetDtype(x)), ib->ZerosLike(indices), ib->Cast(dupdates, ib->GetDtype(updates))};
}

ShapeArray RegenerateOutputShapeFunc(const ShapeArray &inputs) {
  constexpr size_t inputs_num = 4;
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == inputs_num, "inputs num should equal to 4.");

  auto x_shape = inputs.at(kIndex0);
  auto indices_shape = inputs.at(kIndex1);
  auto axis = inputs.at(kIndex2);
  MS_EXCEPTION_IF_CHECK_FAIL(axis.size() == 1, "axis should be a scalar.");
  auto axis_value = axis[0];
  auto batch_dims = inputs.at(kIndex3);
  MS_EXCEPTION_IF_CHECK_FAIL(batch_dims.size() == 1, "batch_dims should be a scalar.");
  auto batch_dims_value = batch_dims[kIndex0];

  auto out_shape = RegenerateOutputShape(x_shape, indices_shape, axis_value, batch_dims_value);
  return {out_shape};
}

ShapeVector RegenerateOutputInferFunc(const ShapeArray &inputs, const std::unordered_set<size_t> &invalid_indices) {
  constexpr size_t inputs_num = 4;
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == inputs_num, "inputs num should equal to 4.");

  auto x = inputs.at(kIndex0);
  auto indices = inputs.at(kIndex1);
  auto batch_dims = inputs.at(kIndex3);
  if (!invalid_indices.empty() || IsDynamicRank(x) || IsDynamicRank(indices) || IsDynamicRank(batch_dims)) {
    return {-1};
  }

  auto x_rank = x.size();
  auto indices_rank = indices.size();
  MS_EXCEPTION_IF_CHECK_FAIL(batch_dims.size() == 1, "batch_dims should be a scalar.");
  auto batch_dims_value = batch_dims[0];

  return {SizeToLong(x_rank + indices_rank - LongToSize(batch_dims_value))};
}

ShapeArray PermsShapeFunc(const ShapeArray &inputs) {
  constexpr size_t inputs_num = 5;
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == inputs_num, "inputs num should equal to 4.");

  auto x_shape = inputs.at(kIndex0);
  auto dout_shape = inputs.at(kIndex1);
  auto indices_shape = inputs.at(kIndex2);
  auto axis = inputs.at(kIndex3);
  MS_EXCEPTION_IF_CHECK_FAIL(axis.size() == 1, "axis should be a scalar.");
  auto axis_value = axis[0];
  auto batch_dims = inputs.at(kIndex4);
  MS_EXCEPTION_IF_CHECK_FAIL(batch_dims.size() == 1, "batch_dims should be a scalar.");
  auto batch_dims_value = batch_dims[0];

  auto perm_1 = GenerateShapeIndex(dout_shape, indices_shape, axis_value, batch_dims_value);
  auto perm_2 = GenerateInverseIndex(x_shape, axis_value, batch_dims_value);

  return {perm_1, perm_2};
}

ShapeVector PermsInferFunc(const ShapeArray &inputs, const std::unordered_set<size_t> &invalid_indices) {
  auto x = inputs.at(kIndex0);
  auto dout = inputs.at(kIndex1);
  if (!invalid_indices.empty() || IsDynamicRank(x) || IsDynamicRank(dout)) {
    return {-1, -1};
  }

  return {SizeToLong(dout.size()), SizeToLong(x.size())};
}

NodePtr CalcNumSegment(const BpropIRBuilder *ib, const NodePtr &x, const NodePtr &axis) {
  MS_EXCEPTION_IF_NULL(ib);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(axis);
  auto shape_func = [](const ShapeArray &inputs) -> ShapeArray {
    auto x_shp = inputs.at(kIndex0);
    auto axis_v = inputs.at(kIndex1)[0];
    axis_v = RecorrectAxis(axis_v, x_shp.size());
    return {{x_shp[LongToSize(axis_v)]}};
  };
  auto rank_func = [](const ShapeArray &inputs, const std::unordered_set<size_t> &) -> ShapeVector { return {1}; };
  auto num_segment = ib->ShapeCalc({x, axis}, shape_func, rank_func, {1})[0];
  if (num_segment->isa<ValueNode>()) {
    auto num_segment_value = GetIntList(num_segment);
    MS_EXCEPTION_IF_CHECK_FAIL(num_segment_value.size() == 1,
                               "The num_segment should be a int for gradient of Gather.");
    num_segment = ib->Value(num_segment_value[0]);
  } else {
    num_segment = ib->Reshape(num_segment, ShapeVector{});
  }
  return num_segment;
}

ShapeArray GatherReshapeShapeFunc(const ShapeArray &inputs) {
  constexpr size_t inputs_num = 5;
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == inputs_num, "inputs num should equal to 5.");

  auto values_shape = inputs.at(0);
  auto indices_shape = inputs.at(1);
  auto x_shape = inputs.at(2);
  auto axis = inputs.at(3);
  MS_EXCEPTION_IF_CHECK_FAIL(axis.size() == 1, "axis should be a scalar.");
  auto axis_value = axis[0];
  if (axis_value < 0) {
    axis_value += SizeToLong(x_shape.size());
  }

  auto batch_dims = inputs.at(4);
  MS_EXCEPTION_IF_CHECK_FAIL(batch_dims.size() == 1, "batch_dims should be a scalar.");
  auto batch_dims_value = batch_dims[0];

  MS_EXCEPTION_IF_CHECK_FAIL(x_shape.size() > LongToSize(axis_value), "axis should within interval: [0, params_rank).");
  MS_EXCEPTION_IF_CHECK_FAIL(axis_value >= batch_dims_value, "axis can not less than batch_dims.");
  int64_t batch_size = 1;
  for (int64_t i = 0; i < batch_dims_value; i++) {
    batch_size *= x_shape[i];
  }
  auto axis_dim = x_shape[axis_value];

  std::vector<int64_t> values_reshape = {-1};
  (void)values_reshape.insert(values_reshape.end(), values_shape.begin() + batch_dims_value, values_shape.end());

  std::vector<int64_t> indices_reshape = {-1};
  (void)indices_reshape.insert(indices_reshape.end(), indices_shape.begin() + batch_dims_value, indices_shape.end());

  std::vector<int64_t> delta_reshape = {batch_size};
  auto indices_rank = SizeToLong(indices_reshape.size());
  for (int64_t i = 0; i < indices_rank - 1; i++) {
    delta_reshape.push_back(1);
  }

  std::vector<int64_t> params_grad_reshape(values_shape.begin(), values_shape.begin() + batch_dims_value);
  params_grad_reshape.push_back(axis_dim);
  (void)params_grad_reshape.insert(params_grad_reshape.end(), values_reshape.begin() + indices_rank,
                                   values_reshape.end());

  ShapeArray res = {values_reshape, indices_reshape, delta_reshape, params_grad_reshape};
  return res;
}

ShapeVector GatherReshapeInferFunc(const ShapeArray &inputs, const std::unordered_set<size_t> &invalid_indices) {
  constexpr size_t inputs_num = 5;
  MS_EXCEPTION_IF_CHECK_FAIL(inputs.size() == inputs_num, "inputs num should equal to 5.");

  auto values = inputs.at(0);
  auto indices = inputs.at(1);
  auto params_grad = inputs.at(2);
  auto batch_dims = inputs.at(4);

  constexpr size_t return_num = 4;
  if (!invalid_indices.empty() || IsDynamicRank(values) || IsDynamicRank(indices) || IsDynamicRank(params_grad) ||
      IsDynamicRank(batch_dims)) {
    return ShapeVector(return_num, -1);
  }

  MS_EXCEPTION_IF_CHECK_FAIL(batch_dims.size() == 1, "batch_dims should be a scalar.");
  auto batch_dims_value = batch_dims[0];

  auto values_rank = SizeToLong(values.size()) - batch_dims_value + 1;
  auto indices_rank = SizeToLong(indices.size()) - batch_dims_value + 1;
  auto delta_rank = indices_rank;
  auto params_grad_rank = SizeToLong(params_grad.size());

  ShapeVector res = {values_rank, indices_rank, delta_rank, params_grad_rank};
  return res;
}

NodePtr CalBatchGather(const BpropIRBuilder *ib, const NodePtr &values, const NodePtr &indices, const NodePtr &x,
                       int64_t axis, int64_t batch_dims) {
  auto x_shape = ib->Shape(x, true);
  auto batch_size = ib->Tensor(1, kInt64);
  for (int64_t i = 0; i < batch_dims; i++) {
    batch_size = ib->Mul(batch_size, ib->TensorGetItem(x_shape, i));
  }
  auto axis_dim = ib->TensorGetItem(x_shape, axis);

  auto reshape_shape = ib->ShapeCalc({values, indices, x, ib->Tensor(axis), ib->Tensor(batch_dims)},
                                     GatherReshapeShapeFunc, GatherReshapeInferFunc, {3, 4});
  constexpr size_t reshape_size = 4;
  MS_EXCEPTION_IF_CHECK_FAIL(reshape_shape.size() == reshape_size, "reshape_shape should equal to 4.");
  auto values_reshape = reshape_shape[0];
  auto indices_reshape = reshape_shape[1];
  auto delta_reshape = reshape_shape[2];
  auto params_grad_reshape = reshape_shape[3];

  auto values_rshp = ib->Reshape(values, values_reshape);
  auto indices_rshp = ib->Reshape(indices, indices_reshape);
  auto limit = ib->Cast(ib->Mul(batch_size, axis_dim), kInt64);
  constexpr int64_t range_max_len = 1000000;
  auto delta = ib->Emit("Range", {ib->Tensor(0, kInt64), limit, ib->Cast(axis_dim, kInt64)},
                        {{"maxlen", MakeValue(range_max_len)}});
  delta = ib->Reshape(delta, delta_reshape);
  indices_rshp = ib->Add(indices_rshp, delta);
  auto params_grad = ib->Emit("UnsortedSegmentSum", {values_rshp, indices_rshp, limit});
  params_grad = ib->Reshape(params_grad, params_grad_reshape);
  return params_grad;
}

bool IsMutable(const NodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(node->get());
  ValuePtr value_ptr = nullptr;
  if (node->isa<ValueNode>()) {
    value_ptr = node->get<ValueNodePtr>()->value();
  } else {
    auto abstract = node->abstract();
    if (abstract != nullptr) {
      value_ptr = abstract->BuildValue();
    }
  }
  if (value_ptr != nullptr &&
      (value_ptr->isa<ValueSequence>() || value_ptr->isa<Scalar>() || value_ptr->isa<tensor::Tensor>())) {
    return false;
  }
  return true;
}

NodePtrList BinopGatherCommon(const BpropIRBuilder *ib) {
  auto batch_dims = ib->GetAttr<int64_t>(kAttrBatchDims);
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto axis = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto x_shp = ib->GetShape(x);
  auto out_shp = ib->GetShape(dout);
  auto ind_shp = ib->GetShape(indices);

  if (out_shp.empty()) {
    dout = ib->Emit("ExpandDims", {dout, ib->Tensor(-1)});
  }

  int64_t axis_v = 0;
  MS_EXCEPTION_IF_NULL(axis);
  MS_EXCEPTION_IF_NULL(axis->abstract());
  auto axis_tmp = axis->abstract()->BuildValue();
  MS_EXCEPTION_IF_NULL(axis_tmp);
  if (axis_tmp->isa<tensor::Tensor>()) {
    axis_v = CheckAndConvertUtils::CheckTensorIntValue("axis value", axis_tmp, "Gather")[0];
  } else {
    axis_v = CheckRange(GetIntValue(axis), SizeToLong(x_shp.size()));
  }
  if (batch_dims < 0) {
    batch_dims += SizeToLong(ind_shp.size());
  }

  auto is_axis_mutable = IsMutable(axis);
  if ((!is_axis_mutable && (IsDynamicRank(x_shp) || IsDynamicRank(ind_shp) || IsDynamicRank(out_shp))) ||
      (is_axis_mutable && (IsDynamic(x_shp) || IsDynamic(ind_shp) || IsDynamic(out_shp)))) {
    auto batch_dims_tensor = ib->Tensor(batch_dims, kInt64);
    if (ind_shp.empty()) {
      indices = ib->Emit("ExpandDims", {indices, ib->Tensor(-1)});

      auto out_shp1 = ib->ShapeCalc({x, indices, axis, batch_dims_tensor}, RegenerateOutputShapeFunc,
                                    RegenerateOutputInferFunc, {kIndex2, kIndex3})[0];
      dout = ib->Reshape(dout, out_shp1);
    }

    // Calculate perm.
    auto perms =
      ib->ShapeCalc({x, dout, indices, axis, batch_dims_tensor}, PermsShapeFunc, PermsInferFunc, {kIndex3, kIndex4});
    const size_t perm_num = 2;
    MS_EXCEPTION_IF_CHECK_FAIL(perms.size() == perm_num, "Perms number should be 2 for gradient of Gather.");
    auto perm_1 = perms[0];
    auto perm_2 = perms[1];
    auto values_transpose = ib->Transpose(dout, perm_1);
    NodePtr x_grad = nullptr;
    if (batch_dims > 0) {
      x_grad = CalBatchGather(ib, values_transpose, indices, x, axis_v, batch_dims);
    } else {
      auto num_segment = CalcNumSegment(ib, x, axis);
      x_grad = ib->Emit("UnsortedSegmentSum", {values_transpose, indices, num_segment});
    }
    x_grad = ib->Transpose(x_grad, perm_2);
    return {x_grad, ib->ZerosLike(indices), ib->ZerosLike(axis)};
  }

  if (ind_shp.empty()) {
    indices = ib->Emit("ExpandDims", {indices, ib->Tensor(-1)});
    ind_shp = ib->GetShape(indices);
    auto out_shp1 = RegenerateOutputShape(x_shp, ind_shp, axis_v);
    dout = ib->Reshape(dout, out_shp1);
  }

  out_shp = ib->GetShape(dout);
  auto perm_1 = GenerateShapeIndex(out_shp, ind_shp, axis_v, batch_dims);
  auto values_transpose = ib->Transpose(dout, perm_1);
  NodePtr x_grad = nullptr;
  if (batch_dims > 0) {
    x_grad = CalBatchGather(ib, values_transpose, indices, x, axis_v, batch_dims);
  } else {
    auto num_segment = CalcNumSegment(ib, x, axis);
    x_grad = ib->Emit("UnsortedSegmentSum", {values_transpose, indices, num_segment});
  }
  auto perm_2 = GenerateInverseIndex(x_shp, axis_v, batch_dims);
  auto params_grad = ib->Transpose(x_grad, perm_2);
  return {params_grad, ib->ZerosLike(indices), ib->ZerosLike(axis)};
}

ShapeArray ConcatOffsetCal(const ShapeArray &input_shapes, size_t axis_s) {
  ShapeArray res;
  auto rank = input_shapes[0].size();
  auto input_num = input_shapes.size();
  int64_t sum_axis = 0;
  for (size_t i = 0; i < input_num; ++i) {
    std::vector<int64_t> offset(rank, 0);
    offset[axis_s] = sum_axis;
    sum_axis += input_shapes.at(i)[axis_s];
    res.push_back(offset);
  }
  return res;
}

NodePtrList ConcatBpropStatic(const BpropIRBuilder *ib, const NodePtr &dout, const ShapeArray &input_shapes,
                              int64_t axis, bool is_list) {
  auto rank = input_shapes[0].size();
  auto axis_s = LongToSize(RecorrectAxis(axis, rank));

  bool is_uniform = true;
  auto input_nums = input_shapes.size();
  for (size_t i = 0; i < input_nums; ++i) {
    if (input_shapes[i].size() != rank) {
      MS_EXCEPTION(ValueError) << "For gradient of 'Concat', input shapes [" << i
                               << "] and input shapes [0] must have same rank, but got: " << input_shapes[i].size()
                               << " vs " << rank;
    }
    if (input_shapes[i][axis_s] != input_shapes[0][axis_s]) {
      is_uniform = false;
    }
  }

  NodePtrList res;
  if (is_uniform) {
    auto long_nums = SizeToLong(input_nums);
    auto dx = ib->Emit(
      kSplitOpName, {dout},
      {{kAttrAxis, MakeValue(axis)}, {kAttrOutputNum, MakeValue(long_nums)}, {"num_split", MakeValue(long_nums)}});
    // Split output is a tuple.
    if (!is_list) {
      return {dx};
    }

    for (size_t i = 0; i < input_nums; ++i) {
      res.push_back(ib->TupleGetItem(dx, i));
    }
  } else {
    auto offsets = ConcatOffsetCal(input_shapes, axis_s);
    for (size_t i = 0; i < input_nums; ++i) {
      auto offset_value = ib->Value(offsets[i]);
      auto slice_out = ib->Emit(kSliceOpName, {dout, offset_value, ib->Value(input_shapes[i])});
      res.push_back(slice_out);
    }
  }

  if (is_list) {
    return {ib->MakeList(res)};
  }
  return {ib->MakeTuple(res)};
}

NodePtrList StackBpropFunc(const BpropIRBuilder *ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto num = ib->GetAttr("num");
  auto ret = ib->Emit("Unstack", {dout}, {{"num", num}, {"axis", ib->GetAttr("axis")}});

  auto x_abs = x->get()->abstract();
  MS_EXCEPTION_IF_NULL(x_abs);
  bool is_list = x_abs->isa<abstract::AbstractList>();
  if (is_list) {
    NodePtrList res;
    auto num_v = LongToSize(GetValue<int64_t>(num));
    for (size_t i = 0; i < num_v; ++i) {
      res.push_back(ib->TupleGetItem(ret, i));
    }
    return {ib->MakeList(res)};
  }
  return {ret};
}

NodePtrList BinopGatherDGradCommon(const BpropIRBuilder *ib, const std::string &op_name) {
  auto dim = LongToSize(GetValue<int64_t>(ib->GetAttr("dim")));
  auto index = ib->GetInput(kIndex0);
  auto x = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  ShapeVector x_shp;
  if (op_name == "GatherDGrad") {
    x_shp = GetValue<ShapeVector>(ib->GetAttr("shape"));
  } else {
    x_shp = ib->GetShape(x);
  }
  auto index_shp = ib->GetShape(index);
  int64_t dim_before_axis = 1;
  for (size_t i = 0; i < dim; ++i) {
    dim_before_axis *= x_shp[i];
  }
  auto dim_at_axis_index = index_shp[dim];
  auto dim_at_axis_output = x_shp[dim];
  int64_t dim_after_axis = 1;
  for (size_t i = dim + 1; i < x_shp.size(); ++i) {
    dim_after_axis *= x_shp[i];
  }
  auto element = (dim_before_axis * dim_at_axis_index) * dim_after_axis;
  auto index_type = ib->GetDtype(index);
  auto id = ib->Tensor(Range(element), index_type);
  auto i = ib->FloorDiv(id, ib->Tensor((dim_at_axis_index * dim_after_axis), index_type));
  auto k = ib->FloorMod(id, ib->Tensor(dim_after_axis, index_type));
  auto less = ib->Less(index, ib->Tensor(0, index_type));
  auto j = ib->Cast(less, index_type);
  auto j_read = ib->Add((ib->Mul(ib->Tensor(dim_at_axis_index, index_type), j)), index);
  auto j_read_reshape = ib->Reshape(j_read, {-1});
  auto i_after = ib->Mul(i, ib->Tensor(dim_at_axis_output * dim_after_axis, index_type));
  auto read_id = ib->Add((ib->Add(i_after, (ib->Mul(j_read_reshape, ib->Tensor(dim_after_axis, index_type))))), k);
  auto dout_reshape = ib->Reshape(dout, {-1});
  auto dx = ib->Gather(dout_reshape, read_id, ib->Tensor(0));
  dx = ib->Reshape(dx, ib->GetShape(x));
  return {ib->ZerosLike(index), dx};
}
}  // namespace

REG_BPROP_BUILDERS_BEGIN(GradArrayOps)
REG_BPROP_BUILDER("GatherD").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dim = ib->GetInput(kIndex1);
  auto index = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx = ib->Emit("GatherDGradV2", {x, dim, index, dout});
  return {dx, ib->ZerosLike(dim), ib->ZerosLike(index)};
});

REG_BPROP_BUILDER("GatherDGrad").SetUnusedInputs({i1, i2}).SetBody(BODYFUNC(ib) {
  return BinopGatherDGradCommon(ib, "GatherDGrad");
});

REG_BPROP_BUILDER("GatherDGradV2").SetUnusedInputs({i1, i2}).SetBody(BODYFUNC(ib) {
  return BinopGatherDGradCommon(ib, "GatherDGradV2");
});

REG_BPROP_BUILDER("SparseGatherV2").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto axis = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto x_shp = ib->GetShape(x);
  auto axis_int = CheckRange(GetValue<int64_t>(axis->get<ValueNodePtr>()->value()), SizeToLong(x_shp.size()));
  if (axis_int == 0) {
    ShapeVector values_shape{ib->GetSize(indices)};
    if (x_shp.size() > 1) {
      (void)values_shape.insert(values_shape.end(), x_shp.begin() + 1, x_shp.end());
    }
    auto values = ib->Reshape(dout, values_shape);
    auto indices_new = ib->Reshape(indices, {values_shape[0]});
    auto row_tensor = ib->MakeTuple({indices_new, values, ib->Value<ShapeVector>(x_shp)});
    return {row_tensor, ib->ZerosLike(indices), ib->ZerosLike(axis)};
  }
  auto out_shp = ib->GetShape(dout);
  auto ind_shp = ib->GetShape(indices);
  if (out_shp.size() == 0) {
    dout = ib->ExpandDims(dout, -1);
  }
  if (ind_shp.size() == 0) {
    indices = ib->ExpandDims(indices, -1);
  }
  out_shp = ib->GetShape(dout);
  ind_shp = ib->GetShape(indices);
  auto perm_1 = GenerateShapeIndex(out_shp, ind_shp, axis_int);
  auto values_transpose = ib->Transpose(dout, perm_1);
  auto params_grad =
    ib->Emit("UnsortedSegmentSum", {values_transpose, indices, ib->Value<int64_t>(x_shp[LongToSize(axis_int)])});
  auto perm_2 = GenerateInverseIndex(x_shp, axis_int);
  params_grad = ib->Transpose(params_grad, perm_2);
  return {params_grad, ib->ZerosLike(indices), ib->ZerosLike(axis)};
});

REG_BPROP_BUILDER("Sort").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto axis = GetValue<int64_t>(ib->GetAttr("axis"));
  auto descending = GetValue<bool>(ib->GetAttr("descending"));
  auto input_x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);

  auto shape_func = [axis](const ShapeArray &inputs) -> ShapeArray {
    auto x_shape = inputs[0];
    auto x_rank = x_shape.size();
    auto recorrect_axis = RecorrectAxis(axis, x_rank);
    ShapeVector transposition;
    ShapeVector invert_perm;
    if (LongToSize(recorrect_axis + 1) == x_rank) {
      // A (0, 1, 2, ...) will change Transpose as a copy-like operator.
      // This can delete two control flow block.
      transposition = Range(x_rank);
      invert_perm = Range(x_rank);
    } else {
      transposition = GetTransposition(recorrect_axis, x_rank);
      invert_perm = InvertPermutation(transposition);
    }

    auto k = x_shape.at(LongToSize(recorrect_axis));
    return {{k}, transposition, invert_perm};
  };

  auto infer_func = [](const ShapeArray &inputs, const std::unordered_set<size_t> &invalid_indices) -> ShapeVector {
    auto x = inputs.at(0);
    if (!invalid_indices.empty() || IsDynamicRank(x)) {
      return {1, -1, -1};
    }

    auto x_rank = SizeToLong(x.size());
    return {1, x_rank, x_rank};
  };

  auto res1 = ib->ShapeCalc({input_x}, shape_func, infer_func, {});
  auto k = res1[0];
  if (k->abstract()->isa<abstract::AbstractSequence>()) {
    if (k->isa<ValueNode>()) {
      auto value = GetIntList(k);
      k = ib->Tensor(value.at(0), kInt64);
    } else {
      k = ib->TupleGetItem(k, 0);
    }
  }
  auto transposition = ib->TupleToTensor(res1[1]);
  auto invert_perm = ib->TupleToTensor(res1[2]);
  auto dvalue = ib->TupleGetItem(dout, 0);
  if (!descending) {
    input_x = ib->Neg(input_x);
    dvalue = ib->Neg(dvalue);
  }

  auto top_k_input = ib->Transpose(input_x, transposition);
  auto tmp = ib->Emit("TopK", {top_k_input, k}, {{"sorted", MakeValue(true)}});
  auto indices = ib->TupleGetItem(tmp, 1);

  auto shape_func1 = [](const ShapeArray &inputs) -> ShapeArray {
    auto indices_shape = inputs[0];
    auto top_k_input_shape = inputs[1];

    auto indices_rank = indices_shape.size();
    auto top_k_input_rank = top_k_input_shape.size();
    if (indices_rank < 1 || top_k_input_rank < 1) {
      MS_LOG(EXCEPTION) << "For Sort, indices rank and top k rank should not less than 1, but got " << indices_rank
                        << " and " << top_k_input_rank;
    }
    auto ind_lastdim = indices_shape.at(indices_rank - 1);
    auto in_lastdim = top_k_input_shape.at(top_k_input_rank - 1);
    auto x_size = std::accumulate(top_k_input_shape.begin(), top_k_input_shape.end(), 1, std::multiplies<int64_t>());

    auto outer_dim = std::accumulate(indices_shape.begin(), indices_shape.end() - 1, 1, std::multiplies<int64_t>());
    return {top_k_input_shape, {-1, ind_lastdim}, {in_lastdim}, {x_size}, {outer_dim * in_lastdim}};
  };

  auto infer_func1 = [](const ShapeArray &inputs, const std::unordered_set<size_t> &invalid_indices) -> ShapeVector {
    if (invalid_indices.count(1) != 0) {
      return {-1, 2, 1, 1, 1};
    }

    auto top_k_input_shape = inputs[1];
    auto top_k_input_rank = top_k_input_shape.size();
    return {SizeToLong(top_k_input_rank), 2, 1, 1, 1};
  };

  auto res = ib->ShapeCalc({indices, top_k_input}, shape_func1, infer_func1);
  auto indices_dtype = ib->GetDtype(indices);
  auto range_flatten_index =
    ib->Cast(ib->Range(ib->Tensor(0, kInt64), ib->TupleToTensor(res[4]), ib->TupleToTensor(res[2])), indices_dtype);
  range_flatten_index = ib->ExpandDims(range_flatten_index, -1);
  auto ind_2d = ib->Reshape(indices, res[1]);
  auto ind = ib->Reshape(ib->Add(ind_2d, range_flatten_index), {-1});

  dvalue = ib->Transpose(dvalue, invert_perm);
  auto ind_expand = ib->ExpandDims(ind, -1);
  auto scatter = ib->Emit("ScatterNd", {ind_expand, ib->Reshape(dvalue, {-1}), res[3]});
  auto out_grad = ib->Reshape(scatter, res[0]);
  auto dx = ib->Transpose(out_grad, invert_perm);

  if (!descending) {
    dx = ib->Neg(dx);
  }
  return NodePtrList{dx};
});

REG_BPROP_BUILDER("Identity").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  return {dout};
});

REG_BPROP_BUILDER("Range").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(BODYFUNC(ib) {
  auto start = ib->GetInput(kIndex0);
  auto limit = ib->GetInput(kIndex1);
  auto delta = ib->GetInput(kIndex2);
  return {ib->ZerosLike(start), ib->ZerosLike(limit), ib->ZerosLike(delta)};
});

REG_BPROP_BUILDER("Pack").SetUnusedInputs({i0, i1}).SetBody(StackBpropFunc);
REG_BPROP_BUILDER("Stack").SetUnusedInputs({i0, i1}).SetBody(StackBpropFunc);

REG_BPROP_BUILDER("ReverseV2").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("ReverseV2", {dout}, {{"axis", ib->GetAttr("axis")}});
  return {dx};
});

REG_BPROP_BUILDER("Unstack").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto out = ib->Emit("Stack", {dout}, {{"axis", ib->GetAttr("axis")}});
  return {out};
});

REG_BPROP_BUILDER("StridedSlice").SetUnusedInputs({i0, i4}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto begin = ib->GetInput(kIndex1);
  auto end = ib->GetInput(kIndex2);
  auto strides = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  auto x_shape_vec = ib->GetShape(x);

  NodePtr x_shape_node;
  if (IsDynamic(x_shape_vec)) {
    x_shape_node = ib->Shape(x);
  } else {
    x_shape_node = ib->EmitValue(MakeValue(x_shape_vec));
  }
  auto dx = ib->Emit("StridedSliceGrad", {dout, x_shape_node, begin, end, strides},
                     {{"begin_mask", ib->GetAttr("begin_mask")},
                      {"end_mask", ib->GetAttr("end_mask")},
                      {"ellipsis_mask", ib->GetAttr("ellipsis_mask")},
                      {"new_axis_mask", ib->GetAttr("new_axis_mask")},
                      {"shrink_axis_mask", ib->GetAttr("shrink_axis_mask")}});
  auto dbegin = ib->ZerosLike(begin);
  auto dend = ib->ZerosLike(end);
  auto dstrides = ib->ZerosLike(strides);
  return {dx, dbegin, dend, dstrides};
});

REG_BPROP_BUILDER("StridedSliceGrad").SetUnusedInputs({i0, i1, i5}).SetBody(BODYFUNC(ib) {
  auto shapex = ib->GetInput(kIndex1);
  auto begin = ib->GetInput(kIndex2);
  auto end = ib->GetInput(kIndex3);
  auto strides = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex6);
  return {ib->Emit("StridedSlice", {dout, begin, end, strides},
                   {{"begin_mask", ib->GetAttr("begin_mask")},
                    {"end_mask", ib->GetAttr("end_mask")},
                    {"ellipsis_mask", ib->GetAttr("ellipsis_mask")},
                    {"new_axis_mask", ib->GetAttr("new_axis_mask")},
                    {"shrink_axis_mask", ib->GetAttr("shrink_axis_mask")}}),
          ib->ZerosLike(shapex), ib->ZerosLike(begin), ib->ZerosLike(end), ib->ZerosLike(strides)};
});

REG_BPROP_BUILDER("Eye").SetUnusedInputs({i0, i1, i3, i4}).SetBody(BODYFUNC(ib) {
  auto n = ib->GetInput(kIndex0);
  auto m = ib->GetInput(kIndex1);
  auto t = ib->GetInput(kIndex2);
  return {ib->ZerosLike(n), ib->ZerosLike(m), t};
});

REG_BPROP_BUILDER("Select").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  auto cond = ib->GetInput(kIndex0);
  auto x = ib->GetInput(kIndex1);
  auto y = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  return {ib->ZerosLike(cond), ib->Select(cond, dout, ib->ZerosLike(x)), ib->Select(cond, ib->ZerosLike(y), dout)};
});

REG_BPROP_BUILDER("OnesLike").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("ZerosLike").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("ResizeNearestNeighbor").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto x_shape = ib->GetShape(x);
  NodePtr shape;
  if (!IsDynamic(x_shape)) {
    ShapeVector new_shape;
    for (size_t i = 2; i < x_shape.size(); i++) {
      new_shape.push_back(x_shape[i]);
    }
    shape = ib->EmitValue(MakeValue(new_shape));
  } else {
    auto shape_func = [](const ShapeArray &inputs) -> ShapeArray {
      auto shape = inputs[0];
      ShapeVector res;
      for (size_t i = 2; i < shape.size(); ++i) {
        res.push_back(shape[i]);
      }
      return {res};
    };

    auto infer_func = [](const ShapeArray &inputs, const std::unordered_set<size_t> &invalid_indices) -> ShapeVector {
      auto x = inputs.at(0);
      if (!invalid_indices.empty() || IsDynamicRank(x)) {
        return {-1};
      }
      auto rank = SizeToLong(x.size());
      return {rank > 2 ? (rank - 2) : 0};
    };

    shape = ib->ShapeCalc({x}, shape_func, infer_func, {})[0];
  }

  auto out = ib->Emit("ResizeNearestNeighborGrad", {dout, shape}, {{"align_corners", ib->GetAttr("align_corners")}});
  return {out};
});

REG_BPROP_BUILDER("GatherNd").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto x_shp = ib->GetShape(x);
  NodePtr shp;
  if (IsDynamic(x_shp)) {
    shp = ib->Shape(x, true);
  } else {
    shp = ib->EmitValue(MakeValue(x_shp));
  }
  return {ib->Emit("ScatterNd", {indices, dout, shp}), ib->ZerosLike(indices)};
});

REG_BPROP_BUILDER("ScatterNd").SetUnusedInputs({i1, i2, i3}).SetBody(BODYFUNC(ib) {
  auto indices = ib->GetInput(kIndex0);
  auto shape = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  return {ib->ZerosLike(indices), ib->Emit("GatherNd", {dout, indices}), ib->ZerosLike(shape)};
});

REG_BPROP_BUILDER("ScatterNdUpdate").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  auto indices = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  return {dout, ib->ZerosLike(indices), ib->Emit("GatherNd", {dout, indices})};
});

REG_BPROP_BUILDER("ScatterNonAliasingAdd").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  auto indices = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  return {dout, ib->ZerosLike(indices), ib->Emit("GatherNd", {dout, indices})};
});

REG_BPROP_BUILDER("TensorScatterUpdate").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  auto indices = ib->GetInput(kIndex1);
  auto update = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto x_grad = ib->Emit("TensorScatterUpdate", {dout, indices, ib->ZerosLike(update)});
  auto update_grad = ib->Emit("GatherNd", {dout, indices});
  return {x_grad, ib->ZerosLike(indices), update_grad};
});

REG_BPROP_BUILDER("Flatten").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto x_shape = ib->GetShape(x);
  if (IsDynamic(x_shape)) {
    return {ib->Reshape(dout, ib->Shape(x))};
  }
  return {ib->Reshape(dout, x_shape)};
});

REG_BPROP_BUILDER("Reshape").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto shp = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto shape_x = ib->GetShape(x);
  NodePtr dx;
  if (!IsDynamic(shape_x)) {
    dx = ib->Reshape(dout, shape_x);
  } else {
    dx = ib->Reshape(dout, ib->Shape(x));
  }
  return {dx, ib->ZerosLike(shp)};
});

REG_BPROP_BUILDER("NonZero").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("Argmax").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("Argmin").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("Diag").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  return {ib->Emit("DiagPart", {dout})};
});

REG_BPROP_BUILDER("DiagPart").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  return {ib->Emit("Diag", {dout})};
});

REG_BPROP_BUILDER("SpaceToBatch").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto dx =
    ib->Emit("BatchToSpace", {dout}, {{"block_size", ib->GetAttr("block_size")}, {"crops", ib->GetAttr("paddings")}});
  return {dx};
});

REG_BPROP_BUILDER("BatchToSpace").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto dx =
    ib->Emit("SpaceToBatch", {dout}, {{"block_size", ib->GetAttr("block_size")}, {"paddings", ib->GetAttr("crops")}});
  return {dx};
});

REG_BPROP_BUILDER("ReverseSequence").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto seq_lengths = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("ReverseSequence", {dout, seq_lengths},
                     {{"batch_dim", ib->GetAttr("batch_dim")}, {"seq_dim", ib->GetAttr("seq_dim")}});
  return {dx, ib->ZerosLike(seq_lengths)};
});

REG_BPROP_BUILDER("TensorScatterAdd").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  auto indices = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  auto update_grad = ib->Emit("GatherNd", {dout, indices});
  return {dout, ib->ZerosLike(indices), update_grad};
});

REG_BPROP_BUILDER("Concat").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto axis = ib->GetAttr<int64_t>(kAttrAxis);
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto input_shapes = ib->GetShapes(x);
  if (input_shapes.empty()) {
    MS_EXCEPTION(ValueError) << "For gradient of 'Concat', 'x' can not be empty";
  }

  bool is_dynamic = std::any_of(input_shapes.cbegin(), input_shapes.cend(),
                                [](const std::vector<int64_t> &shape) { return IsDynamic(shape); });
  auto x_abs = x->get()->abstract();
  MS_EXCEPTION_IF_NULL(x_abs);
  bool is_list = x_abs->isa<abstract::AbstractList>();

  if (!is_dynamic) {
    return ConcatBpropStatic(ib, dout, input_shapes, axis, is_list);
  }

  auto input_nums = input_shapes.size();

  auto shape_func = [axis](const ShapeArray &inputs) -> ShapeArray {
    auto rank = inputs[0].size();
    auto axis_s = LongToSize(RecorrectAxis(axis, rank));
    return ConcatOffsetCal(inputs, axis_s);
  };

  auto infer_func = [](const ShapeArray &inputs, const std::unordered_set<size_t> &invalid_indices) -> ShapeVector {
    auto x = inputs.at(0);
    auto input_num = inputs.size();
    if (!invalid_indices.empty() || IsDynamicRank(x)) {
      return ShapeVector(input_num, -1);
    }
    return ShapeVector(input_num, SizeToLong(x.size()));
  };

  NodePtrList x_tuple;
  for (size_t i = 0; i < input_nums; ++i) {
    x_tuple.push_back(ib->TupleGetItem(x, i));
  }
  auto concat_offset = ib->ShapeCalc(x_tuple, shape_func, infer_func, {});
  NodePtrList res;
  for (size_t i = 0; i < input_nums; ++i) {
    auto input = ib->Shape(ib->TupleGetItem(x, i));
    auto slice_out = ib->Emit(kSliceOpName, {dout, concat_offset.at(i), input});
    res.push_back(slice_out);
  }

  if (is_list) {
    return {ib->MakeList(res)};
  }
  return {ib->MakeTuple(res)};
});

REG_BPROP_BUILDER("Mvlgamma").SetUnusedInputs({i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("MvlgammaGrad", {dout, x}, {{"p", ib->GetAttr("p")}});
  return {dx};
});

REG_BPROP_BUILDER("TensorScatterDiv").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto update = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto in_grad = ib->Emit("TensorScatterDiv", {dout, indices, update});
  auto gather_update = ib->Emit("GatherNd", {dout, indices});
  auto gather_x = ib->Emit("GatherNd", {x, indices});
  auto mul_result = ib->Mul(update, update);
  auto neg_result = ib->Emit("Neg", {mul_result});
  auto update_grad = ib->Mul(gather_update, (ib->Div(gather_x, neg_result)));
  return {in_grad, ib->ZerosLike(indices), update_grad};
});

REG_BPROP_BUILDER("TensorScatterSub").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  auto indices = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  auto update_grad = ib->Emit("Neg", {ib->Emit("GatherNd", {dout, indices})});
  return {dout, ib->ZerosLike(indices), update_grad};
});

REG_BPROP_BUILDER("TensorScatterMul").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto update = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto gather_update = ib->Emit("GatherNd", {dout, indices});
  auto gather_x = ib->Emit("GatherNd", {x, indices});
  auto dx = ib->Emit("TensorScatterMul", {dout, indices, update});
  auto d_update = ib->Mul(gather_x, gather_update);
  return {dx, ib->ZerosLike(indices), d_update};
});

REG_BPROP_BUILDER("TensorScatterMax").SetBody(TensorScatterPossibleReplacement);
REG_BPROP_BUILDER("TensorScatterMin").SetBody(TensorScatterPossibleReplacement);

REG_BPROP_BUILDER("IndexFill").SetUnusedInputs({i0, i4}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dim = ib->GetInput(kIndex1);
  auto indices = ib->GetInput(kIndex2);
  auto value = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  auto zero_value = ib->ZerosLike(value);
  auto x_grad = ib->Emit("IndexFill", {dout, dim, indices, zero_value});
  NodePtr value_grad;
  if (ib->GetShape(x).empty()) {
    value_grad = dout;
  } else {
    auto tmp = ib->Gather(dout, indices, dim);
    value_grad = ib->ReduceSum(tmp, ShapeVector());
  }
  return {x_grad, ib->ZerosLike(dim), ib->ZerosLike(indices), value_grad};
});

REG_BPROP_BUILDER("UnsortedSegmentSum").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  auto segment_ids = ib->GetInput(kIndex1);
  auto num_segments = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  return {GatherDropNegatives(ib, dout, segment_ids, nullptr, nullptr)[0], ib->ZerosLike(segment_ids),
          ib->ZerosLike(num_segments)};
});

REG_BPROP_BUILDER("UnsortedSegmentMin").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto segment_ids = ib->GetInput(kIndex1);
  auto num_segments = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  return UnsortedSegmentMinOrMaxGrad(ib, x, segment_ids, num_segments, out, dout);
});

REG_BPROP_BUILDER("UnsortedSegmentMax").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto segment_ids = ib->GetInput(kIndex1);
  auto num_segments = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);
  return UnsortedSegmentMinOrMaxGrad(ib, x, segment_ids, num_segments, out, dout);
});

REG_BPROP_BUILDER("UnsortedSegmentProd").SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto segment_ids = ib->GetInput(kIndex1);
  auto num_segments = ib->GetInput(kIndex2);
  auto out = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex4);

  NodePtr is_zero = nullptr;
  auto x_dtype = ib->GetDtype(x);
  MS_EXCEPTION_IF_NULL(x_dtype);
  auto x_dtype_id = x_dtype->type_id();
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    MS_EXCEPTION(TypeError) << "For 'UnsortedSegmentProd', complex number is not supported for gradient currently.";
  }
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    is_zero = ib->Equal(x, ib->Tensor(0, x_dtype));
  } else {
    is_zero = ib->Equal(ib->Cast(x, kFloat32), ib->Tensor(0, kFloat32));
  }

  auto num_zero = ib->Emit("UnsortedSegmentSum", {ib->Cast(is_zero, kInt32), segment_ids, num_segments});
  auto grad = ib->Select(ib->Greater(num_zero, ib->Tensor(1, ib->GetDtype(num_zero))), ib->ZerosLike(dout), dout);
  NodePtr non_zero_data = nullptr;
  if (x_dtype_id == kNumberTypeComplex64 || x_dtype_id == kNumberTypeComplex128) {
    non_zero_data = ib->Select(is_zero, ib->Emit("OnesLike", {x}), x);
  } else {
    auto temp_var = ib->Emit("OnesLike", {ib->Cast(x, kFloat32)});
    non_zero_data = ib->Select(is_zero, ib->Cast(temp_var, x_dtype_id), x);
  }
  auto non_zero_prod = ib->Emit("UnsortedSegmentProd", {non_zero_data, segment_ids, num_segments});
  auto zero_clipped_indices = ib->Maximum(segment_ids, ib->ZerosLike(segment_ids));
  auto gathered_prod = ib->Gather(out, zero_clipped_indices, 0);
  auto gathered_non_zero_prod = ib->Gather(non_zero_prod, zero_clipped_indices, 0);

  NodePtr prod_divided_by_x = nullptr;
  if (x_dtype_id == kNumberTypeUInt32 || x_dtype_id == kNumberTypeUInt64) {
    prod_divided_by_x = ib->RealDiv(ib->Cast(gathered_prod, kFloat32), ib->Cast(x, kFloat32));
  } else {
    prod_divided_by_x = ib->RealDiv(gathered_prod, x);
  }
  auto partial_derivative =
    ib->Select(is_zero, gathered_non_zero_prod, ib->Cast(prod_divided_by_x, ib->GetDtype(gathered_non_zero_prod)));

  auto temp_outs = GatherDropNegatives(ib, grad, segment_ids, zero_clipped_indices, nullptr);
  MS_EXCEPTION_IF_CHECK_FAIL(temp_outs.size() > 0, "Outputs should not be empty.");
  auto gathered_grad = temp_outs[0];
  NodePtr dx = nullptr;
  if (x_dtype_id == kNumberTypeUInt32 || x_dtype_id == kNumberTypeUInt64) {
    auto temp_dx = ib->Mul(ib->Cast(gathered_grad, kFloat32), ib->Cast(partial_derivative, kFloat32));
    dx = ib->Cast(temp_dx, x_dtype);
  } else {
    dx = ib->Mul(gathered_grad, partial_derivative);
  }

  return {dx, ib->ZerosLike(segment_ids), ib->ZerosLike(num_segments)};
});

REG_BPROP_BUILDER("SpaceToBatchND").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("BatchToSpaceND", {dout},
                     {{"block_shape", ib->GetAttr("block_shape")}, {"crops", ib->GetAttr("paddings")}});
  return {dx};
});

REG_BPROP_BUILDER("BatchToSpaceND").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("SpaceToBatchND", {dout},
                     {{"block_shape", ib->GetAttr("block_shape")}, {"paddings", ib->GetAttr("crops")}});
  return {dx};
});

REG_BPROP_BUILDER("BroadcastTo").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto broadcast_shape = ib->GetAttr<ShapeVector>("shape");

  auto x_shape = ib->GetShape(x);
  auto dout_shape = ib->GetShape(dout);

  bool input_dynamic = IsDynamic(x_shape) || IsDynamic(dout_shape);
  if (!input_dynamic && x_shape == dout_shape) {
    return {dout};
  }

  auto dout_dtype = ib->GetDtype(dout)->type_id();

  NodePtr dx;
  if (!input_dynamic && !IsDynamic(broadcast_shape)) {
    auto tuple_out = BroadcastGradientArgs(broadcast_shape, x_shape);
    MS_EXCEPTION_IF_CHECK_FAIL(!tuple_out.empty(), "BroadcastGradientArgs out should not be empty!");
    auto reduction_axes = tuple_out[kIndex1];
    NodePtr reduced_grad;
    if (dout_dtype == kNumberTypeInt16 || dout_dtype == kNumberTypeInt32 || dout_dtype == kNumberTypeInt64) {
      auto dout_cast = ib->Cast(dout, kFloat32);
      reduced_grad = ib->ReduceSum(dout_cast, reduction_axes, true);
      reduced_grad = ib->Cast(reduced_grad, ib->GetDtype(dout));
    } else {
      reduced_grad = ib->ReduceSum(dout, reduction_axes, true);
    }
    dx = ib->Reshape(reduced_grad, x_shape);
  } else {
    auto x_shape_node = ib->Shape(x, true);
    auto broadcast_shape_node = ib->Shape(dout, true);
    auto brod = ib->Emit("DynamicBroadcastGradientArgs", {broadcast_shape_node, x_shape_node});
    auto reduction_axes = ib->TupleGetItem(brod, 1);
    NodePtr reduced_grad;
    if (dout_dtype == kNumberTypeInt16 || dout_dtype == kNumberTypeInt32 || dout_dtype == kNumberTypeInt64) {
      auto dout_cast = ib->Cast(dout, kFloat32);
      reduced_grad = ib->ReduceSum(dout_cast, reduction_axes, true, true);
      reduced_grad = ib->Cast(reduced_grad, ib->GetDtype(dout));
    } else {
      reduced_grad = ib->ReduceSum(dout, reduction_axes, true, true);
    }
    dx = ib->Reshape(reduced_grad, x_shape_node);
  }

  return {dx};
});

REG_BPROP_BUILDER("SpaceToDepth").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  return {ib->Emit("DepthToSpace", {dout},
                   {{"block_size", ib->GetAttr("block_size")},
                    {"data_format", MakeValue("NCHW")},
                    {"format", ib->GetAttr("format")}})};
});

REG_BPROP_BUILDER("DepthToSpace").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  return {ib->Emit("SpaceToDepth", {dout},
                   {{"block_size", ib->GetAttr("block_size")},
                    {"data_format", MakeValue("NCHW")},
                    {"format", ib->GetAttr("format")}})};
});

REG_BPROP_BUILDER("ScatterMax").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  auto indices = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  return {dout, ib->ZerosLike(indices), ib->Gather(dout, indices, 0)};
});

REG_BPROP_BUILDER("ScatterMin").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  auto indices = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  return {dout, ib->ZerosLike(indices), ib->Gather(dout, indices, 0)};
});

REG_BPROP_BUILDER("ScatterUpdate").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  auto indices = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  return {dout, ib->ZerosLike(indices), ib->Gather(dout, indices, 0)};
});

REG_BPROP_BUILDER("Fills").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto value = ib->GetInput(kIndex1);
  return {ib->ZerosLike(x), ib->ZerosLike(value)};
});

REG_BPROP_BUILDER("Cast").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto t = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto x_dtype = ib->GetDtype(x);
  NodePtr dx;
  if (dout->abstract()->isa<abstract::AbstractRowTensor>()) {
    auto row_tensor_values = ib->Emit("RowTensorGetValues", {dout});
    auto value = ib->Cast(row_tensor_values, x_dtype);
    auto indices = ib->Emit("RowTensorGetIndices", {dout});
    auto dense_shape = ib->Emit("RowTensorGetDenseShape", {dout});
    dx = ib->Emit("MakeRowTensor", {indices, value, dense_shape});
  } else {
    dx = ib->Cast(dout, x_dtype);
  }
  return {dx, ib->ZerosLike(t)};
});

REG_BPROP_BUILDER("ExpandDims").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto axis = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto shape_x = ib->GetShape(x);
  NodePtr dx;
  if (IsDynamic(shape_x)) {
    dx = ib->Reshape(dout, ib->Shape(x));
  } else {
    dx = ib->Reshape(dout, shape_x);
  }
  return {dx, ib->ZerosLike(axis)};
});

REG_BPROP_BUILDER("Squeeze").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto shapex = ib->GetShape(x);
  if (IsDynamic(shapex)) {
    return {ib->Reshape(dout, ib->Shape(x))};
  }
  return {ib->Reshape(dout, shapex)};
});

REG_BPROP_BUILDER("Padding").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex2);
  auto shp = ib->GetShape(x);
  if (!IsDynamic(shp)) {
    std::vector<int64_t> begin(shp.size(), 0);
    auto dx = ib->Emit("Slice", {dout, ib->Value<ShapeVector>(begin), ib->Value<ShapeVector>(shp)});
    return {dx};
  }

  auto shape_node = ib->Shape(x);
  auto begin_node = ib->ZerosLike(shape_node);
  return {ib->Emit("Slice", {dout, begin_node, shape_node})};
});

REG_BPROP_BUILDER("Transpose").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto perm = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto shape_func = [](const ShapeArray &inputs) -> ShapeArray {
    auto perm = inputs[0];
    std::vector<int64_t> new_perm;
    (void)std::transform(perm.begin(), perm.end(), std::back_inserter(new_perm),
                         [&perm](const int64_t v) { return v >= 0 ? v : v + SizeToLong(perm.size()); });
    auto res_perm = InvertPermutation(new_perm);
    return {res_perm};
  };

  auto infer_func = [](const ShapeArray &inputs, const std::unordered_set<size_t> &invalid_indices) -> ShapeVector {
    auto x = inputs.at(0);
    if (!invalid_indices.empty() || IsDynamicRank(x)) {
      return {-1};
    }
    return {SizeToLong(x.size())};
  };

  auto res_perm = ib->ShapeCalc({perm}, shape_func, infer_func, {0})[0];
  return {ib->Transpose(dout, res_perm), ib->ZerosLike(perm)};
});

REG_BPROP_BUILDER("Slice").SetUnusedInputs({i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto begin = ib->GetInput(kIndex1);
  auto size = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto dx = ib->Emit("SliceGrad", {dout, x, begin, size});
  return {dx, ib->ZerosLike(begin), ib->ZerosLike(size)};
});

REG_BPROP_BUILDER("Split").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("Concat", {dout}, {{"axis", ib->GetAttr("axis")}});
  return {dx};
});

REG_BPROP_BUILDER("Tile").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto input_multiples = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);

  auto shape_func = [](const ShapeArray &inputs) -> ShapeArray {
    // {x_shape, multiples}
    auto r_shape = TileShape(inputs.at(1), inputs.at(0));
    ShapeVector axis;
    size_t axis_sz = r_shape.size() / 2;
    axis.reserve(axis_sz);
    for (int64_t i = 0; i < static_cast<int64_t>(axis_sz); ++i) {
      axis.push_back(i * 2);
    }

    return {r_shape, axis};
  };

  auto infer_func = [](const ShapeArray &inputs, const std::unordered_set<size_t> &invalid_indices) -> ShapeVector {
    auto x = inputs.at(0);
    auto multiples = inputs.at(1);
    if (!invalid_indices.empty() || IsDynamicRank(x) || IsDynamicRank(multiples)) {
      return {-1, -1};
    }
    auto x_sz = static_cast<int64_t>(x.size());
    auto multiples_sz = static_cast<int64_t>(multiples.size());
    auto max_sz = x_sz > multiples_sz ? x_sz : multiples_sz;
    return {2 * max_sz, max_sz};
  };

  auto calc_res = ib->ShapeCalc({x, input_multiples}, shape_func, infer_func, {1});
  auto r_shape = calc_res[0];
  auto axis = calc_res[1];
  auto dout_reshaped = ib->Reshape(dout, r_shape);
  auto dout_dtype = ib->GetDtype(dout_reshaped)->type_id();
  NodePtr dx;
  auto need_reduce = ib->NeedReduce(r_shape, axis, false);
  if (need_reduce.first) {
    if (dout_dtype == kNumberTypeInt16 || dout_dtype == kNumberTypeInt32 || dout_dtype == kNumberTypeInt64) {
      dout_reshaped = ib->Cast(dout_reshaped, kFloat32);
      dx = ib->ReduceSum(dout_reshaped, axis);
      dx = ib->Cast(dx, dout_dtype);
    } else {
      dx = ib->ReduceSum(dout_reshaped, axis);
    }
  } else {
    dx = ib->Reshape(dout_reshaped, need_reduce.second);
  }
  auto shape_x = ib->Shape(x);
  dx = ib->Reshape(dx, shape_x);
  return {dx, ib->ZerosLike(input_multiples)};
});

REG_BPROP_BUILDER("Gather").SetUnusedInputs({i0, i3}).SetBody(BinopGatherCommon);
REG_BPROP_BUILDER("GatherV2").SetUnusedInputs({i0, i3}).SetBody(BinopGatherCommon);

REG_BPROP_BUILDER("Fill").SetUnusedInputs({i0, i1, i2, i3, i4}).SetBody(BODYFUNC(ib) {
  auto dtype = ib->GetInput(kIndex0);
  auto dims = ib->GetInput(kIndex1);
  auto x = ib->GetInput(kIndex2);
  return {ib->ZerosLike(dtype), ib->ZerosLike(dims), ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("MatrixDiagV3").SetUnusedInputs({i0, i2, i3, i4, i5}).SetBody(BODYFUNC(ib) {
  auto k = ib->GetInput(kIndex1);
  auto num_rows = ib->GetInput(kIndex2);
  auto num_cols = ib->GetInput(kIndex3);
  auto padding_value = ib->GetInput(kIndex4);
  auto dout = ib->GetInput(kIndex6);
  auto part = ib->Emit("MatrixDiagPartV3", {dout, k, ib->Tensor(0, ib->GetDtype(dout))},
                       {{"align", ib->GetAttr("align")}, {"max_length", MakeValue<int64_t>(diag_max_length)}});
  return {part, ib->ZerosLike(k), ib->ZerosLike(num_rows), ib->ZerosLike(num_cols), ib->ZerosLike(padding_value)};
});

REG_BPROP_BUILDER("MatrixDiagPartV3").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  auto align = ib->GetAttr("align");
  auto x = ib->GetInput(kIndex0);
  auto k = ib->GetInput(kIndex1);
  auto padding_value = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto x_shape = ib->GetShape(x);
  bool is_dynamic_case = IsDynamicRank(x_shape);
  ShapeVector sub_shape;
  if (!is_dynamic_case) {
    size_t begin = (x_shape.size() < 2) ? 0 : (x_shape.size() - 2);
    for (; begin < x_shape.size(); ++begin) {
      sub_shape.push_back(x_shape[begin]);
    }
    is_dynamic_case = IsDynamic(sub_shape);
  }

  NodePtr diag = nullptr;
  if (!is_dynamic_case) {
    if (sub_shape.size() < 2) {
      MS_LOG(EXCEPTION) << "For gradient of MatrixDiagPartV3, rank should be greater than 2";
    }
    auto row = x_shape[x_shape.size() - 2];
    auto col = x_shape[x_shape.size() - 1];
    diag = ib->Emit("MatrixDiagV3",
                    {dout, k, ib->Tensor(row, kInt32), ib->Tensor(col, kInt32), ib->Tensor(0, ib->GetDtype(dout))},
                    {{"align", align}});
  } else {
    diag = ib->Emit("MatrixSetDiagV3", {ib->ZerosLike(x), dout, k},
                    {{"align", align}, {"max_length", MakeValue<int64_t>(diag_max_length)}});
  }
  return {diag, ib->ZerosLike(k), ib->ZerosLike(padding_value)};
});

REG_BPROP_BUILDER("MatrixSetDiagV3").SetUnusedInputs({i0, i1, i3}).SetBody(BODYFUNC(ib) {
  auto align = ib->GetAttr("align");
  auto diagonal = ib->GetInput(kIndex1);
  auto k = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto max_length = MakeValue<int64_t>(diag_max_length);
  auto diagonal_cal = ib->Emit("MatrixDiagPartV3", {dout, k, ib->Tensor(0, ib->GetDtype(dout))},
                               {{"align", align}, {"max_length", max_length}});
  auto diagonal_shape = ib->GetShape(diagonal);
  NodePtr x_cal;
  auto dout_type = ib->GetDtypeId(dout);
  if (IsDynamic(diagonal_shape)) {
    auto diagonal_temp = ib->Cast(diagonal, dout_type);
    x_cal = ib->Emit("MatrixSetDiagV3", {dout, ib->ZerosLike(diagonal_temp), k},
                     {{"align", align}, {"max_length", max_length}});
  } else {
    x_cal = ib->Emit("MatrixSetDiagV3", {dout, ib->Fill(static_cast<int64_t>(0), diagonal_shape, dout_type), k},
                     {{"align", align}, {"max_length", max_length}});
  }
  return {x_cal, diagonal_cal, ib->ZerosLike(k)};
});

REG_BPROP_BUILDER("LogNormalReverse").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto input_data = ib->GetInput(kIndex0);
  return {ib->ZerosLike(input_data)};
});

REG_BPROP_BUILDER("Shape").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("Rank").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("DynamicShape").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("TensorShape").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("DType").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  return {ib->ZerosLike(x)};
});

REG_BPROP_BUILDER("StridedSliceV2").SetUnusedInputs({i0, i4}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto begin = ib->GetInput(kIndex1);
  auto end = ib->GetInput(kIndex2);
  auto strides = ib->GetInput(kIndex3);
  auto dout = ib->GetInput(kIndex5);
  auto x_shape_vec = ib->GetShape(x);
  NodePtr x_shape;
  if (IsDynamic(x_shape_vec)) {
    x_shape = ib->Shape(x);
  } else {
    x_shape = ib->Tensor(x_shape_vec);
  }
  auto dx = ib->Emit("StridedSliceV2Grad", {x_shape, begin, end, strides, dout},
                     {{"begin_mask", ib->GetAttr("begin_mask")},
                      {"end_mask", ib->GetAttr("end_mask")},
                      {"ellipsis_mask", ib->GetAttr("ellipsis_mask")},
                      {"new_axis_mask", ib->GetAttr("new_axis_mask")},
                      {"shrink_axis_mask", ib->GetAttr("shrink_axis_mask")}});
  return {dx, ib->ZerosLike(begin), ib->ZerosLike(end), ib->ZerosLike(strides)};
});

REG_BPROP_BUILDER("MaskedFill").SetUnusedInputs({i2, i3}).SetBody(BODYFUNC(ib) {
  auto input_data = ib->GetInput(kIndex0);
  auto mask = ib->GetInput(kIndex1);
  auto value = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  mask = ib->Cast(mask, kFloat32);
  auto dinput = ib->Mul(dout, ib->Sub((ib->Tensor(1, ib->GetDtype(mask))), mask));
  auto dvalue = ib->Mul(dout, mask);
  auto bout = BinopGradCommon(ib, input_data, mask, dinput, dvalue);

  auto dvalue_shape = dvalue->shape();
  if (IsDynamicRank(dvalue_shape)) {
    auto axis_node = ib->Range(ib->Shape(dvalue, true));
    dvalue = ib->ReduceSum(bout[1], axis_node);
  } else {
    dvalue = ib->ReduceSum(bout[1]);
  }

  dinput = ib->Cast(bout[0], ib->GetDtype(input_data));
  if (value->isa<ValueNode>()) {
    dvalue = ib->ZerosLike(value);
  } else {
    dvalue = ib->Cast(dvalue, ib->GetDtype(value));
  }
  return {dinput, ib->ZerosLike(mask), dvalue};
});

REG_BPROP_BUILDER("Coalesce").SetUnusedInputs({i0, i1, i2, i3}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex4);
  auto d1 = ib->TupleGetItem(dout, 0);
  auto d2 = ib->TupleGetItem(dout, 1);
  auto d3 = ib->TupleGetItem(dout, 2);
  return {d1, d2, d3};
});

REG_BPROP_BUILDER("ConjugateTranspose").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto perm = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto tmp_perm = GetIntList(perm);
  auto tmp_perm_sz = SizeToLong(tmp_perm.size());
  std::vector<int64_t> new_perm;
  (void)std::transform(tmp_perm.begin(), tmp_perm.end(), std::back_inserter(new_perm),
                       [&tmp_perm, tmp_perm_sz](const int64_t v) { return v >= 0 ? v : v + tmp_perm_sz; });
  auto res_perm = InvertPermutation(new_perm);
  return {ib->Emit("ConjugateTranspose", {dout, ib->Value<ShapeVector>(res_perm)}), ib->ZerosLike(perm)};
});

REG_BPROP_BUILDER("Triu").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto diagonal = GetValue<int64_t>(ib->GetAttr("diagonal"));
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("Triu", {dout}, {{"diagonal", MakeValue(diagonal)}});
  return {dx};
});

REG_BPROP_BUILDER("CheckNumerics").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  return {ib->Emit("CheckNumerics", {dout})};
});

REG_BPROP_BUILDER("IdentityN").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto dout = ib->GetInput(kIndex2);
  return {dout};
});

REG_BPROP_BUILDER("ResizeNearestNeighborV2").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto align_corners = GetValue<bool>(ib->GetAttr("align_corners"));
  auto half_pixel_centers = GetValue<bool>(ib->GetAttr("half_pixel_centers"));
  auto data_format = GetValue<std::string>(ib->GetAttr("format"));
  auto x = ib->GetInput(kIndex0);
  auto dout = ib->GetInput(kIndex3);

  bool is_nchw = (data_format == "NCHW");
  auto shape_func = [is_nchw](const ShapeArray &inputs) -> ShapeArray {
    auto x_shape = inputs.at(0);
    ShapeVector grad_in_size = GetShapeByRange(x_shape, 1, 3);
    if (is_nchw) {
      grad_in_size = GetShapeByRange(x_shape, 2, 4);
    }
    const size_t kTwo = 2;
    if (grad_in_size.size() != kTwo) {
      MS_LOG(EXCEPTION) << "For ResizeNearestNeighborV2Grad, size's rank should be 2, but got " << grad_in_size.size();
    }
    return {grad_in_size};
  };
  auto infer_func = [](const ShapeArray &inputs, const std::unordered_set<size_t> &) -> ShapeVector { return {2}; };
  auto grad_in_size = ib->ShapeCalc({x}, shape_func, infer_func)[0];
  if (grad_in_size->isa<ValueNode>()) {
    grad_in_size = ib->Tensor(GetIntList(grad_in_size), kInt64);
  }
  auto dx = ib->Emit("ResizeNearestNeighborV2Grad", {dout, grad_in_size},
                     {{"align_corners", MakeValue(align_corners)},
                      {"half_pixel_centers", MakeValue(half_pixel_centers)},
                      {"format", MakeValue(data_format)}});
  return {dx, ib->ZerosLike(grad_in_size)};
});

REG_BPROP_BUILDER("Tril").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto diagonal = GetValue<int64_t>(ib->GetAttr("diagonal"));
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("Tril", {dout}, {{"diagonal", MakeValue(diagonal)}});
  return {dx};
});

REG_BPROP_BUILDER("SegmentSum").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto segment_ids = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dout_type = ib->GetDtype(dout);
  std::set<TypePtr> type_list = {kInt8, kInt16, kInt64, kUInt8, kUInt16, kUInt32, kUInt64};
  if (CheckType(dout_type, type_list)) {
    dout = ib->Cast(dout, kInt32);
  }
  if (dout_type->type_id() == TypeId::kNumberTypeFloat64) {
    dout = ib->Cast(dout, kFloat32);
  }
  return {ib->Cast(ib->Gather(dout, segment_ids, ib->Tensor(0)), dout_type), ib->ZerosLike(segment_ids)};
});

REG_BPROP_BUILDER("EmbeddingLookup").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto indices = ib->GetInput(kIndex1);
  auto offset = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto x_shp = ib->GetShape(x);
  auto offset_v = GetIntValue(offset);
  auto new_indices = ib->Sub(indices, ib->Tensor(offset_v, ib->GetDtype(indices)));
  auto indices_size = ib->GetSize(new_indices);
  ShapeVector new_indices_shape;
  ShapeVector x_shp_tail;
  ShapeVector actual_dout_shape;
  if (indices_size > 0) {
    new_indices_shape.push_back(indices_size);
    new_indices = ib->Reshape(new_indices, new_indices_shape);
  }
  int64_t x_rank = static_cast<int64_t>(x_shp.size());
  auto start1 = x_rank <= 1 ? x_shp.end() : x_shp.begin() + 1;
  (void)std::copy(start1, x_shp.end(), std::back_inserter(x_shp_tail));
  (void)std::copy(new_indices_shape.begin(), new_indices_shape.end(), std::back_inserter(actual_dout_shape));
  (void)std::copy(x_shp_tail.begin(), x_shp_tail.end(), std::back_inserter(actual_dout_shape));
  auto actual_dout = ib->Reshape(dout, actual_dout_shape);
  return {ib->MakeTuple({new_indices, actual_dout, ib->Value<ShapeVector>(x_shp)}), ib->ZerosLike(indices),
          ib->ZerosLike(offset)};
});

REG_BPROP_BUILDER("MaskedSelect").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto mask = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("MaskedSelectGrad", {x, mask, dout});
  return {dx, ib->ZerosLike(mask)};
});

REG_BPROP_BUILDER("SplitV").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto split_dim = GetValue<int64_t>(ib->GetAttr("split_dim"));
  auto dout = ib->GetInput(kIndex2);
  auto dx = ib->Emit("Concat", {dout}, {{"axis", MakeValue(split_dim)}});
  return {dx};
});

REG_BPROP_BUILDER("Col2Im").SetUnusedInputs({i0, i1, i2}).SetBody(BODYFUNC(ib) {
  auto ksizes = GetValue<std::vector<int64_t>>(ib->GetAttr("kernel_size"));
  auto dilations = GetValue<std::vector<int64_t>>(ib->GetAttr("dilation"));
  auto strides = GetValue<std::vector<int64_t>>(ib->GetAttr("stride"));
  auto pads = GetValue<std::vector<int64_t>>(ib->GetAttr("padding"));
  auto output_size = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dx = ib->Emit("Im2Col", {dout},
                     {{"ksizes", MakeValue(ksizes)},
                      {"dilations", MakeValue(dilations)},
                      {"strides", MakeValue(strides)},
                      {"padding_mode", MakeValue("CALCULATED")},
                      {"pads", MakeValue(pads)}});
  return {dx, ib->ZerosLike(output_size)};
});

REG_BPROP_BUILDER("ExtractVolumePatches").SetUnusedInputs({i0, i1}).SetBody(BODYFUNC(ib) {
  auto ksize = GetValue<std::vector<int64_t>>(ib->GetAttr("kernel_size"));
  auto ksize_d = ksize.at(2);
  auto ksize_h = ksize.at(3);
  auto ksize_w = ksize.at(4);
  auto x = ib->GetInput(kIndex0);
  auto out = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex2);
  auto x_shape = ib->GetShape(x);
  auto x_n = x_shape.at(0);
  auto x_c = x_shape.at(1);
  auto x_d = x_shape.at(2);
  auto x_h = x_shape.at(3);
  auto x_w = x_shape.at(4);
  auto x_indices_num = 1 + ((x_d * x_h) * x_w);
  auto x_idx = ib->Tensor(Range(1, x_indices_num), kFloat16);
  x_idx = ib->Reshape(x_idx, {1, 1, x_d, x_h, x_w});
  auto x_idx_patched = ib->Emit("ExtractVolumePatches", {x_idx},
                                {{"kernel_size", ib->GetAttr("kernel_size")},
                                 {"strides", ib->GetAttr("strides")},
                                 {"padding", ib->GetAttr("padding")}});
  x_idx_patched = ib->Transpose(x_idx_patched, {0, 2, 3, 4, 1});
  x_idx_patched = ib->Cast(x_idx_patched, kInt32);
  auto out_shape = ib->GetShape(out);
  auto out_d = out_shape.at(2);
  auto out_h = out_shape.at(3);
  auto out_w = out_shape.at(4);
  auto out_indices_num = ((((out_d * out_h) * out_w) * ksize_d) * ksize_h) * ksize_w;
  auto out_idx = ib->Tensor(Range(0, out_indices_num), kInt32);
  out_idx = ib->Reshape(out_idx, {1, out_d, out_h, out_w, (ksize_d * ksize_h) * ksize_w});
  auto idx_tensor =
    ib->Emit("Concat", {ib->MakeTuple({ib->ExpandDims(x_idx_patched, -1), ib->ExpandDims(out_idx, -1)})},
             {{"axis", MakeValue<int64_t>(-1)}});
  auto idx_map = ib->Reshape(idx_tensor, {-1, 2});
  std::vector<int64_t> sp_shape = {x_indices_num, out_indices_num};
  std::vector<int64_t> ones(out_indices_num, 1);
  auto sp_mat_full =
    ib->Emit("ScatterNd", {idx_map, ib->Tensor(ones, ib->GetDtype(dout)), ib->Value<ShapeVector>(sp_shape)});
  auto sp_tensor = ib->Emit("Slice", {sp_mat_full, ib->Value<ShapeVector>({1, 0}),
                                      ib->Value<ShapeVector>({x_indices_num - 1, out_indices_num})});
  auto grad = ib->Transpose(dout, {0, 2, 3, 4, 1});
  grad = ib->Reshape(grad, {x_n, out_d, out_h, out_w, ksize_d, ksize_h, ksize_w, x_c});
  auto grad_expended = ib->Transpose(grad, {1, 2, 3, 4, 5, 6, 0, 7});
  auto grad_flat = ib->Reshape(grad_expended, {-1, x_n * x_c});
  auto jac = ib->MatMul(sp_tensor, grad_flat, false, false);
  auto dx = ib->Reshape(jac, {x_d, x_h, x_w, x_n, x_c});
  dx = ib->Transpose(dx, {3, 4, 0, 1, 2});
  return {dx};
});

REG_BPROP_BUILDER("AffineGrid").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto align_corners = GetValue<bool>(ib->GetAttr("align_corners"));
  auto output_size = GetIntList(ib->GetInput(kIndex1));
  auto dout = ib->GetInput(kIndex3);
  auto start = ib->Tensor(-1, kFloat32);
  auto stop = ib->Tensor(1, kFloat32);
  auto zero = ib->Tensor(0, kFloat32);
  constexpr int64_t c0 = 0;
  constexpr int64_t c1 = 1;
  constexpr int64_t c2 = 2;
  constexpr int64_t c3 = 3;
  constexpr int64_t c4 = 4;
  ShapeVector perm1{c1, c0};
  ShapeVector perm2{c0, c2, c1};
  if (output_size.size() == kDim5) {
    const auto n_value = output_size[kIndex0];
    const auto d_value = output_size[kIndex2];
    const auto h_value = output_size[kIndex3];
    const auto w_value = output_size[kIndex4];
    auto vecx = (w_value != 1) ? ib->Emit("LinSpace", {start, stop, ib->Value(w_value)}) : zero;
    auto vecy = (h_value != 1) ? ib->Emit("LinSpace", {start, stop, ib->Value(h_value)}) : zero;
    auto vecz = (d_value != 1) ? ib->Emit("LinSpace", {start, stop, ib->Value(d_value)}) : zero;
    if (!align_corners) {
      vecx = (vecx * ib->Tensor(w_value - 1, kFloat32)) / ib->Tensor(w_value, kFloat32);
      vecy = (vecy * ib->Tensor(h_value - 1, kFloat32)) / ib->Tensor(h_value, kFloat32);
      vecz = (vecz * ib->Tensor(d_value - 1, kFloat32)) / ib->Tensor(d_value, kFloat32);
    }
    auto out = (h_value * d_value != 1) ? ib->Tile(vecx, {h_value * d_value, 1}) : vecx;
    auto one = ib->Reshape(out, {h_value * w_value * d_value, 1});
    out = (w_value == 1) ? ib->ExpandDims(vecy, 0) : ib->Tile(vecy, {w_value, 1});
    out = ib->Transpose(out, perm1);
    if (d_value != 1) {
      out = ib->Tile(out, {d_value, 1});
    }
    auto two = ib->Reshape(out, {h_value * w_value * d_value, 1});
    out = (w_value * h_value != 1) ? ib->Tile(vecz, {w_value * h_value, 1}) : ib->ExpandDims(vecz, 0);
    out = ib->Transpose(out, perm1);
    auto tre = ib->Reshape(out, {h_value * w_value * d_value, 1});
    auto fou = ib->Emit("OnesLike", {tre});
    auto output = ib->Concat({one, two, tre, fou}, 1);
    output = ib->Transpose(output, perm1);
    if (n_value != 1) {
      output = ib->Tile(output, {n_value, 1});
    }
    output = ib->Reshape(output, {n_value, c4, h_value * w_value * d_value});
    dout = ib->Reshape(dout, {n_value, d_value * h_value * w_value, c3});
    dout = ib->Cast(dout, kFloat32);
    auto dtheta = ib->BatchMatMul(output, dout);
    dtheta = ib->Transpose(dtheta, perm2);
    return {dtheta, tre};
  } else if (output_size.size() == kDim4) {
    auto x_shape = ib->GetShape(dout);
    const auto n_value = x_shape[kIndex0];
    const auto h_value = x_shape[kIndex1];
    const auto w_value = x_shape[kIndex2];
    auto vecx = (w_value != 1) ? ib->Emit("LinSpace", {start, stop, ib->Value(w_value)}) : zero;
    auto vecy = (h_value != 1) ? ib->Emit("LinSpace", {start, stop, ib->Value(h_value)}) : zero;
    if (!align_corners) {
      vecx = (vecx * ib->Tensor(w_value - 1, kFloat32)) / ib->Tensor(w_value, kFloat32);
      vecy = (vecy * ib->Tensor(h_value - 1, kFloat32)) / ib->Tensor(h_value, kFloat32);
    }
    auto out = (h_value != 1) ? ib->Tile(vecx, {h_value, 1}) : vecx;
    auto one = ib->Reshape(out, {h_value * w_value, 1});
    out = (w_value == 1) ? ib->ExpandDims(vecy, 0) : ib->Tile(vecy, {w_value, 1});
    out = ib->Transpose(out, perm1);
    auto two = ib->Reshape(out, {h_value * w_value, 1});
    auto tre = ib->Emit("OnesLike", {two});
    auto output = ib->Concat({one, two, tre}, 1);
    output = ib->Transpose(output, perm1);
    output = ib->Tile(output, {n_value, 1});
    output = ib->Reshape(output, {n_value, c3, h_value * w_value});
    dout = ib->Reshape(dout, {n_value, h_value * w_value, c2});
    dout = ib->Cast(dout, kFloat32);
    auto dtheta = ib->BatchMatMul(output, dout);
    dtheta = ib->Transpose(dtheta, perm2);
    return {dtheta, tre};
  }
  MS_LOG(EXCEPTION) << "For op[" << ib->name() << "], the length of output_size should be 4 or 5, but got "
                    << output_size.size();
});

REG_BPROP_BUILDER("SegmentMax").SetBody(SegmentMinOrMaxGrad);
REG_BPROP_BUILDER("SegmentMin").SetBody(SegmentMinOrMaxGrad);

REG_BPROP_BUILDER("TensorScatterElements").SetUnusedInputs({i0, i3}).SetBody(BODYFUNC(ib) {
  auto indices = ib->GetInput(kIndex1);
  auto update = ib->GetInput(kIndex2);
  auto dout = ib->GetInput(kIndex4);
  auto axis = ib->GetAttr("axis");
  auto x_grad = ib->Emit("TensorScatterElements", {dout, indices, ib->ZerosLike(update)},
                         {{"axis", axis}, {"reduction", ib->GetAttr("reduction")}});
  auto update_grad = ib->Emit("GatherD", {dout, ib->EmitValue(axis), indices});
  return {x_grad, ib->ZerosLike(indices), update_grad};
});

REG_BPROP_BUILDER("ScatterAddWithAxis").SetUnusedInputs({i0, i2, i3}).SetBody(BODYFUNC(ib) {
  auto axis = ib->GetAttr("axis");
  auto indices = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex4);
  auto dout_shape = ib->GetShape(dout);
  auto index_shape = ib->GetShape(indices);
  NodePtr update_grad = nullptr;
  if (dout_shape != index_shape) {
    ShapeVector slice_list(dout_shape.size(), 0);
    std::vector<ShapeVector> pad_list;
    pad_list.reserve(dout_shape.size());
    for (size_t i = 0; i < dout_shape.size(); i++) {
      (void)pad_list.emplace_back(ShapeVector{0, dout_shape[i] - index_shape[i]});
    }
    auto out_index = ib->Emit("Pad", {indices}, {{"paddings", MakeValue(pad_list)}});
    auto out_gather = ib->Emit("GatherD", {dout, ib->EmitValue(axis), out_index});
    update_grad = ib->Emit("Slice", {out_gather, ib->Value(slice_list), ib->Value(index_shape)});
  } else {
    update_grad = ib->Emit("GatherD", {dout, ib->EmitValue(axis), indices});
  }
  return {dout, ib->ZerosLike(indices), update_grad};
});

REG_BPROP_BUILDER("Expand").SetUnusedInputs({i0, i2}).SetBody(BODYFUNC(ib) {
  auto x = ib->GetInput(kIndex0);
  auto shape = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);
  auto dout_shape = ib->GetShape(dout);
  auto dshape = ib->ZerosLike(shape);
  if (dout_shape.empty()) {
    return {ib->ReduceSum(dout), dshape};
  }
  auto x_shape = ib->GetShape(x);
  auto leading_dims = dout_shape.size() - x_shape.size();
  auto reduce_dims = Range(SizeToLong(leading_dims));
  for (size_t j = leading_dims; j < dout_shape.size(); ++j) {
    if (x_shape[j - leading_dims] == 1 && dout_shape[j] != 1) {
      reduce_dims.push_back(j);
    }
  }
  if (!reduce_dims.empty()) {
    dout = ib->ReduceSum(dout, reduce_dims, true);
  }
  auto dx = leading_dims > 0 ? ib->Reshape(dout, x_shape) : dout;
  return {dx, dshape};
});

REG_BPROP_BUILDER("SegmentMean").SetUnusedInputs({i2}).SetBody(BODYFUNC(ib) {
  auto input_x = ib->GetInput(kIndex0);
  auto segment_ids = ib->GetInput(kIndex1);
  auto dout = ib->GetInput(kIndex3);

  auto shape_func = [](const ShapeArray &inputs) -> ShapeArray {
    auto x_rank = inputs.at(0).size();
    auto segment_ids_shape = inputs.at(1);
    ShapeVector ones_shape(segment_ids_shape.begin(), segment_ids_shape.end());
    if (x_rank < 1) {
      MS_LOG(EXCEPTION) << "For SegmentMean's gradient, the rank of input x should be greater or equal to one, but got "
                        << x_rank;
    }
    ShapeVector rank_shape(x_rank - 1, 1LL);
    ones_shape.insert(ones_shape.end(), rank_shape.begin(), rank_shape.end());
    return {ones_shape};
  };

  auto infer_func = [](const ShapeArray &inputs, const std::unordered_set<size_t> &invalid_indices) -> ShapeVector {
    auto x = inputs.at(0);
    auto segment_ids = inputs.at(1);
    if (!invalid_indices.empty() || IsDynamicRank(x) || IsDynamicRank(segment_ids)) {
      return {-1};
    }
    auto x_rank = x.size();
    if (x_rank < 1) {
      MS_LOG(EXCEPTION) << "For SegmentMean's gradient, the rank of input x should be greater or equal to one, but got "
                        << x_rank;
    }
    auto segment_ids_rank = segment_ids.size();
    return {SizeToLong(x_rank - 1 + segment_ids_rank)};
  };

  auto ones_shape = ib->ShapeCalc({input_x, segment_ids}, shape_func, infer_func, {})[0];
  auto ones = ib->Fill(1.0, ones_shape, TypeId::kNumberTypeFloat32);

  auto input_x_type = ib->GetDtype(input_x);
  if (input_x_type->type_id() != kNumberTypeFloat32) {
    input_x = ib->Cast(input_x, kFloat32);
    dout = ib->Cast(dout, kFloat32);
  }
  const int64_t max_len = 1000000;
  auto scaled_grad = ib->Div(dout, ib->Emit("SegmentSum", {ones, segment_ids}, {{"max_length", MakeValue(max_len)}}));
  auto dx = ib->Gather(scaled_grad, segment_ids, 0);
  if (input_x_type->type_id() != kNumberTypeFloat32) {
    dx = ib->Cast(dx, input_x_type);
  }
  return {dx, ib->ZerosLike(segment_ids)};
});
REG_BPROP_BUILDERS_END
}  // namespace mindspore::expander::bprop
