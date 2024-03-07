#pragma once

// @generated by torchgen/gen.py from Operator.h

#include <tuple>
#include <vector>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {
namespace _ops {


struct TORCH_API _mixed_dtypes_linear {
  using schema = at::Tensor (const at::Tensor &, const at::Tensor &, const at::Tensor &, const c10::optional<at::Tensor> &, c10::optional<c10::string_view>);
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::_mixed_dtypes_linear")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "_mixed_dtypes_linear(Tensor input, Tensor weight, Tensor scale, *, Tensor? bias=None, str? activation=None) -> Tensor")
  static at::Tensor call(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & scale, const c10::optional<at::Tensor> & bias, c10::optional<c10::string_view> activation);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & input, const at::Tensor & weight, const at::Tensor & scale, const c10::optional<at::Tensor> & bias, c10::optional<c10::string_view> activation);
};

}} // namespace at::_ops