#pragma once

// @generated by torchgen/gen.py from NativeFunction.h

#include <c10/core/Scalar.h>
#include <c10/core/Storage.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>
#include <c10/core/QScheme.h>
#include <ATen/core/Reduction.h>
#include <ATen/core/Tensor.h>
#include <tuple>
#include <vector>


namespace at {
namespace native {
TORCH_API at::Tensor masked_scatter(const at::Tensor & self, const at::Tensor & mask, const at::Tensor & source);
TORCH_API at::Tensor & masked_scatter_out(const at::Tensor & self, const at::Tensor & mask, const at::Tensor & source, at::Tensor & out);
TORCH_API at::Tensor & masked_scatter__cpu(at::Tensor & self, const at::Tensor & mask, const at::Tensor & source);
TORCH_API at::Tensor & masked_scatter__cuda(at::Tensor & self, const at::Tensor & mask, const at::Tensor & source);
TORCH_API at::Tensor & masked_scatter__mps(at::Tensor & self, const at::Tensor & mask, const at::Tensor & source);
} // namespace native
} // namespace at
