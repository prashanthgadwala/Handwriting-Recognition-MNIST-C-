#pragma once
// @generated by torchgen/gen.py from DispatchKeyFunction.h

// NB: The implementing C++ file is RegisterDispatchKey.cpp

// The only #includes we need are for custom classes that have defaults in the C++ API
#include <c10/core/MemoryFormat.h>
#include <c10/core/Scalar.h>
#include <ATen/core/Reduction.h>

// Forward declarations of any types needed in the operator signatures.
// We can't directly include these classes because it will cause circular include dependencies.
// This file is included by TensorBody.h, which defines the Tensor class.
#include <ATen/core/ATen_fwd.h>

namespace at {

namespace mps {

TORCH_API at::Tensor scatter(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src);
TORCH_API at::Tensor & scatter_out(at::Tensor & out, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src);
TORCH_API at::Tensor & scatter_outf(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, at::Tensor & out);
TORCH_API at::Tensor & scatter_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src);
TORCH_API at::Tensor scatter(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value);
TORCH_API at::Tensor & scatter_out(at::Tensor & out, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value);
TORCH_API at::Tensor & scatter_outf(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, at::Tensor & out);
TORCH_API at::Tensor & scatter_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value);
TORCH_API at::Tensor scatter(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce);
TORCH_API at::Tensor & scatter_out(at::Tensor & out, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce);
TORCH_API at::Tensor & scatter_outf(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce, at::Tensor & out);
TORCH_API at::Tensor & scatter_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & src, c10::string_view reduce);
TORCH_API at::Tensor scatter(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce);
TORCH_API at::Tensor & scatter_out(at::Tensor & out, const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce);
TORCH_API at::Tensor & scatter_outf(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce, at::Tensor & out);
TORCH_API at::Tensor & scatter_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Scalar & value, c10::string_view reduce);

} // namespace mps
} // namespace at
