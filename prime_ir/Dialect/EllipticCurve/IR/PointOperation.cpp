/* Copyright 2026 The PrimeIR Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "prime_ir/Dialect/EllipticCurve/IR/PointOperation.h"

namespace mlir::prime_ir::elliptic_curve {

template class PointOperationBase<PointKind::kAffine>;
template class PointOperationBase<PointKind::kJacobian>;
template class PointOperationBase<PointKind::kXYZZ>;
template class PointOperationBase<PointKind::kEdAffine>;
template class PointOperationBase<PointKind::kEdExtended>;

PointKind PointOperation::getKind() const {
  return std::visit([](const auto &v) -> PointKind { return v.getKind(); },
                    operation);
}

PointOperation PointOperation::operator-() const {
  return std::visit([](const auto &v) -> PointOperation { return -v; },
                    operation);
}

namespace {

template <typename F>
PointOperation applyUnaryOp(const PointOperation::OperationType &operation,
                            F op) {
  return std::visit([&](const auto &v) -> PointOperation { return op(v); },
                    operation);
}

template <typename F>
PointOperation applyBinaryOp(const PointOperation::OperationType &a,
                             const PointOperation::OperationType &b, F op) {
  return std::visit(
      [&](const auto &lhs, const auto &rhs) -> PointOperation {
        using LHS = std::decay_t<decltype(lhs)>;
        using RHS = std::decay_t<decltype(rhs)>;
        // See the matching comment in PointCodeGen.cpp.
        constexpr bool kRealizable = std::is_same_v<LHS, RHS> ||
                                     isAffine(LHS::getKind()) ||
                                     isAffine(RHS::getKind());
        if constexpr (!isSameFamily(LHS::getKind(), RHS::getKind())) {
          llvm_unreachable("Cross-family binary op not supported");
        } else if constexpr (kRealizable) {
          return op(lhs, rhs);
        }
        llvm_unreachable("Unsupported field type in binary operator");
      },
      a, b);
}

} // namespace

PointOperation PointOperation::add(const PointOperation &other,
                                   PointKind outputKind) const {
  PointKind lhsKind = getKind();
  PointKind rhsKind = other.getKind();
  if (lhsKind == PointKind::kAffine && lhsKind == rhsKind) {
    if (outputKind == PointKind::kXYZZ) {
      return std::get<AffinePointOperation>(operation).AddToXyzz(
          std::get<AffinePointOperation>(other.operation));
    }
  }

  return applyBinaryOp(operation, other.operation,
                       [](const auto &a, const auto &b) { return a + b; });
}

PointOperation PointOperation::sub(const PointOperation &other,
                                   PointKind outputKind) const {
  PointKind lhsKind = getKind();
  PointKind rhsKind = other.getKind();
  if (lhsKind == PointKind::kAffine && lhsKind == rhsKind) {
    if (outputKind == PointKind::kXYZZ) {
      // sub(a, b) = add(a, negate(b))
      return std::get<AffinePointOperation>(operation).AddToXyzz(
          -std::get<AffinePointOperation>(other.operation));
    }
  }

  return applyBinaryOp(operation, other.operation,
                       [](const auto &a, const auto &b) { return a - b; });
}

PointOperation PointOperation::dbl(PointKind outputKind) const {
  PointKind kind = getKind();
  if (kind == PointKind::kAffine) {
    if (outputKind == PointKind::kXYZZ) {
      return std::get<AffinePointOperation>(operation).DoubleToXyzz();
    }
  }
  return applyUnaryOp(operation, [](const auto &v) { return v.Double(); });
}

namespace {

template <PointKind Kind>
PointOperation convertImpl(const PointOperation::OperationType &operation) {
  return std::visit(
      [](const auto &v) -> PointOperation {
        return v.template convert<Kind>();
      },
      operation);
}

} // namespace

PointOperation PointOperation::convert(PointKind outputKind) const {
  switch (outputKind) {
  case PointKind::kAffine:
    return convertImpl<PointKind::kAffine>(operation);
  case PointKind::kJacobian:
    return convertImpl<PointKind::kJacobian>(operation);
  case PointKind::kXYZZ:
    return convertImpl<PointKind::kXYZZ>(operation);
  case PointKind::kEdAffine:
    return convertImpl<PointKind::kEdAffine>(operation);
  case PointKind::kEdExtended:
    return convertImpl<PointKind::kEdExtended>(operation);
  }
  llvm_unreachable("Unsupported point kind");
}

} // namespace mlir::prime_ir::elliptic_curve
