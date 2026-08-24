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

#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/PointCodeGen.h"

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"

namespace mlir::prime_ir::elliptic_curve {

template class PointCodeGenBase<PointKind::kAffine>;
template class PointCodeGenBase<PointKind::kJacobian>;
template class PointCodeGenBase<PointKind::kXYZZ>;
template class PointCodeGenBase<PointKind::kEdAffine>;
template class PointCodeGenBase<PointKind::kEdExtended>;

PointCodeGen::PointCodeGen(Type type, Value value) {
  if (isa<AffineType>(type)) {
    codeGen = AffinePointCodeGen(value);
  } else if (isa<JacobianType>(type)) {
    codeGen = JacobianPointCodeGen(value);
  } else if (isa<XYZZType>(type)) {
    codeGen = XYZZPointCodeGen(value);
  } else if (isa<EdAffineType>(type)) {
    codeGen = EdAffinePointCodeGen(value);
  } else if (isa<EdExtendedType>(type)) {
    codeGen = EdExtendedPointCodeGen(value);
  } else {
    llvm_unreachable("Unsupported point type");
  }
}

PointCodeGen::operator Value() const {
  return std::visit(
      [](const auto &v) -> Value { return static_cast<Value>(v); }, codeGen);
}

PointKind PointCodeGen::getKind() const {
  return std::visit([](const auto &v) -> PointKind { return v.getKind(); },
                    codeGen);
}

namespace {

template <typename F>
PointCodeGen applyUnaryOp(const PointCodeGen::CodeGenType &codeGen, F op) {
  return std::visit([&](const auto &v) -> PointCodeGen { return op(v); },
                    codeGen);
}

template <typename F>
PointCodeGen applyBinaryOp(const PointCodeGen::CodeGenType &a,
                           const PointCodeGen::CodeGenType &b, F op) {
  return std::visit(
      [&](const auto &lhs, const auto &rhs) -> PointCodeGen {
        using LHS = std::decay_t<decltype(lhs)>;
        using RHS = std::decay_t<decltype(rhs)>;
        // Either the same coordinate system on both sides, or a mixed
        // addition with one affine operand. Classified by point kind rather
        // than by naming concrete classes: is_same_v against
        // AffinePointCodeGen covers only short Weierstrass, which silently
        // excluded Edwards from the mixed-addition arm.
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

PointCodeGen PointCodeGen::add(const PointCodeGen &other,
                               PointKind outputKind) const {
  PointKind lhsKind = getKind();
  PointKind rhsKind = other.getKind();
  if (lhsKind == PointKind::kAffine && lhsKind == rhsKind) {
    if (outputKind == PointKind::kXYZZ) {
      return std::get<AffinePointCodeGen>(codeGen).AddToXyzz(
          std::get<AffinePointCodeGen>(other.codeGen));
    }
  }

  PointCodeGen result =
      applyBinaryOp(codeGen, other.codeGen,
                    [](const auto &a, const auto &b) { return a + b; });
  return result.convert(outputKind);
}

PointCodeGen PointCodeGen::dbl(PointKind outputKind) const {
  PointKind kind = getKind();
  if (kind == PointKind::kAffine) {
    if (outputKind == PointKind::kXYZZ) {
      return std::get<AffinePointCodeGen>(codeGen).DoubleToXyzz();
    }
  }
  PointCodeGen result =
      applyUnaryOp(codeGen, [](const auto &v) { return v.Double(); });
  return result.convert(outputKind);
}

namespace {

template <PointKind Kind>
PointCodeGen convertImpl(const PointCodeGen::CodeGenType &codeGen) {
  return std::visit(
      [](const auto &v) -> PointCodeGen { return v.template convert<Kind>(); },
      codeGen);
}

} // namespace

PointCodeGen PointCodeGen::convert(PointKind outputKind) const {
  switch (outputKind) {
  case PointKind::kAffine:
    return convertImpl<PointKind::kAffine>(codeGen);
  case PointKind::kJacobian:
    return convertImpl<PointKind::kJacobian>(codeGen);
  case PointKind::kXYZZ:
    return convertImpl<PointKind::kXYZZ>(codeGen);
  case PointKind::kEdAffine:
    return convertImpl<PointKind::kEdAffine>(codeGen);
  case PointKind::kEdExtended:
    return convertImpl<PointKind::kEdExtended>(codeGen);
  }
  llvm_unreachable("Unsupported point kind");
}

} // namespace mlir::prime_ir::elliptic_curve
