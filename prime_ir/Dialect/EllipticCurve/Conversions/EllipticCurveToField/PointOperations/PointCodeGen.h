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

#ifndef PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_POINTOPERATIONS_POINTCODEGEN_H_
#define PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_POINTOPERATIONS_POINTCODEGEN_H_

#include <array>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/ConversionUtils.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/FieldCodeGen.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/PointCodeGenBaseForward.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveAttributes.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Dialect/EllipticCurve/IR/PointOperationSelector.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Utils/BuilderContext.h"
#include "prime_ir/Utils/ControlFlowOperation.h"
#include "zk_dtypes/include/geometry/curve_type.h"
#include "zk_dtypes/include/geometry/point_traits.h"

namespace zk_dtypes {

template <mlir::prime_ir::elliptic_curve::PointKind Kind>
class PointTraits<
    mlir::prime_ir::elliptic_curve::PointCodeGenBase<Kind>,
    std::enable_if_t<!mlir::prime_ir::elliptic_curve::isEdwards(Kind)>> {
public:
  constexpr static CurveType kType = CurveType::kShortWeierstrass;

  using AffinePoint = mlir::prime_ir::elliptic_curve::PointCodeGenBase<
      mlir::prime_ir::elliptic_curve::PointKind::kAffine>;
  using JacobianPoint = mlir::prime_ir::elliptic_curve::PointCodeGenBase<
      mlir::prime_ir::elliptic_curve::PointKind::kJacobian>;
  using PointXyzz = mlir::prime_ir::elliptic_curve::PointCodeGenBase<
      mlir::prime_ir::elliptic_curve::PointKind::kXYZZ>;
  using BaseField = mlir::prime_ir::elliptic_curve::FieldCodeGen;
};

template <mlir::prime_ir::elliptic_curve::PointKind Kind>
class PointTraits<
    mlir::prime_ir::elliptic_curve::PointCodeGenBase<Kind>,
    std::enable_if_t<mlir::prime_ir::elliptic_curve::isEdwards(Kind)>> {
public:
  constexpr static CurveType kType = CurveType::kTwistedEdwards;

  using AffinePoint = mlir::prime_ir::elliptic_curve::PointCodeGenBase<
      mlir::prime_ir::elliptic_curve::PointKind::kEdAffine>;
  using ExtendedPoint = mlir::prime_ir::elliptic_curve::PointCodeGenBase<
      mlir::prime_ir::elliptic_curve::PointKind::kEdExtended>;
  using BaseField = mlir::prime_ir::elliptic_curve::FieldCodeGen;
};

} // namespace zk_dtypes

namespace mlir::prime_ir::elliptic_curve {

template <PointKind Kind>
class PointCodeGenBase : public PointOperationSelector<Kind>::template Type<
                             PointCodeGenBase<Kind>> {
public:
  static constexpr size_t kNumCoords = getNumCoords(Kind);

  PointCodeGenBase() = default;
  explicit PointCodeGenBase(Value value) : value(value) {}

  // Static so it is usable as LHS::getKind() in the if-constexpr dispatch
  // in PointCodeGen.cpp; still callable on an instance.
  static constexpr PointKind getKind() { return Kind; }
  operator Value() const { return value; }

  template <PointKind Kind2>
  PointCodeGenBase<Kind2> convert() const {
    if constexpr (Kind == Kind2) {
      return *this;
    } else if constexpr (isEdwards(Kind) != isEdwards(Kind2)) {
      llvm_unreachable("Cross-family conversion not supported");
      return PointCodeGenBase<Kind2>();
      // NOLINTNEXTLINE(readability/braces)
    } else if constexpr (Kind == PointKind::kEdExtended &&
                         Kind2 == PointKind::kEdAffine) {
      return this->ToAffine();
      // NOLINTNEXTLINE(readability/braces)
    } else if constexpr (Kind == PointKind::kEdAffine &&
                         Kind2 == PointKind::kEdExtended) {
      return this->ToExtended();
    } else if constexpr (Kind2 == PointKind::kAffine) {
      return this->ToAffine();
    } else if constexpr (Kind2 == PointKind::kJacobian) {
      return this->ToJacobian();
    } else {
      static_assert(Kind2 == PointKind::kXYZZ, "Unsupported point kind");
      return this->ToXyzz();
    }
  }

protected:
  template <typename, typename>
  friend class zk_dtypes::AffinePointOperation;
  template <typename, typename>
  friend class zk_dtypes::JacobianPointOperation;
  template <typename, typename>
  friend class zk_dtypes::PointXyzzOperation;
  template <typename, typename>
  friend class zk_dtypes::ExtendedPointOperation;

  std::array<FieldCodeGen, kNumCoords> ToCoords() const {
    ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
    Operation::result_range coords = toCoords(*b, value);
    std::array<FieldCodeGen, kNumCoords> ret;
    for (size_t i = 0; i < kNumCoords; ++i) {
      ret[i] = FieldCodeGen(coords[i]);
    }
    return ret;
  }

  PointCodeGenBase
  FromCoords(const std::array<FieldCodeGen, kNumCoords> &coords_array) const {
    SmallVector<Value, kNumCoords> coords(coords_array.begin(),
                                          coords_array.end());
    ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
    return PointCodeGenBase(fromCoords(*b, value.getType(), coords));
  }

  auto CreateAffinePoint(const std::array<FieldCodeGen, 2> &coords) const {
    if constexpr (isEdwards(Kind)) {
      return createPoint<PointKind::kEdAffine, 2>(coords);
    } else {
      return createPoint<PointKind::kAffine, 2>(coords);
    }
  }

  PointCodeGenBase<PointKind::kJacobian>
  CreateJacobianPoint(const std::array<FieldCodeGen, 3> &coords) const {
    return createPoint<PointKind::kJacobian, 3>(coords);
  }

  PointCodeGenBase<PointKind::kXYZZ>
  CreatePointXyzz(const std::array<FieldCodeGen, 4> &coords) const {
    return createPoint<PointKind::kXYZZ, 4>(coords);
  }

  PointCodeGenBase<PointKind::kEdExtended>
  CreateExtendedPoint(const std::array<FieldCodeGen, 4> &coords) const {
    return createPoint<PointKind::kEdExtended, 4>(coords);
  }

  auto MaybeConvertToAffine(mlir::ValueRange values) const {
    if constexpr (isEdwards(Kind)) {
      return PointCodeGenBase<PointKind::kEdAffine>(values[0]);
    } else {
      return PointCodeGenBase<PointKind::kAffine>(values[0]);
    }
  }

  PointCodeGenBase<PointKind::kEdAffine> &&
  MaybeConvertToAffine(PointCodeGenBase<PointKind::kEdAffine> &&point) const {
    return std::move(point);
  }

  PointCodeGenBase<PointKind::kJacobian>
  MaybeConvertToJacobian(mlir::ValueRange values) const {
    return PointCodeGenBase<PointKind::kJacobian>(values[0]);
  }

  PointCodeGenBase<PointKind::kXYZZ>
  MaybeConvertToXyzz(mlir::ValueRange values) const {
    return PointCodeGenBase<PointKind::kXYZZ>(values[0]);
  }

  PointCodeGenBase<PointKind::kEdExtended>
  MaybeConvertToExtended(mlir::ValueRange values) const {
    return PointCodeGenBase<PointKind::kEdExtended>(values[0]);
  }

  PointCodeGenBase<PointKind::kEdExtended> &&MaybeConvertToExtended(
      PointCodeGenBase<PointKind::kEdExtended> &&point) const {
    return std::move(point);
  }

  Value IsAZero() const {
    Type baseFieldType =
        cast<PointTypeInterface>(value.getType()).getBaseFieldType();
    ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
    Value zero = field::createFieldZero(baseFieldType, *b);
    return field::CmpOp::create(*b, arith::CmpIPredicate::eq, GetA(), zero);
  }

  FieldCodeGen getCurveParamCodeGen(
      llvm::function_ref<TypedAttr(ShortWeierstrassAttr)> swFn,
      llvm::function_ref<TypedAttr(TwistedEdwardsAttr)> teFn) const {
    ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
    auto pti = cast<PointTypeInterface>(value.getType());
    TypedAttr attr = llvm::TypeSwitch<Attribute, TypedAttr>(pti.getCurveAttr())
                         .Case<ShortWeierstrassAttr>(swFn)
                         .Case<TwistedEdwardsAttr>(teFn);
    return FieldCodeGen(
        field::ConstantOp::create(*b, pti.getBaseFieldType(), attr));
  }

  FieldCodeGen GetA() const {
    return getCurveParamCodeGen(
        [](ShortWeierstrassAttr sw) { return sw.getA(); },
        [](TwistedEdwardsAttr te) { return te.getA(); });
  }

  FieldCodeGen GetD() const {
    ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
    auto teAttr = cast<TwistedEdwardsAttr>(
        cast<PointTypeInterface>(value.getType()).getCurveAttr());
    return FieldCodeGen(
        field::ConstantOp::create(*b, teAttr.getBaseField(), teAttr.getD()));
  }

  zk_dtypes::ControlFlowOperation<Value> GetCFOperation() const { return {}; }

  template <PointKind Kind2, size_t N>
  PointCodeGenBase<Kind2>
  createPoint(const std::array<FieldCodeGen, N> &coords_array) const {
    SmallVector<Value, N> coords(coords_array.begin(), coords_array.end());
    return createPoint<Kind2>(coords);
  }

  template <PointKind Kind2>
  PointCodeGenBase<Kind2> createPoint(mlir::ValueRange coords) const {
    Type type;
    auto curveAttr = cast<PointTypeInterface>(value.getType()).getCurveAttr();
    if constexpr (Kind2 == PointKind::kEdAffine) {
      auto teCurve = cast<TwistedEdwardsAttr>(curveAttr);
      type = EdAffineType::get(value.getContext(), teCurve);
    } else if constexpr (Kind2 == PointKind::kEdExtended) {
      auto teCurve = cast<TwistedEdwardsAttr>(curveAttr);
      type = EdExtendedType::get(value.getContext(), teCurve);
    } else {
      auto swCurve = cast<ShortWeierstrassAttr>(curveAttr);
      if constexpr (Kind2 == PointKind::kAffine) {
        type = AffineType::get(value.getContext(), swCurve);
      } else if constexpr (Kind2 == PointKind::kJacobian) {
        type = JacobianType::get(value.getContext(), swCurve);
      } else if constexpr (Kind2 == PointKind::kXYZZ) {
        type = XYZZType::get(value.getContext(), swCurve);
      }
    }
    ImplicitLocOpBuilder *b = BuilderContext::GetInstance().Top();
    return PointCodeGenBase<Kind2>(fromCoords(*b, type, coords));
  }

  Value value;
};

extern template class PointCodeGenBase<PointKind::kAffine>;
extern template class PointCodeGenBase<PointKind::kJacobian>;
extern template class PointCodeGenBase<PointKind::kXYZZ>;
extern template class PointCodeGenBase<PointKind::kEdAffine>;
extern template class PointCodeGenBase<PointKind::kEdExtended>;

using AffinePointCodeGen = PointCodeGenBase<PointKind::kAffine>;
using JacobianPointCodeGen = PointCodeGenBase<PointKind::kJacobian>;
using XYZZPointCodeGen = PointCodeGenBase<PointKind::kXYZZ>;
using EdAffinePointCodeGen = PointCodeGenBase<PointKind::kEdAffine>;
using EdExtendedPointCodeGen = PointCodeGenBase<PointKind::kEdExtended>;

class PointCodeGen {
public:
  using CodeGenType =
      std::variant<AffinePointCodeGen, JacobianPointCodeGen, XYZZPointCodeGen,
                   EdAffinePointCodeGen, EdExtendedPointCodeGen>;

  template <typename T, typename = std::enable_if_t<
                            !std::is_same_v<std::decay_t<T>, FieldCodeGen> &&
                            std::is_constructible_v<CodeGenType, T>>>
  PointCodeGen(T &&cg) // NOLINT(runtime/explicit)
      : codeGen(std::forward<T>(cg)) {}
  PointCodeGen(Type type, Value value);
  ~PointCodeGen() = default;

  operator Value() const;
  PointKind getKind() const;

  PointCodeGen add(const PointCodeGen &other, PointKind outputKind) const;
  PointCodeGen dbl(PointKind outputKind) const;
  PointCodeGen convert(PointKind outputKind) const;

private:
  CodeGenType codeGen;
};

} // namespace mlir::prime_ir::elliptic_curve

// NOLINTNEXTLINE(whitespace/line_length)
#endif // PRIME_IR_DIALECT_ELLIPTICCURVE_CONVERSIONS_ELLIPTICCURVETOFIELD_POINTOPERATIONS_POINTCODEGEN_H_
