/* Copyright 2025 The PrimeIR Authors.

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

#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h"

#include <utility>

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/ConversionUtils.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/MSM/Pippengers/Generic.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PairingOperations/PairingCodeGen.h"
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/PointOperations/PointCodeGen.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Dialect/EllipticCurve/IR/KnownCurves.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Utils/BitSerialAlgorithm.h"
#include "prime_ir/Utils/ConversionUtils.h"
#include "prime_ir/Utils/LoweringMode.h"

namespace mlir::prime_ir::elliptic_curve {

#define GEN_PASS_DEF_ELLIPTICCURVETOFIELD
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.h.inc"

// ---- AOT Runtime helpers ----

static llvm::StringRef pointKindToString(PointKind kind) {
  switch (kind) {
  case PointKind::kAffine:
    return "affine";
  case PointKind::kJacobian:
    return "jacobian";
  case PointKind::kXYZZ:
    return "xyzz";
  case PointKind::kEdAffine:
    return "ed_affine";
  case PointKind::kEdExtended:
    return "ed_extended";
  }
  llvm_unreachable("unknown PointKind");
}

// Return "_mont" suffix if the field uses Montgomery form, else "".
static std::string getMontSuffix(Type baseFieldType) {
  return field::isMontgomery(baseFieldType) ? "_mont" : "";
}

// Build AOT runtime function name from curve alias and point kind.
// Example: "ec_add_bn254_g1_xyzz", "ec_add_bn254_g2_xyzz_mont"
static std::optional<std::string> getAOTRuntimeFuncName(llvm::StringRef op,
                                                        Type pointType) {
  auto pti = dyn_cast<PointTypeInterface>(pointType);
  if (!pti)
    return std::nullopt;

  auto curveAttr = dyn_cast<ShortWeierstrassAttr>(pti.getCurveAttr());
  if (!curveAttr)
    return std::nullopt;
  auto alias = getKnownCurveAlias(curveAttr);
  if (!alias)
    return std::nullopt;

  return ("ec_" + op + "_" + *alias + "_" +
          pointKindToString(pti.getPointKind()) +
          getMontSuffix(pti.getBaseFieldType()))
      .str();
}

// Build cross-type AOT function name:
// ec_{op}_{curve}_{inputKind}_to_{outputKind}{mont} Example:
// "ec_add_bn254_g1_affine_to_jacobian_mont"
static std::optional<std::string>
getCrossTypeAOTFuncName(llvm::StringRef op, Type inputType, Type outputType) {
  auto inputPti = dyn_cast<PointTypeInterface>(inputType);
  auto outputPti = dyn_cast<PointTypeInterface>(outputType);
  if (!inputPti || !outputPti)
    return std::nullopt;

  auto curveAttr = dyn_cast<ShortWeierstrassAttr>(outputPti.getCurveAttr());
  if (!curveAttr)
    return std::nullopt;
  auto alias = getKnownCurveAlias(curveAttr);
  if (!alias)
    return std::nullopt;

  return ("ec_" + op + "_" + *alias + "_" +
          pointKindToString(inputPti.getPointKind()) + "_to_" +
          pointKindToString(outputPti.getPointKind()) +
          getMontSuffix(outputPti.getBaseFieldType()))
      .str();
}

// Build AOT runtime name for point type conversions.
// Example: "ec_jacobian_to_affine_bn254_g1", "ec_xyzz_to_affine_bn254_g2_mont"
static std::optional<std::string>
getConvertFuncName(PointTypeInterface inputPti, PointTypeInterface outputPti) {
  auto curveAttr = dyn_cast<ShortWeierstrassAttr>(inputPti.getCurveAttr());
  if (!curveAttr)
    return std::nullopt;
  auto alias = getKnownCurveAlias(curveAttr);
  if (!alias)
    return std::nullopt;

  return ("ec_" + pointKindToString(inputPti.getPointKind()) + "_to_" +
          pointKindToString(outputPti.getPointKind()) + "_" + *alias +
          getMontSuffix(inputPti.getBaseFieldType()))
      .str();
}

// Check if AOTRuntime should be used for this operation.
// When inlineConstOps is true, ops with 2+ operands where at least one is
// constant skip AOT to allow constant folding / strength reduction.
static bool shouldUseAOTRuntime(Operation *op, Type pointType,
                                LoweringMode mode, bool inlineConstOps) {
  if (mode == LoweringMode::Inline)
    return false;
  if (inlineConstOps && hasConstantOperand(op))
    return false;
  if (mode == LoweringMode::AOTRuntime)
    return true;
  // Auto: AOT for extension field curves (G2), inline for prime field (G1)
  auto pti = dyn_cast<PointTypeInterface>(pointType);
  if (!pti)
    return false;
  return isa<field::ExtensionFieldType>(pti.getBaseFieldType());
}

struct ConvertConstant : public OpConversionPattern<ConstantOp> {
  explicit ConvertConstant(MLIRContext *context)
      : OpConversionPattern<ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type resultType = op.getType();
    Type pointType = getElementTypeOrSelf(resultType);
    auto pointTypeInterface = cast<PointTypeInterface>(pointType);
    Type baseFieldType = pointTypeInterface.getBaseFieldType();

    // Convert coordinate attributes to field constants
    // Unified structure: ArrayAttr<ArrayAttr<Attr>> where outer array contains
    // points and inner array contains coordinates per point
    ArrayAttr coordsAttr = op.getCoords();

    // Helper to create a single point from its coordinate attributes
    auto createPointFromCoords =
        [&](ArrayAttr pointCoords) -> FailureOr<Value> {
      SmallVector<Value> coordValues;
      for (auto coordAttr : pointCoords) {
        Value coordValue;
        if (auto intAttr = dyn_cast<IntegerAttr>(coordAttr)) {
          // Prime field coordinate
          coordValue = field::ConstantOp::create(b, baseFieldType, intAttr);
        } else if (auto denseAttr = dyn_cast<DenseIntElementsAttr>(coordAttr)) {
          // Extension field coordinate
          coordValue = field::ConstantOp::create(b, baseFieldType, denseAttr);
        } else {
          return failure();
        }
        coordValues.push_back(coordValue);
      }
      return fromCoords(b, pointType, coordValues);
    };

    if (auto shapedType = dyn_cast<ShapedType>(resultType)) {
      // Tensor constant: create flat integer tensor, bitcast to field tensor,
      // then bitcast to point tensor
      unsigned numCoords = pointTypeInterface.getNumCoords();
      int64_t numPoints = shapedType.getNumElements();

      // Determine extension field degree and get base prime field
      unsigned extDegree = field::getExtensionDegree(baseFieldType);
      auto primeFieldType = field::getBasePrimeField(baseFieldType);

      // Collect all coordinate values into a flat vector
      SmallVector<APInt> flatValues;
      flatValues.reserve(numPoints * numCoords * extDegree);

      for (auto pointAttr : coordsAttr) {
        auto pointCoords = cast<ArrayAttr>(pointAttr);
        for (auto coordAttr : pointCoords) {
          if (auto intAttr = dyn_cast<IntegerAttr>(coordAttr)) {
            // Prime field coordinate
            flatValues.push_back(intAttr.getValue());
          } else if (auto denseAttr =
                         dyn_cast<DenseIntElementsAttr>(coordAttr)) {
            // Extension field coordinate - flatten all coefficients
            for (const APInt &coeff : denseAttr.getValues<APInt>())
              flatValues.push_back(coeff);
          } else {
            return op.emitError() << "unsupported coordinate attribute type";
          }
        }
      }

      // Create flat integer tensor constant using arith.constant
      int64_t flatSize = numPoints * numCoords * extDegree;
      auto intTensorType =
          RankedTensorType::get({flatSize}, primeFieldType.getStorageType());
      auto flatDenseAttr = DenseIntElementsAttr::get(intTensorType, flatValues);
      auto intConstant =
          arith::ConstantOp::create(b, intTensorType, flatDenseAttr);

      // Bitcast integer tensor to prime field tensor
      auto pfTensorType = RankedTensorType::get({flatSize}, primeFieldType);
      auto pfTensor =
          field::BitcastOp::create(b, pfTensorType, intConstant.getResult());

      // Use elliptic_curve.bitcast to convert prime field tensor to point
      // tensor. This can be properly bufferized unlike
      // UnrealizedConversionCastOp.
      Value fieldTensor = pfTensor.getResult();
      Value tensorResult =
          BitcastOp::create(b, shapedType, fieldTensor).getResult();
      rewriter.replaceOp(op, tensorResult);
    } else {
      // Scalar constant: create single point
      ArrayAttr pointCoords = cast<ArrayAttr>(coordsAttr[0]);
      auto pointOrFailure = createPointFromCoords(pointCoords);
      if (failed(pointOrFailure))
        return op.emitError() << "unsupported coordinate attribute type";
      rewriter.replaceOp(op, *pointOrFailure);
    }

    return success();
  }
};

struct ConvertIsZero : public OpConversionPattern<IsZeroOp> {
  explicit ConvertIsZero(MLIRContext *context)
      : OpConversionPattern<IsZeroOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IsZeroOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Operation::result_range coords = toCoords(b, op.getInput());
    Type baseFieldType =
        cast<PointTypeInterface>(op.getInput().getType()).getBaseFieldType();
    Value zeroBF = field::createFieldZero(baseFieldType, b);

    Value isZero;
    Type inputType = op.getInput().getType();
    if (isa<AffineType>(inputType)) {
      // SW affine identity: (0, 0)
      Value xIsZero =
          field::CmpOp::create(b, arith::CmpIPredicate::eq, coords[0], zeroBF);
      Value yIsZero =
          field::CmpOp::create(b, arith::CmpIPredicate::eq, coords[1], zeroBF);
      isZero = arith::AndIOp::create(b, xIsZero, yIsZero);
    } else if (isa<EdAffineType>(inputType)) {
      // Edwards affine identity: (0, 1)
      Value oneBF = field::createFieldOne(baseFieldType, b);
      Value xIsZero =
          field::CmpOp::create(b, arith::CmpIPredicate::eq, coords[0], zeroBF);
      Value yIsOne =
          field::CmpOp::create(b, arith::CmpIPredicate::eq, coords[1], oneBF);
      isZero = arith::AndIOp::create(b, xIsZero, yIsOne);
    } else if (isa<EdExtendedType>(inputType)) {
      // Edwards extended identity: (0, c, c, 0) for any c != 0.
      // Check X == 0 and Y == Z (projective equivalence to (0, 1, 1, 0)).
      Value xIsZero =
          field::CmpOp::create(b, arith::CmpIPredicate::eq, coords[0], zeroBF);
      Value yEqZ = field::CmpOp::create(b, arith::CmpIPredicate::eq, coords[1],
                                        coords[2]);
      isZero = arith::AndIOp::create(b, xIsZero, yEqZ);
    } else {
      // SW projective: identity has Z == 0
      isZero =
          field::CmpOp::create(b, arith::CmpIPredicate::eq, coords[2], zeroBF);
    }
    rewriter.replaceOp(op, isZero);
    return success();
  }
};

struct ConvertConvertPointType
    : public OpConversionPattern<ConvertPointTypeOp> {
  explicit ConvertConvertPointType(MLIRContext *context,
                                   AOTConfig aotConfig = {})
      : OpConversionPattern<ConvertPointTypeOp>(context), aotConfig(aotConfig) {
    // The wrapped-scalar normalization below re-emits a scalar
    // convert_point_type that this same pattern must legalize; the scalar op
    // cannot match the normalization branch again, so recursion is bounded.
    setHasBoundedRewriteRecursion();
  }

  LogicalResult
  matchAndRewrite(ConvertPointTypeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type inputType = op.getInput().getType();
    Type outputType = getElementTypeOrSelf(op.getType());
    auto loc = op.getLoc();

    // Normalize wrapped scalars: a statically single-element tensor (any
    // rank, including rank-0) is a scalar in disguise — extract it, convert
    // as a scalar, and wrap the result back. The scalar convert_point_type
    // re-enters this pattern and takes the AOT or inline path per lowering
    // mode. The batch path below saves nothing at N = 1 (Montgomery's trick
    // costs 3(N-1) muls + 1 inverse).
    if (auto tensorType = dyn_cast<RankedTensorType>(inputType);
        tensorType && tensorType.hasStaticShape() &&
        tensorType.getNumElements() == 1) {
      Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
      SmallVector<Value> indices(tensorType.getRank(), zeroIdx);
      Value scalar =
          tensor::ExtractOp::create(rewriter, loc, adaptor.getInput(), indices);
      Value converted =
          ConvertPointTypeOp::create(rewriter, loc, outputType, scalar);
      rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(
          op, cast<RankedTensorType>(op.getType()), converted);
      return success();
    }

    // Tensor batch path: Jacobian/XYZZ → Affine with batch field.inverse.
    // The batch inverse avoids per-element scalar inverse, enabling
    // Montgomery's trick (3(N-1) muls + 1 inverse instead of N inverses).
    if (auto tensorType = dyn_cast<RankedTensorType>(inputType);
        tensorType && tensorType.getRank() > 0) {
      auto inputPti = cast<PointTypeInterface>(tensorType.getElementType());
      auto outputPti = cast<PointTypeInterface>(outputType);
      if (outputPti.getPointKind() == PointKind::kAffine &&
          inputPti.getPointKind() != PointKind::kAffine) {
        return rewriteTensorToAffine(op, adaptor, rewriter, tensorType,
                                     inputPti, outputPti);
      }
    }

    // Any remaining tensor form converts element-wise with the inline
    // scalar codegen regardless of mode: unlike the ->affine batch above
    // these directions share no work across elements (no inverse to
    // amortize), so a per-point AOT call would be pure overhead — and
    // jacobian<->xyzz have no runtime symbol to call anyway.
    if (auto tensorType = dyn_cast<RankedTensorType>(inputType)) {
      ImplicitLocOpBuilder b(loc, rewriter);
      SmallVector<Value> dynExtents;
      for (int64_t i = 0; i < tensorType.getRank(); ++i) {
        if (tensorType.isDynamicDim(i)) {
          Value idx = arith::ConstantIndexOp::create(b, i);
          dynExtents.push_back(
              tensor::DimOp::create(b, adaptor.getInput(), idx));
        }
      }
      auto outputTensorType =
          RankedTensorType::get(tensorType.getShape(), outputType);
      PointKind outKind = cast<PointTypeInterface>(outputType).getPointKind();
      Value result =
          tensor::GenerateOp::create(
              b, outputTensorType, dynExtents,
              [&](OpBuilder &nb, Location nloc, ValueRange ivs) {
                ImplicitLocOpBuilder lb(nloc, nb);
                ScopedBuilderContext scopedBuilderContext(&lb);
                Value pt =
                    tensor::ExtractOp::create(lb, adaptor.getInput(), ivs);
                PointCodeGen codeGen(tensorType.getElementType(), pt);
                tensor::YieldOp::create(lb, codeGen.convert(outKind));
              })
              .getResult();
      rewriter.replaceOp(op, result);
      return success();
    }

    // AOT runtime path for scalar point type conversions.
    if (shouldUseAOTRuntime(op, outputType, aotConfig.mode,
                            aotConfig.inlineConstOps)) {
      auto inputPti = cast<PointTypeInterface>(getElementTypeOrSelf(inputType));
      auto outputPti = cast<PointTypeInterface>(outputType);
      auto funcName = getConvertFuncName(inputPti, outputPti);
      if (funcName) {
        rewriter.replaceOp(op, emitAOTFuncCall(op, *funcName, op.getType(),
                                               {op.getInput()}, rewriter));
        return success();
      }
    }

    // Inline scalar path.
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ScopedBuilderContext scopedBuilderContext(&b);

    PointCodeGen inputCodeGen(inputType, adaptor.getInput());
    PointKind outKind = cast<PointTypeInterface>(outputType).getPointKind();
    rewriter.replaceOp(op, {inputCodeGen.convert(outKind)});
    return success();
  }

private:
  /// Lower tensor<N x !ec.jacobian> → tensor<N x !ec.affine> using batch
  /// field.inverse on the Z coordinates.
  ///
  /// Jacobian (X, Y, Z) → Affine (X / Z², Y / Z³):
  ///   z_inv = field.inverse(Z)          ← batch on tensor<N x !F>
  ///   z_inv² = z_inv * z_inv
  ///   z_inv³ = z_inv² * z_inv
  ///   x = X * z_inv²
  ///   y = Y * z_inv³
  ///
  /// XYZZ (X, Y, ZZ, ZZZ) → Affine (X / ZZ, Y / ZZZ):
  ///   z_inv³ = field.inverse(ZZZ)       ← batch on tensor<N x !F>
  ///   per-element: z_inv = z_inv³ * ZZ, z_inv² = z_inv²
  ///   x = X * z_inv², y = Y * z_inv³
  LogicalResult rewriteTensorToAffine(ConvertPointTypeOp op, OpAdaptor adaptor,
                                      ConversionPatternRewriter &rewriter,
                                      RankedTensorType tensorType,
                                      PointTypeInterface inputPti,
                                      PointTypeInterface outputPti) const {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Type fieldType = inputPti.getBaseFieldType();
    // Shape-generic: field.inverse collapses the multi-dim column to 1-D
    // internally, so any rank keeps the single batched inverse.
    ArrayRef<int64_t> shape = tensorType.getShape();
    SmallVector<Value> dynExtentsStorage;
    for (int64_t i = 0; i < tensorType.getRank(); ++i) {
      if (tensorType.isDynamicDim(i)) {
        Value idx = arith::ConstantIndexOp::create(b, i);
        dynExtentsStorage.push_back(
            tensor::DimOp::create(b, adaptor.getInput(), idx));
      }
    }
    ValueRange dynExtents = dynExtentsStorage;
    auto colType = RankedTensorType::get(shape, fieldType);

    PointKind inKind = inputPti.getPointKind();
    if (inKind != PointKind::kJacobian && inKind != PointKind::kXYZZ) {
      return rewriter.notifyMatchFailure(
          op, "unsupported input point kind for batch conversion");
    }

    // Extract coordinate columns via tensor.generate + to_coords.
    // No bitcast/expand_shape/extract_slice — avoids buffer aliasing
    // issues that cause use-after-free during bufferization.
    auto makeCol = [&](unsigned coordIdx) -> Value {
      return tensor::GenerateOp::create(
                 b, colType, dynExtents,
                 [&](OpBuilder &nb, Location loc, ValueRange ivs) {
                   ImplicitLocOpBuilder lb(loc, nb);
                   Value pt =
                       tensor::ExtractOp::create(lb, adaptor.getInput(), ivs);
                   auto coords = toCoords(lb, pt);
                   tensor::YieldOp::create(lb, coords[coordIdx]);
                 })
          .getResult();
    };

    // Fused output: batch inverse → single tensor.generate that computes
    // affine coordinates and assembles the output point in one pass.
    auto outputTensorType = RankedTensorType::get(shape, cast<Type>(outputPti));
    Value result;
    if (inKind == PointKind::kJacobian) {
      Value colZ = makeCol(2);
      Value zInv = field::InverseOp::create(b, colType, colZ);
      Value zInv2 = field::MulOp::create(b, colType, zInv, zInv);
      Value zInv3 = field::MulOp::create(b, colType, zInv2, zInv);
      result = tensor::GenerateOp::create(
                   b, outputTensorType, dynExtents,
                   [&](OpBuilder &nb, Location loc, ValueRange ivs) {
                     ImplicitLocOpBuilder lb(loc, nb);
                     Value pt =
                         tensor::ExtractOp::create(lb, adaptor.getInput(), ivs);
                     auto coords = toCoords(lb, pt);
                     Value zi2 = tensor::ExtractOp::create(lb, zInv2, ivs);
                     Value zi3 = tensor::ExtractOp::create(lb, zInv3, ivs);
                     Value x = field::MulOp::create(lb, coords[0], zi2);
                     Value y = field::MulOp::create(lb, coords[1], zi3);
                     Value resPt = fromCoords(lb, outputPti, ValueRange{x, y});
                     tensor::YieldOp::create(lb, resPt);
                   })
                   .getResult();
    } else {
      // XYZZ→Affine: single BatchInverse on ZZZ only.
      // zInvCubic = BatchInverse(ZZZ) = Z⁻³
      // Per-element: zInv = zInvCubic * ZZ, zInvSq = zInv²,
      //              x = X * zInvSq, y = Y * zInvCubic.
      Value colZZZ = makeCol(3);
      Value zInvCubic = field::InverseOp::create(b, colType, colZZZ);
      result = tensor::GenerateOp::create(
                   b, outputTensorType, dynExtents,
                   [&](OpBuilder &nb, Location loc, ValueRange ivs) {
                     ImplicitLocOpBuilder lb(loc, nb);
                     Value pt =
                         tensor::ExtractOp::create(lb, adaptor.getInput(), ivs);
                     auto coords = toCoords(lb, pt);
                     Value zi3 = tensor::ExtractOp::create(lb, zInvCubic, ivs);
                     // Z⁻¹ = Z⁻³ * Z² = zInvCubic * ZZ
                     Value zInv = field::MulOp::create(lb, zi3, coords[2]);
                     Value zInvSq = field::MulOp::create(lb, zInv, zInv);
                     Value x = field::MulOp::create(lb, coords[0], zInvSq);
                     Value y = field::MulOp::create(lb, coords[1], zi3);
                     Value resPt = fromCoords(lb, outputPti, ValueRange{x, y});
                     tensor::YieldOp::create(lb, resPt);
                   })
                   .getResult();
    }

    rewriter.replaceOp(op, result);
    return success();
  }

  AOTConfig aotConfig;
};

///////////// POINT ARITHMETIC OPERATIONS //////////////

struct ConvertAdd : public OpConversionPattern<AddOp> {
  explicit ConvertAdd(MLIRContext *context, AOTConfig aotConfig = {})
      : OpConversionPattern<AddOp>(context), aotConfig(aotConfig) {}

  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type outputType = getElementTypeOrSelf(op.getType());

    // AOT runtime path: emit func.call for known curves.
    if (shouldUseAOTRuntime(op, outputType, aotConfig.mode,
                            aotConfig.inlineConstOps)) {
      Type lhsType = getElementTypeOrSelf(op->getOperandTypes()[0]);
      Type rhsType = getElementTypeOrSelf(op->getOperandTypes()[1]);
      std::optional<std::string> funcName;
      if (lhsType == rhsType) {
        if (lhsType == outputType)
          funcName = getAOTRuntimeFuncName("add", outputType);
        else
          funcName = getCrossTypeAOTFuncName("add", lhsType, outputType);
      } else {
        funcName = getAOTRuntimeFuncName("mixed_add", outputType);
      }
      if (funcName) {
        rewriter.replaceOp(op, emitAOTFuncCall(op, *funcName, op.getType(),
                                               {op.getLhs(), op.getRhs()},
                                               rewriter));
        return success();
      }
    }

    // Inline path (original).
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ScopedBuilderContext scopedBuilderContext(&b);

    Type lhsPointType = getElementTypeOrSelf(op->getOperandTypes()[0]);
    PointCodeGen lhsCodeGen(lhsPointType, adaptor.getLhs());
    Type rhsPointType = getElementTypeOrSelf(op->getOperandTypes()[1]);
    PointCodeGen rhsCodeGen(rhsPointType, adaptor.getRhs());
    PointKind outputKind = cast<PointTypeInterface>(outputType).getPointKind();
    rewriter.replaceOp(op, {lhsCodeGen.add(rhsCodeGen, outputKind)});
    return success();
  }

  AOTConfig aotConfig;
};

struct ConvertDouble : public OpConversionPattern<DoubleOp> {
  explicit ConvertDouble(MLIRContext *context, AOTConfig aotConfig = {})
      : OpConversionPattern<DoubleOp>(context), aotConfig(aotConfig) {}

  LogicalResult
  matchAndRewrite(DoubleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type outputType = getElementTypeOrSelf(op.getType());

    if (shouldUseAOTRuntime(op, outputType, aotConfig.mode,
                            aotConfig.inlineConstOps)) {
      Type inputType = getElementTypeOrSelf(op.getInput().getType());
      std::optional<std::string> funcName;
      if (inputType == outputType)
        funcName = getAOTRuntimeFuncName("double", outputType);
      else
        funcName = getCrossTypeAOTFuncName("double", inputType, outputType);
      if (funcName) {
        rewriter.replaceOp(op, emitAOTFuncCall(op, *funcName, op.getType(),
                                               {op.getInput()}, rewriter));
        return success();
      }
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ScopedBuilderContext scopedBuilderContext(&b);

    Type inputType = op.getInput().getType();
    PointCodeGen inputCodeGen(inputType, adaptor.getInput());
    PointKind outputKind = cast<PointTypeInterface>(outputType).getPointKind();
    rewriter.replaceOp(op, {inputCodeGen.dbl(outputKind)});
    return success();
  }

  AOTConfig aotConfig;
};

struct ConvertNegate : public OpConversionPattern<NegateOp> {
  explicit ConvertNegate(MLIRContext *context)
      : OpConversionPattern<NegateOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(NegateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Operation::result_range coords = toCoords(b, op.getInput());
    SmallVector<Value> outputCoords(coords);

    Type elementType = getElementTypeOrSelf(op.getType());
    if (isa<EdAffineType>(elementType)) {
      // Twisted Edwards affine: -(x, y) = (-x, y)
      outputCoords[0] = field::NegateOp::create(b, coords[0]);
    } else if (isa<EdExtendedType>(elementType)) {
      // Twisted Edwards extended: -(X, Y, Z, T) = (-X, Y, Z, -T)
      outputCoords[0] = field::NegateOp::create(b, coords[0]);
      outputCoords[3] = field::NegateOp::create(b, coords[3]);
    } else {
      // Short Weierstrass: negate Y coordinate.
      outputCoords[1] = field::NegateOp::create(b, coords[1]);
    }

    rewriter.replaceOp(op, fromCoords(b, op.getType(), outputCoords));
    return success();
  }
};

struct ConvertSub : public OpConversionPattern<SubOp> {
  explicit ConvertSub(MLIRContext *context)
      : OpConversionPattern<SubOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value negP2 = NegateOp::create(b, op.getRhs());
    Value result = AddOp::create(b, op.getType(), op.getLhs(), negP2);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertCmp : public OpConversionPattern<CmpOp> {
  explicit ConvertCmp(MLIRContext *context)
      : OpConversionPattern<CmpOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Operation::result_range lhsCoords = toCoords(b, lhs);
    Operation::result_range rhsCoords = toCoords(b, rhs);
    llvm::SmallVector<Value, 4> cmps;
    for (auto [lhsCoord, rhsCoord] : llvm::zip(lhsCoords, rhsCoords)) {
      cmps.push_back(
          field::CmpOp::create(b, op.getPredicate(), lhsCoord, rhsCoord));
    }
    Value result;
    if (op.getPredicate() == arith::CmpIPredicate::eq) {
      result = combineCmps<arith::AndIOp>(b, cmps);
    } else if (op.getPredicate() == arith::CmpIPredicate::ne) {
      result = combineCmps<arith::OrIOp>(b, cmps);
    } else {
      llvm_unreachable(
          "Unsupported comparison predicate for EllipticCurve point type");
    }
    rewriter.replaceOp(op, result);
    return success();
  }

  template <typename Op>
  Value combineCmps(ImplicitLocOpBuilder &b, ValueRange cmps) const {
    Op result = Op::create(b, cmps[0], cmps[1]);
    if (cmps.size() == 3) {
      result = Op::create(b, result, cmps[2]);
    } else if (cmps.size() == 4) {
      Op result2 = Op::create(b, cmps[2], cmps[3]);
      result = Op::create(b, result, result2);
    }
    return result;
  }
};

namespace {
/// Converts a prime field value to its integer representation.
/// For field::ConstantOp: extracts the mathematical value via maybeToStandard.
/// For dynamic values: emits FromMontOp (if Montgomery) + BitcastOp.
Value fieldPrimeToInteger(ImplicitLocOpBuilder &b, Value fieldVal) {
  auto pfType = cast<field::PrimeFieldType>(fieldVal.getType());
  auto intType = pfType.getStorageType();

  if (auto constOp = fieldVal.getDefiningOp<field::ConstantOp>()) {
    Attribute stdAttr =
        field::maybeToStandard(constOp.getType(), constOp.getValue());
    return arith::ConstantIntOp::create(b, intType,
                                        cast<IntegerAttr>(stdAttr).getValue());
  }

  Value std = pfType.isMontgomery()
                  ? field::FromMontOp::create(
                        b, field::getStandardFormType(pfType), fieldVal)
                  : fieldVal;
  return field::BitcastOp::create(b, TypeRange{intType}, std);
}
} // namespace

// Currently implements Double-and-Add algorithm
// TODO(ashjeong): implement GLV
struct ConvertScalarMul : public OpConversionPattern<ScalarMulOp> {
  explicit ConvertScalarMul(MLIRContext *context, AOTConfig aotConfig = {})
      : OpConversionPattern<ScalarMulOp>(context), aotConfig(aotConfig) {}

  LogicalResult
  matchAndRewrite(ScalarMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value point = op.getPoint();
    Value scalarPF = op.getScalar();

    Type outputType = op.getType();

    // By point kind, not isa<AffineType>: that names only the short Weierstrass
    // affine type, so an Edwards affine point would skip the widening below.
    const bool pointIsAffine =
        isAffine(cast<PointTypeInterface>(point.getType()).getPointKind());

    // AOT runtime path: emit func.call for scalar multiply.
    // Use "scalar_mul" for affine input, "scalar_mul_jac" for projective
    // input (the name predates the twisted Edwards representations).
    if (shouldUseAOTRuntime(op, outputType, aotConfig.mode,
                            aotConfig.inlineConstOps)) {
      std::string opName = pointIsAffine ? "scalar_mul" : "scalar_mul_jac";
      auto funcName = getAOTRuntimeFuncName(opName, outputType);
      if (funcName) {
        rewriter.replaceOp(op, emitAOTFuncCall(op, *funcName, op.getType(),
                                               {scalarPF, point}, rewriter));
        return success();
      }
    }

    // Inline implementation: Double-and-Add algorithm
    Value zeroPoint = createZeroPoint(b, outputType);
    Value initialPoint = pointIsAffine
                             ? ConvertPointTypeOp::create(b, outputType, point)
                             : point;

    Value scalarInt = fieldPrimeToInteger(b, scalarPF);
    Value result = generateBitSerialLoop(
        b, scalarInt, initialPoint, zeroPoint,
        [outputType](ImplicitLocOpBuilder &b, Value v) {
          return DoubleOp::create(b, outputType, v);
        },
        [outputType](ImplicitLocOpBuilder &b, Value acc, Value v) {
          return AddOp::create(b, outputType, acc, v);
        });
    rewriter.replaceOp(op, result);
    return success();
  }

private:
  AOTConfig aotConfig;
};

// Currently implements Pippenger's
struct ConvertMSM : public OpConversionPattern<MSMOp> {
  explicit ConvertMSM(MLIRContext *context)
      : OpConversionPattern<MSMOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MSMOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value scalars = op.getScalars();
    Value points = op.getPoints();

    Type outputType = op.getType();

    PippengersGeneric pippengers(scalars, points, outputType, b,
                                 adaptor.getParallel(), adaptor.getDegree(),
                                 adaptor.getWindowBits());

    rewriter.replaceOp(op, pippengers.generate());
    return success();
  }
};

struct ConvertPairingCheck : public OpConversionPattern<PairingCheckOp> {
  explicit ConvertPairingCheck(MLIRContext *context)
      : OpConversionPattern<PairingCheckOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PairingCheckOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Value g1Points = op.getG1Points();
    Value g2Points = op.getG2Points();

    // Get pairing curve family from G1/G2 attributes.
    auto g1PointType = cast<PointTypeInterface>(
        cast<RankedTensorType>(g1Points.getType()).getElementType());
    auto g2PointType = cast<PointTypeInterface>(
        cast<RankedTensorType>(g2Points.getType()).getElementType());
    auto g1CurveAttr = cast<ShortWeierstrassAttr>(g1PointType.getCurveAttr());
    auto g2CurveAttr = cast<ShortWeierstrassAttr>(g2PointType.getCurveAttr());

    auto family = getKnownPairingCurveFamily(g1CurveAttr, g2CurveAttr);
    if (!family)
      return op.emitError() << "unsupported pairing curve family";

    bool isMontgomery =
        cast<field::PrimeFieldType>(g1PointType.getBaseFieldType())
            .isMontgomery();

    Value result =
        emitBN254PairingCheck(b, *family, g1Points, g2Points, isMontgomery);
    rewriter.replaceOp(op, result);
    return success();
  }
};

namespace rewrites {
// In an inner namespace to avoid conflicts with canonicalization patterns
#include "prime_ir/Dialect/EllipticCurve/Conversions/EllipticCurveToField/EllipticCurveToField.cpp.inc"
} // namespace rewrites

struct EllipticCurveToField
    : impl::EllipticCurveToFieldBase<EllipticCurveToField> {
  using EllipticCurveToFieldBase::EllipticCurveToFieldBase;

  void runOnOperation() override;
};

void EllipticCurveToField::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();

  // Parse lowering mode option
  std::optional<LoweringMode> parsedMode =
      mlir::prime_ir::parseLoweringMode(loweringMode);
  if (!parsedMode) {
    module.emitError() << "invalid lowering-mode option: '" << loweringMode
                       << "' (expected 'inline', 'auto', or 'aot_runtime')";
    return signalPassFailure();
  }
  LoweringMode mode = *parsedMode;

  ConversionTarget target(*context);
  target.addIllegalOp<
      // clang-format off
      AddOp,
      ConstantOp,
      CmpOp,
      ConvertPointTypeOp,
      DoubleOp,
      IsZeroOp,
      MSMOp,
      NegateOp,
      PairingCheckOp,
      ScalarMulOp,
      SubOp
      // clang-format on
      >();

  target.addLegalDialect<
      // clang-format off
      arith::ArithDialect,
      bufferization::BufferizationDialect,
      field::FieldDialect,
      func::FuncDialect,
      linalg::LinalgDialect,
      memref::MemRefDialect,
      scf::SCFDialect,
      tensor::TensorDialect
      // clang-format on
      >();

  target.addLegalOp<
      // clang-format off
      BitcastOp,
      FromCoordsOp,
      ToCoordsOp,
      func::FuncOp,
      func::CallOp,
      func::ReturnOp,
      UnrealizedConversionCastOp
      // clang-format on
      >();

  RewritePatternSet patterns(context);
  linalg::populateLinalgNamedOpsGeneralizationPatterns(patterns);
  rewrites::populateWithGenerated(patterns);

  // Register patterns with mode-aware lowering (inline vs AOT runtime).
  patterns.add<ConvertScalarMul>(context, AOTConfig{mode, inlineConstantOps});
  patterns.add<ConvertAdd>(context, AOTConfig{mode, inlineConstantOps});
  patterns.add<ConvertDouble>(context, AOTConfig{mode, inlineConstantOps});
  patterns.add<ConvertConvertPointType>(context,
                                        AOTConfig{mode, inlineConstantOps});
  // NOTE: We intentionally do NOT add function signature conversion patterns.
  // Converting function signatures would require converting all operations that
  // use EC tensor types (e.g., linalg.reduce), which is beyond the scope of
  // EC-to-Field. The EC tensor types in function signatures will remain and be
  // handled by later passes or explicit bitcasts.

  patterns.add<
      // clang-format off
      ConvertConstant,
      ConvertCmp,
      ConvertIsZero,
      ConvertMSM,
      ConvertNegate,
      ConvertPairingCheck,
      ConvertSub
      // clang-format on
      >(context);

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace mlir::prime_ir::elliptic_curve
