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

#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h"

#include <algorithm>
#include <utility>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/ConversionUtils.h"
#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/FieldCodeGen.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithOps.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"
#include "prime_ir/Dialect/TensorExt/IR/TensorExtOps.h"
#include "prime_ir/IR/Attributes.h"
#include "prime_ir/Utils/BitSerialAlgorithm.h"
#include "prime_ir/Utils/BuilderContext.h"
#include "prime_ir/Utils/ConversionUtils.h"
#include "prime_ir/Utils/KnownModulus.h"
#include "prime_ir/Utils/LoweringMode.h"
#include "prime_ir/Utils/ShapedTypeConverter.h"

namespace mlir::prime_ir::field {

#define GEN_PASS_DEF_FIELDTOMODARITH
#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.h.inc"

namespace {

// Algorithm for a prime-field scalar inverse. Mirrors the `loweringMode`
// string-option-to-enum idiom (see Utils/LoweringMode.h).
enum class InverseAlgorithm { BernsteinYang, Fermat, Auto };

// Returns std::nullopt for unrecognized values so the caller can fail fast
// instead of silently selecting a default.
std::optional<InverseAlgorithm> parseInverseAlgorithm(StringRef algorithm) {
  if (algorithm == "bernstein-yang")
    return InverseAlgorithm::BernsteinYang;
  if (algorithm == "fermat")
    return InverseAlgorithm::Fermat;
  if (algorithm == "auto")
    return InverseAlgorithm::Auto;
  return std::nullopt;
}

// ---- AOT Runtime helpers ----

// Check if AOTRuntime should be used for this extension field operation.
// AOT is used for extension fields that are computationally expensive:
//   degree >= 4 (quartic+, always expensive regardless of prime size)
//   OR degree >= 2 AND prime > 64-bit (large-prime quadratic/cubic)
static bool shouldUseFieldAOTRuntime(Operation *op, Type fieldType,
                                     LoweringMode mode, bool inlineConstOps) {
  if (mode == LoweringMode::Inline)
    return false;
  if (inlineConstOps && hasConstantOperand(op))
    return false;
  auto efType = dyn_cast<ExtensionFieldType>(fieldType);
  if (!efType)
    return false;
  unsigned degree = efType.getDegreeOverPrime();
  unsigned primeBits = efType.getBasePrimeField().getTypeSizeInBits();
  if (degree < 2)
    return false;
  bool expensive = degree >= 4 || primeBits > 64;
  if (!expensive)
    return false;
  // Skip AOT for mixed-type operations (e.g., PF × EF). AOT functions expect
  // homogeneous types, and inline expansion is actually more efficient for
  // scalar broadcast anyway.
  for (Value operand : op->getOperands()) {
    if (getElementTypeOrSelf(operand.getType()) != fieldType)
      return false;
  }
  return mode == LoweringMode::AOTRuntime || mode == LoweringMode::Auto;
}

// Build the zk_dtypes-style extension suffix by walking the tower.
// ef<2x ef<3x pf>> → "x3x2",  ef<2x pf> → "x2",  ef<4x pf> → "x4"
static std::string getExtensionTowerSuffix(ExtensionFieldType efType) {
  std::string suffix;
  Type cur = efType;
  while (auto ef = dyn_cast<ExtensionFieldType>(cur)) {
    suffix = "x" + std::to_string(ef.getDegree()) + suffix;
    cur = ef.getBaseField();
  }
  return suffix;
}

// Build AOT runtime function name for extension field operations.
// Pattern: "ef_<op>_<prime_alias><tower_suffix>[_mont]"
// Follows zk_dtypes tower naming: bn254_bfx2, babybearx4, mersenne31x2x2.
// Example: "ef_mul_bn254_bfx2_mont", "ef_inverse_mersenne31x2x2"
static std::optional<std::string> getFieldAOTFuncName(llvm::StringRef op,
                                                      Type fieldType) {
  auto efType = dyn_cast<ExtensionFieldType>(fieldType);
  if (!efType)
    return std::nullopt;

  // Tower extensions (base field is itself an extension, e.g.,
  // !field.ef<3x !field.ef<2x !PF, ...>, ...>) are not yet AOT-compiled.
  // Only direct extensions over prime fields (e.g., !field.ef<2x !PF, ...>).
  if (efType.isTower())
    return std::nullopt;

  auto baseAlias =
      getKnownModulusAlias(efType.getBasePrimeField().getModulus().getValue());
  if (!baseAlias)
    return std::nullopt;

  std::string towerSuffix = getExtensionTowerSuffix(efType);
  std::string montSuffix =
      isMontgomery(efType.getBasePrimeField()) ? "_mont" : "";
  return ("ef_" + op + "_" + *baseAlias + towerSuffix + montSuffix).str();
}

class FieldToModArithTypeConverter : public ShapedTypeConverter {
public:
  explicit FieldToModArithTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
    addConversion([](PrimeFieldType type) -> Type {
      return convertPrimeFieldType(type);
    });
    addConversion([](ShapedType type) -> Type {
      if (auto primeFieldType =
              dyn_cast<PrimeFieldType>(type.getElementType())) {
        return convertShapedType(type, type.getShape(),
                                 convertPrimeFieldType(primeFieldType));
      } else if (auto vectorType =
                     dyn_cast<VectorType>(type.getElementType())) {
        if (auto primeFieldType =
                dyn_cast<PrimeFieldType>(vectorType.getElementType())) {
          return convertShapedType(
              type, type.getShape(),
              vectorType.cloneWith(vectorType.getShape(),
                                   convertPrimeFieldType(primeFieldType)));
        }
      }
      return type;
    });
  }
};

struct ConvertConstant : public OpConversionPattern<ConstantOp> {
  explicit ConvertConstant(MLIRContext *context)
      : OpConversionPattern<ConstantOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Skip binary field operations - they are handled by BinaryFieldToArith
    if (isa<BinaryFieldType>(getElementTypeOrSelf(op.getType()))) {
      return failure();
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Case 1: Prime field constants (scalar or tensor). The PF value attr is
    // already storage-int (IntegerAttr / DenseIntElementsAttr), which is what
    // mod_arith.constant expects.
    if (auto pfType =
            dyn_cast<PrimeFieldType>(getElementTypeOrSelf(op.getType()))) {
      auto cval = mod_arith::ConstantOp::create(
          b, typeConverter->convertType(op.getType()), op.getValueAttr());
      rewriter.replaceOp(op, cval);
      return success();
    }

    // Case 2: Tensor of extension field constants
    // Use efficient approach: create prime field tensor constant + bitcast
    if (auto shapedType = dyn_cast<ShapedType>(op.getType())) {
      auto efType = dyn_cast<ExtensionFieldType>(shapedType.getElementType());
      if (!efType) {
        op.emitOpError(
            "unsupported shaped type with non-extension field element type: ")
            << shapedType.getElementType();
        return failure();
      }

      // For tower extensions, use the underlying prime field
      auto modType = cast<mod_arith::ModArithType>(
          typeConverter->convertType(efType.getBasePrimeField()));
      unsigned degree = efType.getDegreeOverPrime();

      auto denseAttr =
          cast<DenseIntElementsAttr>(prime_ir::maybeConvertPrimeIRToBuiltinAttr(
              cast<DenseElementsAttr>(op.getValueAttr())));
      auto allValues = denseAttr.getValues<APInt>();

      // Create a flattened prime field tensor constant
      // tensor<K x !EF> with degree N becomes tensor<K*N x !ModArith>
      // tensor<!EF> (rank-0) becomes tensor<N x !ModArith>
      SmallVector<int64_t> flatShape(shapedType.getShape());
      if (flatShape.empty()) {
        flatShape.push_back(degree);
      } else {
        flatShape.back() *= degree;
      }

      auto flatTensorType = RankedTensorType::get(flatShape, modType);
      SmallVector<APInt> flatCoeffs(allValues.begin(), allValues.end());
      auto flatDenseAttr = DenseIntElementsAttr::get(
          flatTensorType.clone(modType.getStorageType()), flatCoeffs);
      auto flatConstant =
          mod_arith::ConstantOp::create(b, flatTensorType, flatDenseAttr);

      // Bitcast the flattened prime field tensor to extension field tensor
      auto efTensorType = RankedTensorType::get(shapedType.getShape(), efType);
      auto bitcast = BitcastOp::create(b, efTensorType, flatConstant);
      rewriter.replaceOp(op, bitcast);
      return success();
    }

    // Case 3: Scalar extension field constant
    auto efType = dyn_cast<ExtensionFieldType>(op.getType());
    if (!efType) {
      op.emitOpError("unsupported output type");
      return failure();
    }

    // For tower extensions, use the underlying prime field
    auto modType = cast<mod_arith::ModArithType>(
        typeConverter->convertType(efType.getBasePrimeField()));

    auto denseAttr = cast<DenseIntElementsAttr>(op.getValueAttr());
    SmallVector<Value> primeCoeffs;
    for (auto coeff : denseAttr.getValues<APInt>()) {
      auto coeffAttr = IntegerAttr::get(modType.getStorageType(), coeff);
      primeCoeffs.push_back(
          mod_arith::ConstantOp::create(b, modType, coeffAttr));
    }
    // Use fromPrimeCoeffs to properly handle tower extension fields
    rewriter.replaceOp(op, fromPrimeCoeffs(b, efType, primeCoeffs));
    return success();
  }
};

struct ConvertBitcast : public OpConversionPattern<BitcastOp> {
  explicit ConvertBitcast(MLIRContext *context)
      : OpConversionPattern<BitcastOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type inputType = op.getInput().getType();
    Type outputType = op.getType();

    // Check if this is a tensor reinterpret bitcast (extension field <-> prime
    // field tensors)
    if (isTensorReinterpretBitcast(inputType, outputType)) {
      auto outputShaped = cast<ShapedType>(outputType);
      return convertTensorBitcast(op, adaptor, rewriter, outputShaped);
    }

    // Scalar bitcast: just convert to mod_arith.bitcast
    auto bitcast = mod_arith::BitcastOp::create(
        b, typeConverter->convertType(op.getType()), adaptor.getInput());
    rewriter.replaceOp(op, bitcast);
    return success();
  }

private:
  LogicalResult convertTensorBitcast(BitcastOp op, OpAdaptor adaptor,
                                     ConversionPatternRewriter &rewriter,
                                     ShapedType outputTensorType) const {
    // Get the converted output type
    Type convertedOutputType = typeConverter->convertType(outputTensorType);
    if (!convertedOutputType) {
      return op.emitOpError("failed to convert output type");
    }

    // Keep the field.bitcast op but with updated types.
    // This preserves zero-copy semantics by deferring the actual memory
    // reinterpretation to the bufferization stage, where it will be
    // converted to a memref-level bitcast, and then to LLVM pointer casts.
    //
    // NOTE: We intentionally do NOT extract/reconstruct tensor elements here
    // as that would cause memory copies.
    rewriter.replaceOpWithNewOp<BitcastOp>(op, convertedOutputType,
                                           adaptor.getInput());
    return success();
  }
};

struct ConvertToMont : public OpConversionPattern<ToMontOp> {
  explicit ConvertToMont(MLIRContext *context)
      : OpConversionPattern<ToMontOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ToMontOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type fieldType = getElementTypeOrSelf(op.getOutput());
    if (isa<PrimeFieldType>(fieldType)) {
      Type modArithType = typeConverter->convertType(op.getType());
      auto extracted =
          mod_arith::ToMontOp::create(b, modArithType, adaptor.getInput());
      rewriter.replaceOp(op, extracted);
      return success();
    }
    if (auto efType = dyn_cast<ExtensionFieldType>(fieldType)) {
      // Use getBasePrimeField() to handle both direct and tower extensions
      auto basePrimeField = efType.getBasePrimeField();
      Type baseModArithType = typeConverter->convertType(basePrimeField);
      auto coeffs = toModArithCoeffs(b, adaptor.getInput());

      SmallVector<Value> montCoeffs;
      for (auto coeff : coeffs) {
        montCoeffs.push_back(
            mod_arith::ToMontOp::create(b, baseModArithType, coeff));
      }
      rewriter.replaceOp(op, fromCoeffs(b, fieldType, montCoeffs));
      return success();
    }
    return failure();
  }
};

struct ConvertFromMont : public OpConversionPattern<FromMontOp> {
  explicit ConvertFromMont(MLIRContext *context)
      : OpConversionPattern<FromMontOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FromMontOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type fieldType = getElementTypeOrSelf(op.getOutput());
    if (isa<PrimeFieldType>(fieldType)) {
      Type resultType = typeConverter->convertType(op.getType());
      auto extracted =
          mod_arith::FromMontOp::create(b, resultType, adaptor.getInput());
      rewriter.replaceOp(op, extracted);
      return success();
    }
    if (auto efType = dyn_cast<ExtensionFieldType>(fieldType)) {
      // Use getBasePrimeField() to handle both direct and tower extensions
      auto basePrimeField = efType.getBasePrimeField();
      Type baseModArithType = typeConverter->convertType(basePrimeField);
      auto coeffs = toModArithCoeffs(b, adaptor.getInput());

      SmallVector<Value> stdCoeffs;
      for (auto coeff : coeffs) {
        stdCoeffs.push_back(
            mod_arith::FromMontOp::create(b, baseModArithType, coeff));
      }
      rewriter.replaceOp(op, fromCoeffs(b, fieldType, stdCoeffs));
      return success();
    }
    return failure();
  }
};

template <typename OpT, typename Derived>
struct ConvertFieldOpBase : public OpConversionPattern<OpT> {
  using Base = ConvertFieldOpBase;
  using OpAdaptor = typename OpConversionPattern<OpT>::OpAdaptor;

  ConvertFieldOpBase(const TypeConverter &converter, MLIRContext *context,
                     AOTConfig aotConfig = {})
      : OpConversionPattern<OpT>(converter, context), aotConfig(aotConfig) {
    this->setHasBoundedRewriteRecursion(true);
  }

  LogicalResult
  matchAndRewrite(OpT op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (isa<ShapedType>(op.getOutput().getType()) &&
        isa<ExtensionFieldType>(getElementTypeOrSelf(op.getOutput())))
      return failure();

    // AOT runtime path: emit func.call to pre-compiled function.
    Type fieldType = getElementTypeOrSelf(op.getOutput());
    if (shouldUseFieldAOTRuntime(op, fieldType, aotConfig.mode,
                                 aotConfig.inlineConstOps)) {
      auto funcName =
          static_cast<const Derived *>(this)->getAOTFuncName(fieldType);
      if (funcName) {
        rewriter.replaceOp(op, emitAOTFuncCall(op, *funcName,
                                               op.getOutput().getType(),
                                               op->getOperands(), rewriter));
        return success();
      }
    }

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    ScopedBuilderContext scopedBuilderContext(&b);

    rewriter.replaceOp(
        op,
        {static_cast<const Derived *>(this)->emitInlineCodeGen(op, adaptor)});
    return success();
  }

protected:
  AOTConfig aotConfig;
};

/// Lowers field.inverse for both scalar and tensor types.
///
/// Scalar: delegates to ConvertFieldOpBase (AOT or inline codegen).
///
/// Tensor (batch, default): Montgomery's batch inversion trick.
///   Given a₀, a₁, ..., aₙ₋₁:
///     Forward:  prefix[0] = a₀; prefix[i] = prefix[i-1] * a[i]
///     Inverse:  inv = prefix[n-1]⁻¹
///     Backward: result[i] = inv * prefix[i-1]; inv = inv * a[i]  (i = n-1..1)
///               result[0] = inv
///   Uses O(3(n-1)) field multiplications + 1 field inversion instead of n
///   independent inversions. Optimal for CPU (sequential, fewer total muls).
///
/// Tensor (elementwise, opt-in): per-element scalar inverse via linalg.generic.
///   Each element is independently inverted (Fermat's little theorem: a^(p-2)).
///   Preferred for GPU where embarrassingly parallel per-element inverse
///   outperforms sequential batch inverse.
struct ConvertInverse : ConvertFieldOpBase<InverseOp, ConvertInverse> {
  using Base = ConvertFieldOpBase<InverseOp, ConvertInverse>;

  ConvertInverse(const TypeConverter &converter, MLIRContext *context,
                 AOTConfig aotConfig, bool useElementwiseInverse,
                 InverseAlgorithm inverseAlgorithm)
      : Base(converter, context, aotConfig),
        useElementwiseInverse(useElementwiseInverse),
        inverseAlgorithm(inverseAlgorithm) {}

  std::optional<std::string> getAOTFuncName(Type fieldType) const {
    return getFieldAOTFuncName("inverse", fieldType);
  }
  Value emitInlineCodeGen(InverseOp op, OpAdaptor adaptor) const {
    Type fieldType = getElementTypeOrSelf(op.getOutput());
    return emitScalarInverse(fieldType, adaptor.getInput());
  }

  // Past this width the chain loses to safegcd, so `auto` stops taking it.
  static constexpr unsigned kFermatMaxModulusBits = 64;

  bool preferFermatChain(const APInt &modulus) const {
    if (inverseAlgorithm == InverseAlgorithm::Fermat)
      return true;
    return inverseAlgorithm == InverseAlgorithm::Auto &&
           PrimeFieldCodeGen::hasInverseChain(modulus) &&
           modulus.getActiveBits() <= kFermatMaxModulusBits;
  }

  Value emitScalarInverse(Type fieldType, Value value) const {
    FieldCodeGen cg(fieldType, value, this->typeConverter);
    if (auto pf = dyn_cast<PrimeFieldType>(fieldType)) {
      APInt modulus = pf.getModulus().getValue();
      // Chain-able primes route through cg.inverse() below. PrimeFieldType
      // rejects power-of-2 moduli, so p - 2 >= 1 — never 0, as pow() requires.
      if (inverseAlgorithm == InverseAlgorithm::Fermat &&
          !PrimeFieldCodeGen::hasInverseChain(modulus))
        return cg.pow(modulus - 2);
    }
    return cg.inverse();
  }

  LogicalResult
  matchAndRewrite(InverseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Skip binary field operations - they are handled by BinaryFieldToArith
    if (isa<BinaryFieldType>(getElementTypeOrSelf(op.getType()))) {
      return failure();
    }

    // Scalar, per-element, and extension lowerings all emit a base-field
    // inverse internally; carry the choice, keyed on the base prime, to each.
    APInt baseModulus = getBasePrimeField(getElementTypeOrSelf(op.getOutput()))
                            .getModulus()
                            .getValue();
    ScopedPreferFermatChain preferFermat(preferFermatChain(baseModulus));

    auto tensorType = dyn_cast<RankedTensorType>(op.getOutput().getType());
    if (!tensorType)
      return Base::matchAndRewrite(op, adaptor, rewriter);

    if (useElementwiseInverse)
      return emitElementwiseInverse(op, adaptor, rewriter, tensorType);
    return emitBatchInverse(op, rewriter, tensorType);
  }

private:
  LogicalResult emitBatchInverse(InverseOp op,
                                 ConversionPatternRewriter &rewriter,
                                 RankedTensorType tensorType) const {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Value input = op.getInput();
    Type elemType = tensorType.getElementType();

    // Static zero-element tensor: no-op.
    if (tensorType.hasStaticShape() && tensorType.getNumElements() == 0) {
      rewriter.replaceOp(op, input);
      return success();
    }

    // Scalar (rank-0): extract, invert, wrap back.
    if (tensorType.getRank() == 0) {
      Value elem = tensor::ExtractOp::create(b, input, ValueRange{});
      Value inv = InverseOp::create(b, elem);
      Value result = tensor::FromElementsOp::create(b, tensorType, inv);
      rewriter.replaceOp(op, result);
      return success();
    }

    // Collapse multi-dimensional tensors to 1-D for linear indexing.
    // For dynamic shapes, collapse is only valid for rank > 1 with a
    // single dynamic dim (or all static except one).
    bool needsReshape = tensorType.getRank() != 1;
    if (needsReshape) {
      int64_t n = tensorType.hasStaticShape() ? tensorType.getNumElements()
                                              : ShapedType::kDynamic;
      auto flatType = RankedTensorType::get({n}, elemType);
      SmallVector<ReassociationIndices> reassoc = {
          llvm::to_vector(llvm::seq<int64_t>(0, tensorType.getRank()))};
      input = tensor::CollapseShapeOp::create(b, flatType, input, reassoc);
    }

    // Get the dimension size (static or dynamic).
    Value c0 = arith::ConstantIndexOp::create(b, 0);
    Value c1 = arith::ConstantIndexOp::create(b, 1);
    Value cN = tensor::DimOp::create(b, input, c0);

    // Montgomery's batch inversion (works for any n >= 1).
    // Forward pass: build prefix products.
    // prefix[0] = a[0]; prefix[i] = prefix[i-1] * a[i]
    Value a0 = tensor::ExtractOp::create(b, input, ValueRange{c0});
    Value prefixInit = tensor::EmptyOp::create(
        b, tensor::getMixedSizes(b, b.getLoc(), input), elemType);
    prefixInit = tensor::InsertOp::create(b, a0, prefixInit, ValueRange{c0});

    auto fwdLoop = scf::ForOp::create(
        b, c1, cN, c1, ValueRange{prefixInit, a0},
        [&](OpBuilder &nb, Location loc, Value iv, ValueRange iterArgs) {
          ImplicitLocOpBuilder lb(loc, nb);
          Value prefixTensor = iterArgs[0];
          Value runningProduct = iterArgs[1];
          Value elem = tensor::ExtractOp::create(lb, input, ValueRange{iv});
          Value product = MulOp::create(lb, runningProduct, elem);
          Value updated = tensor::InsertOp::create(lb, product, prefixTensor,
                                                   ValueRange{iv});
          scf::YieldOp::create(lb, ValueRange{updated, product});
        });
    Value prefixTensor = fwdLoop.getResult(0);
    Value totalProduct = fwdLoop.getResult(1);

    // Single scalar inverse.
    Value inv = InverseOp::create(b, totalProduct);

    // Backward pass: recover individual inverses.
    // for i in [n-1, n-2, ..., 1]:
    //   result[i] = inv * prefix[i-1]; inv = inv * a[i]
    Value cNm1 = arith::SubIOp::create(b, cN, c1);
    Value resultInit = tensor::EmptyOp::create(
        b, tensor::getMixedSizes(b, b.getLoc(), input), elemType);

    auto bwdLoop = scf::ForOp::create(
        b, c0, cNm1, c1, ValueRange{resultInit, inv},
        [&](OpBuilder &nb, Location loc, Value iv, ValueRange iterArgs) {
          ImplicitLocOpBuilder lb(loc, nb);
          Value resultTensor = iterArgs[0];
          Value curInv = iterArgs[1];
          // Map forward index iv to reverse index: currIdx = n-1-iv
          Value currIdx = arith::SubIOp::create(lb, cNm1, iv);
          Value prevIdx = arith::SubIOp::create(lb, currIdx, c1);
          Value prevPrefix =
              tensor::ExtractOp::create(lb, prefixTensor, ValueRange{prevIdx});
          Value elemInv = MulOp::create(lb, curInv, prevPrefix);
          Value updated = tensor::InsertOp::create(lb, elemInv, resultTensor,
                                                   ValueRange{currIdx});
          Value origElem =
              tensor::ExtractOp::create(lb, input, ValueRange{currIdx});
          Value newInv = MulOp::create(lb, curInv, origElem);
          scf::YieldOp::create(lb, ValueRange{updated, newInv});
        });

    // result[0] = final inv (= a₀⁻¹).
    Value result = bwdLoop.getResult(0);
    inv = bwdLoop.getResult(1);
    result = tensor::InsertOp::create(b, inv, result, ValueRange{c0});

    if (needsReshape) {
      SmallVector<ReassociationIndices> reassoc = {
          llvm::to_vector(llvm::seq<int64_t>(0, tensorType.getRank()))};
      result = tensor::ExpandShapeOp::create(b, tensorType, result, reassoc);
    }

    rewriter.replaceOp(op, result);
    return success();
  }

  /// Per-element scalar inverse via linalg.generic.
  /// Each element is independently inverted, enabling GPU parallelization.
  /// Uses FieldCodeGen directly to emit mod_arith ops in the body,
  /// avoiding recursive conversion issues with linalg.yield terminator.
  LogicalResult emitElementwiseInverse(InverseOp op, OpAdaptor adaptor,
                                       ConversionPatternRewriter &rewriter,
                                       RankedTensorType tensorType) const {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    Type fieldType = tensorType.getElementType();
    Value input = adaptor.getInput();
    int64_t rank = tensorType.getRank();
    auto convertedTensorType =
        cast<RankedTensorType>(this->typeConverter->convertType(tensorType));
    Type convertedElemType = convertedTensorType.getElementType();

    // Rank-0 tensor: extract, invert, wrap back.
    if (rank == 0) {
      Value elem = tensor::ExtractOp::create(b, input, ValueRange{});
      ScopedBuilderContext scopedCtx(&b);
      Value inv = emitScalarInverse(fieldType, elem);
      Value result =
          tensor::FromElementsOp::create(b, convertedTensorType, inv);
      rewriter.replaceOp(op, result);
      return success();
    }

    // Build identity indexing maps and parallel iterator types.
    SmallVector<AffineMap> indexingMaps(
        2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);

    // Use getMixedSizes to support both static and dynamic shapes.
    SmallVector<OpFoldResult> mixedSizes =
        tensor::getMixedSizes(b, b.getLoc(), input);
    Value init = tensor::EmptyOp::create(b, mixedSizes, convertedElemType);
    auto genericOp = linalg::GenericOp::create(
        b, /*resultTensorTypes=*/TypeRange{convertedTensorType},
        /*inputs=*/ValueRange{input},
        /*outputs=*/ValueRange{init}, indexingMaps, iteratorTypes,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          ImplicitLocOpBuilder lb(nestedLoc, nestedBuilder);
          ScopedBuilderContext scopedCtx(&lb);
          Value inv = emitScalarInverse(fieldType, args[0]);
          linalg::YieldOp::create(lb, ValueRange{inv});
        });
    rewriter.replaceOp(op, genericOp.getResults());
    return success();
  }

  bool useElementwiseInverse;
  InverseAlgorithm inverseAlgorithm;
};

struct ConvertNegate : ConvertFieldOpBase<NegateOp, ConvertNegate> {
  using Base::Base;
  std::optional<std::string> getAOTFuncName(Type fieldType) const {
    return std::nullopt; // cheap op, always inline
  }
  Value emitInlineCodeGen(NegateOp op, OpAdaptor adaptor) const {
    Type fieldType = getElementTypeOrSelf(op.getOutput());
    return -FieldCodeGen(fieldType, adaptor.getInput(), this->typeConverter);
  }
};

struct ConvertAdd : ConvertFieldOpBase<AddOp, ConvertAdd> {
  using Base::Base;
  std::optional<std::string> getAOTFuncName(Type fieldType) const {
    return std::nullopt; // cheap op, always inline
  }
  Value emitInlineCodeGen(AddOp op, OpAdaptor adaptor) const {
    Type lhsType = getElementTypeOrSelf(op.getLhs().getType());
    Type rhsType = getElementTypeOrSelf(op.getRhs().getType());
    FieldCodeGen lhs(lhsType, adaptor.getLhs(), this->typeConverter);
    FieldCodeGen rhs(rhsType, adaptor.getRhs(), this->typeConverter);
    return lhs + rhs;
  }
};

struct ConvertDouble : ConvertFieldOpBase<DoubleOp, ConvertDouble> {
  using Base::Base;
  std::optional<std::string> getAOTFuncName(Type) const { return std::nullopt; }
  Value emitInlineCodeGen(DoubleOp op, OpAdaptor adaptor) const {
    Type fieldType = getElementTypeOrSelf(op.getOutput());
    return FieldCodeGen(fieldType, adaptor.getInput(), this->typeConverter)
        .dbl();
  }
};

struct ConvertSub : ConvertFieldOpBase<SubOp, ConvertSub> {
  using Base::Base;
  std::optional<std::string> getAOTFuncName(Type fieldType) const {
    return std::nullopt; // cheap op, always inline
  }
  Value emitInlineCodeGen(SubOp op, OpAdaptor adaptor) const {
    Type lhsType = getElementTypeOrSelf(op.getLhs().getType());
    Type rhsType = getElementTypeOrSelf(op.getRhs().getType());
    FieldCodeGen lhs(lhsType, adaptor.getLhs(), this->typeConverter);
    FieldCodeGen rhs(rhsType, adaptor.getRhs(), this->typeConverter);
    return lhs - rhs;
  }
};

struct ConvertMul : ConvertFieldOpBase<MulOp, ConvertMul> {
  using Base::Base;
  std::optional<std::string> getAOTFuncName(Type fieldType) const {
    return getFieldAOTFuncName("mul", fieldType);
  }
  Value emitInlineCodeGen(MulOp op, OpAdaptor adaptor) const {
    Type lhsType = getElementTypeOrSelf(op.getLhs().getType());
    Type rhsType = getElementTypeOrSelf(op.getRhs().getType());
    FieldCodeGen lhs(lhsType, adaptor.getLhs(), this->typeConverter);
    FieldCodeGen rhs(rhsType, adaptor.getRhs(), this->typeConverter);
    return lhs * rhs;
  }
};

struct ConvertSquare : ConvertFieldOpBase<SquareOp, ConvertSquare> {
  using Base::Base;
  std::optional<std::string> getAOTFuncName(Type fieldType) const {
    return getFieldAOTFuncName("square", fieldType);
  }
  Value emitInlineCodeGen(SquareOp op, OpAdaptor adaptor) const {
    Type fieldType = getElementTypeOrSelf(op.getOutput());
    return FieldCodeGen(fieldType, adaptor.getInput(), this->typeConverter)
        .square();
  }
};

struct ConvertPowUI : public OpConversionPattern<PowUIOp> {
  explicit ConvertPowUI(MLIRContext *context)
      : OpConversionPattern<PowUIOp>(context) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PowUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto base = op.getBase();
    auto exp = op.getExp();
    auto fieldType = getElementTypeOrSelf(base);

    APInt modulus;
    Value init;
    if (auto pfType = dyn_cast<PrimeFieldType>(fieldType)) {
      modulus = pfType.getModulus().getValue();
      Type intType = pfType.getStorageType();
      Type stdType = getStandardFormType(pfType);
      Type montType = getMontgomeryFormType(pfType);
      if (auto vecType = dyn_cast<VectorType>(base.getType())) {
        intType = vecType.cloneWith(vecType.getShape(), intType);
        stdType = vecType.cloneWith(vecType.getShape(), stdType);
        montType = vecType.cloneWith(vecType.getShape(), montType);
      }
      init = createScalarOrSplatConstant(b, b.getLoc(), intType, 1);
      init = BitcastOp::create(b, stdType, init);
      if (pfType.isMontgomery()) {
        init = ToMontOp::create(b, montType, init);
      }
    } else if (auto efType = dyn_cast<ExtensionFieldType>(fieldType)) {
      modulus = efType.getBasePrimeField().getModulus().getValue();
      init = field::createFieldOne(efType, b);
    } else {
      op.emitOpError("unsupported output type");
      return failure();
    }

    unsigned expBitWidth = cast<IntegerType>(exp.getType()).getWidth();
    bool allowUnroll = op.getUnroll().value_or(true);

    auto emitBitSerialLoop = [&](Value exp) {
      return generateBitSerialLoop(
          b, exp, base, init,
          [](ImplicitLocOpBuilder &b, Value v) {
            return SquareOp::create(b, v);
          },
          [](ImplicitLocOpBuilder &b, Value acc, Value v) {
            return MulOp::create(b, acc, v);
          },
          allowUnroll);
    };

    // Reduce exponent using Fermat's little theorem:
    // x^(p-1) ≡ 1 (prime field), x^(pᵈ-1) ≡ 1 (extension field)
    modulus = modulus.zext(std::max(expBitWidth, modulus.getBitWidth()));
    APInt order;
    if (auto efType = dyn_cast<ExtensionFieldType>(fieldType)) {
      unsigned degreeOverPrime = efType.getDegreeOverPrime();
      modulus = modulus.zext(modulus.getBitWidth() * degreeOverPrime);
      order = modulus;
      for (unsigned i = 1; i < degreeOverPrime; ++i)
        order = order * modulus;
      order = order - 1;
    } else {
      order = modulus - 1;
    }
    auto intType = IntegerType::get(op.getContext(), order.getBitWidth());

    // Match the literal on the exponent as the op carries it. Widening it to
    // the order's width first would hide an `arith.constant` behind an
    // `arith.extui`, sending every field whose modulus is wider than the
    // exponent type down the runtime `remui` path -- and into the bit-serial
    // loop -- for an exponent that was a literal all along.
    if (auto expConstOp = exp.getDefiningOp<arith::ConstantOp>()) {
      APInt cExp = cast<IntegerAttr>(expConstOp.getValue()).getValue();
      exp = arith::ConstantIntOp::create(
          b, intType, cExp.zext(order.getBitWidth()).urem(order));
    } else {
      if (order.getBitWidth() > expBitWidth)
        exp = arith::ExtUIOp::create(b, intType, exp);
      exp = arith::RemUIOp::create(
          b, exp, arith::ConstantIntOp::create(b, intType, order));
    }

    rewriter.replaceOp(op, emitBitSerialLoop(exp));
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

    Type fieldType = getElementTypeOrSelf(op.getLhs());
    arith::CmpIPredicate predicate = op.getPredicate();
    if (isa<PrimeFieldType>(fieldType)) {
      rewriter.replaceOp(op, compareOnStdDomain(b, fieldType, predicate,
                                                adaptor.getLhs(),
                                                adaptor.getRhs()));
      return success();
    } else if (auto efType = dyn_cast<ExtensionFieldType>(fieldType)) {
      // Recursively flatten tower extensions to prime-level coefficients.
      auto lhsPrimeCoeffs = flattenToPrimeCoeffs(b, adaptor.getLhs());
      auto rhsPrimeCoeffs = flattenToPrimeCoeffs(b, adaptor.getRhs());
      unsigned n = efType.getDegreeOverPrime();
      assert(lhsPrimeCoeffs.size() == n && rhsPrimeCoeffs.size() == n);
      PrimeFieldType basePF = efType.getBasePrimeField();
      SmallVector<Value> cmpResults;
      for (unsigned i = 0; i < n; ++i) {
        cmpResults.push_back(compareOnStdDomain(
            b, basePF, predicate, lhsPrimeCoeffs[i], rhsPrimeCoeffs[i]));
      }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
      Value result = cmpResults.front();
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
      if (predicate == arith::CmpIPredicate::eq) {
        for (unsigned i = 1; i < n; ++i) {
          result = arith::AndIOp::create(b, result, cmpResults[i]);
        }
      } else if (predicate == arith::CmpIPredicate::ne) {
        for (unsigned i = 1; i < n; ++i) {
          result = arith::OrIOp::create(b, result, cmpResults[i]);
        }
      } else {
        llvm_unreachable(
            "Unsupported comparison predicate for extension field type");
      }
      rewriter.replaceOp(op, result);
      return success();
    }
    return failure();
  }

  // Recursively flatten a (possibly tower) extension field value to all its
  // prime-level coefficients (mod_arith values).
  SmallVector<Value> flattenToPrimeCoeffs(ImplicitLocOpBuilder &b,
                                          Value val) const {
    if (isa<mod_arith::ModArithType>(val.getType())) {
      return {val};
    }
    auto coeffs = toModArithCoeffs(b, val);
    SmallVector<Value> result;
    for (Value c : coeffs) {
      auto sub = flattenToPrimeCoeffs(b, c);
      result.append(sub.begin(), sub.end());
    }
    return result;
  }

  Value compareOnStdDomain(ImplicitLocOpBuilder &b, Type fieldType,
                           arith::CmpIPredicate predicate, Value lhs,
                           Value rhs) const {
    if (isMontgomery(fieldType)) {
      auto modArithLhsType = cast<mod_arith::ModArithType>(lhs.getType());
      auto stdModArithType = mod_arith::ModArithType::get(
          modArithLhsType.getContext(), modArithLhsType.getModulus(),
          /*isMontgomery=*/false);

      Value standardLhs =
          mod_arith::FromMontOp::create(b, stdModArithType, lhs);
      Value standardRhs =
          mod_arith::FromMontOp::create(b, stdModArithType, rhs);

      return mod_arith::CmpOp::create(b, predicate, standardLhs, standardRhs);
    } else {
      return mod_arith::CmpOp::create(b, predicate, lhs, rhs);
    }
  }
};

namespace rewrites {
// In an inner namespace to avoid conflicts with canonicalization patterns
#include "prime_ir/Dialect/Field/Conversions/FieldToModArith/FieldToModArith.cpp.inc"
} // namespace rewrites

// Check if a type contains a binary field (tower `bf` or flat `ghash`). These
// are left for the BinaryFieldToArith pass, not lowered to mod_arith here.
bool containsBinaryFieldType(Type type) {
  Type elemType = getElementTypeOrSelf(type);
  return isa<BinaryFieldType>(elemType);
}

// Check if any type in the operation contains BinaryFieldType
bool operationContainsBinaryFieldType(Operation *op) {
  // Check result types
  for (Type type : op->getResultTypes()) {
    if (containsBinaryFieldType(type))
      return true;
  }
  // Check operand types
  for (Value operand : op->getOperands()) {
    if (containsBinaryFieldType(operand.getType()))
      return true;
  }
  return false;
}

} // namespace

struct FieldToModArith : impl::FieldToModArithBase<FieldToModArith> {
  using FieldToModArithBase::FieldToModArithBase;

  void runOnOperation() override;
};

void FieldToModArith::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp module = getOperation();
  FieldToModArithTypeConverter typeConverter(context);

  std::optional<LoweringMode> parsedMode =
      mlir::prime_ir::parseLoweringMode(loweringMode);
  if (!parsedMode) {
    module.emitError() << "invalid lowering-mode option: '" << loweringMode
                       << "' (expected 'inline', 'auto', or 'aot_runtime')";
    return signalPassFailure();
  }
  LoweringMode mode = *parsedMode;

  std::optional<InverseAlgorithm> inverseAlgo =
      parseInverseAlgorithm(inverseAlgorithm);
  if (!inverseAlgo) {
    module.emitError() << "invalid inverse-algorithm option: '"
                       << inverseAlgorithm
                       << "' (expected 'bernstein-yang', 'fermat', or 'auto')";
    return signalPassFailure();
  }

  ConversionTarget target(*context);

  // Mark field operations as dynamically legal if they contain BinaryFieldType
  // (those will be handled by BinaryFieldToArith pass instead). InverseOp is
  // included: FieldCodeGen only models prime/extension fields, so a
  // binary-field inverse reaching ConvertInverse would assert.
  target.addDynamicallyLegalOp<ConstantOp, CmpOp, PowUIOp, InverseOp>(
      [](Operation *op) { return operationContainsBinaryFieldType(op); });

  // ConvertFieldOpBase patterns cannot inline-codegen shaped extension field
  // types. ElementwiseMappable ops (add, sub, ...) are scalarized by
  // convert-elementwise-to-linalg. Mark them as legal here so
  // field-to-mod-arith passes through these ops without failing.
  // Note: InverseOp is NOT listed — ConvertInverse handles all (non-binary)
  // tensor inverses via Montgomery's batch inversion trick.
  target
      .addDynamicallyLegalOp<AddOp, SubOp, MulOp, NegateOp, DoubleOp, SquareOp>(
          [](Operation *op) {
            if (operationContainsBinaryFieldType(op))
              return true;
            return llvm::any_of(op->getResultTypes(), [](Type t) {
              return isa<ShapedType>(t) &&
                     isa<ExtensionFieldType>(getElementTypeOrSelf(t));
            });
          });

  // Mark remaining field dialect ops as illegal (prime/extension field ops)
  target.addIllegalDialect<FieldDialect>();
  target.addLegalDialect<mod_arith::ModArithDialect>();
  target.addLegalDialect<func::FuncDialect>();
  target.addLegalOp<func::FuncOp, func::CallOp, func::ReturnOp>();

  RewritePatternSet patterns(context);
  rewrites::populateWithGenerated(patterns);

  AOTConfig aotConfig{mode, inlineConstantOps};
  patterns.add<
      // clang-format off
      ConvertAdd,
      ConvertDouble,
      ConvertMul,
      ConvertNegate,
      ConvertSquare,
      ConvertSub
      // clang-format on
      >(typeConverter, context, aotConfig);
  patterns.add<ConvertInverse>(typeConverter, context, aotConfig,
                               useElementwiseInverse, *inverseAlgo);

  patterns.add<
      // clang-format off
      ConvertBitcast,
      ConvertConstant,
      ConvertCmp,
      ConvertFromMont,
      ConvertPowUI,
      ConvertToMont
      // clang-format on
      >(typeConverter, context);

  // Catch-all: converts any op whose operands/results carry field types.
  // Op-specific patterns above have root-op-name priority in the applicator.
  patterns.add<ConvertAny<void>>(typeConverter, context);

  addStructuralConversionPatterns(typeConverter, patterns, target);

  // Any op not explicitly registered is dynamically legal iff its types are
  // already converted.  This covers ops from downstream or unrelated dialects.
  target.markUnknownOpDynamicallyLegal(
      [&](Operation *op) { return typeConverter.isLegal(op); });

  // Field dialect ops that stay as field ops after type conversion (they keep
  // the same op name but with mod_arith element types) need an explicit
  // override because addIllegalDialect<FieldDialect> would otherwise reject
  // them even after successful conversion.
  target.addDynamicallyLegalOp<BitcastOp, ExtFromCoeffsOp, ExtToCoeffsOp>(
      [&](auto op) { return typeConverter.isLegal(op); });

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

} // namespace mlir::prime_ir::field
