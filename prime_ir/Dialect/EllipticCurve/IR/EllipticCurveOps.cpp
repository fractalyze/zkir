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

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveAttributes.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Dialect/EllipticCurve/IR/KnownCurves.h"
#include "prime_ir/Dialect/EllipticCurve/IR/PointOperation.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/FieldTypes.h"
#include "prime_ir/Dialect/ModArith/IR/ModArithTypes.h"
#include "prime_ir/Utils/AssemblyFormatUtils.h"
#include "prime_ir/Utils/BitcastOpUtils.h"
#include "prime_ir/Utils/ConstantFolder.h"

// IWYU pragma: begin_keep
// Headers needed for EllipticCurveCanonicalization.cpp.inc
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/Matchers.h"
// IWYU pragma: end_keep

namespace mlir::prime_ir::elliptic_curve {

namespace {
template <typename OpType>
LogicalResult verifyMSMPointTypes(OpType op, Type inputType, Type outputType) {
  // Named rather than left to fall through the short-Weierstrass cases below:
  // no backend carries a twisted-Edwards MSM, so accepting one here would push
  // the failure down to a lowering that cannot express it.
  if (isa<EdAffineType, EdExtendedType>(inputType) ||
      isa<EdAffineType, EdExtendedType>(outputType)) {
    return op.emitError() << "MSM does not support twisted Edwards points";
  }
  if (isa<JacobianType>(inputType) && isa<AffineType>(outputType)) {
    return op.emitError()
           << "jacobian input points require a jacobian or xyzz output type";
  } else if (isa<XYZZType>(inputType) &&
             isa<AffineType, JacobianType>(outputType)) {
    return op.emitError() << "xyzz input points require an xyzz output type";
  }
  return success();
}

template <typename OpType>
LogicalResult verifyPointCoordTypes(OpType op, Type pointType, Type coordType) {
  Type ptBaseField = cast<PointTypeInterface>(pointType).getBaseFieldType();
  if (isa<field::FieldDialect>(coordType.getDialect())) {
    if (ptBaseField != coordType) {
      return op.emitError() << "coord must be base field of point; but got "
                            << coordType << " expected " << ptBaseField;
    }
    return success();
  }
  if (auto pfType = dyn_cast<field::PrimeFieldType>(ptBaseField)) {
    if (auto modArithType = dyn_cast<mod_arith::ModArithType>(coordType)) {
      if (pfType.getModulus() != modArithType.getModulus())
        return op.emitError()
               << "output must have the same modulus as the base "
                  "field of input";
      if (pfType.isMontgomery() != modArithType.isMontgomery())
        return op.emitError()
               << "output must have the same montgomery form as the "
                  "base field of input";
    } else if (auto intType = dyn_cast<IntegerType>(coordType)) {
      if (intType.getWidth() != pfType.getTypeSizeInBits())
        return op.emitError()
               << "output must have the same bitwidth as the base "
                  "field of input";
    } else {
      return op.emitError()
             << "output must be a mod_arith type or an integer "
                "type if the base field is a prime field; but got "
             << coordType;
    }
  } else {
    auto efType = cast<field::ExtensionFieldType>(ptBaseField);
    if (auto structType = dyn_cast<LLVM::LLVMStructType>(coordType)) {
      unsigned degree = efType.getDegree();
      if (structType.getBody().size() != degree)
        return op.emitError()
               << "output struct must have " << degree
               << " elements for extension field of degree " << degree;
      // NOTE: In case of extension fields, the types are not lowered to modular
      // types since struct cannot contain modular types so we can ignore that
      // case.
      auto baseFieldStorageType = efType.getBasePrimeField().getStorageType();
      for (unsigned i = 0; i < degree; ++i) {
        if (structType.getBody()[i] != baseFieldStorageType)
          return op.emitError() << "output struct element must have the same "
                                   "bitwidth as the base field of input";
      }
    } else {
      return op.emitError()
             << "output must be a struct type for extension field; but got "
             << coordType;
    }
  }
  return success();
}
} // namespace

Value createZeroPoint(ImplicitLocOpBuilder &b, Type pointType) {
  auto baseFieldType = cast<PointTypeInterface>(pointType).getBaseFieldType();
  Value zeroBF = field::createFieldZero(baseFieldType, b);
  Value oneBF = field::createFieldOne(baseFieldType, b);
  if (isa<AffineType>(pointType)) {
    return FromCoordsOp::create(b, pointType, ValueRange{zeroBF, zeroBF});
  }
  if (isa<JacobianType>(pointType)) {
    return FromCoordsOp::create(b, pointType, ValueRange{oneBF, oneBF, zeroBF});
  }
  if (isa<XYZZType>(pointType)) {
    return FromCoordsOp::create(b, pointType,
                                ValueRange{oneBF, oneBF, zeroBF, zeroBF});
  }
  // Twisted Edwards has a real affine identity, (0, 1), rather than the
  // off-curve sentinel short Weierstrass uses for the point at infinity —
  // a*0^2 + 1^2 = 1 = 1 + d*0^2*1^2. Extended carries it as x = X/Z, y = Y/Z,
  // T = XY/Z, so (0 : 1 : 1 : 0).
  if (isa<EdAffineType>(pointType)) {
    return FromCoordsOp::create(b, pointType, ValueRange{zeroBF, oneBF});
  }
  if (isa<EdExtendedType>(pointType)) {
    return FromCoordsOp::create(b, pointType,
                                ValueRange{zeroBF, oneBF, oneBF, zeroBF});
  }
  llvm_unreachable("Unsupported point type for createZeroPoint");
}

/////////////// VERIFY OPS /////////////////

LogicalResult FromCoordsOp::verify() {
  Type outputType = getType();
  unsigned numCoords = cast<PointTypeInterface>(outputType).getNumCoords();
  if (numCoords != getCoords().size()) {
    return emitError() << outputType << " should have " << numCoords
                       << " coordinates";
  }
  return verifyPointCoordTypes(*this, outputType, getCoords()[0].getType());
}

LogicalResult ToCoordsOp::verify() {
  Type inputType = getInput().getType();
  unsigned numCoords =
      cast<PointTypeInterface>(getElementTypeOrSelf(inputType)).getNumCoords();
  if (numCoords != getOutput().size()) {
    return emitError() << inputType << " should have " << numCoords
                       << " coordinates";
  }
  return verifyPointCoordTypes(*this, inputType, getType(0));
}

LogicalResult IsZeroOp::verify() {
  Type inputType = getInput().getType();
  if (isa<PointTypeInterface>(getElementTypeOrSelf(inputType)))
    return success();
  return emitError() << "invalid input type";
}

namespace {
std::optional<PointKind> getPointKindOf(Type type) {
  if (auto point = dyn_cast<PointTypeInterface>(type)) {
    return point.getPointKind();
  }
  return std::nullopt;
}

// Stated over PointKind rather than the concrete types so the rule reads the
// same for both curve families: affine is not closed under the group law, so
// an affine operand widens the result to a projective kind of its own family
// (jacobian or xyzz for short Weierstrass, extended for twisted Edwards).
template <typename OpType>
LogicalResult verifyBinaryOp(OpType op) {
  Type lhsType = getElementTypeOrSelf(op.getLhs().getType());
  Type rhsType = getElementTypeOrSelf(op.getRhs().getType());
  Type outputType = getElementTypeOrSelf(op.getType());
  std::optional<PointKind> lhs = getPointKindOf(lhsType);
  std::optional<PointKind> rhs = getPointKindOf(rhsType);
  std::optional<PointKind> out = getPointKindOf(outputType);
  if (!lhs || !rhs || !out) {
    return op->emitError() << "input or output types are wrong";
  }
  if (!isSameFamily(*lhs, *rhs) || !isSameFamily(*lhs, *out)) {
    return op->emitError()
           << "cannot mix short Weierstrass and twisted Edwards points";
  }
  if (isAffine(*lhs) || isAffine(*rhs)) {
    // affine, affine -> any projective kind of the family
    if (lhsType == rhsType && !isAffine(*out)) {
      return success();
    }
    // affine, projective -> that same projective type
    if (!isAffine(*out) && (lhsType == outputType || rhsType == outputType)) {
      return success();
    }
  } else if (lhsType == rhsType && rhsType == outputType) {
    // projective, projective -> the same projective type
    return success();
  }
  // TODO(ashjeong): check the curves of given types are the same
  return op->emitError() << "input or output types are wrong";
}

// Shared by the two unary group operations: an affine input widens to a
// projective kind of the same family, and any other kind passes through.
LogicalResult verifyAffineWidens(Operation *op, Type inputType, Type outputType,
                                 StringRef what) {
  std::optional<PointKind> in = getPointKindOf(inputType);
  std::optional<PointKind> out = getPointKindOf(outputType);
  if (in && out && isSameFamily(*in, *out) &&
      ((isAffine(*in) && !isAffine(*out)) || inputType == outputType)) {
    return success();
  }
  // TODO(ashjeong): check curves/fields are the same
  return op->emitError() << "wrong output type given " << what;
}
} // namespace

LogicalResult AddOp::verify() { return verifyBinaryOp(*this); }

LogicalResult SubOp::verify() { return verifyBinaryOp(*this); }

LogicalResult DoubleOp::verify() {
  return verifyAffineWidens(*this, getElementTypeOrSelf(getInput()),
                            getElementTypeOrSelf(getType()), "input type");
}

LogicalResult ScalarMulOp::verify() {
  return verifyAffineWidens(*this, getElementTypeOrSelf(getPoint()),
                            getElementTypeOrSelf(getType()), "point type");
}

LogicalResult CmpOp::verify() {
  arith::CmpIPredicate predicate = getPredicate();
  if (predicate == arith::CmpIPredicate::eq ||
      predicate == arith::CmpIPredicate::ne) {
    return success();
  } else {
    return emitOpError() << "only 'eq' and 'ne' comparisons are supported for "
                            "elliptic curve points";
  }
}

LogicalResult MSMOp::verify() {
  TensorType scalarsType = getScalars().getType();
  TensorType pointsType = getPoints().getType();
  if (scalarsType.getRank() != pointsType.getRank()) {
    return emitError() << "scalars and points must have the same rank";
  }
  Type inputType = pointsType.getElementType();
  Type outputType = getType();
  if (failed(verifyMSMPointTypes(*this, inputType, outputType))) {
    return failure();
  }

  int32_t degree = getDegree();
  if (degree <= 0) {
    return emitError() << "degree must be greater than 0";
  }

  if (scalarsType.hasStaticShape() && pointsType.hasStaticShape()) {
    if (scalarsType.getNumElements() != pointsType.getNumElements()) {
      return emitError()
             << "scalars and points must have the same number of elements";
    }
    if (scalarsType.getNumElements() > (int64_t{1} << degree)) {
      return emitError() << "scalars must have at least 2^degree elements";
    }
  } else if (scalarsType.hasStaticShape() && !pointsType.hasStaticShape()) {
    return emitError() << "scalars has static shape and points does not";
  } else if (!scalarsType.hasStaticShape() && pointsType.hasStaticShape()) {
    return emitError() << "points has static shape and scalars does not";
  }

  int32_t windowBits = getWindowBits();
  if (windowBits < 0) {
    return emitError() << "window bits must be greater than or equal to 0";
  }

  return success();
  // TODO(ashjeong): check curves/fields are the same
}

LogicalResult PairingCheckOp::verify() {
  auto g1TensorType = getG1Points().getType();
  auto g2TensorType = getG2Points().getType();
  auto g1PointType = cast<PointTypeInterface>(g1TensorType.getElementType());
  auto g2PointType = cast<PointTypeInterface>(g2TensorType.getElementType());

  // Both tensors must have matching first dimension
  if (g1TensorType.hasStaticShape() && g2TensorType.hasStaticShape()) {
    if (g1TensorType.getNumElements() != g2TensorType.getNumElements()) {
      return emitError()
             << "g1 and g2 tensors must have the same number of elements, "
                "but got "
             << g1TensorType.getNumElements() << " and "
             << g2TensorType.getNumElements();
    }
  }

  // G1 points must be over a prime field
  Type g1BaseField = g1PointType.getBaseFieldType();
  if (!isa<field::PrimeFieldType>(g1BaseField)) {
    return emitError() << "g1 points must be over a prime base field, but got "
                       << g1BaseField;
  }

  // G2 points must be over a degree-2 extension field
  Type g2BaseField = g2PointType.getBaseFieldType();
  auto g2ExtField = dyn_cast<field::ExtensionFieldType>(g2BaseField);
  if (!g2ExtField) {
    return emitError()
           << "g2 points must be over an extension base field, but got "
           << g2BaseField;
  }
  if (g2ExtField.getDegree() != 2) {
    return emitError()
           << "g2 points must be over a degree-2 extension field, but got "
              "degree "
           << g2ExtField.getDegree();
  }

  // Both curves must be known pairing-friendly short Weierstrass curves
  auto g1CurveAttr = dyn_cast<ShortWeierstrassAttr>(g1PointType.getCurveAttr());
  auto g2CurveAttr = dyn_cast<ShortWeierstrassAttr>(g2PointType.getCurveAttr());
  if (!g1CurveAttr || !g2CurveAttr) {
    return emitError() << "both curves must be short Weierstrass curves";
  }

  auto g1Alias = getKnownCurveAlias(g1CurveAttr);
  auto g2Alias = getKnownCurveAlias(g2CurveAttr);
  if (!g1Alias || !g2Alias) {
    return emitError() << "both curves must be known pairing-friendly curves";
  }

  // Verify they form a valid pairing pair (e.g., bn254_g1 + bn254_g2)
  llvm::StringRef g1Ref(*g1Alias);
  llvm::StringRef g2Ref(*g2Alias);
  if (!g1Ref.ends_with("_g1")) {
    return emitError() << "g1 points must be on a G1 curve, but identified as "
                       << *g1Alias;
  }
  if (!g2Ref.ends_with("_g2")) {
    return emitError() << "g2 points must be on a G2 curve, but identified as "
                       << *g2Alias;
  }

  // Ensure both curves are from the same family
  llvm::StringRef g1Family = g1Ref.drop_back(3);
  llvm::StringRef g2Family = g2Ref.drop_back(3);
  if (g1Family != g2Family) {
    return emitError()
           << "g1 and g2 curves must be from the same curve family, but got "
           << *g1Alias << " and " << *g2Alias;
  }

  return success();
}

//////////////// VERIFY POINT CONVERSIONS ////////////////

LogicalResult ConvertPointTypeOp::verify() {
  Type inputType = getInput().getType();
  Type outputType = getType();
  if (inputType == outputType)
    return emitError() << "Converting on same types";
  // TODO(ashjeong): check curves are the same
  return success();
}

//////////////// VERIFY AND CAST BITCAST ////////////////

namespace {
// Type check helpers for bitcast verification
// Accepts field types and their lower-level representations (mod_arith and
// integers) to support lowering pipelines.
static bool isFieldType(Type t) {
  return isa<field::PrimeFieldType, field::ExtensionFieldType,
             mod_arith::ModArithType, IntegerType>(t);
}

static bool isPointType(Type t) { return isa<PointTypeInterface>(t); }
} // namespace

// Check if types are compatible for bitcast between field tensor and EC point
// tensor.
bool BitcastOp::areCastCompatible(TypeRange inputs, TypeRange outputs) {
  if (inputs.size() != 1 || outputs.size() != 1)
    return false;

  Type inputType = inputs[0];
  Type outputType = outputs[0];

  // Both must be shaped types (tensors)
  auto inputShaped = dyn_cast<ShapedType>(inputType);
  auto outputShaped = dyn_cast<ShapedType>(outputType);
  if (!inputShaped || !outputShaped)
    return false;

  // One must be a field type tensor, the other a point type tensor
  Type inputElemType = inputShaped.getElementType();
  Type outputElemType = outputShaped.getElementType();

  PointTypeInterface pointType;
  ShapedType fieldTensor;
  ShapedType pointTensor;

  if (isFieldType(inputElemType) && isPointType(outputElemType)) {
    fieldTensor = inputShaped;
    pointTensor = outputShaped;
    pointType = cast<PointTypeInterface>(outputElemType);
  } else if (isPointType(inputElemType) && isFieldType(outputElemType)) {
    fieldTensor = outputShaped;
    pointTensor = inputShaped;
    pointType = cast<PointTypeInterface>(inputElemType);
  } else {
    return false;
  }

  unsigned numCoords = pointType.getNumCoords();
  Type baseFieldType = pointType.getBaseFieldType();
  unsigned extDegree = field::getExtensionDegree(baseFieldType);
  unsigned K = numCoords * extDegree;

  // A point's K coordinates sit contiguously, so the constraint is on element
  // counts and not on any distinguished dimension: batching points into rows
  // reinterprets the same bytes as laying them out flat.
  if (fieldTensor.hasStaticShape() && pointTensor.hasStaticShape())
    return fieldTensor.getNumElements() == pointTensor.getNumElements() * K;

  // Only the field side is known: it must at least divide into whole points.
  if (fieldTensor.hasStaticShape())
    return fieldTensor.getNumElements() % K == 0;

  // A dynamic field extent can accommodate any point count; trust the user.
  return true;
}

LogicalResult BitcastOp::verify() {
  Type inputType = getInput().getType();
  Type outputType = getOutput().getType();

  // A reinterpret only describes the same bytes when the source is packed, and
  // a point occupies k field elements, so the offset must land on a point: the
  // descriptor rebuild in the LLVM lowering divides it by k.
  auto inMt = dyn_cast<MemRefType>(inputType);
  auto outMt = dyn_cast<MemRefType>(outputType);
  unsigned ratio = 1;
  if (inMt && outMt) {
    if (auto pointType = dyn_cast<PointTypeInterface>(outMt.getElementType())) {
      ratio = pointType.getNumCoords() *
              field::getExtensionDegree(pointType.getBaseFieldType());
    }
    // The rebuild stamps the result extents in as constants, so a dynamic
    // result would become a descriptor full of the dynamic sentinel. A
    // diagnostic from the conversion pattern is rolled back with the pattern
    // and never reaches the user, so it has to be refused here.
    if (!outMt.hasStaticShape()) {
      return emitOpError("cannot rebuild a descriptor for a dynamically "
                         "shaped result, but got ")
             << outMt;
    }
  }
  if (failed(verifyBitcastMemRefLayout(*this, inputType, outputType, ratio,
                                       "a point"))) {
    return failure();
  }

  if (areCastCompatible(TypeRange{inputType}, TypeRange{outputType}))
    return success();

  // Provide detailed error messages
  auto inputShaped = dyn_cast<ShapedType>(inputType);
  auto outputShaped = dyn_cast<ShapedType>(outputType);

  if (!inputShaped || !outputShaped) {
    return emitOpError() << "bitcast requires tensor types; got " << inputType
                         << " and " << outputType;
  }

  Type inputElemType = inputShaped.getElementType();
  Type outputElemType = outputShaped.getElementType();

  if (!((isFieldType(inputElemType) && isPointType(outputElemType)) ||
        (isPointType(inputElemType) && isFieldType(outputElemType)))) {
    return emitOpError() << "bitcast requires one field tensor and one point "
                            "tensor; got "
                         << inputType << " and " << outputType;
  }

  if (!inputShaped.hasStaticShape() || !outputShaped.hasStaticShape()) {
    return emitOpError() << "bitcast requires static shapes";
  }

  return emitOpError() << "element count mismatch between " << inputType
                       << " and " << outputType;
}

//////////////// PARSE AND PRINT CONSTANT ////////////////

namespace {

// Helper to get extension field degree if the base field is an extension field.
// Returns std::nullopt for prime fields.
// Note: Named differently from field::getExtensionDegree which returns unsigned
// with a default of 1.
std::optional<unsigned> getOptionalExtensionDegree(Type baseFieldType) {
  if (auto efType = dyn_cast<field::ExtensionFieldType>(baseFieldType))
    return efType.getDegreeOverPrime();
  return std::nullopt;
}

// Helper to create a coordinate attribute from APInt coefficients.
// For prime fields: creates IntegerAttr
// For extension fields: creates DenseIntElementsAttr
Attribute createCoordAttr(Type baseFieldType, ArrayRef<APInt> coeffs) {
  if (auto pfType = dyn_cast<field::PrimeFieldType>(baseFieldType)) {
    assert(coeffs.size() == 1 && "prime field should have single coefficient");
    return IntegerAttr::get(pfType.getStorageType(), coeffs[0]);
  }
  auto efType = cast<field::ExtensionFieldType>(baseFieldType);
  auto tensorType =
      RankedTensorType::get({static_cast<int64_t>(coeffs.size())},
                            efType.getBasePrimeField().getStorageType());
  return DenseIntElementsAttr::get(tensorType, coeffs);
}

// Helper to reconstruct ArrayAttr structure from flat parsed values.
// Converts flat values [x0, y0, z0, x1, y1, z1, ...] into
// ArrayAttr<ArrayAttr<Attr>> where each inner ArrayAttr represents one point.
FailureOr<ArrayAttr> reconstructCoordsAttr(MLIRContext *ctx,
                                           ArrayRef<APInt> flatValues,
                                           PointTypeInterface pointType,
                                           int64_t numPoints) {
  Type baseFieldType = pointType.getBaseFieldType();
  unsigned numCoords = pointType.getNumCoords();
  unsigned extDegree = getOptionalExtensionDegree(baseFieldType).value_or(1);
  unsigned valuesPerPoint = numCoords * extDegree;

  if (flatValues.size() != static_cast<size_t>(numPoints) * valuesPerPoint)
    return failure();

  SmallVector<Attribute> pointAttrs;
  size_t idx = 0;

  for (int64_t p = 0; p < numPoints; ++p) {
    SmallVector<Attribute> coordAttrs;
    for (unsigned c = 0; c < numCoords; ++c) {
      ArrayRef<APInt> coeffs(flatValues.begin() + idx, extDegree);
      coordAttrs.push_back(createCoordAttr(baseFieldType, coeffs));
      idx += extDegree;
    }
    pointAttrs.push_back(ArrayAttr::get(ctx, coordAttrs));
  }

  return ArrayAttr::get(ctx, pointAttrs);
}

// Helper: recursively parse dense<[...]> values and track shape.
// Returns flat list of APInt values and the parsed shape.
ParseResult parseDenseValues(OpAsmParser &parser,
                             SmallVector<APInt> &parsedInts,
                             SmallVector<int64_t> &parsedShape) {
  auto parseNested = [&](auto &&self, int level = 0) -> ParseResult {
    int64_t count = 0;
    auto checkpoint = parser.getCurrentLocation();
    do {
      APInt val;
      OptionalParseResult res = parser.parseOptionalInteger(val);
      if (res.has_value()) {
        if (failed(*res))
          return failure();
        parsedInts.push_back(std::move(val));
        ++count;
      } else if (succeeded(parser.parseOptionalLSquare())) {
        if (failed(self(self, level + 1)))
          return failure();
        ++count;
      } else {
        // Neither integer nor nested list - could be empty list or end of list
        break;
      }
    } while (succeeded(parser.parseOptionalComma()));

    if (static_cast<int64_t>(parsedShape.size()) <= level)
      parsedShape.resize(level + 1);
    if (parsedShape[level] == 0)
      parsedShape[level] = count;
    else if (parsedShape[level] != count)
      return parser.emitError(checkpoint, "non-uniform array at dimension ")
             << level;
    return parser.parseRSquare();
  };

  if (failed(parser.parseLSquare()) || failed(parseNested(parseNested)))
    return failure();
  return success();
}

// Helper to get modulus from base field type
FailureOr<APInt> getModulusFromBaseField(OpAsmParser &parser,
                                         Type baseFieldType) {
  if (auto pfType = dyn_cast<field::PrimeFieldType>(baseFieldType)) {
    return pfType.getModulus().getValue();
  }
  if (auto efType = dyn_cast<field::ExtensionFieldType>(baseFieldType)) {
    return efType.getBasePrimeField().getModulus().getValue();
  }
  parser.emitError(parser.getCurrentLocation(), "unsupported base field type");
  return failure();
}

} // namespace

// Unified parsing for EC constants in dense<...> format.
// Supports both scalar (!jacobian) and tensor (tensor<Nx!jacobian>) types.
// Format:
//   Scalar prime field: dense<[x, y, z]> : !jacobian
//   Scalar ext field:   dense<[[x0, x1], [y0, y1]]> : !affine_ext2
//   Tensor prime field: dense<[[x0, y0], [x1, y1]]> : tensor<2x!affine>
//   Tensor ext field:   dense<[[[a0,a1],[b0,b1]], ...]> :
//   tensor<Nx!affine_ext2>
ParseResult parseEllipticCurveConstant(OpAsmParser &parser,
                                       OperationState &result) {
  // 1. Parse dense<[...]> format
  if (failed(parser.parseKeyword("dense")) || failed(parser.parseLess()))
    return failure();

  SmallVector<APInt> parsedInts;
  SmallVector<int64_t> parsedShape;
  if (failed(parseDenseValues(parser, parsedInts, parsedShape)))
    return failure();

  // 2. Parse type
  Type parsedType;
  if (failed(parser.parseGreater()) ||
      failed(parser.parseColonType(parsedType)))
    return failure();

  // 3. Determine scalar vs tensor and extract point type
  bool isTensor = isa<ShapedType>(parsedType);
  int64_t numPoints = 1;
  PointTypeInterface pointType;

  if (isTensor) {
    auto shapedType = cast<ShapedType>(parsedType);
    numPoints = shapedType.getNumElements();
    pointType = dyn_cast<PointTypeInterface>(shapedType.getElementType());
    if (!pointType)
      return parser.emitError(parser.getCurrentLocation(),
                              "expected point type element");
  } else {
    pointType = dyn_cast<PointTypeInterface>(parsedType);
    if (!pointType)
      return parser.emitError(parser.getCurrentLocation(),
                              "expected point type");
  }

  // 4. Validate shape
  // Expected shapes:
  //   Scalar prime field: [numCoords]
  //   Scalar ext field:   [numCoords, degree]
  //   Tensor prime field: [numPoints, numCoords]
  //   Tensor ext field:   [numPoints, numCoords, degree]
  unsigned numCoords = pointType.getNumCoords();
  Type baseFieldType = pointType.getBaseFieldType();
  auto extDegree = getOptionalExtensionDegree(baseFieldType);

  SmallVector<int64_t> expectedShape;
  if (isTensor) {
    auto shapedType = cast<ShapedType>(parsedType);
    expectedShape.append(shapedType.getShape().begin(),
                         shapedType.getShape().end());
  }
  expectedShape.push_back(static_cast<int64_t>(numCoords));
  if (extDegree)
    expectedShape.push_back(static_cast<int64_t>(*extDegree));

  if (expectedShape.size() != parsedShape.size() ||
      !std::equal(expectedShape.begin(), expectedShape.end(),
                  parsedShape.begin())) {
    return parser.emitError(parser.getCurrentLocation())
           << "shape mismatch: expected ["
           << llvm::make_range(expectedShape.begin(), expectedShape.end())
           << "] but got ["
           << llvm::make_range(parsedShape.begin(), parsedShape.end()) << "]";
  }

  // 5. Validate modular values
  auto modulusOrErr = getModulusFromBaseField(parser, baseFieldType);
  if (failed(modulusOrErr))
    return failure();
  APInt modulus = *modulusOrErr;

  for (APInt &val : parsedInts) {
    if (failed(validateModularInteger(parser, modulus, val)))
      return failure();
  }

  // 6. Reconstruct attribute (always ArrayAttr<ArrayAttr<Attr>>)
  auto coordsAttr = reconstructCoordsAttr(parser.getContext(), parsedInts,
                                          pointType, numPoints);
  if (failed(coordsAttr))
    return parser.emitError(parser.getCurrentLocation(),
                            "failed to reconstruct coords attribute");

  result.addAttribute("coords", *coordsAttr);
  result.addTypes(parsedType);
  return success();
}

ParseResult ConstantOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseEllipticCurveConstant(parser, result);
}

void ConstantOp::print(OpAsmPrinter &p) {
  p << " dense<";
  Type type = getType();
  auto coords = getCoords();
  bool isTensor = isa<ShapedType>(type);

  // Unified structure: ArrayAttr<ArrayAttr<Attr>>
  // - Scalar: [[x, y, z]] -> print as [x, y, z]
  // - Tensor: [[x0, y0], [x1, y1]] -> print as [[x0, y0], [x1, y1]]
  if (isTensor) {
    p << "[";
  }

  for (size_t i = 0; i < coords.size(); ++i) {
    if (i > 0)
      p << ", ";
    auto pointCoords = cast<ArrayAttr>(coords[i]);
    p << "[";
    for (size_t j = 0; j < pointCoords.size(); ++j) {
      if (j > 0)
        p << ", ";
      p.printAttributeWithoutType(pointCoords[j]);
    }
    p << "]";
  }

  if (isTensor) {
    p << "]";
  }

  p << "> : ";
  p.printType(type);
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return adaptor.getCoords();
}

// static
ConstantOp ConstantOp::materialize(OpBuilder &builder, Attribute value,
                                   Type type, Location loc) {
  if (!isa<PointTypeInterface>(getElementTypeOrSelf(type))) {
    return nullptr;
  }

  if (auto arrayAttr = dyn_cast<ArrayAttr>(value)) {
    return ConstantOp::create(builder, loc, type, arrayAttr);
  }
  return nullptr;
}

Operation *EllipticCurveDialect::materializeConstant(OpBuilder &builder,
                                                     Attribute value, Type type,
                                                     Location loc) {
  return ConstantOp::materialize(builder, value, type, loc);
}

//////////////// CANONICALIZATION PATTERNS ////////////////

namespace {

struct FromCoordsOfToCoords : public mlir::OpRewritePattern<FromCoordsOp> {
  using OpRewritePattern<FromCoordsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FromCoordsOp op,
                                PatternRewriter &rewriter) const override {
    // Match: elliptic_curve.from_coords(elliptic_curve.to_coords(arg))
    if (op.getOperands().empty())
      return failure();

    auto extToCoordsOp = op.getOperands().front().getDefiningOp<ToCoordsOp>();
    if (!extToCoordsOp)
      return failure();

    // The operands must be exactly the results of the ToCoordsOp, in order.
    if (op.getOperands() != extToCoordsOp->getResults())
      return failure();

    rewriter.replaceOp(op, extToCoordsOp->getOperands());
    return success();
  }
};

struct ToCoordsOfFromCoords : public mlir::OpRewritePattern<ToCoordsOp> {
  using OpRewritePattern<ToCoordsOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ToCoordsOp op,
                                PatternRewriter &rewriter) const override {
    // Match:
    // elliptic_curve.to_coords(elliptic_curve.from_coords(arg...))
    auto extFromCoordOp = op.getOperand().getDefiningOp<FromCoordsOp>();
    if (!extFromCoordOp)
      return failure();

    rewriter.replaceOp(op, extFromCoordOp->getOperands());
    return success();
  }
};

} // namespace

void FromCoordsOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                               MLIRContext *context) {
  patterns.add<FromCoordsOfToCoords>(context);
}

void ToCoordsOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                             MLIRContext *context) {
  patterns.add<ToCoordsOfFromCoords>(context);
}

namespace {

//===----------------------------------------------------------------------===//
// Constant folding infrastructure for EllipticCurve operations
//===----------------------------------------------------------------------===//

// Extract a FieldOperation from an attribute (IntegerAttr or DenseElementsAttr)
field::FieldOperation getFieldOpFromAttr(Attribute attr, Type fieldType) {
  return TypeSwitch<Attribute, field::FieldOperation>(attr)
      .Case<IntegerAttr>([&](auto intAttr) {
        return field::FieldOperation::fromUnchecked(intAttr.getValue(),
                                                    fieldType);
      })
      .Case<DenseIntElementsAttr>([&](auto denseAttr) {
        auto values = denseAttr.template getValues<APInt>();
        const SmallVector<APInt> coeffs(values.begin(), values.end());
        return field::FieldOperation::fromUnchecked(coeffs, fieldType);
      });
}

// Convert ArrayAttr (coordinates) to PointOperation
template <PointKind Kind>
PointOperationBase<Kind> pointOpFromArrayAttr(ArrayAttr coordsAttr,
                                              PointTypeInterface pointType) {
  constexpr size_t kNumCoords = PointOperationBase<Kind>::kNumCoords;
  assert(
      coordsAttr.size() == kNumCoords &&
      "ArrayAttr size must match the number of coordinates for the PointKind");

  Type baseFieldType = pointType.getBaseFieldType();
  std::array<field::FieldOperation, kNumCoords> coords;

  for (size_t i = 0; i < kNumCoords; ++i) {
    auto fieldOp = getFieldOpFromAttr(coordsAttr[i], baseFieldType);
    coords[i] = fieldOp;
  }

  return PointOperationBase<Kind>::fromUnchecked(coords, pointType);
}

// Convert a single point's coords to ArrayAttr (inner array)
template <PointKind Kind>
ArrayAttr pointCoordsToArrayAttr(MLIRContext *ctx,
                                 const PointOperationBase<Kind> &op) {
  Type baseFieldType = op.getPointType().getBaseFieldType();
  SmallVector<Attribute> attrs;

  for (const auto &coord : op.getCoords()) {
    // PrimeFieldOperation converts to APInt, ExtensionFieldOperation to
    // SmallVector<APInt>
    Attribute attr =
        TypeSwitch<Type, Attribute>(baseFieldType)
            .template Case<field::PrimeFieldType>([&](auto pfType) {
              return IntegerAttr::get(pfType.getStorageType(),
                                      static_cast<APInt>(coord));
            })
            .template Case<field::ExtensionFieldType>([&](auto efType) {
              SmallVector<APInt> coeffs =
                  static_cast<SmallVector<APInt>>(coord);
              auto tensorType = RankedTensorType::get(
                  {static_cast<int64_t>(coeffs.size())},
                  efType.getBasePrimeField().getStorageType());
              return DenseIntElementsAttr::get(tensorType, coeffs);
            });
    attrs.push_back(attr);
  }

  return ArrayAttr::get(ctx, attrs);
}

// Convert PointOperation to unified ArrayAttr<ArrayAttr<Attr>> structure
// Returns ArrayAttr containing single inner ArrayAttr of coordinate attributes
template <PointKind Kind>
ArrayAttr pointOpToArrayAttr(MLIRContext *ctx,
                             const PointOperationBase<Kind> &op) {
  ArrayAttr pointCoords = pointCoordsToArrayAttr(ctx, op);
  return ArrayAttr::get(ctx, {pointCoords});
}

// Create a PointOperation from ArrayAttr based on PointKind.
// Unified structure: ArrayAttr<ArrayAttr<Attr>> where outer array contains
// points and inner array contains coordinates per point.
PointOperation createPointOp(ArrayAttr coordsAttr,
                             PointTypeInterface pointType) {
  // Extract the inner array (single point) from unified structure
  ArrayAttr pointCoords = cast<ArrayAttr>(coordsAttr[0]);

  // By kind, not by coordinate count -- see getNumCoords in PointKind.h.
  switch (pointType.getPointKind()) {
  case PointKind::kAffine:
    return pointOpFromArrayAttr<PointKind::kAffine>(pointCoords, pointType);
  case PointKind::kJacobian:
    return pointOpFromArrayAttr<PointKind::kJacobian>(pointCoords, pointType);
  case PointKind::kXYZZ:
    return pointOpFromArrayAttr<PointKind::kXYZZ>(pointCoords, pointType);
  case PointKind::kEdAffine:
    return pointOpFromArrayAttr<PointKind::kEdAffine>(pointCoords, pointType);
  case PointKind::kEdExtended:
    return pointOpFromArrayAttr<PointKind::kEdExtended>(pointCoords, pointType);
  }
  llvm_unreachable("Unsupported point kind");
}

// Convert a PointOperation back to ArrayAttr
ArrayAttr toArrayAttr(MLIRContext *ctx, const PointOperation &op) {
  return std::visit(
      [ctx](const auto &pointOp) -> ArrayAttr {
        return pointOpToArrayAttr(ctx, pointOp);
      },
      op.getOperation());
}

// Helper to convert PointOperations to ArrayAttr<ArrayAttr<Attr>> for tensor
// folding.
ArrayAttr pointOperationsToArrayAttr(MLIRContext *ctx,
                                     ArrayRef<PointOperation> values) {
  SmallVector<Attribute> pointAttrs;
  for (const auto &value : values) {
    ArrayAttr pointCoords = std::visit(
        [ctx](const auto &pointOp) -> ArrayAttr {
          return pointCoordsToArrayAttr(ctx, pointOp);
        },
        value.getOperation());
    pointAttrs.push_back(pointCoords);
  }
  return ArrayAttr::get(ctx, pointAttrs);
}

//===----------------------------------------------------------------------===//
// Config & Mixins following ConstantFolder.h pattern
//===----------------------------------------------------------------------===//

struct EllipticCurveConstantFolderConfig {
  using NativeInputType = PointOperation;
  using NativeOutputType = PointOperation;
  using ScalarAttr = ArrayAttr;
  using TensorAttr = ArrayAttr;
};

template <typename BaseDelegate>
class EllipticCurvePointFolderMixin : public BaseDelegate {
public:
  explicit EllipticCurvePointFolderMixin(PointTypeInterface inputPointType)
      : inputPointType(inputPointType), ctx(inputPointType.getContext()) {}

  PointOperation getNativeInput(ArrayAttr attr) const final {
    return createPointOp(attr, inputPointType);
  }

  OpFoldResult getScalarAttr(const PointOperation &value) const final {
    return toArrayAttr(ctx, value);
  }

  OpFoldResult getTensorAttr(ShapedType type,
                             ArrayRef<PointOperation> values) const final {
    return pointOperationsToArrayAttr(ctx, values);
  }

protected:
  PointTypeInterface inputPointType;
  MLIRContext *ctx; // not owned
};

//===----------------------------------------------------------------------===//
// Unary Folder - different input/output type (e.g., DoubleOp)
//===----------------------------------------------------------------------===//

template <typename Func>
class GenericUnaryEllipticCurveFolder
    : public EllipticCurvePointFolderMixin<
          UnaryConstantFolder<EllipticCurveConstantFolderConfig>::Delegate> {
public:
  GenericUnaryEllipticCurveFolder(PointTypeInterface inputPointType,
                                  PointKind outputKind, Func fn)
      : EllipticCurvePointFolderMixin(inputPointType), outputKind(outputKind),
        fn(fn) {}

  PointOperation operate(const PointOperation &value) const final {
    return fn(value, outputKind);
  }

  // Fold tensor of points (not a virtual override - called directly from
  // helper)
  OpFoldResult foldTensorPoints(ArrayAttr inputAttr,
                                ShapedType outputType) const {
    // Each element in inputAttr is a point's coordinates (ArrayAttr)
    SmallVector<PointOperation, 0> results;
    results.reserve(inputAttr.size());

    for (Attribute pointAttr : inputAttr) {
      // Wrap single point in outer ArrayAttr to match unified structure
      ArrayAttr wrappedPoint =
          ArrayAttr::get(this->ctx, {cast<ArrayAttr>(pointAttr)});
      PointOperation inputOp = this->getNativeInput(wrappedPoint);
      results.push_back(operate(inputOp));
    }

    return this->getTensorAttr(outputType, results);
  }

private:
  PointKind outputKind;
  Func fn;
};

//===----------------------------------------------------------------------===//
// Binary Folder - handles heterogeneous input types (e.g., AddOp, SubOp)
//===----------------------------------------------------------------------===//

template <typename Op, typename Func>
class GenericEllipticCurveBinaryFolder
    : public BinaryConstantFolder<EllipticCurveConstantFolderConfig>::Delegate {
public:
  GenericEllipticCurveBinaryFolder(Op *op, Func fn)
      : op(op), fn(fn), ctx(op->getContext()),
        lhsPointType(cast<PointTypeInterface>(
            getElementTypeOrSelf(op->getLhs().getType()))),
        rhsPointType(cast<PointTypeInterface>(
            getElementTypeOrSelf(op->getRhs().getType()))),
        outputKind(cast<PointTypeInterface>(getElementTypeOrSelf(op->getType()))
                       .getPointKind()) {}

  // This won't be called directly since we override foldScalar(lhs, rhs)
  PointOperation getNativeInput(ArrayAttr attr) const final {
    llvm_unreachable("getNativeInput should not be called directly");
  }

  OpFoldResult getScalarAttr(const PointOperation &value) const final {
    return toArrayAttr(ctx, value);
  }

  OpFoldResult getTensorAttr(ShapedType type,
                             ArrayRef<PointOperation> values) const final {
    return pointOperationsToArrayAttr(ctx, values);
  }

  PointOperation operate(const PointOperation &lhs,
                         const PointOperation &rhs) const final {
    return fn(lhs, rhs, outputKind);
  }

  // Bring base class overloads into scope
  using BinaryConstantFolder<
      EllipticCurveConstantFolderConfig>::Delegate::foldScalar;
  using BinaryConstantFolder<
      EllipticCurveConstantFolderConfig>::Delegate::foldTensor;

  // Override to handle heterogeneous input types
  OpFoldResult foldScalar(ArrayAttr lhsAttr, ArrayAttr rhsAttr) const {
    auto lhsOp = createPointOp(lhsAttr, lhsPointType);
    auto rhsOp = createPointOp(rhsAttr, rhsPointType);
    return getScalarAttr(operate(lhsOp, rhsOp));
  }

  // Fold tensor of points (not a virtual override - called directly from
  // helper)
  OpFoldResult foldTensorPoints(ArrayAttr lhsAttr, ArrayAttr rhsAttr,
                                ShapedType outputType) const {
    if (lhsAttr.size() != rhsAttr.size())
      return {};

    SmallVector<PointOperation, 0> results;
    results.reserve(lhsAttr.size());

    for (auto [lhsPoint, rhsPoint] : llvm::zip(lhsAttr, rhsAttr)) {
      // Wrap single point in outer ArrayAttr to match unified structure
      ArrayAttr wrappedLhs = ArrayAttr::get(ctx, {cast<ArrayAttr>(lhsPoint)});
      ArrayAttr wrappedRhs = ArrayAttr::get(ctx, {cast<ArrayAttr>(rhsPoint)});
      PointOperation lhsOp = createPointOp(wrappedLhs, lhsPointType);
      PointOperation rhsOp = createPointOp(wrappedRhs, rhsPointType);
      results.push_back(operate(lhsOp, rhsOp));
    }

    return getTensorAttr(outputType, results);
  }

private:
  Op *const op; // not owned
  Func fn;
  MLIRContext *ctx; // not owned
  PointTypeInterface lhsPointType;
  PointTypeInterface rhsPointType;
  PointKind outputKind;
};

//===----------------------------------------------------------------------===//
// Helper Functions
//===----------------------------------------------------------------------===//

template <typename Op, typename Func>
OpFoldResult foldUnaryPointOp(Op *op, typename Op::FoldAdaptor adaptor,
                              Func fn) {
  auto inputAttr = dyn_cast_if_present<ArrayAttr>(adaptor.getInput());
  if (!inputAttr)
    return {};

  Type inputType = op->getInput().getType();
  Type outputType = op->getType();

  // Use getElementTypeOrSelf to handle both scalar and tensor types
  auto inputPointType =
      dyn_cast<PointTypeInterface>(getElementTypeOrSelf(inputType));
  if (!inputPointType)
    return {};

  PointKind outputKind =
      cast<PointTypeInterface>(getElementTypeOrSelf(outputType)).getPointKind();

  GenericUnaryEllipticCurveFolder<Func> folder(inputPointType, outputKind, fn);

  // Check if this is a tensor operation
  if (auto shapedType = dyn_cast<ShapedType>(outputType)) {
    return folder.foldTensorPoints(inputAttr, shapedType);
  }
  return folder.foldScalar(inputAttr);
}

template <typename Op, typename Func>
OpFoldResult foldBinaryPointOp(Op *op, typename Op::FoldAdaptor adaptor,
                               Func fn) {
  auto lhsAttr = dyn_cast_if_present<ArrayAttr>(adaptor.getLhs());
  auto rhsAttr = dyn_cast_if_present<ArrayAttr>(adaptor.getRhs());
  if (!lhsAttr || !rhsAttr)
    return {};

  GenericEllipticCurveBinaryFolder<Op, Func> folder(op, fn);

  // Check if this is a tensor operation
  if (auto shapedType = dyn_cast<ShapedType>(op->getType())) {
    return folder.foldTensorPoints(lhsAttr, rhsAttr, shapedType);
  }
  return folder.foldScalar(lhsAttr, rhsAttr);
}

//===----------------------------------------------------------------------===//
// Helper functions for scalar constant matching (used by canonicalization)
//===----------------------------------------------------------------------===//

// Check if a scalar field constant attribute equals the given value.
bool isScalarEqualTo(Attribute attr, uint64_t value) {
  // TODO(chokobole): Implement scalar check once the Scalar Field
  // modulus is accessible via the Curve Attribute.
  return false;
}

// Check if a scalar field constant attribute equals the negation of the given
// value (i.e., modulus - value).
bool isScalarNegativeOf(Attribute attr, uint64_t value) {
  // TODO(chokobole): Implement scalar negation check once the Scalar Field
  // modulus is accessible via the Curve Attribute.
  return false;
}

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveCanonicalization.cpp.inc"
} // namespace

//===----------------------------------------------------------------------===//
// Constant folding
//===----------------------------------------------------------------------===//

OpFoldResult NegateOp::fold(FoldAdaptor adaptor) {
  return foldUnaryPointOp(
      this, adaptor, [](const PointOperation &op, PointKind) { return -op; });
}

OpFoldResult DoubleOp::fold(FoldAdaptor adaptor) {
  return foldUnaryPointOp(this, adaptor,
                          [](const PointOperation &op, PointKind outputKind) {
                            return op.dbl(outputKind);
                          });
}

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  return foldBinaryPointOp(
      this, adaptor,
      [](const PointOperation &lhs, const PointOperation &rhs,
         // NOLINTNEXTLINE(whitespace/newline)
         PointKind outputKind) { return lhs.add(rhs, outputKind); });
}

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  return foldBinaryPointOp(
      this, adaptor,
      [](const PointOperation &lhs, const PointOperation &rhs,
         // NOLINTNEXTLINE(whitespace/newline)
         PointKind outputKind) { return lhs.sub(rhs, outputKind); });
}

#include "prime_ir/Utils/CanonicalizationPatterns.inc"

void AddOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
#define PRIME_IR_ADD_PATTERN(Name) patterns.add<EllipticCurve##Name>(context);
  PRIME_IR_ADDITIVE_GROUP_ADD_PATTERN_LIST(PRIME_IR_ADD_PATTERN)
#undef PRIME_IR_ADD_PATTERN
}

void SubOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                        MLIRContext *context) {
#define PRIME_IR_SUB_PATTERN(Name) patterns.add<EllipticCurve##Name>(context);
  PRIME_IR_ADDITIVE_GROUP_SUB_PATTERN_LIST(PRIME_IR_SUB_PATTERN)
#undef PRIME_IR_SUB_PATTERN
}

void ScalarMulOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                              MLIRContext *context) {
#define PRIME_IR_SCALARMUL_PATTERN(Name)                                       \
  patterns.add<EllipticCurve##Name>(context);
  PRIME_IR_ADDITIVE_GROUP_SCALARMUL_PATTERN_LIST(PRIME_IR_SCALARMUL_PATTERN)
#undef PRIME_IR_SCALARMUL_PATTERN
}

} // namespace mlir::prime_ir::elliptic_curve
