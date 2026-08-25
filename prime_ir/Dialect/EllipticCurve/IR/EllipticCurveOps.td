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

#ifndef PRIME_IR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEOPS_TD_
#define PRIME_IR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEOPS_TD_

include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveDialect.td"
include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.td"

include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/LLVMIR/LLVMTypes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "prime_ir/Dialect/Field/IR/FieldAttributes.td"
include "prime_ir/Dialect/Field/IR/FieldTypes.td"
include "prime_ir/Dialect/ModArith/IR/ModArithTypes.td"
include "prime_ir/Utils/CommonTraits.td"
include "prime_ir/Utils/CommonTypeConstraints.td"

// Base class for all elliptic curve operations.
class EllipticCurve_Op_Base<string mnemonic, list<Trait> traits = []> :
  Op<EllipticCurve_Dialect, mnemonic, traits # [Pure]> {
  let cppNamespace = "::mlir::prime_ir::elliptic_curve";
}

// Base class for elementwise mappable elliptic curve operations.
class EllipticCurve_Op<string mnemonic, list<Trait> traits = []> :
  EllipticCurve_Op_Base<mnemonic, traits # ElementwiseMappable.traits>;

class EllipticCurve_UnaryOp<string mnemonic, list<Trait> traits = []> :
  EllipticCurve_Op<mnemonic, traits> {
  let arguments = (ins PointLike:$input);
  let results = (outs PointLike:$output);
}

class EllipticCurve_BinaryOp<string mnemonic, list<Trait> traits = []> :
  EllipticCurve_Op<mnemonic, traits>,
  Arguments<(ins PointLike:$lhs, PointLike:$rhs)>,
  Results<(outs PointLike:$output)> {
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)";
}

////////////// POINT INITIALIZATIONS //////////////

def EllipticCurve_ConstantOp : EllipticCurve_Op_Base<"constant", [ConstantLike]> {
  let summary = "Define a constant elliptic curve point via coordinates";
  let description = [{
    Defines a constant elliptic curve point from integer coordinates using
    dense attribute syntax. The number of provided coordinates must match
    the point representation:
    - 2 coordinates for Affine and EdAffine points
    - 3 coordinates for Jacobian points
    - 4 coordinates for XYZZ and EdExtended points

    The count does not identify the representation -- two kinds share each of
    2 and 4 -- so the type is what selects the group law, never the count.

    For extension fields, each coordinate is an array of coefficients.
    For tensors, provide an array of points.

    Example:
    ```mlir
    // Scalar points (prime field)
    %0 = elliptic_curve.constant dense<[1, 2]> : !elliptic_curve.affine<#curve>
    %1 = elliptic_curve.constant dense<[1, 2, 1]> : !elliptic_curve.jacobian<#curve>
    %2 = elliptic_curve.constant dense<[1, 2, 1, 1]> : !elliptic_curve.xyzz<#curve>

    // Tensor of points
    %3 = elliptic_curve.constant dense<[[1, 2], [3, 4]]> : tensor<2x!elliptic_curve.affine<#curve>>

    // Extension field (each coordinate is [c0, c1] for degree-2 extension)
    %4 = elliptic_curve.constant dense<[[1, 2], [3, 4]]> : !elliptic_curve.affine<#g2curve>

    // Tensor of extension field points
    %5 = elliptic_curve.constant dense<[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]> : tensor<2x!elliptic_curve.affine<#g2curve>>
    ```
  }];
  let arguments = (ins ArrayAttr:$coords);
  let results = (outs PointLike:$output);
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let extraClassDeclaration = [{
    /// Build the constant op with `value` and `type` if possible, otherwise
    /// returns null.
    static ConstantOp materialize(OpBuilder &builder, Attribute value,
                                  Type type, Location loc);
  }];
  // TODO(chokobole): The current verification DevEx is suboptimal due to
  // Montgomery domain mismatches. If the elliptic curve is defined over a
  // Montgomery base field, but the constant coordinates are provided in the
  // standard (non-Montgomery) domain, the verifier will fail to validate the
  // curve equation.
  //
  // We should enable the verifier once the field constant parser supports
  // automatic domain conversion to ensure consistency between input literals
  // and the target field representation.

}

def EllipticCurve_FromCoordsOp : EllipticCurve_Op_Base<"from_coords"> {
  let summary = "Constructs an elliptic curve point from coordinates";
  let description = [{
    Constructs an elliptic curve point from coordinates in affine(x, y), jacobian(x, y, z), or xyzz(x, y, z², z³) form.

    Example:
    ```
    %0 = elliptic_curve.from_coords %x, %y : elliptic_curve.affine<#curve>
    %1 = elliptic_curve.from_coords %x, %y, %z : elliptic_curve.jacobian<#curve>
    %2 = elliptic_curve.from_coords %x, %y, %zz, %zzz : elliptic_curve.xyzz<#curve>
    ```
  }];
  let arguments = (ins Variadic<AnyTypeOf<[Field_AnyFieldType, ModArith_ModArithType, LLVMStructType, AnySignlessInteger]>>:$coords);
  let results = (outs PointLike:$output);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

////////////// POINT CHECKS //////////////

def EllipticCurve_IsZeroOp : EllipticCurve_Op<"is_zero", [TypesMatchWith<
  "result type has i1 element type and same shape as operands",
  "input", "output", "getI1SameShape($_self)">]> {
  let summary = "Checks whether an elliptic curve point is zero";
  let description = [{
    Outputs a bool true (1) if the elliptic curve point input is zero, and a bool false (0) otherwise.

    Example:
    ```
    %0 = elliptic_curve.is_zero %point : !jacobian
    ```
  }];
  let arguments = (ins PointLike:$input);
  let results = (outs BoolLike:$output);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($input)";
}

/////////// POINT EXTRACTION //////////////

def EllipticCurve_ToCoordsOp : EllipticCurve_Op_Base<"to_coords"> {
  let summary = "Extracts coordinates of a point";
  let description = [{
    Extracts coordinates of an elliptic curve point as integers.

    Example:
    ```
    %coords = elliptic_curve.to_coords %point : elliptic_curve.affine<#curve> -> !PF
    ```
  }];

  let arguments = (ins PointLike:$input);
  let results = (outs Variadic<AnyTypeOf<[Field_AnyFieldType, ModArith_ModArithType, LLVMStructType, AnySignlessInteger]>>:$output);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

////////////// POINT ARITHMETIC OPERATIONS //////////////

// Elliptic curve point addition.
def EllipticCurve_AddOp : EllipticCurve_BinaryOp<"add", [Commutative]> {
  let summary = "Elliptic curve point addition operation";
  let description = [{
    Computes the sum of two elliptic curve points.

    affine, affine -> jacobian
    affine, affine -> xyzz
    affine, jacobian -> jacobian
    affine, xyzz -> xyzz
    jacobian, jacobian -> jacobian
    xyzz, xyzz -> xyzz
    ed_affine, ed_affine -> ed_extended
    ed_affine, ed_extended -> ed_extended
    ed_extended, ed_extended -> ed_extended

    Both operands and the result must belong to the same curve family.

    Example:
    ```
    %sum = elliptic_curve.add %a, %b : !affine, !jacobian -> !jacobian
    ```
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

// Elliptic curve point subtraction.
def EllipticCurve_SubOp : EllipticCurve_BinaryOp<"sub"> {
  let summary = "Elliptic curve point subtraction operation";
  let description = [{
    Computes the difference between two elliptic curve points.

    affine, affine -> jacobian
    affine, affine -> xyzz
    affine, jacobian -> jacobian
    affine, xyzz -> xyzz
    jacobian, jacobian -> jacobian
    xyzz, xyzz -> xyzz
    ed_affine, ed_affine -> ed_extended
    ed_affine, ed_extended -> ed_extended
    ed_extended, ed_extended -> ed_extended

    Both operands and the result must belong to the same curve family.

    Example:
    ```
    %difference = elliptic_curve.sub %a, %b : !affine, !jacobian -> !jacobian
    ```
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

// Elliptic curve point negation.
def EllipticCurve_NegateOp : EllipticCurve_UnaryOp<"negate", [Involution, SameOperandsAndResultType]> {
  let summary = "Elliptic curve point negation operation";
  let description = [{
    Computes the negation of an elliptic curve point.

    affine -> affine
    jacobian -> jacobian
    xyzz -> xyzz

    Example:
    ```
    %negation = elliptic_curve.negate %a : !jacobian
    ```
  }];
  let hasFolder = 1;
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

// Elliptic curve point doubling.
def EllipticCurve_DoubleOp : EllipticCurve_UnaryOp<"double"> {
  let summary = "Elliptic curve point doubling operation";
  let description = [{
    Computes the double of an elliptic curve point. If an affine point is doubled,
    it becomes jacobian. If a jacobian or xyzz point is doubled, its type remains the same.

    affine -> jacobian
    affine -> xyzz
    jacobian -> jacobian
    xyzz -> xyzz
    ed_affine -> ed_extended
    ed_extended -> ed_extended

    Example:
    ```
    %doubled = elliptic_curve.double %a : !affine -> !jacobian
    ```
  }];
  let hasVerifier = 1;
  let hasFolder = 1;
  let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";
}

////////////// MSM FUNCTIONS //////////////

// Elliptic curve point scalar multiplication.
def EllipticCurve_ScalarMulOp : EllipticCurve_Op<"scalar_mul"> {
  let summary = "Elliptic curve point scalar multiplication";
  let description = [{
    Computes the product of an elliptic curve point and a scalar prime field value.

    affine -> jacobian
    affine -> xyzz
    jacobian -> jacobian
    xyzz -> xyzz
    ed_affine -> ed_extended
    ed_extended -> ed_extended

    Example:
    ```
    %product = elliptic_curve.scalar_mul %f, %affine1 : !PF, !affine -> !jacobian
    ```
  }];
  let arguments = (ins PrimeFieldLike:$scalar, PointLike:$point);
  let results = (outs PointLike:$output);
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
  let assemblyFormat = "operands attr-dict `:` type($scalar) `,` type($point) `->` type($output)";
}

// Elliptic curve point comparison.
def EllipticCurve_CmpOp : EllipticCurve_Op<"cmp", [SameTypeOperands,
  TypesMatchWith<"result type has i1 element type and same shape as operands", "lhs", "result", "getI1SameShape($_self)">]> {
  let summary = "Elliptic curve point comparison operation";
  let description = [{
    **From description of `Arith_CmpIOp`**
    Its first argument is an attribute that defines which type of comparison is
    performed. The following comparisons are supported:

    -   equal (mnemonic: `"eq"`; integer value: `0`)
    -   not equal (mnemonic: `"ne"`; integer value: `1`)

    The result is `1` if the comparison is true and `0` otherwise. For vector or
    tensor operands, the comparison is performed elementwise and the element of
    the result indicates whether the comparison is true for the operand elements
    with the same indices as those of the result.

    Example:
    ```
    !affine = elliptic_curve.affine<#sw>

    %equal = elliptic_curve.cmp eq, %a, %b : !affine
    ```
  }];
  let arguments = (ins Arith_CmpIPredicateAttr:$predicate,
                       PointLike:$lhs,
                       PointLike:$rhs);
  let results = (outs BoolLike:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
}

// Elliptic curve point scalar multiplication.
def EllipticCurve_MSMOp : EllipticCurve_Op_Base<"msm"> {
  let summary = "Elliptic curve multi-scalar multiplication";
  let description = [{
    Computes multi-scalar multiplication (MSM) of elliptic curve points and scalars.

    Output type determines internal computation type of buckets and window sums.
    xyzz is recommended for best performance.

    If both scalars and points have a static shape, the size of scalars and points must be equal.
    Otherwise, both must have a dynamic shape.

    Additionally, 2^{degree} must be greater than or equal to the size of scalars
    if both scalars and points have a static shape.

    window_bits is the number of bits to use for the window. If window_bits is 0, window_bits is computed from the degree.

    When parallel is enabled, window-based parallelization is applied. If term-based parallelization is needed,
    parallel should be disabled and the caller should implement it.

    affine -> affine, jacobian, xyzz
    jacobian -> jacobian, xyzz
    xyzz -> xyzz

    Example:

    If you want to run window-based parallelization, use the parallel attribute as follows:

    ```
    %msm_result = elliptic_curve.msm %scalars, %points degree=2 parallel : tensor<3x!PF>, tensor<3x!affine> -> !xyzz
    ```

    If you want to run serial execution, simply omit the parallel attribute:

    ```
    %msm_result = elliptic_curve.msm %scalars, %points degree=2 : tensor<3x!PF>, tensor<3x!affine> -> !xyzz
    ```

    If you want to configure the MSM with a specific window bits, use the window_bits attribute as follows:

    ```
    %msm_result = elliptic_curve.msm %scalars, %points degree=2 window_bits=4 : tensor<3x!PF>, tensor<3x!affine> -> !xyzz
    ```
  }];
  let arguments = (ins
    RankedTensorOf<[Field_PrimeFieldType]>:$scalars,
    RankedTensorOf<[PointType]>:$points,
    I32Attr:$degree,
    DefaultValuedAttr<I32Attr, "0">:$windowBits,
    UnitAttr:$parallel
  );
  let results = (outs PointType:$output);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `degree` `=` $degree (`window_bits` `=` $windowBits^)? (`parallel` $parallel^)? `:` type($scalars) `,` type($points) `->` type($output)";
}

////////////// PAIRING //////////////

def EllipticCurve_PairingCheckOp : EllipticCurve_Op_Base<"pairing_check"> {
  let summary = "Multi-pairing check: verifies Prod_i e(Pi, Qi) = 1";
  let description = [{
    Verifies that the product of pairings equals the identity in GT:
      e(P1, Q1) * e(P2, Q2) * ... * e(Pn, Qn) = 1

    Both inputs must be 1D tensors of affine points on compatible
    pairing-friendly curves (e.g., BN254 G1 and G2). The G1 points must
    be over a prime base field and the G2 points over a degree-2 extension
    field. Both tensors must have matching first dimension.

    Returns a boolean (i1) indicating whether the pairing equation holds.

    Example:
    ```mlir
    %result = elliptic_curve.pairing_check %g1_points, %g2_points
        : tensor<2x!g1_affine>, tensor<2x!g2_affine> -> i1
    ```
  }];
  let arguments = (ins
    1DTensorOf<[EllipticCurve_AffineType]>:$g1Points,
    1DTensorOf<[EllipticCurve_AffineType]>:$g2Points
  );
  let results = (outs I1:$output);
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($g1Points) `,` type($g2Points) `->` type($output)";
}

////////////// POINT CONVERSIONS //////////////

// Non-elementwise: tensor convert_point_type is lowered directly
// (with batch field.inverse) rather than element-by-element via linalg.map.
def EllipticCurve_ConvertPointTypeOp : EllipticCurve_Op_Base<"convert_point_type"> {
  let arguments = (ins PointLike:$input);
  let results = (outs PointLike:$output);
  let summary = "Convert a point from one form to another.";
  let description = [{
    Converts a point on an elliptic curve from one form to another.
    Possible forms are affine, jacobian, and xyzz.

    Example:
    ```
    !affine = elliptic_curve.affine<#sw>
    !jacobian = elliptic_curve.jacobian<#sw>

    %converted = elliptic_curve.convert_point_type %0 : !affine -> !jacobian
    ```
  }];
  let hasVerifier = 1;
  let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";
}

def EllipticCurve_BitcastOp : EllipticCurve_Op_Base<"bitcast", [
    DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "Bitcast between field tensor and elliptic curve point tensor";
  let description = [{
    Reinterprets a flat tensor of field elements as a tensor of elliptic curve
    points, or vice versa. The underlying memory layout is reinterpreted without
    data movement.

    For a point type with K coordinates over a field type F:
    - `tensor<N*K x !F>` can be cast to `tensor<N x !PointType>` and vice versa
    - Coordinates are stored contiguously: [x0, y0, z0, x1, y1, z1, ...]

    Example:
    ```
    !PF = !field.pf<p:i256>
    #curve = #elliptic_curve.sw<...> : !PF
    !jacobian = !elliptic_curve.jacobian<#curve>

    // Field tensor to point tensor (3 coords per jacobian point)
    %points = elliptic_curve.bitcast %fields : tensor<6x!PF> -> tensor<2x!jacobian>

    // Point tensor to field tensor
    %fields = elliptic_curve.bitcast %points : tensor<2x!jacobian> -> tensor<6x!PF>
    ```
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";
  let hasVerifier = 1;
}

#endif  // PRIME_IR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVEOPS_TD_
