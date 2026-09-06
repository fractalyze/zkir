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

#ifndef PRIME_IR_DIALECT_FIELD_IR_FIELDOPS_TD_
#define PRIME_IR_DIALECT_FIELD_IR_FIELDOPS_TD_

include "prime_ir/Dialect/Field/IR/FieldAttributes.td"
include "prime_ir/Dialect/Field/IR/FieldDialect.td"
include "prime_ir/Dialect/Field/IR/FieldTypes.td"
include "prime_ir/Dialect/ModArith/IR/ModArithAttributes.td"
include "prime_ir/Dialect/ModArith/IR/ModArithTypes.td"
include "prime_ir/Utils/CommonTraits.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Base class for all field operations.
class Field_Op_Base<string mnemonic, list<Trait> traits = []> :
  Op<Field_Dialect, mnemonic, traits # [Pure]>;

// Base class for elementwise mappable field operations.
class Field_Op<string mnemonic, list<Trait> traits = []> :
  Field_Op_Base<mnemonic, traits # ElementwiseMappable.traits>;

// Base class for field arithmetic ops. All ops have one result.
class Field_ArithOp<string mnemonic, list<Trait> traits = []> :
  Field_Op<mnemonic, traits>;

// Base class for unary field arithmetic. Operand and result have the same type.
class Field_UnaryOp<string mnemonic, list<Trait> traits = []> :
  Field_ArithOp<mnemonic, traits # [SameOperandsAndResultType]>,
  Arguments<(ins FieldLike:$input)>,
  Results<(outs FieldLike:$output)> {
  let assemblyFormat = "operands attr-dict `:` type($output)";
}

// Base class for binary field arithmetic. Supports both same-type operands
// (e.g., pf + pf) and mixed-type operands (e.g., ef + pf) where one operand
// is an extension field and the other is its base field. The result type is
// inferred via InferTypeOpInterface.
class Field_BinaryOp<string mnemonic, list<Trait> traits = []> :
  Field_ArithOp<mnemonic,
    traits # [DeclareOpInterfaceMethods<InferTypeOpInterface>]>,
  Arguments<(ins FieldLike:$lhs, FieldLike:$rhs)>,
  Results<(outs FieldLike:$output)> {
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def Field_ConstantOp : Field_Op_Base<"constant", [ConstantLike]> {
  let summary = "Define a constant prime field element via an attribute";
  let description = [{
    Example:
    ```
    %0 = field.constant 123 : !field.pf<7>
    ```
  }];
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs FieldLike:$output);
  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let builders = [
    OpBuilder<(ins "Type":$ty, "uint64_t":$value), [{
      if (auto pfType = dyn_cast<PrimeFieldType>(ty)) {
        return build($_builder, $_state, pfType, value);
      } else if (auto efType = dyn_cast<ExtensionFieldType>(ty)) {
        return build($_builder, $_state, efType, value);
      } else {
        llvm_unreachable("Unsupported field type for constant");
      }
    }]>,
    OpBuilder<(ins "PrimeFieldType":$ty, "llvm::APInt":$value), [{
      return build($_builder, $_state, ty, IntegerAttr::get(ty.getStorageType(), value));
    }]>,
    OpBuilder<(ins "PrimeFieldType":$ty, "uint64_t":$value), [{
      return build($_builder, $_state, ty, IntegerAttr::get(ty.getStorageType(), value));
    }]>,
    OpBuilder<(ins "ExtensionFieldType":$ty, "uint64_t":$c0), [{
      auto primeField = ty.getBasePrimeField();
      unsigned degreeOverPrime = ty.getDegreeOverPrime();
      unsigned bitWidth = primeField.getTypeSizeInBits();
      SmallVector<APInt> coeffs(degreeOverPrime, APInt(bitWidth, 0));
      coeffs[0] = static_cast<APInt>(PrimeFieldOperation::fromUnchecked(APInt(bitWidth, c0), primeField));
      auto tensorType = RankedTensorType::get(ty.getAttrShape(), primeField.getStorageType());
      return build($_builder, $_state, ty, DenseIntElementsAttr::get(tensorType, coeffs));
    }]>,
    OpBuilder<(ins "ExtensionFieldType":$ty, "llvm::ArrayRef<uint64_t>":$coeffs), [{
      assert(coeffs.size() == ty.getDegreeOverPrime() && "coefficient count must match total degree over prime");
      auto primeField = ty.getBasePrimeField();
      unsigned bitWidth = primeField.getTypeSizeInBits();
      SmallVector<APInt> coeffInts;
      for (auto c : coeffs)
        coeffInts.push_back(static_cast<APInt>(PrimeFieldOperation::fromUnchecked(APInt(bitWidth, c), primeField)));
      auto tensorType = RankedTensorType::get(ty.getAttrShape(), primeField.getStorageType());
      return build($_builder, $_state, ty, DenseIntElementsAttr::get(tensorType, coeffInts));
    }]>,
  ];
  let extraClassDeclaration = [{
    /// Build the constant op with `value` and `type` if possible, otherwise
    /// returns null.
    static ConstantOp materialize(OpBuilder &builder, Attribute value,
                                  Type type, Location loc);
  }];
}

def Field_BitcastOp : Field_Op_Base<"bitcast", [DeclareOpInterfaceMethods<CastOpInterface>]> {
  let summary = "Bitcast between field types and integer types";
  let description = [{
    Bitcasts between field types (prime or extension) and integer types.

    There are two modes of operation:

    1. **Scalar/element-wise bitcast**: Between prime field types and integer types
       with the same storage bitwidth. The shapes must match.

    2. **Tensor reinterpret bitcast**: Between extension field tensors and prime
       field tensors where the total storage matches. For an extension field of
       degree N over a prime field PF:
       - `tensor<K x !EF>` can be cast to `tensor<K*N x !PF>` and vice versa
       - The underlying memory layout is reinterpreted without data movement

    Integer to integer bitcast is disallowed as it should be handled by
    `arith.bitcast`.

    Example:
    ```
    !PF = !field.pf<7 : i32>
    !EF3 = !field.ef<3x!PF, 2:i32>

    // Scalar bitcasts (same shape)
    %c0 = field.bitcast %f0 : !PF -> i32
    %f1 = field.bitcast %c0 : i32 -> !PF

    // Tensor reinterpret bitcasts (extension field <-> prime field)
    %pf_tensor = field.bitcast %ef_tensor : tensor<2x!EF3> -> tensor<6x!PF>
    %ef_tensor = field.bitcast %pf_tensor : tensor<6x!PF> -> tensor<2x!EF3>
    ```
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let assemblyFormat = "operands attr-dict `:` type($input) `->` type($output)";
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def Field_CmpOp : Field_Op<"cmp", [SameTypeOperands, TypesMatchWith<
  "result type has i1 element type and same shape as operands",
  "lhs", "result", "getI1SameShape($_self)">]> {
  let summary = "Prime field comparison operation";
  let description = [{
    **From description of `Arith_CmpIOp`**
    Its first argument is an attribute that defines which type of comparison is
    performed. The following comparisons are supported:

    -   equal (mnemonic: `"eq"`; integer value: `0`)
    -   not equal (mnemonic: `"ne"`; integer value: `1`)
    -   unsigned less than (mnemonic: `"ult"`; integer value: `6`)
    -   unsigned less than or equal (mnemonic: `"ule"`; integer value: `7`)
    -   unsigned greater than (mnemonic: `"ugt"`; integer value: `8`)
    -   unsigned greater than or equal (mnemonic: `"uge"`; integer value: `9`)

    The result is `1` if the comparison is true and `0` otherwise. For vector or
    tensor operands, the comparison is performed elementwise and the element of
    the result indicates whether the comparison is true for the operand elements
    with the same indices as those of the result.

    Example:
    ```
    %res = field.cmp slt, %a, %b : !PF
    ```
  }];
  let arguments = (ins Arith_CmpIPredicateAttr:$predicate,
                       FieldLike:$lhs,
                       FieldLike:$rhs);
  let results = (outs BoolLike:$result);
  let hasVerifier = 1;
  let hasFolder = 1;
  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` type($lhs)";
}

def Field_ToMontOp : Field_Op<"to_mont", [TypesMatchWith<
  "input type is standard form of output type", "output", "input",
  "getStandardFormType($_self)">]> {
  let summary = "Convert from standard form to montgomery representation";
  let description = [{
    Computes x * R, where `R` is the montgomery factor.

    Example:
    ```
    %m0 = field.to_mont %c0 : !PF
    ```
  }];
  let arguments = (ins FieldLike:$input);
  let results = (outs FieldLike:$output);
  let hasVerifier = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let assemblyFormat = "$input attr-dict `:` type($output)";
}

def Field_FromMontOp : Field_Op<"from_mont", [TypesMatchWith<
  "input type is montgomery form of output type", "output", "input",
  "getMontgomeryFormType($_self)">]> {
  let summary = "Convert from montgomery representation to standard form";
  let description = [{
    Computes x * R⁻¹, where `R` is the montgomery factor.

    Example:
    ```
    %m0 = field.from_mont %c0 : !PF
    ```
  }];
  let arguments = (ins FieldLike:$input);
  let results = (outs FieldLike:$output);
  let hasVerifier = 1;
  let hasFolder = 1;
  let assemblyFormat = "$input attr-dict `:` type($output)";
}

// Field multiplicative inverse
def Field_InverseOp : Field_Op_Base<"inverse", [SameOperandsAndResultType, Involution]> {
  let summary = "Field multiplicative inverse";
  let description = [{
    Computes the multiplicative inverse of a field element.

    Example:
    ```
    %inv_a = field.inverse %a : !field.pf<primeModulus>
    ```
  }];
  let arguments = (ins FieldLike:$input);
  let results = (outs FieldLike:$output);
  let assemblyFormat = "operands attr-dict `:` type($output)";
  let hasFolder = 1;
}

// Field double.
def Field_DoubleOp : Field_UnaryOp<"double"> {
  let summary = "Field double";
  let description = [{
    Computes the double of a field element.

    Example:
    ```
    %double = field.double %a : !field.pf<primeModulus>
    ```
  }];
  let hasFolder = 1;
}

// Field square.
def Field_SquareOp : Field_UnaryOp<"square"> {
  let summary = "Field square";
  let description = [{
    Computes the square of a field element.

    Example:
    ```
    %square = field.square %a : !field.pf<primeModulus>
    ```
  }];
  let hasFolder = 1;
}

// Field negation
def Field_NegateOp : Field_UnaryOp<"negate", [Involution]> {
  let summary = "Field negation";
  let description = [{
    Computes the negation of a field element.

    Example:
    ```
    %negation = field.negate %a : !field.pf<primeModulus>
    ```
  }];
  let hasFolder = 1;
}

// Field addition.
def Field_AddOp : Field_BinaryOp<"add", [Commutative]> {
  let summary = "Field addition";
  let description = [{
    Computes the sum of two field elements. Supports mixed-type operands
    where one operand is an extension field and the other is its base field.

    Example:
    ```
    %r = field.add %a, %b : !field.pf<primeModulus>
    %r2 = field.add %ext, %base : !field.ef<2x!field.pf<7>, 6>, !field.pf<7>
    ```
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

// Field subtraction.
def Field_SubOp : Field_BinaryOp<"sub"> {
  let summary = "Field subtraction";
  let description = [{
    Computes the difference between two field elements. Supports mixed-type
    operands where one operand is an extension field and the other is its
    base field.

    Example:
    ```
    %r = field.sub %a, %b : !field.pf<primeModulus>
    %r2 = field.sub %ext, %base : !field.ef<2x!field.pf<7>, 6>, !field.pf<7>
    ```
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

// Field multiplication.
def Field_MulOp : Field_BinaryOp<"mul", [Commutative]> {
  let summary = "Field multiplication";
  let description = [{
    Computes the product of two field elements. Supports mixed-type operands
    where one operand is an extension field and the other is its base field
    (scalar multiplication).

    Example:
    ```
    %r = field.mul %a, %b : !field.pf<primeModulus>
    %r2 = field.mul %ext, %base : !field.ef<2x!field.pf<7>, 6>, !field.pf<7>
    ```
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

// Field power.
def Field_PowUIOp : Field_Op<"powui", [TypesMatchWith<
  "base and output must have same type", "base", "output", "$_self">]> {
  let summary = "Field power with unsigned integer exponent";
  let description = [{
    Computes the field element a raised to the power of unsigned integer b.

    The optional `unroll` attribute selects the shape of the lowered
    square-and-multiply schedule when the exponent is a compile-time constant.
    Absent or `true` emits it straight-line; `false` keeps the runtime
    bit-serial loop. Straight-line is the cheaper form for a caller that is
    already unrolled around this op, while a caller that rolls its own rounds
    (a Poseidon permutation applying one S-box per round) executes fewer
    instructions with the loop, and a straight-line chain per S-box inflates
    what LLVM has to simplify. A dynamic exponent always lowers to the loop.

    Example:
    ```
    %power = field.powui %a, %b : !field.pf<primeModulus>, i32
    %rolled = field.powui %a, %b {unroll = false} : !field.pf<primeModulus>, i32
    ```
  }];
  let arguments = (ins FieldLike:$base, SignlessIntegerLike:$exp,
                       OptionalAttr<BoolAttr>:$unroll);
  let results = (outs FieldLike:$output);
  let assemblyFormat = "operands attr-dict `:` type($base) `,` type($exp)";
}

def Field_ExtToCoeffsOp : Field_Op_Base<"ext_to_coeffs"> {
  let summary = "Get the coefficients of an extension field element";
  let description = [{
    Gets the coefficient representation of an extension field element.
    This operation does NOT accept tensors since it cannot be applied elementwise.
    For tower extensions, the coefficients are extension field elements themselves.

    Example:
    ```
    !EF2 = field.ef<2x!PF, #beta>
    %c0, %c1 = field.ext_to_coeffs %ef2 : !EF2 -> !PF, !PF

    !EF3 = field.ef<3x!PF, #xi>
    %c0, %c1, %c2 = field.ext_to_coeffs %ef3 : !EF3 -> !PF, !PF, !PF

    // Tower extension: Fp6 = (Fp2)^3
    !EF6 = field.ef<3x!EF2, #xi>
    %c0, %c1, %c2 = field.ext_to_coeffs %ef6 : !EF6 -> !EF2, !EF2, !EF2
    ```
  }];
  let arguments = (ins Field_ExtensionFieldType:$input);
  let results = (outs Variadic<AnyTypeOf<[Field_PrimeFieldType, Field_ExtensionFieldType, ModArith_ModArithType, AnySignlessInteger]>>:$output);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasCanonicalizer = 1;
  let hasFolder = 1;
  let hasVerifier = 1;
}

def Field_ExtFromCoeffsOp : Field_Op_Base<"ext_from_coeffs"> {
  let summary = "Constructs extension field element from the coefficients";
  let description = [{
    Constructs extension field element from the coefficients.
    This operation does NOT accept tensors since it cannot be applied elementwise.
    For tower extensions, the coefficients are extension field elements themselves.

    Example:
    ```
    !EF2 = field.ef<2x!PF, #beta>
    %ef2 = field.ext_from_coeffs %c0, %c1 : !PF, !PF -> !EF2

    !EF3 = field.ef<3x!PF, #xi>
    %ef3 = field.ext_from_coeffs %c0, %c1, %c2 : !PF, !PF, !PF -> !EF3

    // Tower extension: Fp6 = (Fp2)^3
    !EF6 = field.ef<3x!EF2, #xi>
    %ef6 = field.ext_from_coeffs %c0, %c1, %c2 : !EF2, !EF2, !EF2 -> !EF6
    ```
  }];
  let arguments = (ins Variadic<AnyTypeOf<[Field_PrimeFieldType, Field_ExtensionFieldType, ModArith_ModArithType, AnySignlessInteger]>>:$input);
  let results = (outs Field_ExtensionFieldType:$output);
  let assemblyFormat = "operands attr-dict `:` functional-type(operands, results)";
  let hasCanonicalizer = 1;
  let hasVerifier = 1;
}

#endif  // PRIME_IR_DIALECT_FIELD_IR_FIELDOPS_TD_
