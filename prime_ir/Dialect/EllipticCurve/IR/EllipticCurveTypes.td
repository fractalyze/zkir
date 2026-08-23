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

#ifndef PRIME_IR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVETYPES_TD_
#define PRIME_IR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVETYPES_TD_

include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveAttributes.td"
include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveDialect.td"

include "mlir/IR/DialectBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/ConstantLikeInterface.td"

// Base type definition for EllipticCurve types.
class EllipticCurve_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<EllipticCurve_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

// Point type interface.
// Provides a unified interface for all point types (affine, jacobian, xyzz, etc.)
def EllipticCurve_PointTypeInterface : TypeInterface<"PointTypeInterface"> {
  let description = [{
    Interface for all point types that provides common operations.
  }];

  let methods = [
    InterfaceMethod<"Get number of coordinates",
      "unsigned", "getNumCoords">,
    InterfaceMethod<"Get the base field type",
      "Type", "getBaseFieldType">,
    InterfaceMethod<"Get the curve attribute",
      "Attribute", "getCurveAttr">,
    InterfaceMethod<"Get the point kind",
      "PointKind", "getPointKind">,
  ];
}

def EllipticCurve_AffineType : EllipticCurve_Type<"Affine", "affine", [
  MemRefElementTypeInterface,
  DeclareTypeInterfaceMethods<EllipticCurve_PointTypeInterface>,
  DeclareTypeInterfaceMethods<ConstantLikeInterface>,
  DeclareTypeInterfaceMethods<DenseElementTypeInterface>
  ]> {
  let summary = "An affine point within an elliptic curve";
  let description = [{
    An affine point within an elliptic curve. It must be instantiated with an elliptic curve attribute
    such as `#elliptic_curve.sw` that defines the elliptic curve.

    Example:
    ```
    #bn254 = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !bn254_bf
    !affine = !elliptic_curve.affine<#bn254>
    ```
  }];
  let parameters = (ins EllipticCurve_ShortWeierstrassAttr:$curve);
  let assemblyFormat = "`<` $curve `>`";
}

def EllipticCurve_JacobianType : EllipticCurve_Type<"Jacobian", "jacobian", [
  MemRefElementTypeInterface,
  DeclareTypeInterfaceMethods<EllipticCurve_PointTypeInterface>,
  DeclareTypeInterfaceMethods<ConstantLikeInterface>,
  DeclareTypeInterfaceMethods<DenseElementTypeInterface>
  ]> {
  let summary = "A jacobian point within an elliptic curve";
  let description = [{
    A jacobian point within an elliptic curve. If the point in affine form is represented as (x,y),
    then the jacobian form is (X:Y:Z) where x = X/Z², y = Y/Z³. It must be instantiated with an elliptic
    curve attribute such as `#elliptic_curve.sw` that defines the elliptic curve.

    Example:
    ```
    #bn254 = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !bn254_bf
    !jacobian = !elliptic_curve.jacobian<#bn254>
    ```
  }];
  let parameters = (ins EllipticCurve_ShortWeierstrassAttr:$curve);
  let assemblyFormat = "`<` $curve `>`";
}

def EllipticCurve_XYZZType : EllipticCurve_Type<"XYZZ", "xyzz", [
  MemRefElementTypeInterface,
  DeclareTypeInterfaceMethods<EllipticCurve_PointTypeInterface>,
  DeclareTypeInterfaceMethods<ConstantLikeInterface>,
  DeclareTypeInterfaceMethods<DenseElementTypeInterface>
  ]> {
  let summary = "A xyzz point within an elliptic curve";
  let description = [{
    A xyzz point within an elliptic curve. The xyzz form is a different way of storing
    a jacobian point. While a jacobian point is stored as (X:Y:Z), the xyzz form is
    stored as (X:Y:Z²:Z³) to make computing x = X/Z², y = Y/Z³ easier. It must be instantiated
    with an elliptic curve attribute such as `#elliptic_curve.sw` that defines the elliptic curve.

    Example:
    ```
    #bn254 = #elliptic_curve.sw<0:i256, 3:i256, (1:i256, 2:i256)> : !bn254_bf
    !xyzz = !elliptic_curve.xyzz<#bn254>
    ```
  }];
  let parameters = (ins EllipticCurve_ShortWeierstrassAttr:$curve);
  let assemblyFormat = "`<` $curve `>`";
}

def EllipticCurve_EdAffineType : EllipticCurve_Type<"EdAffine", "ed_affine", [
  MemRefElementTypeInterface,
  DeclareTypeInterfaceMethods<EllipticCurve_PointTypeInterface>,
  DeclareTypeInterfaceMethods<ConstantLikeInterface>,
  DeclareTypeInterfaceMethods<DenseElementTypeInterface>
  ]> {
  let summary = "An affine point on a twisted Edwards curve";
  let description = [{
    An affine point (x, y) on a twisted Edwards curve a*x² + y² = 1 + d*x²*y².
    It must be instantiated with a twisted Edwards curve attribute.

    Example:
    ```
    #ed25519 = #elliptic_curve.te<-1:i256, d:i256, (Gx:i256, Gy:i256)> : !curve25519_bf
    !ed_affine = !elliptic_curve.ed_affine<#ed25519>
    ```
  }];
  let parameters = (ins EllipticCurve_TwistedEdwardsAttr:$curve);
  let assemblyFormat = "`<` $curve `>`";
}

def EllipticCurve_EdExtendedType : EllipticCurve_Type<"EdExtended", "ed_extended", [
  MemRefElementTypeInterface,
  DeclareTypeInterfaceMethods<EllipticCurve_PointTypeInterface>,
  DeclareTypeInterfaceMethods<ConstantLikeInterface>,
  DeclareTypeInterfaceMethods<DenseElementTypeInterface>
  ]> {
  let summary = "An extended point on a twisted Edwards curve";
  let description = [{
    An extended point on a twisted Edwards curve in the form (X:Y:Z:T) where
    x = X/Z, y = Y/Z, T = X*Y/Z (Hisil-Wong-Carter-Dawson 2008). It must be
    instantiated with a twisted Edwards curve attribute `#elliptic_curve.te`.

    Example:
    ```
    #ed25519 = #elliptic_curve.te<-1:i256, d:i256, (Gx:i256, Gy:i256)> : !curve25519_bf
    !extended = !elliptic_curve.ed_extended<#ed25519>
    ```
  }];
  let parameters = (ins EllipticCurve_TwistedEdwardsAttr:$curve);
  let assemblyFormat = "`<` $curve `>`";
}

def PointType : AnyTypeOf<[
    EllipticCurve_AffineType, EllipticCurve_JacobianType,
    EllipticCurve_XYZZType, EllipticCurve_EdAffineType,
    EllipticCurve_EdExtendedType]>;

def AffineLike : TypeOrValueSemanticsContainer<
    EllipticCurve_AffineType, "affine-like">;

def JacobianLike : TypeOrValueSemanticsContainer<
    EllipticCurve_JacobianType, "jacobian-like">;

def XYZZLike : TypeOrValueSemanticsContainer<
    EllipticCurve_XYZZType, "xyzz-like">;

def EdAffineLike : TypeOrValueSemanticsContainer<
    EllipticCurve_EdAffineType, "ed-affine-like">;

def EdExtendedLike : TypeOrValueSemanticsContainer<
    EllipticCurve_EdExtendedType, "ed-extended-like">;

def PointLike : TypeConstraint<Or<[
    AffineLike.predicate, JacobianLike.predicate, XYZZLike.predicate,
    EdAffineLike.predicate, EdExtendedLike.predicate]>,
    "affine-like or jacobian-like or xyzz-like or ed-affine-like or ed-extended-like">;
#endif  // PRIME_IR_DIALECT_ELLIPTICCURVE_IR_ELLIPTICCURVETYPES_TD_
