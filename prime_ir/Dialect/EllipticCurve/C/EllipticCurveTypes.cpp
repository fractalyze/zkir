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

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "prime_ir/Dialect/EllipticCurve/C/EllipticCurveTypes.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveAttributes.h"

using namespace mlir;
using namespace mlir::prime_ir::elliptic_curve;

//===----------------------------------------------------------------------===//
// Affine point types.
//===----------------------------------------------------------------------===//

MlirTypeID primeIRAffineTypeGetTypeID() {
  return wrap(AffineType::getTypeID());
}

bool primeIRTypeIsAnAffine(MlirType type) {
  return llvm::isa<AffineType>(unwrap(type));
}

MlirType primeIRAffineTypeGet(MlirContext ctx, MlirAttribute curve) {
  return wrap(AffineType::get(unwrap(ctx),
                              llvm::cast<ShortWeierstrassAttr>(unwrap(curve))));
}

MlirAttribute primeIRAffineTypeGetCurve(MlirType type) {
  return wrap(llvm::cast<AffineType>(unwrap(type)).getCurve());
}

//===----------------------------------------------------------------------===//
// Jacobian point types.
//===----------------------------------------------------------------------===//

MlirTypeID primeIRJacobianTypeGetTypeID() {
  return wrap(JacobianType::getTypeID());
}

bool primeIRTypeIsAJacobian(MlirType type) {
  return llvm::isa<JacobianType>(unwrap(type));
}

MlirType primeIRJacobianTypeGet(MlirContext ctx, MlirAttribute curve) {
  return wrap(JacobianType::get(
      unwrap(ctx), llvm::cast<ShortWeierstrassAttr>(unwrap(curve))));
}

MlirAttribute primeIRJacobianTypeGetCurve(MlirType type) {
  return wrap(llvm::cast<JacobianType>(unwrap(type)).getCurve());
}

//===----------------------------------------------------------------------===//
// XYZZ point types.
//===----------------------------------------------------------------------===//

MlirTypeID primeIRXYZZTypeGetTypeID() { return wrap(XYZZType::getTypeID()); }

bool primeIRTypeIsAnXYZZ(MlirType type) {
  return llvm::isa<XYZZType>(unwrap(type));
}

MlirType primeIRXYZZTypeGet(MlirContext ctx, MlirAttribute curve) {
  return wrap(XYZZType::get(unwrap(ctx),
                            llvm::cast<ShortWeierstrassAttr>(unwrap(curve))));
}

MlirAttribute primeIRXYZZTypeGetCurve(MlirType type) {
  return wrap(llvm::cast<XYZZType>(unwrap(type)).getCurve());
}

//===----------------------------------------------------------------------===//
// Twisted Edwards point types.
//===----------------------------------------------------------------------===//

MlirTypeID primeIREdAffineTypeGetTypeID(void) {
  return wrap(EdAffineType::getTypeID());
}

bool primeIRTypeIsAnEdAffine(MlirType type) {
  return llvm::isa<EdAffineType>(unwrap(type));
}

MlirType primeIREdAffineTypeGet(MlirContext ctx, MlirAttribute curve) {
  return wrap(EdAffineType::get(unwrap(ctx),
                                llvm::cast<TwistedEdwardsAttr>(unwrap(curve))));
}

MlirAttribute primeIREdAffineTypeGetCurve(MlirType type) {
  return wrap(llvm::cast<EdAffineType>(unwrap(type)).getCurve());
}

MlirTypeID primeIREdExtendedTypeGetTypeID(void) {
  return wrap(EdExtendedType::getTypeID());
}

bool primeIRTypeIsAnEdExtended(MlirType type) {
  return llvm::isa<EdExtendedType>(unwrap(type));
}

MlirType primeIREdExtendedTypeGet(MlirContext ctx, MlirAttribute curve) {
  return wrap(EdExtendedType::get(
      unwrap(ctx), llvm::cast<TwistedEdwardsAttr>(unwrap(curve))));
}

MlirAttribute primeIREdExtendedTypeGetCurve(MlirType type) {
  return wrap(llvm::cast<EdExtendedType>(unwrap(type)).getCurve());
}
