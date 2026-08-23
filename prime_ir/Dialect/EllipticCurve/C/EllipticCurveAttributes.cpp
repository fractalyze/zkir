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

#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveAttributes.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "prime_ir/Dialect/EllipticCurve/C/EllipticCurveAttributes.h"

using namespace mlir;
using namespace mlir::prime_ir::elliptic_curve;

//===----------------------------------------------------------------------===//
// ShortWeierstrass attribute.
//===----------------------------------------------------------------------===//

MlirTypeID primeIRShortWeierstrassAttrGetTypeID() {
  return wrap(ShortWeierstrassAttr::getTypeID());
}

bool primeIRAttributeIsAShortWeierstrass(MlirAttribute attr) {
  return llvm::isa<ShortWeierstrassAttr>(unwrap(attr));
}

MlirAttribute primeIRShortWeierstrassAttrGet(MlirContext ctx,
                                             MlirType baseField,
                                             MlirAttribute a, MlirAttribute b,
                                             MlirAttribute Gx,
                                             MlirAttribute Gy) {
  // Parameters can be IntegerAttr (for prime field/G1) or
  // DenseIntElementsAttr (for extension field/G2)
  return wrap(ShortWeierstrassAttr::get(
      unwrap(ctx), unwrap(baseField), llvm::cast<TypedAttr>(unwrap(a)),
      llvm::cast<TypedAttr>(unwrap(b)), llvm::cast<TypedAttr>(unwrap(Gx)),
      llvm::cast<TypedAttr>(unwrap(Gy))));
}

MlirType primeIRShortWeierstrassAttrGetBaseField(MlirAttribute attr) {
  return wrap(llvm::cast<ShortWeierstrassAttr>(unwrap(attr)).getBaseField());
}

MlirAttribute primeIRShortWeierstrassAttrGetA(MlirAttribute attr) {
  return wrap(llvm::cast<ShortWeierstrassAttr>(unwrap(attr)).getA());
}

MlirAttribute primeIRShortWeierstrassAttrGetB(MlirAttribute attr) {
  return wrap(llvm::cast<ShortWeierstrassAttr>(unwrap(attr)).getB());
}

MlirAttribute primeIRShortWeierstrassAttrGetGx(MlirAttribute attr) {
  return wrap(llvm::cast<ShortWeierstrassAttr>(unwrap(attr)).getGx());
}

MlirAttribute primeIRShortWeierstrassAttrGetGy(MlirAttribute attr) {
  return wrap(llvm::cast<ShortWeierstrassAttr>(unwrap(attr)).getGy());
}

//===----------------------------------------------------------------------===//
// TwistedEdwards attribute.
//===----------------------------------------------------------------------===//

MlirTypeID primeIRTwistedEdwardsAttrGetTypeID(void) {
  return wrap(TwistedEdwardsAttr::getTypeID());
}

bool primeIRAttributeIsATwistedEdwards(MlirAttribute attr) {
  return llvm::isa<TwistedEdwardsAttr>(unwrap(attr));
}

MlirAttribute primeIRTwistedEdwardsAttrGet(MlirContext ctx, MlirType baseField,
                                           MlirAttribute a, MlirAttribute d,
                                           MlirAttribute Gx, MlirAttribute Gy) {
  return wrap(TwistedEdwardsAttr::get(
      unwrap(ctx), unwrap(baseField), llvm::cast<TypedAttr>(unwrap(a)),
      llvm::cast<TypedAttr>(unwrap(d)), llvm::cast<TypedAttr>(unwrap(Gx)),
      llvm::cast<TypedAttr>(unwrap(Gy))));
}

MlirType primeIRTwistedEdwardsAttrGetBaseField(MlirAttribute attr) {
  return wrap(llvm::cast<TwistedEdwardsAttr>(unwrap(attr)).getBaseField());
}

MlirAttribute primeIRTwistedEdwardsAttrGetA(MlirAttribute attr) {
  return wrap(llvm::cast<TwistedEdwardsAttr>(unwrap(attr)).getA());
}

MlirAttribute primeIRTwistedEdwardsAttrGetD(MlirAttribute attr) {
  return wrap(llvm::cast<TwistedEdwardsAttr>(unwrap(attr)).getD());
}

MlirAttribute primeIRTwistedEdwardsAttrGetGx(MlirAttribute attr) {
  return wrap(llvm::cast<TwistedEdwardsAttr>(unwrap(attr)).getGx());
}

MlirAttribute primeIRTwistedEdwardsAttrGetGy(MlirAttribute attr) {
  return wrap(llvm::cast<TwistedEdwardsAttr>(unwrap(attr)).getGy());
}
