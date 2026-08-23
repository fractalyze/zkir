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

#ifndef PRIME_IR_DIALECT_ELLIPTICCURVE_C_ELLIPTICCURVEATTRIBUTES_H_
#define PRIME_IR_DIALECT_ELLIPTICCURVE_C_ELLIPTICCURVEATTRIBUTES_H_

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// ShortWeierstrass attribute.
//===----------------------------------------------------------------------===//

// Returns the typeID of a ShortWeierstrass attribute.
MLIR_CAPI_EXPORTED MlirTypeID primeIRShortWeierstrassAttrGetTypeID(void);

// Checks whether the given attribute is a ShortWeierstrass attribute.
MLIR_CAPI_EXPORTED bool primeIRAttributeIsAShortWeierstrass(MlirAttribute attr);

// Creates a ShortWeierstrass curve attribute.
// Parameters:
//   ctx: The MLIR context
//   baseField: The base field type (!field.pf or !field.ef)
//   a: The 'a' coefficient in y² = x³ + ax + b (TypedAttr)
//   b: The 'b' coefficient in y² = x³ + ax + b (TypedAttr)
//   Gx: The x-coordinate of the generator point (TypedAttr)
//   Gy: The y-coordinate of the generator point (TypedAttr)
// Note: For G1 (prime field), use IntegerAttr.
//       For G2 (extension field), use DenseIntElementsAttr.
MLIR_CAPI_EXPORTED MlirAttribute primeIRShortWeierstrassAttrGet(
    MlirContext ctx, MlirType baseField, MlirAttribute a, MlirAttribute b,
    MlirAttribute Gx, MlirAttribute Gy);

// Returns the base field type of the curve.
MLIR_CAPI_EXPORTED MlirType
primeIRShortWeierstrassAttrGetBaseField(MlirAttribute attr);

// Returns the 'a' coefficient of the curve.
MLIR_CAPI_EXPORTED MlirAttribute
primeIRShortWeierstrassAttrGetA(MlirAttribute attr);

// Returns the 'b' coefficient of the curve.
MLIR_CAPI_EXPORTED MlirAttribute
primeIRShortWeierstrassAttrGetB(MlirAttribute attr);

// Returns the x-coordinate of the generator point.
MLIR_CAPI_EXPORTED MlirAttribute
primeIRShortWeierstrassAttrGetGx(MlirAttribute attr);

// Returns the y-coordinate of the generator point.
MLIR_CAPI_EXPORTED MlirAttribute
primeIRShortWeierstrassAttrGetGy(MlirAttribute attr);

//===----------------------------------------------------------------------===//
// TwistedEdwards attribute.
//===----------------------------------------------------------------------===//

// Returns the typeID of a TwistedEdwards attribute.
MLIR_CAPI_EXPORTED MlirTypeID primeIRTwistedEdwardsAttrGetTypeID(void);

// Checks whether the given attribute is a TwistedEdwards attribute.
MLIR_CAPI_EXPORTED bool primeIRAttributeIsATwistedEdwards(MlirAttribute attr);

// Creates a TwistedEdwards curve attribute.
// Parameters:
//   ctx: The MLIR context
//   baseField: The base field type (!field.pf or !field.ef)
//   a: The 'a' coefficient in a·x² + y² = 1 + d·x²y² (TypedAttr)
//   d: The 'd' coefficient in a·x² + y² = 1 + d·x²y² (TypedAttr)
//   Gx: The x-coordinate of the generator point (TypedAttr)
//   Gy: The y-coordinate of the generator point (TypedAttr)
MLIR_CAPI_EXPORTED MlirAttribute primeIRTwistedEdwardsAttrGet(
    MlirContext ctx, MlirType baseField, MlirAttribute a, MlirAttribute d,
    MlirAttribute Gx, MlirAttribute Gy);

// Returns the base field type of the curve.
MLIR_CAPI_EXPORTED MlirType
primeIRTwistedEdwardsAttrGetBaseField(MlirAttribute attr);

// Returns the 'a' coefficient of the curve.
MLIR_CAPI_EXPORTED MlirAttribute
primeIRTwistedEdwardsAttrGetA(MlirAttribute attr);

// Returns the 'd' coefficient of the curve.
MLIR_CAPI_EXPORTED MlirAttribute
primeIRTwistedEdwardsAttrGetD(MlirAttribute attr);

// Returns the x-coordinate of the generator point.
MLIR_CAPI_EXPORTED MlirAttribute
primeIRTwistedEdwardsAttrGetGx(MlirAttribute attr);

// Returns the y-coordinate of the generator point.
MLIR_CAPI_EXPORTED MlirAttribute
primeIRTwistedEdwardsAttrGetGy(MlirAttribute attr);

#ifdef __cplusplus
}
#endif

#endif // PRIME_IR_DIALECT_ELLIPTICCURVE_C_ELLIPTICCURVEATTRIBUTES_H_
