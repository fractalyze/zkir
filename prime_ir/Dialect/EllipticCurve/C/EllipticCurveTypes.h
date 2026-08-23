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

#ifndef PRIME_IR_DIALECT_ELLIPTICCURVE_C_ELLIPTICCURVETYPES_H_
#define PRIME_IR_DIALECT_ELLIPTICCURVE_C_ELLIPTICCURVETYPES_H_

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Affine point types.
//===----------------------------------------------------------------------===//

// Returns the typeID of an affine point type.
MLIR_CAPI_EXPORTED MlirTypeID primeIRAffineTypeGetTypeID(void);

// Checks whether the given type is an affine point type.
MLIR_CAPI_EXPORTED bool primeIRTypeIsAnAffine(MlirType type);

// Creates an affine point type with the given curve attribute.
MLIR_CAPI_EXPORTED MlirType primeIRAffineTypeGet(MlirContext ctx,
                                                 MlirAttribute curve);

// Returns the curve attribute of the given affine point type.
MLIR_CAPI_EXPORTED MlirAttribute primeIRAffineTypeGetCurve(MlirType type);

//===----------------------------------------------------------------------===//
// Jacobian point types.
//===----------------------------------------------------------------------===//

// Returns the typeID of a jacobian point type.
MLIR_CAPI_EXPORTED MlirTypeID primeIRJacobianTypeGetTypeID(void);

// Checks whether the given type is a jacobian point type.
MLIR_CAPI_EXPORTED bool primeIRTypeIsAJacobian(MlirType type);

// Creates a jacobian point type with the given curve attribute.
MLIR_CAPI_EXPORTED MlirType primeIRJacobianTypeGet(MlirContext ctx,
                                                   MlirAttribute curve);

// Returns the curve attribute of the given jacobian point type.
MLIR_CAPI_EXPORTED MlirAttribute primeIRJacobianTypeGetCurve(MlirType type);

//===----------------------------------------------------------------------===//
// XYZZ point types.
//===----------------------------------------------------------------------===//

// Returns the typeID of an xyzz point type.
MLIR_CAPI_EXPORTED MlirTypeID primeIRXYZZTypeGetTypeID(void);

// Checks whether the given type is an xyzz point type.
MLIR_CAPI_EXPORTED bool primeIRTypeIsAnXYZZ(MlirType type);

// Creates an xyzz point type with the given curve attribute.
MLIR_CAPI_EXPORTED MlirType primeIRXYZZTypeGet(MlirContext ctx,
                                               MlirAttribute curve);

// Returns the curve attribute of the given xyzz point type.
MLIR_CAPI_EXPORTED MlirAttribute primeIRXYZZTypeGetCurve(MlirType type);

//===----------------------------------------------------------------------===//
// Twisted Edwards point types.
//===----------------------------------------------------------------------===//

// Returns the typeID of an ed_affine point type.
MLIR_CAPI_EXPORTED MlirTypeID primeIREdAffineTypeGetTypeID(void);

// Checks whether the given type is an ed_affine point type.
MLIR_CAPI_EXPORTED bool primeIRTypeIsAnEdAffine(MlirType type);

// Creates an ed_affine point type with the given twisted Edwards curve.
MLIR_CAPI_EXPORTED MlirType primeIREdAffineTypeGet(MlirContext ctx,
                                                   MlirAttribute curve);

// Returns the curve attribute of the given ed_affine point type.
MLIR_CAPI_EXPORTED MlirAttribute primeIREdAffineTypeGetCurve(MlirType type);

// Returns the typeID of an ed_extended point type.
MLIR_CAPI_EXPORTED MlirTypeID primeIREdExtendedTypeGetTypeID(void);

// Checks whether the given type is an ed_extended point type.
MLIR_CAPI_EXPORTED bool primeIRTypeIsAnEdExtended(MlirType type);

// Creates an ed_extended point type with the given twisted Edwards curve.
MLIR_CAPI_EXPORTED MlirType primeIREdExtendedTypeGet(MlirContext ctx,
                                                     MlirAttribute curve);

// Returns the curve attribute of the given ed_extended point type.
MLIR_CAPI_EXPORTED MlirAttribute primeIREdExtendedTypeGetCurve(MlirType type);

#ifdef __cplusplus
}
#endif

#endif // PRIME_IR_DIALECT_ELLIPTICCURVE_C_ELLIPTICCURVETYPES_H_
