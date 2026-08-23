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

#ifndef PRIME_IR_DIALECT_ELLIPTICCURVE_PYTHON_ELLIPTICCURVETYPES_H_
#define PRIME_IR_DIALECT_ELLIPTICCURVE_PYTHON_ELLIPTICCURVETYPES_H_

#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "prime_ir/Dialect/EllipticCurve/C/EllipticCurveTypes.h"

namespace mlir::prime_ir::elliptic_curve::python {

class PyAffineType
    : public mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteType<
          PyAffineType> {
public:
  static constexpr IsAFunctionTy isaFunction = primeIRTypeIsAnAffine;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      primeIRAffineTypeGetTypeID;
  static constexpr const char *pyClassName = "AffineType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

class PyJacobianType
    : public mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteType<
          PyJacobianType> {
public:
  static constexpr IsAFunctionTy isaFunction = primeIRTypeIsAJacobian;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      primeIRJacobianTypeGetTypeID;
  static constexpr const char *pyClassName = "JacobianType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

class PyXYZZType
    : public mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteType<
          PyXYZZType> {
public:
  static constexpr IsAFunctionTy isaFunction = primeIRTypeIsAnXYZZ;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      primeIRXYZZTypeGetTypeID;
  static constexpr const char *pyClassName = "XYZZType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

class PyEdAffineType
    : public mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteType<
          PyEdAffineType> {
public:
  static constexpr IsAFunctionTy isaFunction = primeIRTypeIsAnEdAffine;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      primeIREdAffineTypeGetTypeID;
  static constexpr const char *pyClassName = "EdAffineType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

class PyEdExtendedType
    : public mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteType<
          PyEdExtendedType> {
public:
  static constexpr IsAFunctionTy isaFunction = primeIRTypeIsAnEdExtended;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      primeIREdExtendedTypeGetTypeID;
  static constexpr const char *pyClassName = "EdExtendedType";
  using PyConcreteType::PyConcreteType;

  static void bindDerived(ClassTy &c);
};

void populateIRTypes(nanobind::module_ &m);

} // namespace mlir::prime_ir::elliptic_curve::python

#endif // PRIME_IR_DIALECT_ELLIPTICCURVE_PYTHON_ELLIPTICCURVETYPES_H_
