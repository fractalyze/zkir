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

#ifndef PRIME_IR_DIALECT_ELLIPTICCURVE_PYTHON_ELLIPTICCURVEATTRIBUTES_H_
#define PRIME_IR_DIALECT_ELLIPTICCURVE_PYTHON_ELLIPTICCURVEATTRIBUTES_H_

#include "mlir/Bindings/Python/IRCore.h"
#include "mlir/Bindings/Python/Nanobind.h"
#include "prime_ir/Dialect/EllipticCurve/C/EllipticCurveAttributes.h"

namespace mlir::prime_ir::elliptic_curve::python {

class PyShortWeierstrassAttr
    : public mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute<
          PyShortWeierstrassAttr> {
public:
  static constexpr IsAFunctionTy isaFunction =
      primeIRAttributeIsAShortWeierstrass;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      primeIRShortWeierstrassAttrGetTypeID;
  static constexpr const char *pyClassName = "ShortWeierstrassAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c);
};

class PyTwistedEdwardsAttr
    : public mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN::PyConcreteAttribute<
          PyTwistedEdwardsAttr> {
public:
  static constexpr IsAFunctionTy isaFunction =
      primeIRAttributeIsATwistedEdwards;
  static constexpr GetTypeIDFunctionTy getTypeIdFunction =
      primeIRTwistedEdwardsAttrGetTypeID;
  static constexpr const char *pyClassName = "TwistedEdwardsAttr";
  using PyConcreteAttribute::PyConcreteAttribute;

  static void bindDerived(ClassTy &c);
};

void populateIRAttributes(nanobind::module_ &m);

} // namespace mlir::prime_ir::elliptic_curve::python

#endif // PRIME_IR_DIALECT_ELLIPTICCURVE_PYTHON_ELLIPTICCURVEATTRIBUTES_H_
