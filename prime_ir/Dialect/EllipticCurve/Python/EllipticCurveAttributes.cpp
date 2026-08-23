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

#include "prime_ir/Dialect/EllipticCurve/Python/EllipticCurveAttributes.h"

namespace nb = nanobind;
using namespace mlir::python::MLIR_BINDINGS_PYTHON_DOMAIN;

namespace mlir::prime_ir::elliptic_curve::python {

// static
void PyShortWeierstrassAttr::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](PyType &baseField, PyAttribute &a, PyAttribute &b, PyAttribute &Gx,
         PyAttribute &Gy,
         DefaultingPyMlirContext context) -> PyShortWeierstrassAttr {
        MlirAttribute attr = primeIRShortWeierstrassAttrGet(
            context->get(), baseField, a, b, Gx, Gy);
        return PyShortWeierstrassAttr(context->getRef(), attr);
      },
      nb::arg("base_field"), nb::arg("a"), nb::arg("b"), nb::arg("gx"),
      nb::arg("gy"), nb::arg("context") = nb::none(),
      "Create a ShortWeierstrass curve attribute");
  c.def_prop_ro(
      "base_field",
      [](PyShortWeierstrassAttr &self) -> PyType {
        return PyType(self.getContext(),
                      primeIRShortWeierstrassAttrGetBaseField(self));
      },
      "Returns the base field type of the curve");
  c.def_prop_ro(
      "a",
      [](PyShortWeierstrassAttr &self) -> PyAttribute {
        return PyAttribute(self.getContext(),
                           primeIRShortWeierstrassAttrGetA(self));
      },
      "Returns the 'a' coefficient of the curve equation y² = x³ + ax + b");
  c.def_prop_ro(
      "b",
      [](PyShortWeierstrassAttr &self) -> PyAttribute {
        return PyAttribute(self.getContext(),
                           primeIRShortWeierstrassAttrGetB(self));
      },
      "Returns the 'b' coefficient of the curve equation y² = x³ + ax + b");
  c.def_prop_ro(
      "gx",
      [](PyShortWeierstrassAttr &self) -> PyAttribute {
        return PyAttribute(self.getContext(),
                           primeIRShortWeierstrassAttrGetGx(self));
      },
      "Returns the x-coordinate of the generator point");
  c.def_prop_ro(
      "gy",
      [](PyShortWeierstrassAttr &self) -> PyAttribute {
        return PyAttribute(self.getContext(),
                           primeIRShortWeierstrassAttrGetGy(self));
      },
      "Returns the y-coordinate of the generator point");
}

// static
void PyTwistedEdwardsAttr::bindDerived(ClassTy &c) {
  c.def_static(
      "get",
      [](PyType &baseField, PyAttribute &a, PyAttribute &d, PyAttribute &Gx,
         PyAttribute &Gy,
         DefaultingPyMlirContext context) -> PyTwistedEdwardsAttr {
        MlirAttribute attr = primeIRTwistedEdwardsAttrGet(
            context->get(), baseField, a, d, Gx, Gy);
        return PyTwistedEdwardsAttr(context->getRef(), attr);
      },
      nb::arg("base_field"), nb::arg("a"), nb::arg("d"), nb::arg("gx"),
      nb::arg("gy"), nb::arg("context") = nb::none(),
      "Create a TwistedEdwards curve attribute");
  c.def_prop_ro(
      "base_field",
      [](PyTwistedEdwardsAttr &self) -> PyType {
        return PyType(self.getContext(),
                      primeIRTwistedEdwardsAttrGetBaseField(self));
      },
      "Returns the base field type of the curve");
  c.def_prop_ro(
      "a",
      [](PyTwistedEdwardsAttr &self) -> PyAttribute {
        return PyAttribute(self.getContext(),
                           primeIRTwistedEdwardsAttrGetA(self));
      },
      "Returns the 'a' coefficient of the curve");
  c.def_prop_ro(
      "d",
      [](PyTwistedEdwardsAttr &self) -> PyAttribute {
        return PyAttribute(self.getContext(),
                           primeIRTwistedEdwardsAttrGetD(self));
      },
      "Returns the 'd' coefficient of the curve");
  c.def_prop_ro(
      "gx",
      [](PyTwistedEdwardsAttr &self) -> PyAttribute {
        return PyAttribute(self.getContext(),
                           primeIRTwistedEdwardsAttrGetGx(self));
      },
      "Returns the x-coordinate of the generator point");
  c.def_prop_ro(
      "gy",
      [](PyTwistedEdwardsAttr &self) -> PyAttribute {
        return PyAttribute(self.getContext(),
                           primeIRTwistedEdwardsAttrGetGy(self));
      },
      "Returns the y-coordinate of the generator point");
}

void populateIRAttributes(nb::module_ &m) {
  PyShortWeierstrassAttr::bind(m);
  PyTwistedEdwardsAttr::bind(m);
}

} // namespace mlir::prime_ir::elliptic_curve::python
