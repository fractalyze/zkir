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

#include "gtest/gtest.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveDialect.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveOps.h"
#include "prime_ir/Dialect/EllipticCurve/IR/EllipticCurveTypes.h"
#include "prime_ir/Dialect/EllipticCurve/IR/PointOperation.h"
#include "prime_ir/Dialect/Field/IR/FieldDialect.h"
#include "prime_ir/Dialect/Field/IR/FieldOps.h"
#include "prime_ir/Dialect/Field/IR/PrimeFieldOperation.h"
#include "zk_dtypes/include/elliptic_curve/bn/bn254/g1.h"
#include "zk_dtypes/include/elliptic_curve/curve25519/ed25519/g1.h"

namespace mlir::prime_ir::elliptic_curve {
namespace {

class CreateZeroPointTest : public testing::Test {
protected:
  static void SetUpTestSuite() {
    context.loadDialect<EllipticCurveDialect>();
    context.loadDialect<field::FieldDialect>();
    context.loadDialect<func::FuncDialect>();
  }

  // Verifies that createZeroPoint produces a FromCoordsOp with the expected
  // coordinate pattern. |is_zero_pattern| maps each coordinate index to
  // whether it should be the field zero (true) or field one (false).
  void testCreateZeroPoint(Type pointType, std::vector<bool> is_zero_pattern) {
    OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = ModuleOp::create(loc);

    auto funcType = builder.getFunctionType({}, {pointType});
    auto funcOp =
        builder.create<func::FuncOp>(loc, "test_zero_point", funcType);
    Block *entryBlock = funcOp.addEntryBlock();
    module.push_back(funcOp);

    ImplicitLocOpBuilder b(loc, entryBlock, entryBlock->begin());
    Value zeroPoint = createZeroPoint(b, pointType);
    b.create<func::ReturnOp>(ValueRange{zeroPoint});

    ASSERT_TRUE(succeeded(verify(module)));

    auto fromCoordsOp = zeroPoint.getDefiningOp<FromCoordsOp>();
    ASSERT_NE(fromCoordsOp, nullptr);
    ASSERT_EQ(fromCoordsOp.getNumOperands(), is_zero_pattern.size());

    auto baseFieldType = cast<PointTypeInterface>(pointType).getBaseFieldType();
    auto pfType = cast<field::PrimeFieldType>(baseFieldType);

    for (size_t i = 0; i < is_zero_pattern.size(); ++i) {
      auto constOp =
          fromCoordsOp.getOperand(i).getDefiningOp<field::ConstantOp>();
      ASSERT_NE(constOp, nullptr) << "Coord " << i << " is not field.constant";

      auto intAttr = cast<IntegerAttr>(constOp.getValue());
      field::PrimeFieldOperation pfOp(intAttr, pfType);
      if (is_zero_pattern[i]) {
        EXPECT_TRUE(pfOp.isZero()) << "Coord " << i << " should be zero";
      } else {
        EXPECT_TRUE(pfOp.isOne()) << "Coord " << i << " should be one";
      }
    }

    module.erase();
  }

  static MLIRContext context;
};

MLIRContext CreateZeroPointTest::context;

// Affine zero point: (0, 0)
TEST_F(CreateZeroPointTest, Affine) {
  auto attr =
      AffinePointOperation::getPointType<zk_dtypes::bn254::G1AffinePoint>(
          &context);
  testCreateZeroPoint(attr, {true, true});
}

// Jacobian zero point: (1, 1, 0) — Z=0 represents point at infinity
TEST_F(CreateZeroPointTest, Jacobian) {
  auto attr =
      AffinePointOperation::getPointType<zk_dtypes::bn254::G1JacobianPoint>(
          &context);
  testCreateZeroPoint(attr, {false, false, true});
}

// XYZZ zero point: (1, 1, 0, 0) — ZZ=0, ZZZ=0 represents point at infinity
TEST_F(CreateZeroPointTest, XYZZ) {
  auto attr = AffinePointOperation::getPointType<zk_dtypes::bn254::G1PointXyzz>(
      &context);
  testCreateZeroPoint(attr, {false, false, true, true});
}

// Twisted Edwards has a real affine identity, (0, 1), on the curve — unlike
// the short-Weierstrass (0, 0), which is a sentinel precisely because it is
// off the curve. Getting this wrong yields a point that verifies and computes
// the wrong answer, so the pattern is pinned per representation.
TEST_F(CreateZeroPointTest, EdAffine) {
  auto attr =
      AffinePointOperation::getPointType<zk_dtypes::ed25519::G1AffinePoint>(
          &context);
  testCreateZeroPoint(attr, {true, false});
}

// Extended carries x = X/Z, y = Y/Z and T = XY/Z, so (0, 1) is (0 : 1 : 1 : 0).
TEST_F(CreateZeroPointTest, EdExtended) {
  auto attr =
      AffinePointOperation::getPointType<zk_dtypes::ed25519::G1ExtendedPoint>(
          &context);
  testCreateZeroPoint(attr, {true, false, false, true});
}

} // namespace
} // namespace mlir::prime_ir::elliptic_curve
