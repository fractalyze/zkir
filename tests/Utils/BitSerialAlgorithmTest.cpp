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

#include "prime_ir/Utils/BitSerialAlgorithm.h"

#include "gtest/gtest.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir::prime_ir {

class BitSerialAlgorithmTest : public testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<arith::ArithDialect, scf::SCFDialect>();
    loc = UnknownLoc::get(&context);
  }

  struct OpCounts {
    int addIOp = 0;
    int mulIOp = 0;
    int whileOp = 0;
    int ifOp = 0;
  };

  OpCounts countOps(ModuleOp moduleOp) {
    OpCounts counts;
    moduleOp.walk([&](Operation *op) {
      if (isa<arith::AddIOp>(op))
        counts.addIOp++;
      else if (isa<arith::MulIOp>(op))
        counts.mulIOp++;
      else if (isa<scf::WhileOp>(op))
        counts.whileOp++;
      else if (isa<scf::IfOp>(op))
        counts.ifOp++;
    });
    return counts;
  }

  // Callbacks: MulIOp for double (distinguishable), AddIOp for accumulate
  static DoubleCallback getDoubleCallback() {
    return [](ImplicitLocOpBuilder &b, Value v) {
      return b.create<arith::MulIOp>(v, v);
    };
  }

  static AccumulateCallback getAccumulateCallback() {
    return [](ImplicitLocOpBuilder &b, Value acc, Value val) {
      return b.create<arith::AddIOp>(acc, val);
    };
  }

  // Builds a ModuleOp with constant scalar and calls generateBitSerialLoop.
  OwningOpRef<ModuleOp> buildConstantScalarTest(int64_t scalarVal,
                                                bool unroll = true) {
    auto i32Type = IntegerType::get(&context, 32);
    OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
    Block *body = module->getBody();
    ImplicitLocOpBuilder b(loc, body, body->end());

    Value scalar = b.create<arith::ConstantIntOp>(i32Type, scalarVal);
    Value base = b.create<arith::ConstantIntOp>(i32Type, 2);
    Value identity = b.create<arith::ConstantIntOp>(i32Type, 0);

    generateBitSerialLoop(b, scalar, base, identity, getDoubleCallback(),
                          getAccumulateCallback(), unroll);
    return module;
  }

  // Builds a ModuleOp with a dynamic (non-constant) scalar.
  // Uses arith::XOrIOp to produce a non-constant Value.
  OwningOpRef<ModuleOp> buildDynamicScalarTest() {
    auto i32Type = IntegerType::get(&context, 32);
    OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
    Block *body = module->getBody();
    ImplicitLocOpBuilder b(loc, body, body->end());

    Value c1 = b.create<arith::ConstantIntOp>(i32Type, 3);
    Value c2 = b.create<arith::ConstantIntOp>(i32Type, 5);
    Value scalar = b.create<arith::XOrIOp>(c1, c2);
    Value base = b.create<arith::ConstantIntOp>(i32Type, 2);
    Value identity = b.create<arith::ConstantIntOp>(i32Type, 0);

    generateBitSerialLoop(b, scalar, base, identity, getDoubleCallback(),
                          getAccumulateCallback());
    return module;
  }

  MLIRContext context;
  Location loc = UnknownLoc::get(&context);
};

// scalar = 0: result == identity, no ops generated
TEST_F(BitSerialAlgorithmTest, ConstantScalarZero) {
  auto module = buildConstantScalarTest(0);
  auto counts = countOps(*module);
  EXPECT_EQ(counts.addIOp, 0);
  EXPECT_EQ(counts.mulIOp, 0);
  EXPECT_EQ(counts.whileOp, 0);
  EXPECT_EQ(counts.ifOp, 0);
}

// scalar = 1 (binary: 1): 1 accumulate, 0 double
TEST_F(BitSerialAlgorithmTest, ConstantScalarOne) {
  auto module = buildConstantScalarTest(1);
  auto counts = countOps(*module);
  EXPECT_EQ(counts.addIOp, 1);
  EXPECT_EQ(counts.mulIOp, 0);
  EXPECT_EQ(counts.whileOp, 0);
  EXPECT_EQ(counts.ifOp, 0);
}

// scalar = 2 (binary: 10): 1 accumulate, 1 double
TEST_F(BitSerialAlgorithmTest, ConstantScalarTwo) {
  auto module = buildConstantScalarTest(2);
  auto counts = countOps(*module);
  EXPECT_EQ(counts.addIOp, 1);
  EXPECT_EQ(counts.mulIOp, 1);
  EXPECT_EQ(counts.whileOp, 0);
  EXPECT_EQ(counts.ifOp, 0);
}

// scalar = 5 (binary: 101): 2 accumulate, 2 double
TEST_F(BitSerialAlgorithmTest, ConstantScalarFive) {
  auto module = buildConstantScalarTest(5);
  auto counts = countOps(*module);
  EXPECT_EQ(counts.addIOp, 2);
  EXPECT_EQ(counts.mulIOp, 2);
  EXPECT_EQ(counts.whileOp, 0);
  EXPECT_EQ(counts.ifOp, 0);
}

// scalar = 7 (binary: 111): 3 accumulate, 2 double
TEST_F(BitSerialAlgorithmTest, ConstantScalarSeven) {
  auto module = buildConstantScalarTest(7);
  auto counts = countOps(*module);
  EXPECT_EQ(counts.addIOp, 3);
  EXPECT_EQ(counts.mulIOp, 2);
  EXPECT_EQ(counts.whileOp, 0);
  EXPECT_EQ(counts.ifOp, 0);
}

// scalar = 8 (binary: 1000): 1 accumulate, 3 double
TEST_F(BitSerialAlgorithmTest, ConstantScalarEight) {
  auto module = buildConstantScalarTest(8);
  auto counts = countOps(*module);
  EXPECT_EQ(counts.addIOp, 1);
  EXPECT_EQ(counts.mulIOp, 3);
  EXPECT_EQ(counts.whileOp, 0);
  EXPECT_EQ(counts.ifOp, 0);
}

// Dynamic scalar: scf::WhileOp is generated
TEST_F(BitSerialAlgorithmTest, DynamicScalar) {
  auto module = buildDynamicScalarTest();
  auto counts = countOps(*module);
  EXPECT_EQ(counts.whileOp, 1);
  EXPECT_GE(counts.ifOp, 1);
}

// A caller that rolls its own rounds asks for the loop even though the trip
// count is known: only the two ops of one loop body are emitted, against the
// three accumulates and two doubles the unrolled form spells out for 7.
TEST_F(BitSerialAlgorithmTest, ConstantScalarUnrollDisabled) {
  auto module = buildConstantScalarTest(7, /*unroll=*/false);
  auto counts = countOps(*module);
  EXPECT_EQ(counts.whileOp, 1);
  EXPECT_GE(counts.ifOp, 1);
  EXPECT_EQ(counts.addIOp, 2);
  EXPECT_EQ(counts.mulIOp, 1);
}

} // namespace mlir::prime_ir
