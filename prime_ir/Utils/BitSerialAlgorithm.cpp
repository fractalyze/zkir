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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::prime_ir {

namespace {

// Generates unrolled straight-line IR for a constant scalar.
// Uses LSB-first binary method matching the runtime loop structure.
Value generateConstantUnrolled(ImplicitLocOpBuilder &b, const APInt &scalarVal,
                               Value base, Value identity,
                               DoubleCallback &doubleOp,
                               AccumulateCallback &accumulateOp) {
  if (scalarVal.isZero()) {
    return identity;
  }

  APInt currScalar = scalarVal;
  APInt one(currScalar.getBitWidth(), 1);
  Value result = identity;
  Value factor = base;

  while (!currScalar.isZero()) {
    if ((currScalar & one).getBoolValue()) {
      result = accumulateOp(b, result, factor);
    }
    currScalar = currScalar.lshr(1);
    if (!currScalar.isZero()) {
      factor = doubleOp(b, factor);
    }
  }

  return result;
}

} // namespace

Value generateBitSerialLoop(ImplicitLocOpBuilder &b, Value scalar, Value base,
                            Value identity, DoubleCallback doubleOp,
                            AccumulateCallback accumulateOp, bool unroll) {
  // If scalar is a compile-time constant, unroll into straight-line IR.
  if (unroll) {
    if (auto constOp = scalar.getDefiningOp<arith::ConstantOp>()) {
      APInt scalarVal = cast<IntegerAttr>(constOp.getValue()).getValue();
      return generateConstantUnrolled(b, scalarVal, base, identity, doubleOp,
                                      accumulateOp);
    }
  }

  Type scalarType = scalar.getType();
  Type baseType = base.getType();

  Value one = arith::ConstantIntOp::create(b, scalarType, 1);
  Value zero = arith::ConstantIntOp::create(b, scalarType, 0);

  // First iteration unrolling: handle LSB before entering the loop
  Value result = identity;
  Value lsb = arith::AndIOp::create(b, scalar, one);
  Value lsbSet = arith::CmpIOp::create(b, arith::CmpIPredicate::ne, lsb, zero);

  auto firstIfOp = scf::IfOp::create(
      b, lsbSet,
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder ib(loc, builder);
        Value newResult = accumulateOp(ib, result, base);
        scf::YieldOp::create(ib, newResult);
      },
      [&](OpBuilder &builder, Location loc) {
        ImplicitLocOpBuilder ib(loc, builder);
        scf::YieldOp::create(ib, result);
      });

  result = firstIfOp.getResult(0);
  scalar = arith::ShRUIOp::create(b, scalar, one);

  // Main loop: process remaining bits
  auto whileOp = scf::WhileOp::create(
      b, /*resultTypes=*/TypeRange{scalarType, baseType, baseType},
      /*operands=*/ValueRange{scalar, base, result},
      /*beforeBuilder=*/
      [&](OpBuilder &beforeBuilder, Location beforeLoc, ValueRange args) {
        ImplicitLocOpBuilder ib(beforeLoc, beforeBuilder);
        Value currentScalar = args[0];
        // Continue while scalar > 0
        Value cond = arith::CmpIOp::create(ib, arith::CmpIPredicate::ugt,
                                           currentScalar, zero);
        scf::ConditionOp::create(ib, cond, args);
      },
      /*afterBuilder=*/
      [&](OpBuilder &afterBuilder, Location afterLoc, ValueRange args) {
        ImplicitLocOpBuilder ib(afterLoc, afterBuilder);
        Value currentScalar = args[0];
        Value currentBase = args[1];
        Value currentResult = args[2];

        // Double the base
        Value doubledBase = doubleOp(ib, currentBase);

        // Check if current bit is set
        Value bit = arith::AndIOp::create(ib, currentScalar, one);
        Value bitSet =
            arith::CmpIOp::create(ib, arith::CmpIPredicate::ne, bit, zero);

        // Conditionally accumulate
        auto ifOp = scf::IfOp::create(
            ib, bitSet,
            [&](OpBuilder &builder, Location loc) {
              ImplicitLocOpBuilder ib2(loc, builder);
              Value newResult = accumulateOp(ib2, currentResult, doubledBase);
              scf::YieldOp::create(ib2, newResult);
            },
            [&](OpBuilder &builder, Location loc) {
              ImplicitLocOpBuilder ib2(loc, builder);
              scf::YieldOp::create(ib2, currentResult);
            });

        // Shift scalar right
        Value shiftedScalar = arith::ShRUIOp::create(ib, currentScalar, one);

        scf::YieldOp::create(
            ib, ValueRange{shiftedScalar, doubledBase, ifOp.getResult(0)});
      });

  return whileOp.getResult(2);
}

} // namespace mlir::prime_ir
