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

#ifndef PRIME_IR_UTILS_BITSERIALALGORITHM_H_
#define PRIME_IR_UTILS_BITSERIALALGORITHM_H_

#include <functional>

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Value.h"

namespace mlir::prime_ir {

// Callback for the "doubling" operation in a bit-serial algorithm.
// For field exponentiation: square operation
// For EC scalar multiplication: point doubling
using DoubleCallback =
    std::function<Value(ImplicitLocOpBuilder &b, Value current)>;

// Callback for the "accumulation" operation in a bit-serial algorithm.
// For field exponentiation: multiplication
// For EC scalar multiplication: point addition
using AccumulateCallback =
    std::function<Value(ImplicitLocOpBuilder &b, Value accumulator, Value val)>;

// Generates an optimized LSB-first binary method (double-and-add /
// square-and-multiply).
//
// When `scalar` is defined by an arith::ConstantOp and `allowUnroll` is set,
// the loop is fully unrolled into straight-line IR without scf::WhileOp or
// scf::IfOp. When `scalar` is dynamic, or `allowUnroll` is clear, a runtime
// scf::WhileOp loop with first-iteration unrolling is generated.
//
// Clearing `allowUnroll` costs the caller the constant folding the unrolled
// form exposes, and buys back the instruction count a rolled loop executes;
// which side wins depends on whether the caller is itself unrolled around
// this call.
//
// Algorithm:
//   result = identity
//   if (scalar & 1) result = accumulate(result, base)
//   scalar >>= 1
//   while (scalar > 0):
//     base = double(base)
//     if (scalar & 1) result = accumulate(result, base)
//     scalar >>= 1
//   return result
Value generateBitSerialLoop(ImplicitLocOpBuilder &b, Value scalar, Value base,
                            Value identity, DoubleCallback doubleOp,
                            AccumulateCallback accumulateOp,
                            bool allowUnroll = true);

} // namespace mlir::prime_ir

#endif // PRIME_IR_UTILS_BITSERIALALGORITHM_H_
