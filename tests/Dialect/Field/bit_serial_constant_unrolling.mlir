// Copyright 2026 The PrimeIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

// Tests that field.powui with constant exponents is unrolled into straight-line
// IR (no scf.while/scf.if), while dynamic exponents produce a runtime loop.
// A `unroll = false` attribute keeps the loop for a constant exponent.

// RUN: prime-ir-opt -field-to-mod-arith -canonicalize %s | FileCheck %s -enable-var-scope

!PF = !field.pf<7:i32>
!GL = !field.pf<18446744069414584321:i64>
!GL3 = !field.ef<3x!GL, 7:i64>

// exp = 0 → Fermat: 0 % 6 = 0 → identity
// CHECK-LABEL: @test_powui_const_zero
// CHECK-SAME: (%{{.*}}: [[T:.*]]) -> [[T]]
// CHECK-NEXT: %[[ONE:.*]] = mod_arith.constant 1 : [[T]]
// CHECK-NEXT: return %[[ONE]] : [[T]]
func.func @test_powui_const_zero(%base: !PF) -> !PF {
  %exp = arith.constant 0 : i32
  %res = field.powui %base, %exp : !PF, i32
  return %res : !PF
}

// exp = 1 → Fermat: 1 % 6 = 1 → base
// CHECK-LABEL: @test_powui_const_one
// CHECK-SAME: (%[[BASE:.*]]: [[T:.*]]) -> [[T]]
// CHECK-NEXT: return %[[BASE]] : [[T]]
func.func @test_powui_const_one(%base: !PF) -> !PF {
  %exp = arith.constant 1 : i32
  %res = field.powui %base, %exp : !PF, i32
  return %res : !PF
}

// exp = 3 (11₂) → Fermat: 3 % 6 = 3 → square + mul
// CHECK-LABEL: @test_powui_const_three
// CHECK-SAME: (%[[BASE:.*]]: [[T:.*]]) -> [[T]]
// CHECK-NEXT: %[[SQ:.*]] = mod_arith.square %[[BASE]] : [[T]]
// CHECK-NEXT: %[[MUL:.*]] = mod_arith.mul %[[BASE]], %[[SQ]] : [[T]]
// CHECK-NEXT: return %[[MUL]] : [[T]]
func.func @test_powui_const_three(%base: !PF) -> !PF {
  %exp = arith.constant 3 : i32
  %res = field.powui %base, %exp : !PF, i32
  return %res : !PF
}

// exp = 5 (101₂) → Fermat: 5 % 6 = 5 → 2 squares + mul
// CHECK-LABEL: @test_powui_const_five
// CHECK-SAME: (%[[BASE:.*]]: [[T:.*]]) -> [[T]]
// CHECK-NEXT: %[[SQ1:.*]] = mod_arith.square %[[BASE]] : [[T]]
// CHECK-NEXT: %[[SQ2:.*]] = mod_arith.square %[[SQ1]] : [[T]]
// CHECK-NEXT: %[[MUL:.*]] = mod_arith.mul %[[BASE]], %[[SQ2]] : [[T]]
// CHECK-NEXT: return %[[MUL]] : [[T]]
func.func @test_powui_const_five(%base: !PF) -> !PF {
  %exp = arith.constant 5 : i32
  %res = field.powui %base, %exp : !PF, i32
  return %res : !PF
}

// exp = 6 → Fermat: 6 % 6 = 0 → identity (x⁶ ≡ 1 mod 7 by Fermat)
// CHECK-LABEL: @test_powui_const_six
// CHECK-SAME: (%{{.*}}: [[T:.*]]) -> [[T]]
// CHECK-NEXT: %[[ONE:.*]] = mod_arith.constant 1 : [[T]]
// CHECK-NEXT: return %[[ONE]] : [[T]]
func.func @test_powui_const_six(%base: !PF) -> !PF {
  %exp = arith.constant 6 : i32
  %res = field.powui %base, %exp : !PF, i32
  return %res : !PF
}

// exp = 7 → Fermat: 7 % 6 = 1 → base (x⁷ ≡ x mod 7 by Fermat)
// CHECK-LABEL: @test_powui_const_seven
// CHECK-SAME: (%[[BASE:.*]]: [[T:.*]]) -> [[T]]
// CHECK-NEXT: return %[[BASE]] : [[T]]
func.func @test_powui_const_seven(%base: !PF) -> !PF {
  %exp = arith.constant 7 : i32
  %res = field.powui %base, %exp : !PF, i32
  return %res : !PF
}

// Dynamic exponent: must produce scf.while loop
// CHECK-LABEL: @test_powui_dynamic
// CHECK-NOT: mod_arith.square
// CHECK: scf.while
// CHECK: scf.condition
// CHECK: mod_arith.square
// CHECK: scf.if
// CHECK: mod_arith.mul
func.func @test_powui_dynamic(%base: !PF, %exp: i32) -> !PF {
  %res = field.powui %base, %exp : !PF, i32
  return %res : !PF
}

// A modulus wider than the exponent type must still reach the constant path.
// The exponent is reduced mod p-1 before any widening, so the literal stays
// visible; reducing after an arith.extui hid it and fell back to the loop.

// exp = 7 (111₂), p-1 = 2⁶⁴-2³² ≫ 7 → no reduction → 2 squares + 2 muls
// CHECK-LABEL: @test_powui_wide_modulus_const_seven
// CHECK-SAME: (%[[BASE:.*]]: [[T:.*]]) -> [[T]]
// CHECK-NEXT: %[[SQ1:.*]] = mod_arith.square %[[BASE]] : [[T]]
// CHECK-NEXT: %[[MUL1:.*]] = mod_arith.mul %[[BASE]], %[[SQ1]] : [[T]]
// CHECK-NEXT: %[[SQ2:.*]] = mod_arith.square %[[SQ1]] : [[T]]
// CHECK-NEXT: %[[MUL2:.*]] = mod_arith.mul %[[MUL1]], %[[SQ2]] : [[T]]
// CHECK-NEXT: return %[[MUL2]] : [[T]]
func.func @test_powui_wide_modulus_const_seven(%base: !GL) -> !GL {
  %exp = arith.constant 7 : i32
  %res = field.powui %base, %exp : !GL, i32
  return %res : !GL
}

// The order is p³-1 here, wider still than both the modulus and the exponent.
// CHECK-LABEL: @test_powui_wide_extension_const_five
// CHECK-NOT: scf.while
// CHECK: return
func.func @test_powui_wide_extension_const_five(%base: !GL3) -> !GL3 {
  %exp = arith.constant 5 : i32
  %res = field.powui %base, %exp : !GL3, i32
  return %res : !GL3
}

// unroll = false keeps the loop even though the exponent is a literal. The
// reduced exponent still reaches it as a constant — 3 is 7 >> 1, the trip
// counter after the loop's first-iteration unrolling.
// CHECK-LABEL: @test_powui_const_no_unroll
// CHECK-NOT: mod_arith.square
// CHECK: %[[C3:.*]] = arith.constant 3 : i64
// CHECK: scf.while (%{{.*}} = %[[C3]]
// CHECK: mod_arith.square
// CHECK: scf.if
// CHECK: mod_arith.mul
func.func @test_powui_const_no_unroll(%base: !GL) -> !GL {
  %exp = arith.constant 7 : i32
  %res = field.powui %base, %exp {unroll = false} : !GL, i32
  return %res : !GL
}
