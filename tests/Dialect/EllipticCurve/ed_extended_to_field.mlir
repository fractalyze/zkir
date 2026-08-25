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

// RUN: cat %S/../../ed25519_defs.mlir %s \
// RUN:   | prime-ir-opt -elliptic-curve-to-field \
// RUN:   | FileCheck %s -enable-var-scope

// Test: Edwards extended point addition lowers to field ops.
// CHECK-LABEL: @test_ed_extended_add
func.func @test_ed_extended_add(%p1: !ed_extended, %p2: !ed_extended) -> !ed_extended {
  // CHECK-NOT: elliptic_curve.add
  // CHECK: field.mul
  // CHECK: field.add
  %result = elliptic_curve.add %p1, %p2 : !ed_extended, !ed_extended -> !ed_extended
  return %result : !ed_extended
}

// Test: Edwards extended point doubling lowers to field ops.
// CHECK-LABEL: @test_ed_extended_double
func.func @test_ed_extended_double(%p1: !ed_extended) -> !ed_extended {
  // CHECK-NOT: elliptic_curve.double
  // CHECK: field.mul
  %result = elliptic_curve.double %p1 : !ed_extended -> !ed_extended
  return %result : !ed_extended
}

// Test: Edwards negation negates X (coord 0) and T (coord 3), preserving Y and Z.
// CHECK-LABEL: @test_ed_extended_negate
func.func @test_ed_extended_negate(%p1: !ed_extended) -> !ed_extended {
  // CHECK-NOT: elliptic_curve.negate
  // CHECK: [[COORDS:%.*]]:4 = elliptic_curve.to_coords
  // CHECK: [[NEG_X:%.*]] = field.negate [[COORDS]]#0
  // CHECK: [[NEG_T:%.*]] = field.negate [[COORDS]]#3
  // CHECK: elliptic_curve.from_coords [[NEG_X]], [[COORDS]]#1, [[COORDS]]#2, [[NEG_T]]
  %result = elliptic_curve.negate %p1 : !ed_extended
  return %result : !ed_extended
}

// Test: Edwards subtraction lowers (via negate + add).
// CHECK-LABEL: @test_ed_extended_sub
func.func @test_ed_extended_sub(%p1: !ed_extended, %p2: !ed_extended) -> !ed_extended {
  // CHECK-NOT: elliptic_curve.sub
  %result = elliptic_curve.sub %p1, %p2 : !ed_extended, !ed_extended -> !ed_extended
  return %result : !ed_extended
}

// Test: Edwards constant point lowers.
// CHECK-LABEL: @test_ed_extended_constant
func.func @test_ed_extended_constant() {
  // CHECK-NOT: elliptic_curve.constant
  // CHECK: field.constant
  // T = Gx * Gy mod p (extended coordinate invariant when Z = 1).
  %gen = elliptic_curve.constant dense<[15112221349535400772501151409588531511454012693041857206046113283949847762202, 46316835694926478169428394003475163141307993866256225615783033603165251855960, 1, 46827403850823179245072216630277197565144205554125654976674165829533817101731]> : !ed_extended
  return
}

// The twisted Edwards affine representation has to reach the same paths the
// short Weierstrass affine one does. Both cases below used to be decided by an
// isa<AffineType> test, which names only the Weierstrass type: mixed addition
// hit an llvm_unreachable, and scalar multiply skipped the widening and left an
// affine value where the loop expected the projective output type.

!EdSF = !field.pf<7237005577332262213973186563042994240857116359379907606001950938285454250989:i256>

// Test: mixed addition, extended + affine -> extended.
// CHECK-LABEL: @test_ed_mixed_add
func.func @test_ed_mixed_add(%p1: !ed_extended, %p2: !ed_affine) -> !ed_extended {
  // CHECK-NOT: elliptic_curve.add
  // CHECK: field.mul
  %result = elliptic_curve.add %p1, %p2 : !ed_extended, !ed_affine -> !ed_extended
  return %result : !ed_extended
}

// Test: affine scalar multiply widens to extended.
// CHECK-LABEL: @test_ed_affine_scalar_mul
func.func @test_ed_affine_scalar_mul(%s: !EdSF, %p: !ed_affine) -> !ed_extended {
  // CHECK-NOT: elliptic_curve.scalar_mul
  %result = elliptic_curve.scalar_mul %s, %p : !EdSF, !ed_affine -> !ed_extended
  return %result : !ed_extended
}

// Test: affine + affine also widens (no mixed operand involved).
// CHECK-LABEL: @test_ed_affine_add
func.func @test_ed_affine_add(%p1: !ed_affine, %p2: !ed_affine) -> !ed_extended {
  // CHECK-NOT: elliptic_curve.add
  // CHECK: field.mul
  %result = elliptic_curve.add %p1, %p2 : !ed_affine, !ed_affine -> !ed_extended
  return %result : !ed_extended
}
