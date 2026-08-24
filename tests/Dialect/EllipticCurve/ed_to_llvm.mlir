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
// RUN:   | prime-ir-opt -convert-to-llvm \
// RUN:   | FileCheck %s -enable-var-scope

// The twisted Edwards point types have to reach LLVM like the short Weierstrass
// ones do. When only the Weierstrass types were registered with the LLVM type
// converter, an Edwards type converted to null instead of failing, and the first
// symptom was a null dereference in MemRefDescriptor::fromStaticShape during
// XLA's CPU fusion lowering -- a SIGSEGV with no diagnostic.

!ed_affine = !elliptic_curve.ed_affine<#ed25519>

// A 2-coordinate Edwards point lowers to the same flat struct shape a
// 2-coordinate Weierstrass affine point does.
// CHECK-LABEL: @test_ed_from_coords
func.func @test_ed_from_coords(%v1: i256, %v2: i256, %v3: i256, %v4: i256) {
  // CHECK-NOT: elliptic_curve.from_coords
  // CHECK: llvm.insertvalue
  %aff = elliptic_curve.from_coords %v1, %v2
      : (i256, i256) -> !ed_affine
  %ext = elliptic_curve.from_coords %v1, %v2, %v3, %v4
      : (i256, i256, i256, i256) -> !ed_extended
  return
}

// CHECK-LABEL: @test_ed_to_coords
func.func @test_ed_to_coords(%aff: !ed_affine, %ext: !ed_extended) {
  // CHECK-NOT: elliptic_curve.to_coords
  // CHECK: llvm.extractvalue
  %aff_coords:2 = elliptic_curve.to_coords %aff
      : (!ed_affine) -> (i256, i256)
  %ext_coords:4 = elliptic_curve.to_coords %ext
      : (!ed_extended) -> (i256, i256, i256, i256)
  return
}

// The signature itself is the regression: an unregistered element type made the
// converted function type null rather than reporting a failure.
// CHECK-LABEL: @test_ed_extended_signature
// CHECK-SAME: !llvm.struct<(i256, i256, i256, i256)>
func.func @test_ed_extended_signature(%ext: !ed_extended) -> !ed_extended {
  return %ext : !ed_extended
}

// CHECK-LABEL: @test_ed_affine_signature
// CHECK-SAME: !llvm.struct<(i256, i256)>
func.func @test_ed_affine_signature(%aff: !ed_affine) -> !ed_affine {
  return %aff : !ed_affine
}
