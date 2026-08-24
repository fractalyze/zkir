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

// RUN: prime-ir-opt -split-input-file -verify-diagnostics %s | FileCheck %s

// The group-law rule is stated once over point kinds, so it has to hold for
// twisted Edwards the same way it does for short Weierstrass: affine is not
// closed, and the result widens to the projective kind of its own family —
// extended here, where short Weierstrass would give jacobian or xyzz.

!EdBF = !field.pf<57896044618658097711785492504343953926634992332820282019728792003956564819949:i256>
#ed25519 = #elliptic_curve.te<57896044618658097711785492504343953926634992332820282019728792003956564819948:i256, 37095705934669439343138083508754565189542113879843219016388785533085940283555:i256, (15112221349535400772501151409588531511454012693041857206046113283949847762202:i256, 46316835694926478169428394003475163141307993866256225615783033603165251855960:i256)> : !EdBF
!EdSF = !field.pf<7237005577332262213973186563042994240857116359379907606001950938285454250989:i256>
!ed_affine = !elliptic_curve.ed_affine<#ed25519>
!ed_extended = !elliptic_curve.ed_extended<#ed25519>

// CHECK-LABEL: @ed_affine_add_widens_to_extended
func.func @ed_affine_add_widens_to_extended(%a: !ed_affine, %b: !ed_affine) -> !ed_extended {
  // CHECK: elliptic_curve.add
  %r = elliptic_curve.add %a, %b : !ed_affine, !ed_affine -> !ed_extended
  return %r : !ed_extended
}

// CHECK-LABEL: @ed_mixed_add_keeps_extended
func.func @ed_mixed_add_keeps_extended(%a: !ed_affine, %b: !ed_extended) -> !ed_extended {
  // CHECK: elliptic_curve.add
  %r = elliptic_curve.add %a, %b : !ed_affine, !ed_extended -> !ed_extended
  return %r : !ed_extended
}

// CHECK-LABEL: @ed_extended_sub
func.func @ed_extended_sub(%a: !ed_extended, %b: !ed_extended) -> !ed_extended {
  // CHECK: elliptic_curve.sub
  %r = elliptic_curve.sub %a, %b : !ed_extended, !ed_extended -> !ed_extended
  return %r : !ed_extended
}

// CHECK-LABEL: @ed_affine_double_widens
func.func @ed_affine_double_widens(%a: !ed_affine) -> !ed_extended {
  // CHECK: elliptic_curve.double
  %r = elliptic_curve.double %a : !ed_affine -> !ed_extended
  return %r : !ed_extended
}

// CHECK-LABEL: @ed_affine_scalar_mul_widens
func.func @ed_affine_scalar_mul_widens(%s: !EdSF, %p: !ed_affine) -> !ed_extended {
  // CHECK: elliptic_curve.scalar_mul
  %r = elliptic_curve.scalar_mul %s, %p : !EdSF, !ed_affine -> !ed_extended
  return %r : !ed_extended
}

// -----

!EdBF = !field.pf<57896044618658097711785492504343953926634992332820282019728792003956564819949:i256>
#ed25519 = #elliptic_curve.te<57896044618658097711785492504343953926634992332820282019728792003956564819948:i256, 37095705934669439343138083508754565189542113879843219016388785533085940283555:i256, (15112221349535400772501151409588531511454012693041857206046113283949847762202:i256, 46316835694926478169428394003475163141307993866256225615783033603165251855960:i256)> : !EdBF
!ed_affine = !elliptic_curve.ed_affine<#ed25519>

// Affine is no more closed under the group law here than it is for short
// Weierstrass, so an affine result has to be rejected rather than silently
// producing a point off the representation.
func.func @ed_affine_result_is_not_closed(%a: !ed_affine, %b: !ed_affine) -> !ed_affine {
  // expected-error @+1 {{input or output types are wrong}}
  %r = elliptic_curve.add %a, %b : !ed_affine, !ed_affine -> !ed_affine
  return %r : !ed_affine
}

// -----

!EdBF = !field.pf<57896044618658097711785492504343953926634992332820282019728792003956564819949:i256>
#ed25519 = #elliptic_curve.te<57896044618658097711785492504343953926634992332820282019728792003956564819948:i256, 37095705934669439343138083508754565189542113879843219016388785533085940283555:i256, (15112221349535400772501151409588531511454012693041857206046113283949847762202:i256, 46316835694926478169428394003475163141307993866256225615783033603165251855960:i256)> : !EdBF
#sw = #elliptic_curve.sw<0:i256, 7:i256, (2:i256, 2439533663544638029669143078078524294665610446926061315433185154815338255773:i256)> : !EdBF
!ed_extended = !elliptic_curve.ed_extended<#ed25519>
!sw_affine = !elliptic_curve.affine<#sw>

// The two families have disjoint coordinate systems. Worth pinning because the
// affine-operand arm matches on the *other* operand's type, so a mixed pair
// lines up with it structurally even though the operation is meaningless.
func.func @families_do_not_mix(%a: !sw_affine, %b: !ed_extended) -> !ed_extended {
  // expected-error @+1 {{cannot mix short Weierstrass and twisted Edwards points}}
  %r = elliptic_curve.add %a, %b : !sw_affine, !ed_extended -> !ed_extended
  return %r : !ed_extended
}

// -----

!EdBF = !field.pf<57896044618658097711785492504343953926634992332820282019728792003956564819949:i256>
#ed25519 = #elliptic_curve.te<57896044618658097711785492504343953926634992332820282019728792003956564819948:i256, 37095705934669439343138083508754565189542113879843219016388785533085940283555:i256, (15112221349535400772501151409588531511454012693041857206046113283949847762202:i256, 46316835694926478169428394003475163141307993866256225615783033603165251855960:i256)> : !EdBF
!EdSF = !field.pf<7237005577332262213973186563042994240857116359379907606001950938285454250989:i256>
!ed_affine = !elliptic_curve.ed_affine<#ed25519>

// No backend carries a twisted-Edwards MSM, so it is rejected here rather than
// at a lowering that cannot express it.
func.func @msm_rejects_edwards(%s: tensor<4x!EdSF>, %p: tensor<4x!ed_affine>) -> !ed_affine {
  // expected-error @+1 {{MSM does not support twisted Edwards points}}
  %r = elliptic_curve.msm %s, %p degree=1 : tensor<4x!EdSF>, tensor<4x!ed_affine> -> !ed_affine
  return %r : !ed_affine
}
