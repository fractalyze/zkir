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

// Shared Ed25519 type aliases for MLIR tests.

// ===== Field types =====

// Curve25519 base field: p = 2^255 - 19
!EdBF = !field.pf<57896044618658097711785492504343953926634992332820282019728792003956564819949:i256>

// ===== Ed25519 EC types (standard) =====

// a = -1 mod p, d = -121665/121666 mod p, G = (Gx, Gy)
#ed25519 = #elliptic_curve.te<57896044618658097711785492504343953926634992332820282019728792003956564819948:i256, 37095705934669439343138083508754565189542113879843219016388785533085940283555:i256, (15112221349535400772501151409588531511454012693041857206046113283949847762202:i256, 46316835694926478169428394003475163141307993866256225615783033603165251855960:i256)> : !EdBF
!ed_affine = !elliptic_curve.ed_affine<#ed25519>
!ed_extended = !elliptic_curve.ed_extended<#ed25519>
