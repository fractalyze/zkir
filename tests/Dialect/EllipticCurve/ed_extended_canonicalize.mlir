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
// RUN:   | prime-ir-opt -canonicalize \
// RUN:   | FileCheck %s -enable-var-scope

// CHECK-LABEL: @test_ed_add_self_is_double
// CHECK-SAME: (%[[ARG0:.*]]: [[EDEXT:.*]]) -> [[EDEXT]] {
func.func @test_ed_add_self_is_double(%point: !ed_extended) -> !ed_extended {
  // CHECK: %[[DOUBLE:.*]] = elliptic_curve.double %[[ARG0]] : [[EDEXT]] -> [[EDEXT]]
  // CHECK-NOT: elliptic_curve.add
  // CHECK: return %[[DOUBLE]] : [[EDEXT]]
  %sum = elliptic_curve.add %point, %point : !ed_extended, !ed_extended -> !ed_extended
  return %sum : !ed_extended
}

// CHECK-LABEL: @test_ed_add_after_neg_lhs
// CHECK-SAME: (%[[ARG0:.*]]: [[EDEXT:.*]], %[[ARG1:.*]]: [[EDEXT]]) -> [[EDEXT]] {
func.func @test_ed_add_after_neg_lhs(%point1: !ed_extended, %point2: !ed_extended) -> !ed_extended {
  // CHECK: %[[SUB:.*]] = elliptic_curve.sub %[[ARG1]], %[[ARG0]] : [[EDEXT]], [[EDEXT]] -> [[EDEXT]]
  // CHECK-NOT: elliptic_curve.add
  // CHECK-NOT: elliptic_curve.negate
  // CHECK: return %[[SUB]] : [[EDEXT]]
  %neg = elliptic_curve.negate %point1 : !ed_extended
  %sum = elliptic_curve.add %neg, %point2 : !ed_extended, !ed_extended -> !ed_extended
  return %sum : !ed_extended
}

// CHECK-LABEL: @test_ed_double_negate
// CHECK-SAME: (%[[ARG0:.*]]: [[EDEXT:.*]]) -> [[EDEXT]] {
func.func @test_ed_double_negate(%point: !ed_extended) -> !ed_extended {
  // CHECK: return %[[ARG0]] : [[EDEXT]]
  %neg1 = elliptic_curve.negate %point : !ed_extended
  %neg2 = elliptic_curve.negate %neg1 : !ed_extended
  return %neg2 : !ed_extended
}

// CHECK-LABEL: @test_ed_sub_after_neg_rhs
// CHECK-SAME: (%[[ARG0:.*]]: [[EDEXT:.*]], %[[ARG1:.*]]: [[EDEXT]]) -> [[EDEXT]] {
func.func @test_ed_sub_after_neg_rhs(%point1: !ed_extended, %point2: !ed_extended) -> !ed_extended {
  // CHECK: %[[ADD:.*]] = elliptic_curve.add %[[ARG0]], %[[ARG1]] : [[EDEXT]], [[EDEXT]] -> [[EDEXT]]
  // CHECK-NOT: elliptic_curve.sub
  // CHECK-NOT: elliptic_curve.negate
  // CHECK: return %[[ADD]] : [[EDEXT]]
  %neg = elliptic_curve.negate %point2 : !ed_extended
  %diff = elliptic_curve.sub %point1, %neg : !ed_extended, !ed_extended -> !ed_extended
  return %diff : !ed_extended
}

// The cases above take function arguments, so they never reach the constant
// folder -- a separate path, and the only one that read the coordinate count.
// These fold real constants and check the resulting values: the wrong answer
// was a well-formed constant of the right type, so checking that folding
// happened proves nothing.
//
// The two extended results are projective, so their coordinates depend on the
// formula's route, not only on the point. Each was verified by normalizing
// (X/Z, Y/Z) and checking it equals the expected multiple of the base point,
// satisfies -x^2 + y^2 = 1 + d*x^2*y^2, and preserves T*Z == X*Y. A change to
// the doubling or addition formula legitimately changes these numbers;
// re-derive them the same way rather than copying whatever the new output is.

// double(G) == 2G. Exercises foldUnaryPointOp.
// CHECK-LABEL: @test_ed_fold_double_constant
func.func @test_ed_fold_double_constant() -> !ed_extended {
  // CHECK-NOT: elliptic_curve.double
  // CHECK: elliptic_curve.constant dense<[22227142146053615383686711456592054533481723065238328079491086165754688571991, 23132612897935763947376118816302936961945753855592497212527330206034714001367, 47730969525411543486323345491594608200968822454103728746637676179095643807339, 10919983009863980608562433598283441687065789490543687699070727834902457043353]>
  %g = elliptic_curve.constant dense<[15112221349535400772501151409588531511454012693041857206046113283949847762202, 46316835694926478169428394003475163141307993866256225615783033603165251855960, 1, 46827403850823179245072216630277197565144205554125654976674165829533817101731]> : !ed_extended
  %result = elliptic_curve.double %g : !ed_extended -> !ed_extended
  return %result : !ed_extended
}

// G + 2G == 3G. Exercises GenericEllipticCurveBinaryFolder.
// CHECK-LABEL: @test_ed_fold_add_constants
func.func @test_ed_fold_add_constants() -> !ed_extended {
  // CHECK-NOT: elliptic_curve.add
  // CHECK: elliptic_curve.constant dense<[8730208168533972070470970288646983225609249201301384357661140266392926273419, 1975756425246993437738006527812324286200101983770209055058433275404672302437, 449866727041981856910711739926099867055731189810398506275864610260399601655, 41541903736245094508289808986107247391095121702608974599969833236659985164713]>
  %g = elliptic_curve.constant dense<[15112221349535400772501151409588531511454012693041857206046113283949847762202, 46316835694926478169428394003475163141307993866256225615783033603165251855960, 1, 46827403850823179245072216630277197565144205554125654976674165829533817101731]> : !ed_extended
  %g2 = elliptic_curve.constant dense<[24727413235106541002554574571675588834622768167397638456726423682521233608206, 15549675580280190176352668710449542251549572066445060580507079593062643049417, 1, 16552979481334663544878610556091376071931149008662153799327195285289362371585]> : !ed_extended
  %result = elliptic_curve.add %g, %g2 : !ed_extended, !ed_extended -> !ed_extended
  return %result : !ed_extended
}

// negate(G) == (-x, y). Affine is canonical, so this pins an exact point, and
// it covers ed_affine.
// CHECK-LABEL: @test_ed_fold_negate_affine_constant
func.func @test_ed_fold_negate_affine_constant() -> !ed_affine {
  // CHECK-NOT: elliptic_curve.negate
  // CHECK: elliptic_curve.constant dense<[42783823269122696939284341094755422415180979639778424813682678720006717057747, 46316835694926478169428394003475163141307993866256225615783033603165251855960]>
  %g = elliptic_curve.constant dense<[15112221349535400772501151409588531511454012693041857206046113283949847762202, 46316835694926478169428394003475163141307993866256225615783033603165251855960]> : !ed_affine
  %result = elliptic_curve.negate %g : !ed_affine
  return %result : !ed_affine
}
