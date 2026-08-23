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

// RUN: prime-ir-opt -split-input-file -verify-diagnostics %s

// Every case here reaches curve arithmetic, which resolves coordinates through
// the base prime field and compares APInts against the modulus. Anything the
// parser lets through mismatched aborts instead of reporting, so these all have
// to be rejected while a diagnostic is still possible.

// A coordinate literal defaults to i64 while the base field stores i32.
// expected-error@+1 {{expected 'i32' attribute for gY, but got 'i64'}}
#sw_width = #elliptic_curve.sw<0:i32, 3:i32, (2412:i32, 1)> : !field.pf<7681 : i32>

// -----

// Generator not on y² = x³ + ax + b.
// expected-error@+1 {{a, b, gX, and gY must satisfy the equation}}
#sw_off_curve = #elliptic_curve.sw<0:i32, 3:i32, (2412:i32, 1:i32)> : !field.pf<7681 : i32>

// -----

// Binary fields have no base prime field to resolve coordinates through.
// expected-error@+1 {{elliptic curve base field must be a prime or extension field, but got '!field.bf<7>'}}
#sw_binary_base = #elliptic_curve.sw<0:i8, 3:i8, (1:i8, 2:i8)> : !field.bf<7>

// -----

!pf = !field.pf<21888242871839275222246405745257275088696311157297823662689037894645226208583:i256>
!ef = !field.ef<2x!pf, 21888242871839275222246405745257275088696311157297823662689037894645226208582:i256>

// An Fp2 coefficient vector carries one entry per degree; three is not a
// coordinate over this field.
// expected-error@+1 {{expected 'tensor<2xi256>' attribute for a, but got 'tensor<3xi256>'}}
#sw_ef_shape = #elliptic_curve.sw<dense<0> : tensor<3xi256>, dense<0> : tensor<2xi256>, (dense<0> : tensor<2xi256>, dense<0> : tensor<2xi256>)> : !ef

// -----

// Twisted Edwards runs the same validation before its own curve equation.
// expected-error@+1 {{expected 'i32' attribute for gX, but got 'i64'}}
#te_width = #elliptic_curve.te<1:i32, 2:i32, (1, 0:i32)> : !field.pf<7681 : i32>

// -----

// Twisted Edwards requires a non-zero 'a'.
// expected-error@+1 {{twisted Edwards parameter 'a' must be non-zero}}
#te_zero_a = #elliptic_curve.te<0:i32, 2:i32, (1:i32, 0:i32)> : !field.pf<7681 : i32>

// -----

// ... and a non-zero 'd', or the curve degenerates to a conic.
// expected-error@+1 {{twisted Edwards parameter 'd' must be non-zero}}
#te_zero_d = #elliptic_curve.te<1:i32, 0:i32, (1:i32, 0:i32)> : !field.pf<7681 : i32>

// -----

// Generator not on a·x² + y² = 1 + d·x²y². The Weierstrass analogue is
// #sw_off_curve above; this is the twisted Edwards equation, which takes 'd'
// where that one takes 'b'.
// expected-error@+1 {{a, d, gX, and gY must satisfy the equation}}
#te_off_curve = #elliptic_curve.te<1:i32, 2:i32, (3:i32, 4:i32)> : !field.pf<7681 : i32>
