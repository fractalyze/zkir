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

#ifndef PRIME_IR_DIALECT_ELLIPTICCURVE_IR_POINTKIND_H_
#define PRIME_IR_DIALECT_ELLIPTICCURVE_IR_POINTKIND_H_

#include "zk_dtypes/include/geometry/point_declarations.h"

namespace mlir::prime_ir::elliptic_curve {

// TODO(chokobole): Rename kAffine→kSwAffine, kJacobian→kSwJacobian,
// kXYZZ→kSwXYZZ (and corresponding MLIR types AffineType→SwAffineType, etc.)
// for naming consistency with the Ed-prefixed variants. Separate refactor PR.
enum class PointKind {
  kAffine,
  kJacobian,
  kXYZZ,
  kEdAffine,
  kEdExtended,
};

// Written as default-less switches rather than equality chains: a chain answers
// "no" for a representation added later, which is a silent misclassification,
// whereas an unhandled case here fails the build. //bazel:prime_ir_cc.bzl puts
// -Werror=all on every prime_ir target and -Wall implies -Wswitch, so adding a
// PointKind without updating these is a compile error, not a warning.
//
// The absence of `default:` is therefore load-bearing. Do not add one "for
// safety": it would silence that check at every switch over this enum.
constexpr bool isEdwards(PointKind kind) {
  switch (kind) {
  case PointKind::kEdAffine:
  case PointKind::kEdExtended:
    return true;
  case PointKind::kAffine:
  case PointKind::kJacobian:
  case PointKind::kXYZZ:
    return false;
  }
  return false;
}

// Affine in either family. Affine is not closed under the group law, so these
// are the kinds an add/sub/double/scalar-mul result has to widen away from.
constexpr bool isAffine(PointKind kind) {
  switch (kind) {
  case PointKind::kAffine:
  case PointKind::kEdAffine:
    return true;
  case PointKind::kJacobian:
  case PointKind::kXYZZ:
  case PointKind::kEdExtended:
    return false;
  }
  return false;
}

// The two families have disjoint coordinate systems, so a group operation
// mixing them is meaningless however the individual kinds line up.
constexpr bool isSameFamily(PointKind lhs, PointKind rhs) {
  return isEdwards(lhs) == isEdwards(rhs);
}

// Not injective, so it cannot be inverted: ed_affine is 2 coordinates like
// affine, and ed_extended is 4 like xyzz. Ask a type for its kind rather
// than deriving one from a count.
constexpr size_t getNumCoords(PointKind kind) {
  switch (kind) {
  case PointKind::kAffine:
    return 2;
  case PointKind::kJacobian:
    return 3;
  case PointKind::kXYZZ:
    return 4;
  case PointKind::kEdAffine:
    return 2;
  case PointKind::kEdExtended:
    return 4;
  }
  // Unreachable for any declared kind; falling off the end would be UB.
  return 0;
}

template <typename Point>
constexpr PointKind getPointKind() {
  if constexpr (zk_dtypes::IsAffinePoint<Point> &&
                Point::Curve::kType == zk_dtypes::CurveType::kTwistedEdwards) {
    return PointKind::kEdAffine;
  } else if constexpr (zk_dtypes::IsAffinePoint<Point>) {
    return PointKind::kAffine;
  } else if constexpr (zk_dtypes::IsJacobianPoint<Point>) {
    return PointKind::kJacobian;
  } else if constexpr (zk_dtypes::IsPointXyzz<Point>) {
    return PointKind::kXYZZ;
  } else {
    static_assert(zk_dtypes::IsExtendedPoint<Point>,
                  "Point must be an extended point");
    return PointKind::kEdExtended;
  }
}

} // namespace mlir::prime_ir::elliptic_curve

#endif // PRIME_IR_DIALECT_ELLIPTICCURVE_IR_POINTKIND_H_
