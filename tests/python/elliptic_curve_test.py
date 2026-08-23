from absl.testing import absltest

from prime_ir.mlir.ir import *
from prime_ir.mlir.dialects import field
from prime_ir.mlir.dialects import elliptic_curve


# BN254 base field modulus (Fp) - used for G1/G2 point coordinates (256-bit)
BN254_BASE_FIELD_MODULUS = 21888242871839275222246405745257275088696311157297823662689037894645226208583

# BN254 scalar field modulus (Fr) - used for scalars
BN254_SCALAR_FIELD_MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617

# BN254 G1 curve parameters: y² = x³ + 3
BN254_G1_A = 0
BN254_G1_B = 3

# BN254 G1 generator point
BN254_G1_GX = 1
BN254_G1_GY = 2

# BN254 G2 curve parameters (in Fp2, each coordinate has 2 elements [c0, c1])
# Non-residue for Fp2 (from bn254_field_defs.mlir)
BN254_FP2_NON_RESIDUE = 21888242871839275222246405745257275088696311157297823662689037894645226208582

# G2 a = 0 (in Fp2)
BN254_G2_A = [0, 0]

# G2 b = 3 / (9 + u) where u is the non-residue
# In standard form: b = (19485874751759354771024239261021720505790618469301721065564631296452457478373, 266929791119991161246907387137283842545076965332900288569378510910307636690)
BN254_G2_B = [
    19485874751759354771024239261021720505790618469301721065564631296452457478373,
    266929791119991161246907387137283842545076965332900288569378510910307636690,
]

# G2 generator point in standard form
BN254_G2_GX = [
    10857046999023057135944570762232829481370756359578518086990519993285655852781,
    11559732032986387107991004021392285783925812861821192530917403151452391805634,
]
BN254_G2_GY = [
    8495653923123431417604973247489272438418190587263600148770280649306958101930,
    4082367875863433681332203403145435568316851327593401208105741076214120093531,
]


def _create_prime_field(ctx, modulus_value):
  """Create a 256-bit canonical prime field type for curve coordinates."""
  i256 = IntegerType.get_unsigned(256)
  modulus = IntegerAttr.get(i256, modulus_value)
  return field.PrimeFieldType.get(modulus, False, ctx)


def _create_bn254_prime_field(ctx):
  return _create_prime_field(ctx, BN254_BASE_FIELD_MODULUS)


def _create_ed25519_prime_field(ctx):
  return _create_prime_field(ctx, ED25519_BASE_FIELD_MODULUS)


def _create_bn254_extension_field(ctx, prime_field):
  """Create BN254 extension field (Fq2) type."""
  i256 = IntegerType.get_unsigned(256)
  non_residue = IntegerAttr.get(i256, BN254_FP2_NON_RESIDUE)
  return field.ExtensionFieldType.get(2, prime_field, non_residue, ctx)


def _create_bn254_g1_curve(ctx, base_field):
  """Create BN254 G1 curve attribute."""
  i256 = IntegerType.get_unsigned(256)
  a = IntegerAttr.get(i256, BN254_G1_A)
  b = IntegerAttr.get(i256, BN254_G1_B)
  gx = IntegerAttr.get(i256, BN254_G1_GX)
  gy = IntegerAttr.get(i256, BN254_G1_GY)
  return elliptic_curve.ShortWeierstrassAttr.get(base_field, a, b, gx, gy, ctx)


def _create_fp2_attr(ctx, values):
  """Create Fp2 element as DenseIntElementsAttr with shape [2].

  Uses DenseElementsAttr.get() with a list of IntegerAttr objects to support
  large integers (>64-bit).
  """
  i256 = IntegerType.get_unsigned(256)
  tensor_type = RankedTensorType.get([2], i256)
  # Create IntegerAttr for each element (IntegerAttr supports large integers)
  attrs = [IntegerAttr.get(i256, v) for v in values]
  return DenseElementsAttr.get(attrs, type=tensor_type)


def _create_bn254_g2_curve(ctx, base_field):
  """Create BN254 G2 curve attribute."""
  a = _create_fp2_attr(ctx, BN254_G2_A)
  b = _create_fp2_attr(ctx, BN254_G2_B)
  gx = _create_fp2_attr(ctx, BN254_G2_GX)
  gy = _create_fp2_attr(ctx, BN254_G2_GY)
  return elliptic_curve.ShortWeierstrassAttr.get(base_field, a, b, gx, gy, ctx)


# Ed25519 (RFC 8032 §5.1): a twisted Edwards curve a·x² + y² = 1 + d·x²y² over
# GF(2^255 - 19), with a = -1 and d = -121665/121666.
ED25519_BASE_FIELD_MODULUS = 2**255 - 19
ED25519_A = ED25519_BASE_FIELD_MODULUS - 1
ED25519_D = 37095705934669439343138083508754565189542113879843219016388785533085940283555
ED25519_GX = 15112221349535400772501151409588531511454012693041857206046113283949847762202
ED25519_GY = 46316835694926478169428394003475163141307993866256225615783033603165251855960


def _create_ed25519_curve(ctx, base_field):
  """Create the Ed25519 twisted Edwards curve attribute."""
  i256 = IntegerType.get_unsigned(256)
  a = IntegerAttr.get(i256, ED25519_A)
  d = IntegerAttr.get(i256, ED25519_D)
  gx = IntegerAttr.get(i256, ED25519_GX)
  gy = IntegerAttr.get(i256, ED25519_GY)
  return elliptic_curve.TwistedEdwardsAttr.get(base_field, a, d, gx, gy, ctx)


class EllipticCurveTest(absltest.TestCase):

  def testG1ShortWeierstrassAttr(self):
    """Test creating and reading G1 ShortWeierstrass curve attributes."""
    with Context() as ctx, Location.unknown():
      field.register_dialect(ctx)
      elliptic_curve.register_dialect(ctx)

      base_field = _create_bn254_prime_field(ctx)
      curve = _create_bn254_g1_curve(ctx, base_field)

      # Verify we can read back the curve parameters
      self.assertEqual(curve.base_field, base_field)
      self.assertEqual(int(IntegerAttr(curve.a)), BN254_G1_A)
      self.assertEqual(int(IntegerAttr(curve.b)), BN254_G1_B)
      self.assertEqual(int(IntegerAttr(curve.gx)), BN254_G1_GX)
      self.assertEqual(int(IntegerAttr(curve.gy)), BN254_G1_GY)

  def testG2ShortWeierstrassAttr(self):
    """Test creating and reading G2 ShortWeierstrass curve attributes with Fp2."""
    with Context() as ctx, Location.unknown():
      field.register_dialect(ctx)
      elliptic_curve.register_dialect(ctx)

      prime_field = _create_bn254_prime_field(ctx)
      ext_field = _create_bn254_extension_field(ctx, prime_field)
      curve = _create_bn254_g2_curve(ctx, ext_field)

      # Verify we can read back the curve parameters
      self.assertEqual(curve.base_field, ext_field)

      # Verify G2 coefficients by comparing attributes
      # (DenseIntElementsAttr iteration doesn't support 256-bit integers)
      expected_a = _create_fp2_attr(ctx, BN254_G2_A)
      expected_b = _create_fp2_attr(ctx, BN254_G2_B)
      expected_gx = _create_fp2_attr(ctx, BN254_G2_GX)
      expected_gy = _create_fp2_attr(ctx, BN254_G2_GY)

      self.assertEqual(curve.a, expected_a)
      self.assertEqual(curve.b, expected_b)
      self.assertEqual(curve.gx, expected_gx)
      self.assertEqual(curve.gy, expected_gy)

  def testG1AffineType(self):
    """Test creating G1 affine point types."""
    with Context() as ctx, Location.unknown():
      field.register_dialect(ctx)
      elliptic_curve.register_dialect(ctx)

      base_field = _create_bn254_prime_field(ctx)
      curve = _create_bn254_g1_curve(ctx, base_field)

      # Create affine point type
      affine_type = elliptic_curve.AffineType.get(curve, ctx)
      self.assertEqual(affine_type.curve, curve)

  def testG2AffineType(self):
    """Test creating G2 affine point types."""
    with Context() as ctx, Location.unknown():
      field.register_dialect(ctx)
      elliptic_curve.register_dialect(ctx)

      prime_field = _create_bn254_prime_field(ctx)
      ext_field = _create_bn254_extension_field(ctx, prime_field)
      curve = _create_bn254_g2_curve(ctx, ext_field)

      # Create G2 affine point type
      affine_type = elliptic_curve.AffineType.get(curve, ctx)
      self.assertEqual(affine_type.curve, curve)

  def testJacobianType(self):
    """Test creating jacobian point types."""
    with Context() as ctx, Location.unknown():
      field.register_dialect(ctx)
      elliptic_curve.register_dialect(ctx)

      base_field = _create_bn254_prime_field(ctx)
      curve = _create_bn254_g1_curve(ctx, base_field)

      # Create jacobian point type
      jacobian_type = elliptic_curve.JacobianType.get(curve, ctx)
      self.assertEqual(jacobian_type.curve, curve)

  def testXYZZType(self):
    """Test creating XYZZ point types."""
    with Context() as ctx, Location.unknown():
      field.register_dialect(ctx)
      elliptic_curve.register_dialect(ctx)

      base_field = _create_bn254_prime_field(ctx)
      curve = _create_bn254_g1_curve(ctx, base_field)

      # Create XYZZ point type
      xyzz_type = elliptic_curve.XYZZType.get(curve, ctx)
      self.assertEqual(xyzz_type.curve, curve)

  def testTwistedEdwardsAttr(self):
    """Test creating and reading twisted Edwards curve attributes."""
    with Context() as ctx, Location.unknown():
      field.register_dialect(ctx)
      elliptic_curve.register_dialect(ctx)

      base_field = _create_ed25519_prime_field(ctx)
      curve = _create_ed25519_curve(ctx, base_field)

      self.assertEqual(curve.base_field, base_field)
      self.assertEqual(int(IntegerAttr(curve.a)), ED25519_A)
      self.assertEqual(int(IntegerAttr(curve.d)), ED25519_D)
      self.assertEqual(int(IntegerAttr(curve.gx)), ED25519_GX)
      self.assertEqual(int(IntegerAttr(curve.gy)), ED25519_GY)

  def testEdAffineType(self):
    """Test creating twisted Edwards affine point types."""
    with Context() as ctx, Location.unknown():
      field.register_dialect(ctx)
      elliptic_curve.register_dialect(ctx)

      base_field = _create_ed25519_prime_field(ctx)
      curve = _create_ed25519_curve(ctx, base_field)

      ed_affine_type = elliptic_curve.EdAffineType.get(curve, ctx)
      self.assertEqual(ed_affine_type.curve, curve)

  def testEdExtendedType(self):
    """Test creating twisted Edwards extended point types."""
    with Context() as ctx, Location.unknown():
      field.register_dialect(ctx)
      elliptic_curve.register_dialect(ctx)

      base_field = _create_ed25519_prime_field(ctx)
      curve = _create_ed25519_curve(ctx, base_field)

      ed_extended_type = elliptic_curve.EdExtendedType.get(curve, ctx)
      self.assertEqual(ed_extended_type.curve, curve)


if __name__ == "__main__":
  absltest.main()
