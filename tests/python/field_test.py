from absl.testing import absltest

from prime_ir.mlir.ir import *
from prime_ir.mlir.dialects import field


BABYBEAR_MODULUS = 2**31 - 2**27 + 1
BABYBEAR_R = 2**32


def _createBabybearType(ctx):
  i32 = IntegerType.get_signless(32)
  modulus = IntegerAttr.get(i32, BABYBEAR_MODULUS)
  return field.PrimeFieldType.get(modulus, True, ctx)


def _createBabybearAttribute(value):
  i32 = IntegerType.get_signless(32)
  return IntegerAttr.get(i32, value)


class FieldTest(absltest.TestCase):

  def testTypes(self):
    with Context() as ctx, Location.unknown():
      field.register_dialect(ctx)
      pf = _createBabybearType(ctx)
      self.assertEqual(str(pf), "!field.pf<2013265921 : i32, true>")
      self.assertEqual(int(pf.modulus), BABYBEAR_MODULUS)
      self.assertTrue(pf.is_montgomery)

  def testBinaryFieldTypes(self):
    with Context() as ctx, Location.unknown():
      field.register_dialect(ctx)

      bf8 = field.BinaryFieldType.get(3, context=ctx)
      self.assertEqual(str(bf8), "!field.bf<3>")
      self.assertEqual(bf8.tower_level, 3)
      self.assertFalse(bf8.is_ghash)

      # is_flat selects the flat basis of the level: GHASH at level 7, AES at
      # level 3. Each flat type is distinct from its tower sibling
      # (tower 2*3 = 1 vs ghash 2*3 = 6; tower 2*2 = 3 vs aes 2*2 = 4).
      tower = field.BinaryFieldType.get(7, context=ctx)
      ghash = field.BinaryFieldType.get(7, True, ctx)
      self.assertEqual(str(tower), "!field.bf<7>")
      self.assertEqual(str(ghash), "!field.bf<7, ghash>")
      self.assertFalse(tower.is_ghash)
      self.assertTrue(ghash.is_ghash)
      self.assertFalse(ghash.is_aes)
      self.assertNotEqual(tower, ghash)

      aes = field.BinaryFieldType.get(3, is_flat=True, context=ctx)
      self.assertEqual(str(aes), "!field.bf<3, aes>")
      self.assertEqual(aes.tower_level, 3)
      self.assertTrue(aes.is_aes)
      self.assertFalse(aes.is_ghash)
      self.assertNotEqual(bf8, aes)

  def testExtensionFieldTypes(self):
    with Context() as ctx, Location.unknown():
      field.register_dialect(ctx)
      pf = _createBabybearType(ctx)

      # Test degree 2 (11 is a valid quadratic non-residue)
      non_residue_2 = (11 * BABYBEAR_R) % BABYBEAR_MODULUS
      non_residue_attr_2 = _createBabybearAttribute(non_residue_2)
      ef2_type = field.ExtensionFieldType.get(2, pf, non_residue_attr_2)
      self.assertEqual(
          str(ef2_type),
          "!field.ef<2x!field.pf<2013265921 : i32, true>, 11 : i32>",
      )
      self.assertEqual(ef2_type.degree, 2)
      self.assertEqual(ef2_type.degree_over_prime, 2)
      self.assertFalse(ef2_type.is_tower)
      self.assertEqual(ef2_type.base_field, pf)
      self.assertEqual(ef2_type.base_prime_field, pf)
      self.assertIsInstance(ef2_type.base_field, field.PrimeFieldType)
      self.assertEqual(int(ef2_type.non_residue), non_residue_2)
      self.assertTrue(ef2_type.is_montgomery)

      # Test degree 3 (2 is a valid cubic non-residue)
      non_residue_3 = (2 * BABYBEAR_R) % BABYBEAR_MODULUS
      non_residue_attr_3 = _createBabybearAttribute(non_residue_3)
      ef3_type = field.ExtensionFieldType.get(3, pf, non_residue_attr_3)
      self.assertEqual(
          str(ef3_type),
          "!field.ef<3x!field.pf<2013265921 : i32, true>, 2 : i32>",
      )
      self.assertEqual(ef3_type.degree, 3)
      self.assertEqual(ef3_type.degree_over_prime, 3)
      self.assertFalse(ef3_type.is_tower)
      self.assertEqual(ef3_type.base_field, pf)
      self.assertEqual(ef3_type.base_prime_field, pf)
      self.assertIsInstance(ef3_type.base_field, field.PrimeFieldType)
      self.assertEqual(int(ef3_type.non_residue), non_residue_3)
      self.assertTrue(ef3_type.is_montgomery)

      # Test degree 4 (11 is a valid quartic non-residue)
      non_residue_4 = (11 * BABYBEAR_R) % BABYBEAR_MODULUS
      non_residue_attr_4 = _createBabybearAttribute(non_residue_4)
      ef4_type = field.ExtensionFieldType.get(4, pf, non_residue_attr_4)
      self.assertEqual(
          str(ef4_type),
          "!field.ef<4x!field.pf<2013265921 : i32, true>, 11 : i32>",
      )
      self.assertEqual(ef4_type.degree, 4)
      self.assertEqual(ef4_type.degree_over_prime, 4)
      self.assertFalse(ef4_type.is_tower)
      self.assertEqual(ef4_type.base_field, pf)
      self.assertEqual(ef4_type.base_prime_field, pf)
      self.assertIsInstance(ef4_type.base_field, field.PrimeFieldType)
      self.assertEqual(int(ef4_type.non_residue), non_residue_4)
      self.assertTrue(ef4_type.is_montgomery)

  def testExtensionFieldGeneralModulus(self):
    # The pil2-stark Goldilocks cubic x³ - x - 1 (u³ = 1 + u): a general
    # monic modulus is encoded as a DenseIntElementsAttr of the low
    # coefficients on a prime base. Like the binomial non-residue, the
    # in-memory attribute carries the coefficients in the base field's
    # representation (Montgomery raw form when the base is Montgomery);
    # only the textual form is standard. jax's MLIR bridge must emit the
    # same when zk_dtypes' efinfo carries modulus_low_coeffs.
    goldilocks_modulus = 2**64 - 2**32 + 1
    goldilocks_r = 2**64 % goldilocks_modulus  # Montgomery form of 1
    # IntegerAttr.get casts through int64_t; reinterpret as two's complement
    # for values >= 2^63 (jax's _to_signless_int_attr does the same).
    goldilocks_modulus_signed = goldilocks_modulus - 2**64
    for is_montgomery in (False, True):
      with Context() as ctx, Location.unknown():
        field.register_dialect(ctx)
        i64 = IntegerType.get_signless(64)
        pf = field.PrimeFieldType.get(
            IntegerAttr.get(i64, goldilocks_modulus_signed), is_montgomery, ctx
        )
        one = goldilocks_r if is_montgomery else 1
        coeffs_attr = DenseElementsAttr.get(
            [IntegerAttr.get(i64, c) for c in (one, one, 0)],
            type=RankedTensorType.get([3], i64),
        )
        ef3_type = field.ExtensionFieldType.get(3, pf, coeffs_attr)
        mont_suffix = ", true" if is_montgomery else ""
        self.assertEqual(
            str(ef3_type),
            "!field.ef<3x!field.pf<18446744069414584321 : i64"
            f"{mont_suffix}>, dense<[1, 1, 0]> : tensor<3xi64>>",
        )
        self.assertEqual(ef3_type.degree, 3)
        self.assertEqual(ef3_type.degree_over_prime, 3)
        self.assertFalse(ef3_type.is_tower)
        self.assertEqual(ef3_type.base_field, pf)
        self.assertEqual(ef3_type.is_montgomery, is_montgomery)

  def testExtensionFieldTowerCreation(self):
    with Context() as ctx, Location.unknown():
      field.register_dialect(ctx)
      pf = _createBabybearType(ctx)

      # Build Fp3 = Fp[x]/(x³ - 2)
      nr3 = (2 * BABYBEAR_R) % BABYBEAR_MODULUS
      nr3_attr = _createBabybearAttribute(nr3)
      ef3 = field.ExtensionFieldType.get(3, pf, nr3_attr)
      self.assertEqual(ef3.degree, 3)
      self.assertEqual(ef3.degree_over_prime, 3)
      self.assertFalse(ef3.is_tower)

      # Build tower: Fp6 = (Fp3)[y]/(y² - 11), degree_over_prime = 3 * 2 = 6
      # 11 is a QNR in Fp and remains a QNR in Fp3 (odd-degree extension).
      nr_tower = (11 * BABYBEAR_R) % BABYBEAR_MODULUS
      nr_tower_attr = _createBabybearAttribute(nr_tower)
      ef3x2 = field.ExtensionFieldType.get(2, ef3, nr_tower_attr)
      self.assertEqual(ef3x2.degree, 2)
      self.assertEqual(ef3x2.degree_over_prime, 6)
      self.assertTrue(ef3x2.is_tower)
      self.assertEqual(ef3x2.base_prime_field, pf)
      self.assertIsInstance(ef3x2.base_field, field.ExtensionFieldType)
      self.assertEqual(ef3x2.base_field, ef3)


if __name__ == "__main__":
  absltest.main()
