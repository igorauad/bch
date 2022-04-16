import unittest

import numpy as np

import gf


class TestGf(unittest.TestCase):

    def test_gf16(self):
        # GF(2^4) - Table 2.8 from the book
        gf_table = gf.gf(4, "11001")
        expected_table = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0],
                                   [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0],
                                   [0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 0, 1],
                                   [1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 0],
                                   [0, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1],
                                   [1, 0, 0, 1]])
        np.testing.assert_array_equal(expected_table, gf_table)

    def test_gf64(self):
        # GF(2^6) Table 6.2 from the book
        gf_table = gf.gf(6, "1100001")
        expected_table = np.array([[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0],
                                   [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0],
                                   [0, 1, 1, 0, 0, 0], [0, 0, 1, 1, 0, 0],
                                   [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 1, 1],
                                   [1, 1, 0, 0, 0, 1], [1, 0, 1, 0, 0, 0],
                                   [0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0],
                                   [0, 0, 0, 1, 0, 1], [1, 1, 0, 0, 1, 0],
                                   [0, 1, 1, 0, 0, 1], [1, 1, 1, 1, 0, 0],
                                   [0, 1, 1, 1, 1, 0], [0, 0, 1, 1, 1, 1],
                                   [1, 1, 0, 1, 1, 1], [1, 0, 1, 0, 1, 1],
                                   [1, 0, 0, 1, 0, 1], [1, 0, 0, 0, 1, 0],
                                   [0, 1, 0, 0, 0, 1], [1, 1, 1, 0, 0, 0],
                                   [0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0],
                                   [0, 0, 0, 1, 1, 1], [1, 1, 0, 0, 1, 1],
                                   [1, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 0],
                                   [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1],
                                   [1, 1, 0, 1, 0, 0], [0, 1, 1, 0, 1, 0],
                                   [0, 0, 1, 1, 0, 1], [1, 1, 0, 1, 1, 0],
                                   [0, 1, 1, 0, 1, 1], [1, 1, 1, 1, 0, 1],
                                   [1, 0, 1, 1, 1, 0], [0, 1, 0, 1, 1, 1],
                                   [1, 1, 1, 0, 1, 1], [1, 0, 1, 1, 0, 1],
                                   [1, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 1],
                                   [1, 1, 1, 0, 0, 1], [1, 0, 1, 1, 0, 0],
                                   [0, 1, 0, 1, 1, 0], [0, 0, 1, 0, 1, 1],
                                   [1, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0],
                                   [0, 1, 0, 1, 0, 1], [1, 1, 1, 0, 1, 0],
                                   [0, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 0],
                                   [0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1],
                                   [1, 0, 1, 1, 1, 1], [1, 0, 0, 1, 1, 1],
                                   [1, 0, 0, 0, 1, 1], [1, 0, 0, 0, 0, 1]])
        np.testing.assert_array_equal(expected_table, gf_table)

    def test_packed_impl(self):
        """Confirm the bit-packed implementation produces identical results"""
        for m, poly_str in zip([4, 6], ["11001", "1100001"]):
            gf_table_unpacked = gf.gf(m, poly_str)
            gf_table_packed = gf.gf_packed(m, poly_str)
            # Convert the non-packed table to its packed equivalent. First, pad
            # each row to length 8 by adding (8 - m) columns to the left.
            # Otherwise, np.packbits would pad columns to the right instead.
            gf_table_padded = np.pad(gf_table_unpacked, ((0, 0), (8 - m, 0)))
            expected_table = np.packbits(gf_table_padded, axis=1)
            # Make sure the two implementations match. The packed
            # implementation returns a uint32 1-D array.
            np.testing.assert_array_equal(
                expected_table.astype(np.uint32).reshape(-1), gf_table_packed)

    def test_elem_to_index_lut(self):
        m = 4
        poly_str = "11001"
        gf_table = gf.gf_packed(m, poly_str)
        inv_gf_table = gf.elem_to_index_lut(gf_table)

        # Look-up and inverse look-ups work
        for elem in gf_table:
            exponent = inv_gf_table[elem]
            self.assertEqual(gf_table[exponent], elem)

        # The first element is an exception. By convention, it will be zero in
        # the inverse LUT. By definition, it is zero in the original table.
        self.assertEqual(gf_table[0], 0)
        self.assertEqual(inv_gf_table[0], 0)

        # Since the convention is to represent the coefficient of lowest power
        # on the left of the int number representing alpha^i, it follows that
        # element alpha^(m-1) will have the right-most bit (coefficient of
        # highest power) equal to one. Namely, the int representing alpha^(m-1)
        # is unitary. Thus, the second element of the inverse LUT should be
        # point to the exponent "m - 1".
        #
        # Nevertheless, note the original GF(2^m) table has the elements [0,
        # alpha^0, ..., alpha^(2^m - 2)]. That is, the element associated with
        # exponent 0 is at index=1 (the second position), while the element
        # associated with exponent 1 is at index=2 (the third position), and so
        # on. In other words, the index on the original GF table is equal to
        # the exponent plus one. Hence, index=m represents exponent "m - 1".
        self.assertEqual(gf_table[m], 1)
        self.assertEqual(inv_gf_table[1], m)

    def test_multiply(self):
        # See some cases in Example 2.7
        m = 4
        poly_str = "11001"
        gf_table = gf.gf_packed(m, poly_str)
        inv_gf_table = gf.elem_to_index_lut(gf_table)
        multiplicand_exponents = [
            (0, 4),
            (1, 4),
            (1, 5),
            (1, 6),
            (5, 7),
            (12, 7),
        ]
        result_exponents = [
            4,
            5,
            6,
            7,
            12,
            4,
        ]
        for (exp_a, exp_b), exp_z in zip(multiplicand_exponents,
                                         result_exponents):
            # Get the elements for each exponent while noting alpha^0 is at
            # gf_table[1], alpha^1 at gf_table[2], and so on.
            a = gf_table[exp_a + 1]
            b = gf_table[exp_b + 1]
            z = gf_table[exp_z + 1]
            self.assertAlmostEqual(gf.multiply(gf_table, inv_gf_table, a, b),
                                   z)
        # Multiplication by zero should always result in 0
        for x in range(2**m):
            self.assertAlmostEqual(gf.multiply(gf_table, inv_gf_table, x, 0),
                                   0)

    def test_multiplicative_inverse(self):
        m = 4
        poly_str = "11001"
        gf_table = gf.gf_packed(m, poly_str)
        inv_gf_table = gf.elem_to_index_lut(gf_table)

        # See some cases in Example 2.7
        elem = gf_table[12 + 1]  # alpha^12
        inv = gf_table[3 + 1]  # alpha^3
        self.assertEqual(gf.inverse(gf_table, inv_gf_table, elem), inv)

        elem = gf_table[5 + 1]  # alpha^5
        inv = gf_table[10 + 1]  # alpha^10
        self.assertEqual(gf.inverse(gf_table, inv_gf_table, elem), inv)

        elem = gf_table[1]  # alpha^0
        inv = gf_table[1]  # alpha^0
        self.assertEqual(gf.inverse(gf_table, inv_gf_table, elem), inv)

        elem = gf_table[15]  # alpha^14
        inv = gf_table[2]  # alpha^1
        self.assertEqual(gf.inverse(gf_table, inv_gf_table, elem), inv)

    def test_division(self):
        m = 4
        poly_str = "11001"
        gf_table = gf.gf_packed(m, poly_str)
        inv_gf_table = gf.elem_to_index_lut(gf_table)

        # See some cases in Example 2.7
        a = gf_table[4 + 1]  # alpha^4
        b = gf_table[12 + 1]  # alpha^12
        res = gf_table[7 + 1]  # alpha^7
        self.assertEqual(gf.divide(gf_table, inv_gf_table, a, b), res)

        a = gf_table[12 + 1]  # alpha^12
        b = gf_table[5 + 1]  # alpha^5
        res = gf_table[7 + 1]  # alpha^7
        self.assertEqual(gf.divide(gf_table, inv_gf_table, a, b), res)

        a = gf_table[12 + 1]  # alpha^12
        b = gf_table[1]  # alpha^0
        res = gf_table[12 + 1]  # alpha^12
        self.assertEqual(gf.divide(gf_table, inv_gf_table, a, b), res)

    def test_conjugates(self):
        # See Table 2.9
        m = 4
        self.assertEqual(gf.conjugates(m, 0), [0])  # alpha^0=1
        for i in [1, 2, 4, 8]:
            self.assertEqual(gf.conjugates(m, i), [1, 2, 4, 8])
        for i in [3, 6, 9, 12]:
            self.assertEqual(gf.conjugates(m, i), [3, 6, 9, 12])
        for i in [5, 10]:
            self.assertEqual(gf.conjugates(m, i), [5, 10])
        for i in [7, 11, 13, 14]:
            self.assertEqual(gf.conjugates(m, i), [7, 11, 13, 14])

    def test_min_polynomial(self):
        # See Table 2.9
        m = 4
        poly_str = "11001"
        gf_table = gf.gf_packed(m, poly_str)
        inv_gf_table = gf.elem_to_index_lut(gf_table)

        # Trivial minimal polynomials
        self.assertEqual(
            gf.min_polynomial(gf_table, inv_gf_table, gf_table[0]), [1, 0])
        self.assertEqual(
            gf.min_polynomial(gf_table, inv_gf_table, gf_table[1]), [1, 1])

        for i in [1, 2, 4, 8]:
            beta = gf_table[i + 1]  # alpha^i
            self.assertEqual(gf.min_polynomial(gf_table, inv_gf_table, beta),
                             [1, 0, 0, 1, 1])
        for i in [3, 6, 9, 12]:
            beta = gf_table[i + 1]  # alpha^i
            self.assertEqual(gf.min_polynomial(gf_table, inv_gf_table, beta),
                             [1, 1, 1, 1, 1])
        for i in [5, 10]:
            beta = gf_table[i + 1]  # alpha^i
            self.assertEqual(gf.min_polynomial(gf_table, inv_gf_table, beta),
                             [1, 1, 1])
        for i in [7, 11, 13, 14]:
            beta = gf_table[i + 1]  # alpha^i
            self.assertEqual(gf.min_polynomial(gf_table, inv_gf_table, beta),
                             [1, 1, 0, 0, 1])
