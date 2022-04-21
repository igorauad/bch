import unittest

import numpy as np

from gf import GF, Gf2Poly, Gf2mPoly, gf_mtx


class TestGf(unittest.TestCase):

    def test_gf16(self):
        # GF(2^4) - Table 2.8 from the book
        gf_table = gf_mtx(4, "11001")
        expected_table = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0],
                                   [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 0, 0],
                                   [0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 0, 1],
                                   [1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 1, 0],
                                   [0, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1],
                                   [1, 0, 0, 1]])
        np.testing.assert_array_equal(expected_table, gf_table)

    def test_gf64(self):
        # GF(2^6) Table 6.2 from the book
        gf_table = gf_mtx(6, "1100001")
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

    def test_gf2m_packed_impl(self):
        """Confirm the bit-packed implementation produces identical results"""
        for m, poly_str in zip([4, 6], ["11001", "1100001"]):
            gf_table_unpacked = gf_mtx(m, poly_str)
            field = GF(m, poly_str)
            # Convert the non-packed table to its packed equivalent. First, pad
            # each row to length 8 by adding (8 - m) columns to the left.
            # Otherwise, np.packbits would pad columns to the right instead.
            gf_table_padded = np.pad(gf_table_unpacked, ((0, 0), (8 - m, 0)))
            expected_table = np.packbits(gf_table_padded, axis=1)
            # Make sure the two implementations match. The packed
            # implementation returns a uint32 1-D array.
            np.testing.assert_array_equal(
                expected_table.astype(np.uint32).reshape(-1), field.table)

    def test_gf2m_elem_to_index_lut(self):
        m = 4
        poly_str = "11001"
        field = GF(m, poly_str)

        # Look-up and inverse look-ups work
        for elem in field.table:
            exponent = field.inv_table[elem]
            self.assertEqual(field.table[exponent], elem)

        # The first element is an exception. By convention, it will be zero in
        # the inverse LUT. By definition, it is zero in the original table.
        self.assertEqual(field.table[0], 0)
        self.assertEqual(field.inv_table[0], 0)

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
        self.assertEqual(field.table[m], 1)
        self.assertEqual(field.inv_table[1], m)

    def test_gf2m_multiplication(self):
        # See some cases in Example 2.7
        m = 4
        poly_str = "11001"
        field = GF(m, poly_str)
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
            # Get the elements for each exponent
            a = field.get_element(exp_a)
            b = field.get_element(exp_b)
            z = field.get_element(exp_z)
            self.assertAlmostEqual(field.multiply(a, b), z)
        # Multiplication by zero should always result in 0
        for x in range(2**m):
            self.assertAlmostEqual(field.multiply(x, 0), 0)

    def test_gf2m_inverse(self):
        m = 4
        poly_str = "11001"
        field = GF(m, poly_str)

        # See some cases in Example 2.7
        elem = field.get_element(12)  # alpha^12
        inv = field.get_element(3)  # alpha^3
        self.assertEqual(field.inverse(elem), inv)

        elem = field.get_element(5)  # alpha^5
        inv = field.get_element(10)  # alpha^10
        self.assertEqual(field.inverse(elem), inv)

        elem = field.get_element(0)  # alpha^0
        inv = field.get_element(0)  # alpha^0
        self.assertEqual(field.inverse(elem), inv)

        elem = field.get_element(14)  # alpha^14
        inv = field.get_element(1)  # alpha^1
        self.assertEqual(field.inverse(elem), inv)

    def test_gf2m_division(self):
        m = 4
        poly_str = "11001"
        field = GF(m, poly_str)

        # See some cases in Example 2.7
        a = field.get_element(4)  # alpha^4
        b = field.get_element(12)  # alpha^12
        res = field.get_element(7)  # alpha^7
        self.assertEqual(field.divide(a, b), res)

        a = field.get_element(12)  # alpha^12
        b = field.get_element(5)  # alpha^5
        res = field.get_element(7)  # alpha^7
        self.assertEqual(field.divide(a, b), res)

        a = field.get_element(12)  # alpha^12
        b = field.get_element(0)  # alpha^0
        res = field.get_element(12)  # alpha^12
        self.assertEqual(field.divide(a, b), res)

    def test_gf2m_conjugates(self):
        # See Table 2.9
        m = 4
        poly_str = "11001"
        field = GF(m, poly_str)
        self.assertEqual(field.conjugates(0), [0])  # alpha^0=1
        for i in [1, 2, 4, 8]:
            self.assertEqual(field.conjugates(i), [1, 2, 4, 8])
        for i in [3, 6, 9, 12]:
            self.assertEqual(field.conjugates(i), [3, 6, 9, 12])
        for i in [5, 10]:
            self.assertEqual(field.conjugates(i), [5, 10])
        for i in [7, 11, 13, 14]:
            self.assertEqual(field.conjugates(i), [7, 11, 13, 14])

    def test_gf2m_min_polynomial(self):
        # See Table 2.9
        m = 4
        poly_str = "11001"
        field = GF(m, poly_str)

        # Trivial minimal polynomials
        self.assertEqual(field.min_polynomial(field.table[0]), Gf2Poly([1, 0]))
        self.assertEqual(field.min_polynomial(field.table[1]), Gf2Poly([1, 1]))

        for i in [1, 2, 4, 8]:
            beta = field.get_element(i)  # alpha^i
            self.assertEqual(field.min_polynomial(beta),
                             Gf2Poly([1, 0, 0, 1, 1]))
        for i in [3, 6, 9, 12]:
            beta = field.get_element(i)  # alpha^i
            self.assertEqual(field.min_polynomial(beta),
                             Gf2Poly([1, 1, 1, 1, 1]))
        for i in [5, 10]:
            beta = field.get_element(i)  # alpha^i
            self.assertEqual(field.min_polynomial(beta), Gf2Poly([1, 1, 1]))
        for i in [7, 11, 13, 14]:
            beta = field.get_element(i)  # alpha^i
            self.assertEqual(field.min_polynomial(beta),
                             Gf2Poly([1, 1, 0, 0, 1]))

    def test_gf2_poly_assertion(self):
        Gf2Poly([1, 0, 0, 0, 0, 0, 1])
        Gf2Poly([1, 0, 1])
        Gf2Poly([1, 0])
        Gf2Poly([1])

        with self.assertRaises(ValueError):
            Gf2Poly([1, 2, 0, 0, 0, 0, 1])
        with self.assertRaises(ValueError):
            Gf2Poly([1, 2, 1])
        with self.assertRaises(ValueError):
            Gf2Poly([1, 2])
        with self.assertRaises(ValueError):
            Gf2Poly([2])

    def test_gf2m_poly_assertion(self):
        m = 4
        poly_str = "11001"
        field = GF(m, poly_str)
        alpha_2 = field.get_element(2)
        alpha_5 = field.get_element(5)
        out_of_field = 1 << m

        Gf2mPoly(field, [1, 0, 0, 0, 0, 0, alpha_2])
        Gf2mPoly(field, [1, 0, alpha_5])
        Gf2mPoly(field, [1, alpha_5])
        Gf2mPoly(field, [alpha_5])

        with self.assertRaises(ValueError):
            Gf2mPoly(field, [1, out_of_field, 0, 0, 0, 0, alpha_2])
        with self.assertRaises(ValueError):
            Gf2mPoly(field, [1, out_of_field, alpha_5])
        with self.assertRaises(ValueError):
            Gf2mPoly(field, [1, out_of_field])
        with self.assertRaises(ValueError):
            Gf2mPoly(field, [out_of_field])

    def test_gf2_poly_degree(self):
        self.assertEqual(Gf2Poly([1, 0, 1]).degree, 2)
        self.assertEqual(Gf2Poly([0, 1, 0, 1]).degree, 2)
        self.assertEqual(Gf2Poly([1, 1, 0, 1]).degree, 3)
        self.assertEqual(Gf2Poly([1, 1]).degree, 1)
        self.assertEqual(Gf2Poly([1]).degree, 0)
        self.assertEqual(Gf2Poly([0, 1]).degree, 0)
        self.assertEqual(Gf2Poly([0]).degree, 0)

    def test_gf2m_poly_degree(self):
        m = 4
        poly_str = "11001"
        field = GF(m, poly_str)
        alpha_2 = field.get_element(2)
        alpha_5 = field.get_element(5)

        self.assertEqual(Gf2mPoly(field, [1, 0, alpha_2]).degree, 2)
        self.assertEqual(Gf2mPoly(field, [0, alpha_5, 0, alpha_2]).degree, 2)
        self.assertEqual(Gf2mPoly(field, [1, alpha_5, 0, alpha_2]).degree, 3)
        self.assertEqual(Gf2mPoly(field, [1, alpha_2]).degree, 1)
        self.assertEqual(Gf2mPoly(field, [alpha_2]).degree, 0)
        self.assertEqual(Gf2mPoly(field, [0, alpha_2]).degree, 0)
        self.assertEqual(Gf2mPoly(field, [0]).degree, 0)

    def test_gf2_poly_hamming_weight(self):
        self.assertEqual(Gf2Poly([1, 0, 1]).hamming_weight(), 2)
        self.assertEqual(Gf2Poly([0, 1, 0, 1]).hamming_weight(), 2)
        self.assertEqual(Gf2Poly([1, 1, 0, 1]).hamming_weight(), 3)
        self.assertEqual(Gf2Poly([1, 1]).hamming_weight(), 2)
        self.assertEqual(Gf2Poly([1]).hamming_weight(), 1)
        self.assertEqual(Gf2Poly([0, 1]).hamming_weight(), 1)
        self.assertEqual(Gf2Poly([0]).hamming_weight(), 0)

    def test_gf2_poly_addition(self):
        a = Gf2Poly([1, 0, 1])
        b = Gf2Poly([1, 1])
        c = Gf2Poly([1])
        d = Gf2Poly([1, 1, 0, 1])
        e = Gf2Poly([1, 1, 0, 0, 0])
        f = Gf2Poly([1])
        g = Gf2Poly([])
        self.assertEqual((a + b).coefs, [1, 1, 0])
        self.assertEqual((a + a).coefs, [])
        self.assertEqual((a + c).coefs, [1, 0, 0])
        self.assertEqual((a + d).coefs, [1, 0, 0, 0])
        self.assertEqual((a + e).coefs, [1, 1, 1, 0, 1])
        self.assertEqual((f + f).coefs, [])
        self.assertEqual((f + g).coefs, [1])
        self.assertEqual((g + f).coefs, [1])
        self.assertEqual((g + g).coefs, [])

    def test_gf2m_poly_addition(self):
        m = 4
        poly_str = "11001"
        field = GF(m, poly_str)
        alpha_0 = field.get_element(0)
        alpha_1 = field.get_element(1)
        alpha_4 = field.get_element(4)
        a = Gf2mPoly(field, [alpha_4, 0, 1])
        b = Gf2mPoly(field, [alpha_1, 1])
        c = Gf2mPoly(field, [1])
        d = Gf2mPoly(field, [1, alpha_0, 0, 1])
        e = Gf2mPoly(field, [1, 1, 0, 0, 0])
        f = Gf2mPoly(field, [1])
        g = Gf2mPoly(field, [])
        self.assertEqual((a + b).coefs, [alpha_4, alpha_1, 0])
        self.assertEqual((a + a).coefs, [])
        self.assertEqual((a + c).coefs, [alpha_4, 0, 0])
        self.assertEqual((a + d).coefs, [1, alpha_1, 0, 0])
        self.assertEqual((a + e).coefs, [1, 1, alpha_4, 0, 1])
        self.assertEqual((f + f).coefs, [])
        self.assertEqual((f + g).coefs, [1])
        self.assertEqual((g + f).coefs, [1])
        self.assertEqual((g + g).coefs, [])

    def test_gf2_poly_multiplication(self):
        # Example 6.1
        #
        # phi1(x) = x^4 + x + 1
        # phi3(x) = x^4 + x^3 + x^2 + x + 1
        # g(x) = phi1(x) * phi3(x) = x^8 + x^7 + x^6 + x^4 + 1
        phi1 = Gf2Poly([1, 0, 0, 1, 1])
        phi3 = Gf2Poly([1, 1, 1, 1, 1])
        g2 = Gf2Poly([1, 1, 1, 0, 1, 0, 0, 0, 1])
        self.assertEqual(phi1 * phi3, g2)

        # phi5(x) = x^2 + x + 1
        # g(x) = phi1(x) * phi3(x) * phi5(x)
        #      = x^10 + x^8 + x^5 + x^4 + x^2 + x + 1
        phi5 = Gf2Poly([1, 1, 1])
        g3 = Gf2Poly([1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1])
        self.assertEqual(g2 * phi5, g3)

    def test_gf2_poly_remainder(self):
        # Section 2.3
        #
        # f(x) = 1 + x + x^4 + x^5 + x^6
        # g(x) = 1 + x + x^3
        #
        # f(x) = (x^3 + x^2)*g(x) + (x^2 + x + 1)
        f = Gf2Poly([1, 1, 1, 0, 0, 1, 1])
        g = Gf2Poly([1, 0, 1, 1])
        self.assertEqual(f % g, Gf2Poly([1, 1, 1]))

        # Theorem 2.10: a primitive polynomial of degree m necessarily divides
        # "x^(2^m - 1) + 1". Example for m=3: (x^7 + 1) divided by (x^3 + x +
        # 1) must yield zero remainder.
        a = Gf2Poly([1, 0, 0, 0, 0, 0, 0, 1])
        b = Gf2Poly([1, 0, 1, 1])
        self.assertEqual(a % b, Gf2Poly([]))

        # A zero polynomial divided by a non-zero polynomial should result in
        # zero
        c = Gf2Poly([0, 0, 0])
        d = Gf2Poly([1, 0, 1, 1])
        self.assertEqual(c % d, Gf2Poly([]))