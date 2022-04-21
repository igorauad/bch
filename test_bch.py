import unittest

import bch
from gf import GF, Gf2Poly, Gf2mPoly


class TestBch(unittest.TestCase):

    def test_generator_poly(self):
        # Example 6.1
        m = 4
        poly_str = "11001"
        field = GF(m, poly_str)
        # Double-error-correcting code
        code = bch.Bch(field, t=2)
        self.assertEqual(code.g, Gf2Poly([1, 1, 1, 0, 1, 0, 0, 0, 1]))
        self.assertEqual(code.dmin, 5)
        # Triple-error-correcting code
        code = bch.Bch(field, t=3)
        self.assertEqual(code.g, Gf2Poly([1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1]))
        self.assertEqual(code.dmin, 7)

        # Table 6.4
        m = 6
        poly_str = "1100001"
        field = GF(m, poly_str)
        code = bch.Bch(field, t=1)
        g1 = code.g
        self.assertEqual(g1, Gf2Poly([1, 0, 0, 0, 0, 1, 1]))
        self.assertEqual(code.k, 57)
        code = bch.Bch(field, t=2)
        g2 = code.g
        self.assertEqual(g2, g1 * Gf2Poly([1, 0, 1, 0, 1, 1, 1]))
        self.assertEqual(code.k, 51)
        code = bch.Bch(field, t=3)
        g3 = code.g
        self.assertEqual(g3, g2 * Gf2Poly([1, 1, 0, 0, 1, 1, 1]))
        self.assertEqual(code.k, 45)
        code = bch.Bch(field, t=4)
        g4 = code.g
        self.assertEqual(g4, g3 * Gf2Poly([1, 0, 0, 1, 0, 0, 1]))
        self.assertEqual(code.k, 39)
        code = bch.Bch(field, t=5)
        g5 = code.g
        self.assertEqual(g5, g4 * Gf2Poly([1, 1, 0, 1]))
        self.assertEqual(code.k, 36)
        code = bch.Bch(field, t=6)
        g6 = code.g
        self.assertEqual(g6, g5 * Gf2Poly([1, 1, 0, 1, 1, 0, 1]))
        self.assertEqual(code.k, 30)
        code = bch.Bch(field, t=7)
        g7 = code.g
        self.assertEqual(g7, g6 * Gf2Poly([1, 0, 1, 1, 0, 1, 1]))
        self.assertEqual(code.k, 24)
        code = bch.Bch(field, t=10)
        g10 = code.g
        self.assertEqual(g10, g7 * Gf2Poly([1, 1, 1, 0, 1, 0, 1]))
        self.assertEqual(code.k, 18)
        code = bch.Bch(field, t=11)
        g11 = code.g
        self.assertEqual(g11, g10 * Gf2Poly([1, 1, 1]))
        self.assertEqual(code.k, 16)
        code = bch.Bch(field, t=13)
        g13 = code.g
        self.assertEqual(g13, g11 * Gf2Poly([1, 1, 1, 0, 0, 1, 1]))
        self.assertEqual(code.k, 10)
        code = bch.Bch(field, t=15)
        g15 = code.g
        self.assertEqual(g15, g13 * Gf2Poly([1, 0, 1, 1]))
        self.assertEqual(code.k, 7)

    def test_syndrome(self):
        # Example 6.4
        r = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]  # r(x) = x^8 + 1
        m = 4
        poly_str = "11001"
        field = GF(m, poly_str)
        code = bch.Bch(field, t=2)  # Double-error-correcting code
        S = code.syndrome(r)
        self.assertEqual(S[0], field.get_element(2))
        self.assertEqual(S[1], field.get_element(4))
        self.assertEqual(S[2], field.get_element(7))
        self.assertEqual(S[3], field.get_element(8))

    def test_error_correction(self):
        # Example 6.5
        r = [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]  # x^12 + x^5 + x^3
        m = 4
        poly_str = "11001"
        field = GF(m, poly_str)
        code = bch.Bch(field, t=3)  # Triple-error-correcting code

        S = code.syndrome(r)
        self.assertEqual(S[0], field.get_element(0))
        self.assertEqual(S[1], field.get_element(0))
        self.assertEqual(S[2], field.get_element(10))
        self.assertEqual(S[3], field.get_element(0))
        self.assertEqual(S[4], field.get_element(10))
        self.assertEqual(S[5], field.get_element(5))

        alpha_0 = field.unit
        alpha_5 = field.get_element(5)
        err_loc_poly = code.err_loc_polynomial(S)
        self.assertEqual(err_loc_poly,
                         Gf2mPoly(field, [alpha_5, 0, alpha_0, alpha_0]))

        err_loc_numbers = code.err_loc_numbers(err_loc_poly)
        self.assertEqual(err_loc_numbers, [
            field.get_element(12),
            field.get_element(5),
            field.get_element(3)
        ])

        self.assertEqual(code.correct(r), 15 * [0])
