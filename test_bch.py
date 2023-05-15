import random
import unittest

import numpy as np

import bch
from gf import GF, Gf2Poly, Gf2mPoly


def flip_rand_bits(word, n_errors):
    error_loc = random.sample(range(0, len(word)), n_errors)
    for loc in error_loc:
        word[loc] ^= 1


def get_all_k_bit_messages(k):
    assert k <= 8
    msgs = []
    for i in range(2**k):
        bit_array = np.unpackbits(np.array([i], dtype=np.uint8))
        msg = list(bit_array[(8 - k):])  # k-bit message
        msgs.append(msg)
    return msgs


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

    def test_encode(self):
        m = 6
        t = 15
        poly_str = "1100001"
        field = GF(m, poly_str)
        code = bch.Bch(field, t)
        msgs = get_all_k_bit_messages(code.k)
        for msg in msgs:
            codeword = code.encode(msg)
            self.assertEqual(len(codeword), code.n)
            # The codeword should be a multiple of the generator polynomial.
            rem = Gf2Poly(codeword) % code.g
            self.assertEqual(rem, Gf2Poly([]))  # no remainder

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

    def test_syndrome_no_errors(self):
        m = 4
        poly_str = "11001"
        field = GF(m, poly_str)
        code = bch.Bch(field, t=2)
        msgs = get_all_k_bit_messages(code.k)
        for msg in msgs:
            codeword = code.encode(msg)
            S = code.syndrome(codeword)
            for i in range(2 * code.t):
                self.assertEqual(S[i], field.zero)

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

        self.assertEqual(code.decode(r), code.k * [0])

    def test_error_loc_poly_no_errors(self):
        m = 4
        poly_str = "11001"
        field = GF(m, poly_str)
        code = bch.Bch(field, t=3)
        msgs = get_all_k_bit_messages(code.k)
        for msg in msgs:
            codeword = code.encode(msg)
            S = code.syndrome(codeword)
            err_loc_poly = code.err_loc_polynomial(S)
            # The error-location polynomial should be sigma(x)=1, a polynomial
            # of zero degree (i.e., with no roots).
            self.assertEqual(err_loc_poly, Gf2mPoly(field, [field.unit]))
            self.assertEqual(err_loc_poly.degree, 0)
            # The list of error-location numbers should be empty.
            err_loc_numbers = code.err_loc_numbers(err_loc_poly)
            self.assertEqual(err_loc_numbers, [])

    def test_encode_decode(self):
        # BCH code chosen from Table 6.4
        m = 6
        t = 15
        poly_str = "1100001"
        field = GF(m, poly_str)
        code = bch.Bch(field, t)
        msg = list(np.random.randint(2, size=code.k))
        # Up to t errors can be corrected
        for n_errors in range(t):
            codeword = code.encode(msg)
            flip_rand_bits(codeword, n_errors)
            decoded_msg = code.decode(codeword)
            self.assertEqual(msg, decoded_msg)
        # If there are more than t errors, they cannot be corrected
        codeword = code.encode(msg)
        flip_rand_bits(codeword, t + 1)
        decoded_msg = code.decode(codeword)
        self.assertNotEqual(msg, decoded_msg)

    def test_dvbs2_min_poly(self):
        """Test DVB-S2's min polynomials composing the generator polynomial"""

        # Normal FECFRAME: based on GF(2^16) and the primitive polynomial given
        # by g1(x) from Table 6a in the standard.
        #
        # NOTE: g1(x) is the minimal polynomial of element alpha^1, namely, the
        # primitive polynomial that generates the field.
        m = 16
        t = 12
        poly_str = "10110100000000001"
        field = GF(m, poly_str)
        code = bch.Bch(field, t)
        g1 = Gf2Poly([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1])
        g2 = Gf2Poly([1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1])
        g3 = Gf2Poly([1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1])
        g4 = Gf2Poly([1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1])
        g5 = Gf2Poly([1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1])
        g6 = Gf2Poly([1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1])
        g7 = Gf2Poly([1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1])
        g8 = Gf2Poly([1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1])
        g9 = Gf2Poly([1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1])
        g10 = Gf2Poly([1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1])
        g11 = Gf2Poly([1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1])
        g12 = Gf2Poly([1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1])
        self.assertEqual(code.min_poly[0], g1)
        self.assertEqual(code.min_poly[2], g2)
        self.assertEqual(code.min_poly[4], g3)
        self.assertEqual(code.min_poly[6], g4)
        self.assertEqual(code.min_poly[8], g5)
        self.assertEqual(code.min_poly[10], g6)
        self.assertEqual(code.min_poly[12], g7)
        self.assertEqual(code.min_poly[14], g8)
        self.assertEqual(code.min_poly[16], g9)
        self.assertEqual(code.min_poly[18], g10)
        self.assertEqual(code.min_poly[20], g11)
        self.assertEqual(code.min_poly[22], g12)

        # Short FECFRAME: based on GF(2^14) and the primitive polynomial given
        # by g1(x) from Table 6b in the standard.
        m = 14
        t = 12
        poly_str = "110101000000001"
        field = GF(m, poly_str)
        code = bch.Bch(field, t)
        g1 = Gf2Poly([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1])
        g2 = Gf2Poly([1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1])
        g3 = Gf2Poly([1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1])
        g4 = Gf2Poly([1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1])
        g5 = Gf2Poly([1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        g6 = Gf2Poly([1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1])
        g7 = Gf2Poly([1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1])
        g8 = Gf2Poly([1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1])
        g9 = Gf2Poly([1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        g10 = Gf2Poly([1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1])
        g11 = Gf2Poly([1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1])
        g12 = Gf2Poly([1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1])
        self.assertEqual(code.min_poly[0], g1)  # g1(x)
        self.assertEqual(code.min_poly[2], g2)  # g2(x)
        self.assertEqual(code.min_poly[4], g3)  # g3(x)
        self.assertEqual(code.min_poly[6], g4)  # g4(x)
        self.assertEqual(code.min_poly[8], g5)  # g5(x)
        self.assertEqual(code.min_poly[10], g6)  # g6(x)
        self.assertEqual(code.min_poly[12], g7)  # g7(x)
        self.assertEqual(code.min_poly[14], g8)  # g8(x)
        self.assertEqual(code.min_poly[16], g9)  # g9(x)
        self.assertEqual(code.min_poly[18], g10)  # g10(x)
        self.assertEqual(code.min_poly[20], g11)  # g11(x)
        self.assertEqual(code.min_poly[22], g12)  # g12(x)
