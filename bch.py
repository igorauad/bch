import numpy as np

from gf import GF, Gf2Poly, Gf2mPoly


def _generator_poly(field: GF, t: int):
    """Generator polynomial of a t-error-correcting BCH code

    Parameters
    ----------
    field : GF
        Galois Field object.
    t : int
        Target error correction capability in bits.

    Returns
    -------
    list
        List with the binary coefficients of the generator polynomial, a
        polynomial over GF(2).

    """
    # g(x) is not just the product of the minimal polynomials. It is the LCM of
    # the product, which means each distinct mimimal polynomial should only
    # appear once in the product. Since the conjugates of element beta have the
    # same minimal polynomial, keep track of the conjugates that are already
    # processed and should not be repeated.
    processed_conjugates = set()

    # Iteratively compute the product of the minimal polynomials associated
    # with the elements alpha^j for odd j
    g = Gf2Poly([1])
    for i in range(t):
        exponent = 2 * i + 1  # alpha^(2i + 1)
        if (exponent in processed_conjugates):
            continue
        processed_conjugates.update(field.conjugates(exponent))
        beta = field.get_element(exponent)  # alpha^(2i + 1)
        min_poly = field.min_polynomial(beta)
        g *= min_poly
    assert g.degree <= field.m * t, "The degree of g(x) must be at most m*t"
    return g


class Bch:

    def __init__(self, field: GF, t: int) -> None:
        """Construct the BCH code object

        Parameters
        ----------
        field : GF
            Galois Field object.
        t : int
            Target error correction capability in bits.
        """
        self.field = field  # GF(2^m)
        self.t = t  # error correction capability
        self.n = 2**field.m - 1  # codeword length
        self.g = g = _generator_poly(field, t)  # generator polynomial
        self.dmin = g.hamming_weight()  # see Theorem 3.1
        self.k = self.n - g.degree  # dataword length
        self.nparity = self.n - self.k  # number of parity-check digits
        assert self.dmin > 2 * t, \
            "Minimum distance not greater than 2t"

        # Precompute the minimal polynomials associated with the elements alpha
        # to alpha^(2*t)
        self.min_poly = []
        for i in range(1, 2 * t + 1):
            alpha_i = field.get_element(i)  # element alpha^i
            self.min_poly.append(field.min_polynomial(alpha_i))

    def syndrome(self, r: list) -> np.array:
        """Compute the syndrome of a received codeword

        Since the generator polynomial g(x) is the LCM of the minimal
        polynomials associated with the elements alpha to alpha^(2*t), it
        follows that every valid codeword must have alpha to alpha^(2*t) as
        roots (alpha being the primitive element of GF(2^m)). Hence, when a
        codeword is expressed as a polynomial c(x) over GF(2), the following
        identities must hold:

        c(alpha) = 0
        c(alpha^2) = 0
        ...
        c(alpha^(2*t)) = 0.

        After transmission through a noisy channel, the received codeword can
        be expressed as the sum of c(x) and an error vector e(x), i.e.:

        r(x) = c(x) + e(x).

        Substituting x=alpha to x=alpha^(2*t) yields:

        r(alpha) = e(alpha)
        r(alpha^2) = e(alpha^2)
        ...
        r(alpha^(2*t)) = c(alpha^(2*t)).

        Each such equation is a syndrome symbol (a GF(2^m) symbol), as they
        depend on the error patter only. That is, the syndrome components are:

        S_1 = r(alpha)
        S_2 = r(alpha^2)
        ...
        S_2t = r(alpha^(2*t)).

        So the final question is how to compute r(alpha^i) efficiently if the
        codeword is very long. Since alpha^i is a root of its minimal
        polynomial phi_i(x), we can divide the received codeword polynomial
        r(x) by each minimal polynomial. The result becomes:

        r(x) = q(x) * phi_i(x) + b_i(x),

        where q(x) is the quotient and b_i(x) is the remainder. Now, since
        alpha^i is a root of phi_i(x), it follows that:

        r(alpha^i) = b_i(alpha^i).

        The latter is a more compact expression since b_i(x) has degree "m - 1"
        or less, given the minimal polynomial phi_i(x) has degree m or less.

        Hence, ultimately, the syndrome components become:

        S_1 = b_1(alpha)
        S_2 = b_2(alpha^2)
        ...
        S_2t = b_2t(alpha^(2*t))

        This function computes the above syndrome components as b_i(alpha^i).
        Note b_i(x) is a polynomial over GF(2), which is then evaluated for
        x=alpha^i, namely for x being a GF(2^m) element.

        Parameters
        ----------
        r : list
            Received codeword as a list of binary values with length n.

        Returns
        -------
        np.array
            Array with the 2*t syndrome components S_1 to S_2t.
        """
        assert len(r) == self.n, "Invalid codeword length"
        r_poly = Gf2Poly(r)
        S = np.zeros(2 * self.t, dtype=self.field.dtype)
        for i in range(1, 2 * self.t + 1):
            bi = r_poly % self.min_poly[i - 1]  # a polynomial over GF(2)
            bi_gf2m = Gf2mPoly(self.field, bi)  # cast to a GF(2^m) polynomial
            alpha_i = self.field.get_element(i)  # element alpha^i
            S[i - 1] = bi_gf2m.eval(alpha_i)  # b_i(alpha^i)
        return S

    def err_loc_polynomial(self, S: np.array) -> Gf2mPoly:
        """Compute the error-location polynomial

        Based on the Berlekamp's iterative algorithm described in Section 6.3.

        Parameters
        ----------
        S : np.array
            Array with the 2*t syndrome components S_1 to S_2t.

        Returns
        -------
        Gf2mPoly
            Error-location polynomial.
        """

        # Form a table iteratively with up to 2*t + 1 rows. The first row is
        # associated with iteration "mu = -1", the second with "mu = 0". The
        # first two rows are prefilled and the iterations start for "mu = 1".
        # For convenience, instead of using mu starting from -1 as in the book,
        # the following implementation uses mu starting from zero, i.e., the
        # following "mu" variable is actually "mu - 1" from Section 6.3.
        nrows = 2 * self.t + 1
        mu = 1  # starting iteration

        # Iteratively computed error location polynomial. Starts with
        # sigma^-1(x) = 1 on the first row and sigma^0(x) = 1 on the second.
        sigma = [
            Gf2mPoly(self.field, [self.field.unit]),
            Gf2mPoly(self.field, [self.field.unit])
        ]

        # Discrepancy from equation (6.24), a GF(2^m) value. The first two rows
        # have discrepancies equal to 1 and S_1. However, the second row is
        # computed inside the while loop below, so only the first is prefilled.
        d = np.zeros(nrows, dtype=self.field.dtype)
        d[0] = self.field.unit

        while (mu <= 2 * self.t):
            # mu-th discrepancy from equation (6.24)
            d[mu] = S[mu - 1]  # e.g., for mu=1, pick S[0]
            for j, coef in enumerate(reversed(sigma[mu].coefs[:-1])):
                if (coef):
                    d[mu] ^= self.field.multiply(coef, S[mu - j - 2])

            # Next candidate polynomial
            if d[mu] == 0:
                sigma.append(sigma[mu])
            else:
                # Find another row rho prior to the μ-th row such that the
                # rho-th discrepancy d[rho] is not zero and the difference
                # between the row number rho and the degree of sigma at this
                # row has the largest value
                rho = 0
                max_diff = -1
                for j in range(mu - 1, -1, -1):  # row prior to the μ-th row
                    if d[j] != 0:  # discrepancy is not zero
                        diff = j - sigma[j].degree
                        if diff > max_diff:
                            max_diff = diff
                            rho = j
                # Equation (6.25)
                d_mu_inv_d_rho = self.field.divide(d[mu], d[rho])
                x_mu_minus_rho = Gf2mPoly(self.field,
                                          [self.field.unit] + (mu - rho) * [0])
                sigma.append(sigma[mu] +
                             (x_mu_minus_rho * d_mu_inv_d_rho * sigma[rho]))

            mu += 1

        # If the final polynomial has degree greater than t, there are more
        # than t errors in the received polynomial rc(X), and generaHy it is
        # not possible to locate them.
        return sigma[mu]

    def err_loc_numbers(self, sigma: Gf2mPoly) -> list:
        roots = []
        for elem in self.field.table:
            if sigma.eval(elem) == self.field.zero:
                roots.append(elem)
        return [self.field.inverse(x) for x in roots]

    def correct(self, r: list):
        out = r.copy()
        S = self.syndrome(r)
        err_loc_poly = self.err_loc_polynomial(S)
        err_loc_num = self.err_loc_numbers(err_loc_poly)
        err_loc_exp = [self.field.get_exponent(x) for x in err_loc_num]
        err_loc = [self.n - 1 - x for x in err_loc_exp]
        for pos in err_loc:
            out[pos] ^= 1
        return out
