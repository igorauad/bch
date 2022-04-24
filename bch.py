import numpy as np

from gf import GF, Gf2Poly, Gf2mPoly


def _generator_poly(field: GF, t: int) -> Gf2Poly:
    """Generator polynomial of a t-error-correcting BCH code

    Parameters
    ----------
    field : GF
        Galois Field object.
    t : int
        Target error correction capability in bits.

    Returns
    -------
    Gf2Poly
        Generator polynomial, a polynomial over GF(2).

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
        """Construct the binary BCH code object

        Parameters
        ----------
        field : GF
            GF(2^m) Galois field object.
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

    def encode(self, msg: list) -> list:
        """Encode the given k-bit message into an n-bit codeword

        In a systematic cyclic code, a codeword c(x) can be expressed as
        follows:

            c(x) = x^(n-k)*d(x) + rho(x),

        where d(x) is the polynomial of degree k-1 or less (i.e., k bits)
        representing the original message, and rho(x) is a polynomial of degree
        "n - k - 1" representing the "n - k" parity bits. The multiplication by
        "x^(n - k)" is equivalent to zero-padding the original message with
        "n-k" zeros. These zeros are filled up by the rho(x) term, i.e., the
        parity bits.

        Since the codeword has to be a multiple of the generator polynomial
        g(x), it follows that rho(x) is given by:

            rho(x) = x^(n-k)*d(x) mod g(x),

        Adding the remainder rho(x) to "x^(n-k)*d(x)" is the same as
        subtracting the remainder from "x^(n-k)*d(x)" under GF(2). In the end,
        after this subtraction, the resulting c(x) becomes a multiple of g(x).

        Parameters
        ----------
        msg : list
            Message with k bits.

        Returns
        -------
        list
            Codeword with n bits.
        """
        assert len(msg) == self.k
        padded_msg = Gf2Poly(msg + self.nparity * [0])  # x^(n-k)*d(x)
        parity = padded_msg % self.g  # rho(x)
        # The Gf2Poly class omits the leading zeros of the polynomial. However,
        # here we need to add them back to form self.nparity bits.
        n_padding = self.nparity - parity.degree - 1
        return msg + n_padding * [0] + parity.coefs

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

        Based on the simplified Berlekamp's iterative algorithm described in
        Section 6.4, which works only with binary BCH codes.

        Parameters
        ----------
        S : np.array
            Array with the 2*t syndrome components S_1 to S_2t.

        Returns
        -------
        Gf2mPoly
            Error-location polynomial.
        """

        # Form a table iteratively with up to t + 2 rows.
        nrows = self.t + 2

        # Prefill the values of mu for each row:
        mu_vec = np.zeros(nrows)
        mu_vec[0] = -0.5
        mu_vec[1:] = np.arange(0, self.t + 1)

        # Iteratively computed error-location polynomial. The first two rows
        # are prefilled with "sigma(x) = 1". The third row can be prefilled
        # with the first-degree polynomial "S[0]*x + 1".
        sigma = [
            Gf2mPoly(self.field, [self.field.unit]),
            Gf2mPoly(self.field, [self.field.unit]),
            Gf2mPoly(self.field, [S[0], self.field.unit])
        ]

        # Discrepancy, a GF(2^m) value. The first two rows have discrepancies
        # equal to 1 and S_1 (first syndrome component), respectively.
        d = np.zeros(nrows, dtype=self.field.dtype)
        d[0] = self.field.unit
        d[1] = S[0]

        row = 2
        while (row <= self.t):
            mu = mu_vec[row]
            two_mu = int(2 * mu)

            # Discrepancy from equation (6.42)
            #
            # NOTE: here we compute d_mu instead of d_(mu+1) as in (6.42).
            # Hence, the indexes based on mu in (6.42) have to be adjusted. For
            # instance, S_(2mu + 3) becomes "S_(2*(mu-1) + 3) = S_(2*mu + 1)".
            # Also, the book considers syndrome components S_1 to S_2t, which
            # is equivalent to S[0] to S[2*t - 1] here. Thus, in the end,
            # S_(2mu + 3) from (6.42) becomes S[2*mu] below, while S_(2mu + 2)
            # becomes S[2*mu - 1], and so on.
            d[row] = S[two_mu]  # e.g., for mu=1, pick S[2]
            for j, coef in enumerate(reversed(sigma[row].coefs[:-1])):
                if (coef):
                    d[row] ^= self.field.multiply(coef, S[two_mu - j - 1])

            # Next candidate polynomial
            if d[row] == 0:
                sigma.append(sigma[row])
            else:
                # Find another row rho prior to the μ-th row such that the
                # rho-th discrepancy d[rho] is not zero and the difference
                # between twice the row number (2*rho) and the degree of sigma
                # at this row has the largest value
                row_rho = 0  # row number where mu=rho
                max_diff = -2  # maximum diff "2*rho - sigma[row_rho].degree"
                for j in range(row - 1, -1, -1):  # rows prior to the μ-th row
                    if d[j] != 0:  # discrepancy is not zero
                        diff = (2 * mu_vec[j]) - sigma[j].degree
                        if diff > max_diff:
                            max_diff = diff
                            row_rho = j
                rho = mu_vec[row_rho]  # value of mu at the rho-th row
                # Equation (6.41)
                d_mu_inv_d_rho = self.field.divide(d[row], d[row_rho])
                x_two_mu_minus_rho = Gf2mPoly(self.field, [self.field.unit] +
                                              int(2 * (mu - rho)) * [0])
                sigma.append(sigma[row] + (x_two_mu_minus_rho *
                                           d_mu_inv_d_rho * sigma[row_rho]))

            row += 1

        # If the final polynomial has degree greater than t, there are more
        # than t errors in the received polynomial rc(X), and generally it is
        # not possible to locate them.
        return sigma[row]

    def err_loc_numbers(self, sigma: Gf2mPoly) -> list:
        """Compute the error-location numbers

        The error-location polynomial comes from the product of up to "t" terms
        of the form (1 + beta_j*X), where beta_j is the so-called
        error-location number. Each such term has beta_j^-1 as root since:

            1 + beta_j*X = 1 + beta_j*(beta_j)^-1 = 1 + 1 = 0.

        In other words, the inverse of the error-location number is a root of
        the error-location polynomial.

        This function starts by searching which elements from GF(2^m) are roots
        of the error-location polynomial sigma(x). This search is based on
        trial and error for all non-zero elements of GF(2^m). After that, this
        function obtains the error-location numbers by inverting the roots.

        Parameters
        ----------
        sigma : Gf2mPoly
            Error-location polynomial, a polynomial over GF(2^m).

        Returns
        -------
        list
            List with the error-location numbers (GF(2^m) elements).
        """
        roots = []
        for elem in self.field.table[1:]:
            if sigma.eval(elem) == self.field.zero:
                roots.append(elem)
        return [self.field.inverse(x) for x in roots]

    def decode(self, r: list) -> list:
        """Decode the given n-bit codeword into a k-bit message

        Parameters
        ----------
        r : list
            Noisy n-bit received codeword.

        Returns
        -------
        list
            Decoded k-bit message.
        """
        S = self.syndrome(r)
        # Try to correct errors if there are non-zero syndrome components:
        if np.any(S):
            err_loc_poly = self.err_loc_polynomial(S)
            err_loc_num = self.err_loc_numbers(err_loc_poly)
            # An error-location number alpha^j means there is an error in the
            # polynomial coefficient (bit) multiplying x^j. Here, the
            # convention is that the highest-degree term is on the left (lowest
            # index) of the polynomial coefficient list, while the right-most
            # term (of highest index) is the zero-degree term. Hence, the
            # coefficient multiplying x^j is at the j-th position from right to
            # left, or the (n -j -1)-th position from left to right.
            err_loc_exp = [self.field.get_exponent(x) for x in err_loc_num]
            err_loc = [self.n - 1 - j for j in err_loc_exp]
            for idx in err_loc:
                r[idx] ^= 1
        return r[:self.k]
