from typing import Union

import numpy as np


def _bit_str_to_array(bit_str):
    return np.array(list(map(int, bit_str)), dtype=np.uint8)


def _bit_str_to_int(bit_str):
    return int(bit_str, 2)


def gf_mtx(m, poly_str):
    """Generate the GF(2^m) extension field elements as polynomials over GF(2)

    Implements the logic by representing each element as an array of m bits.
    See the implementation of GF._gen_table() for a more compact implementation
    representing each element as ints with packed bits.

    Parameters
    ----------
    m : int
        Parameter m of GF(2^m), the dimension of the field.
    poly_str : str
        Primitive polynomial of degree m as a bit string with the coefficient
        of lowest power on the left. For instance, a polynomial "1 + x + x^4"
        should be given as "11001".

    Returns
    -------
    np.array
        A (2^m by m) matrix with the 2^m field elements represented as GF(2)
        polynomials of degree "m - 1" (i.e., m bits each) with the coefficients
        of lowest power on the left.

    """
    poly = _bit_str_to_array(poly_str)
    assert (len(poly) == m + 1 and poly[-1] == 1), \
        "The primitive polynomial must have degree m"

    # The GF table is obtained by a linear-feedback shift register (LFSR)
    # configured to multiply any GF(2^m) element by alpha (the primitive
    # element). Taking m=4 as an example, note each element beta can be
    # expressed as a polynomial in alpha of the following form:
    #
    #   beta = b0 + b1*alpha + b2*alpha^2 + b3*alpha^3
    #
    # > NOTE: an element can be expressed as a polynomial in alpha because the
    # > first m powers of alpha, alpha^0 to alpha^(m-1), are represented as
    # > unit-weight binary words. For instance, for m=4, the first four words
    # > associated with alpha^0 to alpha^3 are 1000, 0100, 0010, and 0001.
    # > Equivalently, in polynomial form, these elements are given by x^i.
    # > Hence, there is a unique correspondence between alpha^i and x^i for i
    # > ranging from 0 to m-1.
    #
    # Next, note multiplication of beta by alpha results in:
    #
    #   b0*alpha + b1*alpha^2 + b2*alpha^3 + b3*alpha^4"
    #
    # However, because the primitive polynomial is monic and has degree m=4,
    # the result can be reorganized into an expression for "alpha^m". Also,
    # since the primitive polynomial always has coefficient a0=1 (otherwise it
    # would have x=0 as root), it has form "p(x) = 1 + something + x^m".
    # Finally, since "p(alpha)=0", it follows that "alpha^m = 1 + something".
    # Substituting alpha^4 in the above expression yields:
    #
    # b0*alpha + b1*alpha^2 + b2*alpha^3 + b3*(1 + something),
    #
    # which is equivalent to:
    #
    # b3 + b0*alpha + b1*alpha^2 + b2*alpha^3 + (b3 * something).
    #
    # In other words, the result is the circular-shifted word (the first four
    # terms) plus the term denoted as "(b3 * something)". The latter is given
    # by the primitive polynomial coefficients excluding the one with highest
    # power (power m) and the one with lowest power (power 0), which
    # essentially determines the extra feedback connections of the LFSR in
    # addition to the connection from the last to the first bit. For reference,
    # see Figure 6.3 and the corresponding explanation.
    feedback_weights = np.copy(poly[:-1])  # ignore the highest power
    feedback_weights[0] = 0  # ignore the lowest power

    gf_table = np.zeros((2**m, m), dtype=np.uint8)
    # The first row of the table contains the zero element (the additive
    # identity). The remaining "2^m - 1" rows are given by alpha^i for i=0 to
    # "2^m - 2", where alpha is the primitive element (root of the primitive
    # polynomial). Thus, the second row holds alpha^0, which is the
    # multiplicative identity. Since the adopted convention is to express the
    # lowest power on the left, the multiplicative identity is represented by
    # word (1, 0, 0, 0, ....), as follows:
    gf_table[1, 0] = 1

    # All the other "2^m - 2" elements are generated by shifting the LFSR.
    for i in range(1, 2**m - 1):
        last_bit = gf_table[i, -1]
        gf_table[i + 1, :] = np.logical_xor(np.roll(gf_table[i, :], 1),
                                            last_bit * feedback_weights)

    return gf_table


class GF:
    """Galois Field GF(2^m)
    """

    def __init__(self, m, poly_str, dtype=np.uint32) -> None:
        """Construct the GF(2^m) object

        Parameters
        ----------
        m : int
            Parameter m of GF(2^m), the dimension of the field.
        poly_str : str
            Primitive polynomial of degree m as a bit string with the
            coefficient of lowest power on the left. For instance, a polynomial
            "1 + x + x^4" should be given as "11001".
        dtype : type, optional
            Data type representing the GF(2^m) elements, by default np.uint32.

        """
        if (m <= 1):
            raise ValueError("m must be > 1")
        self.m = m
        self.poly_str = poly_str
        self.dtype = dtype
        self.two_to_m_minus_one = 2**m - 1
        # (2^m by 1) array with [0, alpha^0, alpha^1, ..., alpha^(2^m -2)]:
        self.table = self._gen_table()
        self.zero = self.table[0]  # additive identity
        self.unit = self.table[1]  # multiplicative identity
        self.alpha = self.table[2]  # primitive element
        # (2^m by 1) mapping each GF(2^m) element alpha^i to its index "i + 1":
        self.inv_table = self._gen_inv_table(self.table)

    def _gen_table(self):
        """Generate the GF(2^m) extension field elements as polynomials over GF(2)

        Same as the gf() function but representing each GF(2^m) element as an
        int with packed bits.

        Returns
        -------
        np.array
            A (2^m by 1) array with the 2^m field elements represented as GF(2)
            polynomials of degree "m - 1" (i.e., m bits each) with the
            coefficients of lowest power on the left of a bit-packed int.

        """
        m = self.m
        poly_str = self.poly_str
        max_m = 8 * np.dtype(self.dtype).itemsize
        assert m <= max_m, f"Implementation works up to m={max_m} only"
        poly = int(poly_str[:-1], 2)  # excluding the highest-order term x^m
        gf_table = np.zeros(2**m, dtype=self.dtype)
        # The first element is the additive identity (0), the second is the
        # multiplicative identity (1), and the remaining elements are generated
        # iteratively by the LFSR.
        gf_table[1] = 1 << (m - 1)
        for i in range(1, 2**m - 1):
            gf_table[i + 1] = (gf_table[i] >> 1) ^ ((gf_table[i] & 1) * poly)
        return gf_table

    def _gen_inv_table(self, gf_table):
        """Generate a LUT to map element alpha^i to its GF table index i+1

        All "2^m - 1" non-zero elements from GF(2^m) can be expressed as powers
        "alpha^i" of the primitive element alpha. Multiplication of two
        elements "alpha^i1" and "alpha^i2" is greatly simplified by computing
        "alpha^i1 * alpha^i2 = alpha^(i1 + i2)". For that, it is useful to
        generate a look-up table (LUT) mapping each GF(2^m) element to its
        exponent i.

        This function returns such a LUT. For each element alpha^i, instead of
        mapping to the exponent "i", the generated LUT maps to the element's
        index on the original GF table. Since the GF table produced by
        `_gen_table()` has element alpha^0 at index=1, alpha^1 at index=2, and
        so on, this function generates an array whose value at position
        "alpha^i" is the index "i+1", i.e., "x[alpha^i] = i + 1".

        By convention, the zero element is mapped to index zero, even though it
        cannot be represented in the form alpha^i.

        Parameters
        ----------
        gf_table : np.array
            (2^m by 1) array with the elements [0, alpha^0, alpha^1, ...,
            alpha^(2^m -2)] produced by the _gen_table() function.

        Returns
        -------
        np.array
            A (2^m by 1) array x where "x[0] = 0" and "x[alpha^i] = i + 1".

        """
        assert gf_table[0] == 0, \
            "The GF table must have the zero element at index 0"
        x = np.zeros(gf_table.shape, dtype=gf_table.dtype)
        for alpha_i in range(len(gf_table)):
            x[alpha_i] = np.argwhere(gf_table == alpha_i)[0][0]
        return x

    def get_element(self, i):
        """Get the GF(2^m) element alpha^i given the exponent i

        Parameters
        ----------
        i : int
            Exponent i of the element alpha^i.

        Returns
        -------
        self.dtype
            Element alpha^i.
        """
        # Note alpha^0 is at table[1], alpha^1 at table[2], and so on.
        return self.table[(i % self.two_to_m_minus_one) + 1]

    def get_exponent(self, beta):
        """Get the exponent i of a non-zero GF(2^m) element alpha^i

        Parameters
        ----------
        beta : self.dtype
            Non-zero element beta = alpha^i.

        Returns
        -------
        int
            Exponent i of element beta expressed as a power alpha^i of the
            primitive element alpha.
        """
        assert beta != 0, "beta must be non-zero"
        return self.inv_table[beta] - 1

    def index(self, beta):
        """Get the index of an arbitrary element beta from GF(2^m)

        Parameters
        ----------
        beta : self.dtype
            Target field element.

        Returns
        -------
        int
            Index of the element in the GF elements table.
        """
        return self.inv_table[beta]

    def multiply(self, a, b):
        """Multiply two elements from GF(2^m)

        Converts the GF(2^m) elements a and b to the exponents of their
        representations as powers of the primitive element alpha, i.e., of the
        form alpha^ia and alpha^ib. Next, computes the product as "alpha^(ia +
        ib)". If "ia + ib" exceeds the highest possible exponent "2^m -2", the
        final exponent actually becomes "(ia + ib) mod (2^m - 1)", which is
        within the valid range (from 0 to "2^m - 2"). The modulo reduction is
        equivalent to considering the property "beta^(2^m - 1) = 1", which
        holds for any non-zero element beta from GF(2^m).

        Parameters
        ----------
        a : gf.dtype
            Multiplicand.
        b : gf.dtype
            Multiplier.

        Returns
        -------
        self.dtype
            The product a * b.

        """
        if a == 0 or b == 0:
            return 0
        exp_a = self.get_exponent(a)
        exp_b = self.get_exponent(b)
        return self.get_element(exp_a + exp_b)

    def inverse(self, a):
        """Return the multiplicative inverse of GF(2^m) element a

        Parameters
        ----------
        a : gf.dtype
            Element from GF(2^m) whose inverse is to be computed.

        Returns
        -------
        self.dtype
            The multiplicative inverse of a.

        """
        exp_a = self.get_exponent(a)
        return self.get_element(self.two_to_m_minus_one - exp_a)

    def divide(self, a, b):
        """Divide GF(2^m) element a by GF(2^m) element b

        First, computes the multiplicative inverse of b, i.e., b^-1. Then,
        computes the product a * b^-1.

        Parameters
        ----------
        a : gf.dtype
            Dividend.
        b : gf.dtype
            Divisor.

        Returns
        -------
        self.dtype
            The quotient a / b.

        """
        return self.multiply(a, self.inverse(b))

    def conjugates(self, i):
        """Compute the conjugates of alpha^i from GF(2^m)

        By definition, the conjugates of element alpha^i from GF(2^m) are the
        elements "alpha^i^(2^l)". If alpha^i is the root of an arbitrary
        polynomial over GF(2), its conjugates are also roots of this
        polynomial. Hence, all conjugates have the same associated minimal
        polynomial, and the number of distinct conjugates determines the degree
        of the minimal polynomial.

        The fundamental property used to determine the distinct conjugates is
        that "alpha^(2^m - 1) = 1" for any arbitrary element alpha in GF(2^m).
        This property implies that any exponent "i*(2^l)" exceeding "2^m - 1"
        leads to the same result as the exponent mod (2^m - 1). For instance,
        in GF(2^4), any alpha^x for x exceeding 15 is the same as alpha^y where
        "y = x mod 15".

        As an example, suppose we want to compute the conjugates of alpha^5 in
        GF(2^4). The first conjugate is alpha^5^2 = alpha^10, whose exponent
        does not exceed "2^m - 1 = 15". The second would be "alpha^5^4 =
        alpha^20", which is the same as "(alpha^15)*(alpha^5)", namely the same
        as "alpha^5", since "alpha^15=1". In other words, the second conjugate
        is equal to the original element alpha^5, so the distinct conjugates
        are only alpha^5 and alpha^10. Furthermore, the conjugate "alpha^5^4"
        is more easily computed as "5*(2^2) mod 15", i.e., "20 mod 15 = 5".

        Finally, note that the property "beta^(2^m - 1) = 1" is equivalent to
        the property that "beta^2^m = beta", where beta is any arbitrary
        element "alpha^i" from GF(2^m). By this property, we see that the
        conjugates "alpha^i^(2^l)" are only distinct up to "l = m - 1". That
        is, for "l = m", the conjugate "beta^(2^m)" is the same as the original
        element beta. In other words, the maximum number of distinct conjugates
        is m. Nevertheless, that does not mean every element has m distinct
        conjugates. Some elements have a lower number of distinct conjugates,
        as was the case of alpha^5 from GF(2^4) in the previous example.

        As a sidenote, it is important to emphasize the parenthesis on
        "alpha^i^(2^l)". This expression applies to the non-zero elements
        "alpha^i" from GF(2^m). The l-th conjugate has exponent "(2^l)" with
        parenthesis (to be evaluated first), and it is not the same as
        "alpha^i^2^l". The final exponent is equal to "i*(2^l)", not "i*2*l".

        Parameters
        ----------
        i : int
            Exponent of the target element alpha^i from GF(2^m).

        Returns
        -------
        list
            The exponents of the conjugates of alpha^i, including i. Up to m
            distinct exponents can be returned.

        """
        assert (i <= (self.two_to_m_minus_one - 1)), \
            "The max element in GF(2^m) is alpha^(2^m-2)"
        conjugate_list = [i]  # include the exponent of the original element
        max_distinct_conjugates = self.m
        for j in range(1, max_distinct_conjugates + 1):
            exponent = (i * (2**j)) % self.two_to_m_minus_one
            if (exponent in conjugate_list):
                break
            conjugate_list.append(exponent)
        return sorted(conjugate_list)

    def min_polynomial(self, beta):
        """Compute the minimal polynomial of a GF(2^m) element

        The minimal polynomial of element beta from GF(2^m) is the polynomial
        phi(x) over GF(2) of smallest degree having beta as root. The
        polynomial itself is over GF(2), meaning its coefficients are GF(2)
        elements (i.e., 0 or 1). However, that does not mean the roots of
        phi(x) are in GF(2). On the contrary, the roots are in GF(2^m),
        similarly to how a polynomial with real coefficients can have complex
        roots.

        By definition, beta is a root of the minimal polynomial phi(x) to be
        computed. In addition to beta, the conjugates "beta^(2^l)" of element
        beta are the other roots of phi(x). Hence, the minimal polynomial can
        be computed by the product:

        (x + beta) * (x + beta^2) * (x + beta^4) * ... * (x + beta^(2^(e-1))),

        where e is the degree of phi(x) and the number of distinct conjugates
        associated with element beta. That is, beta^(2^e) = beta, so e is the
        integer after which the conjugates are just repetitions, which is less
        than or equal to m. The rationale is that even though beta^(2^m)=beta
        is a guaranteed point of repetition, that doesn't mean e=m. It is also
        possible to have e less than m.

        This function computes the minimal polynomial of beta by computing the
        above product of "(x + beta^(2^l))" terms. Note these terms are
        polynomials over GF(2^m) of first degree. However, their product should
        result in a polynomial over GF(2).

        Parameters
        ----------
        beta : gf_table.dtype
            Element from GF(2^m) whose minimal polynomial is to be computed.

        """
        if beta == 0:  # 0 is always a root of "f(x) = x"
            return Gf2Poly([0, 1])

        exp_beta = self.get_exponent(beta)  # exponent i of alpha^i = beta
        conjugate_list = self.conjugates(exp_beta)  # exponents i^(2^l)

        # Multiply the terms "(x + beta^(2^l))" iteratively
        prod = Gf2mPoly(self, [self.unit])  # start with the unitary element
        for exp in conjugate_list:
            a = Gf2mPoly(self,
                         [self.get_element(exp), self.unit])  # x + beta^(2^l)
            prod *= a

        # The resulting polynomial should be a polynomial over GF(2), even
        # though it comes from the product of terms "(x + beta^(2^l))", which
        # are polynomials over GF(2^m). Hence, the resulting polynomial should
        # have coefficients that are either the zero or the unit element. Since
        # the unitary element (multiplicative identity) can be expressed by a
        # non-unit int, depending on bit endianess, it is better to look-up its
        # index in the GF table, which is always index=1. The zero element is
        # always at index=0, so the inverse look-up shall lead to a list with
        # zeros and ones, i.e., a polynomial over GF(2).
        return Gf2Poly([self.index(x) for x in prod.coefs])


def _is_gf2_poly(x: list) -> bool:
    """Check if the given polynomial is indeed a polynomial over GF(2)

    A polynomial over GF(2) is a polynomial whose coefficients are either 0 or
    1.

    Parameters
    ----------
    x : list
        List of polynomial coefficients.

    Returns
    -------
    bool
        Whether it is a polynomial over GF(2).
    """
    return set(x).issubset({0, 1})


def _is_gf2m_poly(gf: GF, x: list) -> bool:
    """Check if the given polynomial is indeed a polynomial over GF(2^m)

    A polynomial over GF(2^m) is a polynomial whose coefficients are elements
    from GF(2^m).

    Parameters
    ----------
    gf : GF
        Galois Field object.
    x : list
        List of polynomial coefficients.

    Returns
    -------
    bool
        Whether it is a polynomial over GF(2^m).
    """
    return all([coef in gf.table for coef in x])


def _cut_trailing_gf2_poly_zeros(x: list) -> list:
    """Omit the trailing zeroes of a polynomial coefficient list

    For instance, if "x = [1, 0, 1, 0]", that represents polynomial "x^2 + 1",
    and is the same as list "[1, 0, 1]". This function removes the trailing
    zero and returns "[1, 0, 1]".

    Parameters
    ----------
    x : list
        List of polynomial coefficients.

    Returns
    -------
    list
        Reduced list of polynomial coefficients.
    """
    return x[:(len(x) - x[::-1].index(1))] if any(x) else []


def _cut_trailing_gf2m_poly_zeros(x: list) -> list:
    """Omit the trailing zeroes of a GF(2^m) polynomial coefficient list

    For instance, "[alpha^4, 0, 1, 0]" represents polynomial "x^2 + alpha4",
    and is the same as list "[alpha^4, 0, 1]". This function removes the
    leading zero and returns "[alpha^4, 0, 1]".

    Parameters
    ----------
    x : list
        List of polynomial coefficients.

    Returns
    -------
    list
        Reduced list of polynomial coefficients.
    """
    trailing_zeros = next((i for i, x in enumerate(reversed(x)) if x), None)
    return x[:(len(x) - trailing_zeros)] if trailing_zeros is not None else []


def _add_gf2_poly(a: list, b: list) -> list:
    """Add polynomials over GF(2)

    Parameters
    ----------
    a : list
        Polynomial a(x) given as a list of polynomial coefficients.
    b : list
        Polynomial b(x) given as a list of polynomial coefficients.

    Returns
    -------
    list
        Sum "a(x) + b(x)" as a list of polynomial coefficients.
    """
    n_pad = len(a) - len(b)
    if n_pad > 0:
        b = b + n_pad * [0]
    elif n_pad < 0:
        a = a + (-n_pad * [0])

    return [x ^ y for x, y in zip(a, b)]


def _add_gf2m_poly(gf: GF, a: list, b: list) -> list:
    """Add polynomials over GF(2^m)

    Parameters
    ----------
    a : list
        Polynomial a(x) given as a list of polynomial coefficients.
    b : list
        Polynomial b(x) given as a list of polynomial coefficients.

    Returns
    -------
    list
        Sum "a(x) + b(x)" as a list of polynomial coefficients.
    """
    n_pad = len(a) - len(b)
    if n_pad > 0:
        b = b + n_pad * [0]
    elif n_pad < 0:
        a = a + (-n_pad * [0])

    return [x ^ y for x, y in zip(a, b)]


def _multiply_gf2_poly(a: list, b: list) -> list:
    """Multiply polynomials over GF(2)

    The product between two polynomials over GF(2) is obtained by the
    convolution of their coefficients.

    Note this function's implementation is similar to the multiply_poly()
    function from the gf module. However, while the latter works for any
    GF(2^m), this function is optimized for polynomials over GF(2) only.

    Parameters
    ----------
    a : list
        Polynomial a(x) given as a list of polynomial coefficients.
    b : list
        Polynomial b(x) given as a list of polynomial coefficients.

    Returns
    -------
    list
        Product "a(x) * b(x)" as a list of polynomial coefficients.
    """
    prod = (len(a) + len(b) - 1) * [0]
    for i, a_i in enumerate(a):
        for j, b_j in enumerate(b):
            # In GF(2), multiplication is equivalent to an AND operation, and
            # addition is achieved through a XOR operation.
            prod[i + j] ^= a_i & b_j
    return prod


def _multiply_gf2m_poly(gf, a, b):
    """Multiply polynomials over GF(2^m)

    This function computes the product between two such polynomials, which is
    obtained by convolution of their coefficients.

    Parameters
    ----------
    gf : GF
        Galois Field object.
    a : list
        Polynomial a(x) given as a list of polynomial coefficients.
    b : list
        Polynomial b(x) given as a list of polynomial coefficients.

    Returns
    -------
    list
        Product "a(x) * b(x)" expressed as a list of polynomial coefficients.
    """
    # Recall conv(a,b) has length equal to "len(a) + len(b) - 1"
    prod = (len(a) + len(b) - 1) * [0]
    # Convolution
    for i, a_i in enumerate(a):
        for j, b_j in enumerate(b):
            # Recall each GF(2^m) element is expressed as a polynomial over
            # GF(2). The addition of two such polynomials is addition modulo-2,
            # namely a XOR operation.
            prod[i + j] ^= gf.multiply(a_i, b_j)
    return prod


def _remainder_gf2_poly(a: list, b: list, deg_a: int, deg_b: int) -> list:
    """Compute the remainder of the division a(x) / b(x)

    Parameters
    ----------
    a : list
        Dividend polynomial a(x) given as a list of polynomial coefficients.
    b : list
        Divisor polynomial b(x) given as a list of polynomial coefficients.
    deg_a : int
        Precomputed degree of a(x).
    deg_b : int
        Precomputed degree of b(x).

    Returns
    -------
    list
        Remainder polynomial as a list of coefficients. When the remainder is
        zero, returns an empty list.

    """
    if not any(a):
        return []  # a(x)=0 divided by b(x) is zero

    if deg_a < deg_b:
        return a

    La = len(a)
    Lb = len(b)
    remainder = a.copy()
    nxors = La - Lb + 1
    for i in np.arange(La, La - nxors, -1):
        if (remainder[i - 1]):
            remainder[(i - Lb):i] = np.bitwise_xor(remainder[(i - Lb):i], b)
    return remainder


class Gf2Poly:

    def __init__(self, coefs: list) -> None:
        """Construct a GF(2) polynomial object

        Parameters
        ----------
        coefs : list
            List of polynomial coefficients with the zero-degree coefficient at
            index 0, the first-degree coefficient at index 1, and so on.

        Raises
        ------
        ValueError
            If the list of coefficients contains non-binary elements.
        """
        if not _is_gf2_poly(coefs):
            raise ValueError("Not a polynomial over GF(2)")

        # Polynomial coefficients
        self.coefs = _cut_trailing_gf2_poly_zeros(coefs)

        # Polynomial degree
        self.degree = len(self.coefs) - 1 if any(self.coefs) else -1
        # NOTE: use self.coefs (without leading zeros) to compute the
        # polynomial degree and assume degree -1 for the zero polynomial.

    def __add__(self, x):
        return Gf2Poly(_add_gf2_poly(self.coefs, x.coefs))

    def __mul__(self, x):
        if isinstance(x, Gf2Poly):  # multiply polynomial by polynomial
            return Gf2Poly(_multiply_gf2_poly(self.coefs, x.coefs))
        elif x in [0, 1]:  # multiply polynomial by GF(2) scalar
            return Gf2Poly([x * y for y in self.coefs])
        else:
            raise ValueError("* expects a GF(2) polynomial or scalar")

    def __mod__(self, x):
        return Gf2Poly(
            _remainder_gf2_poly(self.coefs, x.coefs, self.degree, x.degree))

    def __eq__(self, x: object) -> bool:
        return self.coefs == x.coefs

    def hamming_weight(self):
        return sum(self.coefs)


class Gf2mPoly:

    def __init__(self, field: GF, coefs: Union[list, Gf2Poly]) -> None:
        """Construct a GF(2^m) polynomial object

        Parameters
        ----------
        field : GF
            Galois Field object.
        coefs : Union[list, Gf2Poly]
            Coefficients from a list or a GF(2) polynomial to be interpreted as
            a polynomial over GF(2^m), with the zero-degree coefficient at
            index 0, the first-degree coefficient at index 1, and so on.

        Raises
        ------
        ValueError
            If the list of coefficients contains elements that are not in the
            given GF(2^m) field.
        """
        if isinstance(coefs, Gf2Poly):
            coefs = [field.table[x] for x in coefs.coefs]

        if not _is_gf2m_poly(field, coefs):
            raise ValueError("Not a polynomial over GF(2^m)")

        # Polynomial coefficients
        self.coefs = _cut_trailing_gf2m_poly_zeros(coefs)

        # Polynomial degree
        self.degree = len(self.coefs) - 1 if any(self.coefs) else 0
        # NOTE: use self.coefs (without leading zeros) to compute the
        # polynomial degree.

        # GF(2^m) field
        self.field = field

    def __add__(self, x):
        return Gf2mPoly(self.field,
                        _add_gf2m_poly(self.field, self.coefs, x.coefs))

    def __mul__(self, x):
        if isinstance(x, Gf2mPoly):  # multiply polynomial by polynomial
            return Gf2mPoly(
                self.field, _multiply_gf2m_poly(self.field, self.coefs,
                                                x.coefs))
        elif x in self.field.table:  # multiply polynomial by GF(2^m) scalar
            return Gf2mPoly(self.field,
                            [self.field.multiply(x, y) for y in self.coefs])
        else:
            raise ValueError("* expects a GF(2^m) polynomial or scalar")

    def __eq__(self, x: object) -> bool:
        return self.coefs == x.coefs

    def eval(self, x):
        """Evaluate the polynomial p(x) for a given x from GF(2^m)

        Parameters
        ----------
        x : self.field.dtype
            Evaluation value, a GF(2^m) number.

        Returns
        -------
        self.field.dtype
            Evaluated p(x), a GF(2^m) number.
        """
        assert x in self.field.table

        if x == self.field.zero:
            return self.coefs[0]

        # If p(x) has a term "coef * x^j", note it becomes "coef * alpha^(i*j)"
        # when evaluated for "x=alpha^i".
        i = self.field.get_exponent(x)
        res = 0
        for j, coef in enumerate(self.coefs):
            if (coef):
                res ^= self.field.multiply(coef, self.field.get_element(i * j))
        return res
