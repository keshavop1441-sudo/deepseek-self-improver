"""Independent reference solvers (plain Python) used to check model answers."""
from __future__ import annotations

from math import comb, factorial, gcd, isqrt


def _sieve(n: int) -> list[int]:
    if n < 2:
        return []
    s = bytearray([1]) * (n + 1)
    s[0:2] = b"\x00\x00"
    for i in range(2, isqrt(n) + 1):
        if s[i]:
            s[i * i::i] = bytearray(len(s[i * i::i]))
    return [i for i, v in enumerate(s) if v]


def _e1() -> int:
    return sum(i for i in range(1000) if i % 3 == 0 or i % 5 == 0)


def _e2() -> int:
    a, b, t = 1, 2, 0
    while b <= 4_000_000:
        if b % 2 == 0:
            t += b
        a, b = b, a + b
    return t


def _e3() -> int:
    n, f, best = 600851475143, 2, 1
    while f * f <= n:
        while n % f == 0:
            best, n = f, n // f
        f += 1
    return max(best, n) if n > 1 else best


def _e4() -> int:
    return max(a * b for a in range(100, 1000) for b in range(a, 1000) if str(a * b) == str(a * b)[::-1])


def _e5() -> int:
    l = 1
    for i in range(2, 21):
        l = l * i // gcd(l, i)
    return l


def _e6() -> int:
    return sum(range(1, 101)) ** 2 - sum(i * i for i in range(1, 101))


def _e7() -> int:
    return _sieve(200_000)[10000]


def _e9() -> int:
    for a in range(1, 333):
        for b in range(a + 1, 500):
            c = 1000 - a - b
            if c > b and a * a + b * b == c * c:
                return a * b * c
    raise RuntimeError("no triplet")


def _e10() -> int:
    return sum(_sieve(2_000_000 - 1))


def _e15() -> int:
    return comb(40, 20)


def _e16() -> int:
    return sum(int(c) for c in str(2 ** 1000))


def _e20() -> int:
    return sum(int(c) for c in str(factorial(100)))


def _e21() -> int:
    def d(n: int) -> int:
        return sum(i + (n // i if i * i != n else 0) for i in range(2, isqrt(n) + 1) if n % i == 0) + (1 if n > 1 else 0)
    return sum(a for a in range(2, 10000) if d(a) != a and d(d(a)) == a)


def _e25() -> int:
    a, b, i = 1, 1, 2
    while len(str(b)) < 1000:
        a, b, i = b, a + b, i + 1
    return i


def _e28() -> int:
    t, n = 1, 1
    for ring in range(1, 501):
        step = 2 * ring
        for _ in range(4):
            n += step
            t += n
    return t


# problem -> (solver, keyword that must appear in the fetched statement)
EULER: dict[int, tuple] = {
    1: (_e1, "multiples of 3 or 5"), 2: (_e2, "even-valued"), 3: (_e3, "largest prime factor"),
    4: (_e4, "palindrome"), 5: (_e5, "evenly divisible"), 6: (_e6, "square of the sum"),
    7: (_e7, "10001st prime"), 9: (_e9, "Pythagorean triplet"), 10: (_e10, "sum of all the primes below two million"),
    15: (_e15, "20"), 16: (_e16, "2^"), 20: (_e20, "100!"), 21: (_e21, "amicable"),
    25: (_e25, "1000 digits"), 28: (_e28, "spiral"),
}
