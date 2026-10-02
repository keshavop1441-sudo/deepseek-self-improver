"""Task templates for concepts whose *reference page* is retrieved from the internet.

The numbers are generated locally; the page (title/URL/summary) is the real retrieved
source and is recorded as `source_role: concept_reference`. Expected answers are NEVER
stored: the verifier recomputes them from `input_data`.
"""
from __future__ import annotations

import random
from typing import Any, Callable

# (domain, difficulty, question, input_data, verification_method)
Gen = tuple[str, int, str, dict[str, Any], str]

ANSWER_NOTE = " Put only the final value in the `answer` field."


def birthday(rng: random.Random) -> Gen:
    n = rng.randint(8, 35)
    return ("probability", 3,
            f"In a group of {n} people, assume 365 equally likely birthdays and no leap years. What is the "
            f"probability that at least two people share a birthday? Round to 4 decimal places." + ANSWER_NOTE,
            {"kind": "birthday", "n": n, "days": 365, "decimals": 4}, "probability_exact")


def binomial(rng: random.Random) -> Gen:
    n, k = rng.randint(5, 12), None
    k = rng.randint(1, n - 1)
    p = rng.choice(["1/2", "1/3", "1/6", "0.3", "0.25"])
    if rng.random() < 0.5:
        return ("probability", 3, f"X ~ Binomial(n={n}, p={p}). Compute P(X = {k}) to 4 decimal places." + ANSWER_NOTE,
                {"kind": "binomial_pmf", "n": n, "k": k, "p": p, "decimals": 4}, "probability_exact")
    return ("probability", 3, f"X ~ Binomial(n={n}, p={p}). Compute P(X <= {k}) to 4 decimal places." + ANSWER_NOTE,
            {"kind": "binomial_cdf", "n": n, "k": k, "p": p, "decimals": 4}, "probability_exact")


def bayes(rng: random.Random) -> Gen:
    prior, sens, fpr = rng.choice([1, 2, 5]) / 100, rng.choice([90, 95, 99]) / 100, rng.choice([1, 2, 5]) / 100
    return ("probability", 3,
            f"A condition has prevalence {prior:.2f}. A test has sensitivity {sens:.2f} (P(positive | condition)) and "
            f"false-positive rate {fpr:.2f} (P(positive | no condition)). Given a positive test, what is the "
            f"probability the person has the condition? Round to 4 decimal places." + ANSWER_NOTE,
            {"kind": "bayes", "prior": f"{prior:.2f}", "sensitivity": f"{sens:.2f}",
             "false_positive_rate": f"{fpr:.2f}", "decimals": 4}, "probability_exact")


def hypergeo(rng: random.Random) -> Gen:
    N, K, n = 52, rng.choice([4, 13, 26]), rng.randint(3, 7)
    k = rng.randint(1, min(K, n))
    return ("probability", 3,
            f"A deck has {N} cards of which {K} are 'special'. You draw {n} cards without replacement. "
            f"What is the probability exactly {k} are special? Round to 4 decimal places." + ANSWER_NOTE,
            {"kind": "hypergeometric", "N": N, "K": K, "n": n, "k": k, "decimals": 4}, "probability_exact")


def expected_value(rng: random.Random) -> Gen:
    vals = rng.sample(range(-5, 30), 3)
    probs = rng.choice([["1/2", "1/4", "1/4"], ["1/5", "3/10", "1/2"], ["1/10", "2/5", "1/2"]])
    desc = ", ".join(f"{v} with probability {p}" for v, p in zip(vals, probs))
    return ("probability", 2, f"A game pays {desc}. What is the expected payout? Round to 4 decimal places." + ANSWER_NOTE,
            {"kind": "expected_value", "outcomes": [[v, p] for v, p in zip(vals, probs)], "decimals": 4},
            "probability_exact")


def dice(rng: random.Random) -> Gen:
    dice_n, sides = rng.choice([(2, 6), (3, 6), (2, 8)])
    t = rng.randint(dice_n + 1, dice_n * sides - 1)
    return ("probability", 2,
            f"You roll {dice_n} fair {sides}-sided dice. What is the probability that the sum is exactly {t}? "
            f"Round to 4 decimal places." + ANSWER_NOTE,
            {"kind": "dice_sum", "dice": dice_n, "sides": sides, "target": t, "decimals": 4}, "probability_exact")


_VARS = ["p", "q", "r"]


def truth_table(rng: random.Random) -> Gen:
    forms = ["(p or q) and not r", "(not p or q) and (not q or r)", "(p and q) or (not p and r)",
             "not (p and q) or r", "(p or r) and (q or not r)", "not p or (q and r)"]
    f = rng.choice(forms)
    return ("logic", 2,
            f"For the propositional formula `{f}` over variables p, q, r, how many of the 8 truth assignments make "
            f"it true?" + ANSWER_NOTE, {"kind": "sat_count", "formula": f, "variables": _VARS}, "logic_truth_table")


def tautology(rng: random.Random) -> Gen:
    cands = ["(not p or q) or (not q or p)", "(p or q) and (not p)", "not (p and q) or p",
             "(p and q) or not p or not q", "(p or q) or not p", "p and not p", "(not p or q) and p and not q"]
    f = rng.choice(cands)
    return ("logic", 2, f"Is the formula `{f}` a tautology (true for every assignment of p, q)? Answer yes or no.",
            {"kind": "tautology", "formula": f, "variables": ["p", "q"]}, "logic_truth_table")


def entailment(rng: random.Random) -> Gen:
    sets = [(["not p or q", "p"], "q"), (["not p or q", "q"], "p"), (["p or q", "not p"], "q"),
            (["not p or q", "not q or r"], "not p or r"), (["p or q"], "p"), (["p and q"], "p or r")]
    prem, concl = rng.choice(sets)
    return ("logic", 3,
            f"Premises: {'; '.join(prem)}. Does the conclusion `{concl}` logically follow from the premises "
            f"(variables p, q, r)? Answer yes or no.",
            {"kind": "entailment", "premises": prem, "conclusion": concl, "variables": _VARS}, "logic_truth_table")


def gcd_lcm(rng: random.Random) -> Gen:
    a, b = rng.randint(24, 900), rng.randint(24, 900)
    if rng.random() < 0.5:
        return ("numerical_reasoning", 1, f"What is the greatest common divisor of {a} and {b}?" + ANSWER_NOTE,
                {"kind": "gcd", "a": a, "b": b}, "numeric_computation")
    return ("numerical_reasoning", 2, f"What is the least common multiple of {a} and {b}?" + ANSWER_NOTE,
            {"kind": "lcm", "a": a, "b": b}, "numeric_computation")


def modpow(rng: random.Random) -> Gen:
    b, e, m = rng.randint(2, 20), rng.randint(5, 60), rng.choice([7, 11, 13, 17, 19, 23, 97])
    return ("numerical_reasoning", 3, f"Compute {b}^{e} mod {m}." + ANSWER_NOTE,
            {"kind": "modpow", "base": b, "exp": e, "mod": m}, "numeric_computation")


def series(rng: random.Random) -> Gen:
    a, d, n = rng.randint(-10, 20), rng.randint(2, 9), rng.randint(15, 80)
    return ("numerical_reasoning", 2,
            f"An arithmetic progression starts at {a} with common difference {d}. What is the sum of its first {n} terms?"
            + ANSWER_NOTE, {"kind": "arith_series_sum", "first": a, "step": d, "n": n}, "numeric_computation")


def to_base(rng: random.Random) -> Gen:
    n, b = rng.randint(50, 4000), rng.choice([2, 8, 16])
    return ("numerical_reasoning", 2, f"Write the decimal number {n} in base {b} (digits above 9 as capital letters)."
            + ANSWER_NOTE, {"kind": "to_base", "n": n, "base": b}, "numeric_computation")


def percent_change(rng: random.Random) -> Gen:
    old, new = rng.randint(40, 900), rng.randint(40, 900)
    return ("numerical_reasoning", 1,
            f"A quantity changes from {old} to {new}. What is the percentage change (give a percent value) to 2 decimal places?"
            + ANSWER_NOTE, {"kind": "percent_change", "old": old, "new": new, "decimals": 2}, "numeric_computation")


def _poly_str(coeffs: list[int]) -> str:
    terms = []
    for power, c in enumerate(coeffs):
        if c == 0:
            continue
        terms.append(f"{c}*x^{power}" if power > 1 else (f"{c}*x" if power == 1 else f"{c}"))
    return " + ".join(terms).replace("+ -", "- ") or "0"


def derivative(rng: random.Random) -> Gen:
    coeffs = [rng.randint(-9, 9) for _ in range(4)]
    coeffs[3] = rng.choice([c for c in range(-6, 7) if c])
    poly = _poly_str(coeffs)
    return ("mathematics", 2, f"Differentiate f(x) = {poly} with respect to x. Give f'(x) as an expression in x."
            + ANSWER_NOTE, {"kind": "derivative", "poly": poly}, "sympy_symbolic")


def definite_integral(rng: random.Random) -> Gen:
    coeffs = [rng.randint(-5, 9) for _ in range(3)]
    coeffs[2] = rng.choice([1, 2, 3, 4, -2])
    a, b = rng.randint(-2, 1), rng.randint(2, 4)
    poly = _poly_str(coeffs)
    return ("mathematics", 3, f"Evaluate the definite integral of f(x) = {poly} from x = {a} to x = {b}. "
            "Give an exact fraction or decimal to 4 places." + ANSWER_NOTE,
            {"kind": "definite_integral", "poly": poly, "a": a, "b": b, "decimals": 4}, "sympy_symbolic")


def quadratic_roots(rng: random.Random) -> Gen:
    r1, r2 = rng.sample([i for i in range(-9, 10) if i], 2)
    a = rng.choice([1, 1, 2, 3])
    b, c = -a * (r1 + r2), a * r1 * r2
    poly = _poly_str([c, b, a])
    return ("mathematics", 2, f"Find all real roots of {poly} = 0. List every root separated by commas." + ANSWER_NOTE,
            {"kind": "roots", "poly": poly}, "sympy_symbolic")


# ------------------------------------------------------------------ coding generators
def _code_task(name: str, signature: str, spec: str, tests: list, difficulty: int = 2) -> Gen:
    q = (f"Write a Python function `{signature}` that {spec} Return ONLY valid Python source code for the function "
         f"(no tests, no input()) as a string in the `answer` field.")
    return ("coding", difficulty, q, {"function_name": name, "tests": tests}, "code_tests")


def code_fib(rng: random.Random) -> Gen:
    def fib(n: int) -> int:
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a
    ns = sorted(set([0, 1, 2, 10, 20] + [rng.randint(3, 60) for _ in range(3)]))
    return _code_task("fib", "fib(n)", "returns the n-th Fibonacci number with fib(0)=0 and fib(1)=1.",
                      [[[n], fib(n)] for n in ns])


def code_primes(rng: random.Random) -> Gen:
    def primes(n: int) -> list[int]:
        return [i for i in range(2, n + 1) if all(i % j for j in range(2, int(i ** 0.5) + 1))]
    ns = [0, 1, 2, 10, 30] + [rng.randint(31, 150)]
    return _code_task("primes_up_to", "primes_up_to(n)", "returns the list of all primes <= n in increasing order.",
                      [[[n], primes(n)] for n in ns])


def code_palindrome(rng: random.Random) -> Gen:
    cases = ["A man, a plan, a canal: Panama", "race a car", "", "No lemon, no melon", "ab", "Was it a car or a cat I saw?",
             "0P", "12321"]

    def ok(s: str) -> bool:
        t = [c.lower() for c in s if c.isalnum()]
        return t == t[::-1]
    return _code_task("is_palindrome", "is_palindrome(s)",
                      "returns True if string s reads the same forwards and backwards, ignoring case and non-alphanumeric characters.",
                      [[[s], ok(s)] for s in cases])


def code_roman(rng: random.Random) -> Gen:
    def roman(n: int) -> str:
        out = ""
        for v, s in [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"),
                     (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]:
            while n >= v:
                out, n = out + s, n - v
        return out
    ns = [1, 4, 9, 14, 40, 90, 400, 1994, 3999] + [rng.randint(1, 3999)]
    return _code_task("to_roman", "to_roman(n)", "converts an integer 1..3999 to its Roman numeral string.",
                      [[[n], roman(n)] for n in ns], 3)


def code_bsearch(rng: random.Random) -> Gen:
    cases = []
    for _ in range(6):
        xs = sorted(rng.sample(range(-50, 100), rng.randint(0, 12)))
        t = rng.choice(xs) if xs and rng.random() < 0.6 else rng.randint(-60, 110)
        cases.append([[xs, t], xs.index(t) if t in xs else -1])
    return _code_task("binary_search", "binary_search(xs, target)",
                      "returns the index of target in the sorted list xs of distinct ints, or -1 if absent.", cases)


# concept page title (Wikipedia) -> generators
CONCEPTS: dict[str, list[Callable[[random.Random], Gen]]] = {
    "Birthday_problem": [birthday], "Binomial_distribution": [binomial], "Bayes'_theorem": [bayes],
    "Hypergeometric_distribution": [hypergeo], "Expected_value": [expected_value], "Dice": [dice],
    "Truth_table": [truth_table], "Tautology_(logic)": [tautology], "Logical_consequence": [entailment],
    "Greatest_common_divisor": [gcd_lcm], "Modular_exponentiation": [modpow], "Arithmetic_progression": [series],
    "Positional_notation": [to_base], "Percentage": [percent_change],
    "Derivative": [derivative], "Integral": [definite_integral], "Quadratic_equation": [quadratic_roots],
    "Fibonacci_sequence": [code_fib], "Sieve_of_Eratosthenes": [code_primes], "Palindrome": [code_palindrome],
    "Roman_numerals": [code_roman], "Binary_search": [code_bsearch],
}
