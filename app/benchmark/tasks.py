"""Fixed offline benchmark. NEVER used for training, lessons or discovery.

Expected answers are not stored: every item is verified by the same deterministic verifier,
recomputing from `input_data`. Items are fixed literals (no randomness) so scores are comparable
across sessions. Changing anything here changes BENCHMARK_VERSION and invalidates reuse.
"""
from __future__ import annotations

import hashlib

from app.config import DOMAIN_TO_CATEGORY
from app.tasks.model import Task

SRC = "benchmark://fixed-local"
TITLE = "Fixed local benchmark (not internet-sourced)"
NOTE = " Put only the final value in the `answer` field."


def _t(idx: int, domain: str, diff: int, q: str, method: str, data: dict) -> Task:
    return Task(domain=domain, difficulty=diff, question=q + (NOTE if domain != "coding" else ""), source_url=SRC,
                source_title=TITLE, verification_method=method, input_data=data,
                task_id=f"bench_{idx:02d}", retrieved_at="fixed")


def _code(idx: int, name: str, sig: str, spec: str, tests: list) -> Task:
    q = (f"Write a Python function `{sig}` that {spec} Return ONLY valid Python source code for the function as a "
         f"string in the `answer` field.")
    return _t(idx, "coding", 2, q, "code_tests", {"function_name": name, "tests": tests})


def _build() -> list[Task]:
    T: list[Task] = []
    add = lambda *a: T.append(_t(len(T) + 1, *a))
    # numerical (6)
    add("numerical_reasoning", 1, "What is the greatest common divisor of 1071 and 462?", "numeric_computation", {"kind": "gcd", "a": 1071, "b": 462})
    add("numerical_reasoning", 2, "What is the least common multiple of 84 and 126?", "numeric_computation", {"kind": "lcm", "a": 84, "b": 126})
    add("numerical_reasoning", 3, "Compute 7^45 mod 13.", "numeric_computation", {"kind": "modpow", "base": 7, "exp": 45, "mod": 13})
    add("numerical_reasoning", 2, "Write the decimal number 255 in base 2.", "numeric_computation", {"kind": "to_base", "n": 255, "base": 2})
    add("numerical_reasoning", 2, "An arithmetic progression starts at 5 with common difference 3. What is the sum of its first 20 terms?", "numeric_computation", {"kind": "arith_series_sum", "first": 5, "step": 3, "n": 20})
    add("numerical_reasoning", 1, "A quantity changes from 80 to 100. What is the percentage change in percent?", "numeric_computation", {"kind": "percent_change", "old": 80, "new": 100, "decimals": 2})
    # mathematics (6)
    add("mathematics", 2, "Differentiate f(x) = 3*x^3 - 2*x + 5 with respect to x. Give f'(x) as an expression in x.", "sympy_symbolic", {"kind": "derivative", "poly": "3*x^3 - 2*x + 5"})
    add("mathematics", 3, "Evaluate the definite integral of f(x) = 3*x^2 + 2*x from x = 0 to x = 2.", "sympy_symbolic", {"kind": "definite_integral", "poly": "3*x^2 + 2*x", "a": 0, "b": 2, "decimals": 4})
    add("mathematics", 2, "Find all real roots of x^2 - 5*x + 6 = 0. List every root separated by commas.", "sympy_symbolic", {"kind": "roots", "poly": "x^2 - 5*x + 6"})
    add("mathematics", 3, "Find all real roots of 2*x^2 + 2*x - 12 = 0. List every root separated by commas.", "sympy_symbolic", {"kind": "roots", "poly": "2*x^2 + 2*x - 12"})
    add("mathematics", 3, "Find the sum of all the multiples of 3 or 5 below 1000.", "euler_reference", {"problem": 1})
    add("mathematics", 3, "Find the difference between the sum of the squares of the first one hundred natural numbers and the square of their sum (square of the sum minus sum of squares).", "euler_reference", {"problem": 6})
    # statistics / probability (6)
    add("statistics", 2, "Compute the mean of the values 2, 4, 4, 4, 5, 5, 7, 9.", "stats_computation", {"values": ["2", "4", "4", "4", "5", "5", "7", "9"], "statistic": "mean", "decimals": 4})
    add("statistics", 3, "Compute the sample standard deviation (n-1 denominator) of 2, 4, 4, 4, 5, 5, 7, 9 to 4 decimal places.", "stats_computation", {"values": ["2", "4", "4", "4", "5", "5", "7", "9"], "statistic": "sample_stdev", "decimals": 4})
    add("statistics", 3, "X ~ Binomial(n=10, p=1/2). Compute P(X = 3) to 4 decimal places.", "probability_exact", {"kind": "binomial_pmf", "n": 10, "k": 3, "p": "1/2", "decimals": 4})
    add("statistics", 3, "A condition has prevalence 0.01. A test has sensitivity 0.95 and false-positive rate 0.05. Given a positive test, what is the probability of the condition? Round to 4 decimal places.", "probability_exact", {"kind": "bayes", "prior": "0.01", "sensitivity": "0.95", "false_positive_rate": "0.05", "decimals": 4})
    add("statistics", 3, "In a group of 23 people (365 equally likely birthdays), what is the probability at least two share a birthday? Round to 4 decimal places.", "probability_exact", {"kind": "birthday", "n": 23, "days": 365, "decimals": 4})
    add("statistics", 2, "You roll 2 fair 6-sided dice. What is the probability the sum is exactly 7? Round to 4 decimal places.", "probability_exact", {"kind": "dice_sum", "dice": 2, "sides": 6, "target": 7, "decimals": 4})
    # finance (7)
    add("financial_calculation", 2, "What is $1000 worth after 3 years at 5% annual interest compounded annually? Round to 2 decimals.", "finance_formula", {"formula": "future_value", "principal": 1000, "rate_pct": "5", "periods_per_year": 1, "years": 3, "decimals": 2})
    add("financial_calculation", 2, "What is the present value of $5000 received in 5 years at a 4% annual discount rate? Round to 2 decimals.", "finance_formula", {"formula": "present_value", "future_value": 5000, "rate_pct": "4", "years": 5, "decimals": 2})
    add("quant_finance", 3, "A project has cash flows -1000 (t=0), 300 (t=1), 400 (t=2), 500 (t=3). What is the NPV at a 10% discount rate? Round to 2 decimals.", "finance_formula", {"formula": "npv", "rate_pct": "10", "cashflows": [-1000, 300, 400, 500], "decimals": 2})
    add("quant_finance", 3, "A 3-year bond with face value $1000 pays a 5% annual coupon. The yield to maturity is 6% (annual compounding). What is its price? Round to 2 decimals.", "finance_formula", {"formula": "bond_price", "face": 1000, "coupon_rate_pct": "5", "ytm_pct": "6", "years": 3, "decimals": 2})
    add("financial_calculation", 3, "A $20000 loan at 6% nominal annual rate is repaid in equal monthly payments over 5 years. What is the monthly payment? Round to 2 decimals.", "finance_formula", {"formula": "loan_payment", "principal": 20000, "annual_rate_pct": "6", "years": 5, "payments_per_year": 12, "decimals": 2})
    add("quant_finance", 2, "A company has net income of $20 billion on revenue of $100 billion. What is its net margin in percent?", "finance_formula", {"formula": "net_margin_pct", "net_income": 20000000000, "revenue": 100000000000, "decimals": 2})
    add("quant_finance", 3, "An investment grows from $100 to $200 in 5 years. What is the CAGR in percent, rounded to 2 decimals?", "finance_formula", {"formula": "cagr_pct", "begin": 100, "end": 200, "years": 5, "decimals": 2})
    # logic (5)
    add("logic", 2, "For the formula `(p or q) and not r` over variables p, q, r, how many of the 8 truth assignments make it true?", "logic_truth_table", {"kind": "sat_count", "formula": "(p or q) and not r", "variables": ["p", "q", "r"]})
    add("logic", 2, "Is `(not p or q) or (not q or p)` a tautology? Answer yes or no.", "logic_truth_table", {"kind": "tautology", "formula": "(not p or q) or (not q or p)", "variables": ["p", "q"]})
    add("logic", 2, "Premises: not p or q; p. Does the conclusion `q` follow (variables p, q)? Answer yes or no.", "logic_truth_table", {"kind": "entailment", "premises": ["not p or q", "p"], "conclusion": "q", "variables": ["p", "q"]})
    add("logic", 3, "Premises: not p or q; q. Does the conclusion `p` follow (variables p, q)? Answer yes or no.", "logic_truth_table", {"kind": "entailment", "premises": ["not p or q", "q"], "conclusion": "p", "variables": ["p", "q"]})
    add("logic", 3, "For the formula `(p and q) or (not p and r)` over p, q, r, how many of the 8 truth assignments make it true?", "logic_truth_table", {"kind": "sat_count", "formula": "(p and q) or (not p and r)", "variables": ["p", "q", "r"]})
    # coding (5)
    n = len(T)
    T.append(_code(n + 1, "fib", "fib(n)", "returns the n-th Fibonacci number with fib(0)=0 and fib(1)=1.", [[[k], v] for k, v in [(0, 0), (1, 1), (2, 1), (10, 55), (20, 6765)]]))
    T.append(_code(n + 2, "is_palindrome", "is_palindrome(s)", "returns True if s reads the same forwards and backwards ignoring case and non-alphanumeric characters.", [[["A man, a plan, a canal: Panama"], True], [["race a car"], False], [[""], True], [["No lemon, no melon"], True]]))
    T.append(_code(n + 3, "sum_even", "sum_even(xs)", "returns the sum of the even integers in the list xs.", [[[[1, 2, 3, 4]], 6], [[[]], 0], [[[7, 9]], 0], [[[-2, 2, 10]], 10]]))
    T.append(_code(n + 4, "count_vowels", "count_vowels(s)", "returns the number of vowels (a, e, i, o, u, case-insensitive) in string s.", [[["hello"], 2], [[""], 0], [["AEIOU"], 5], [["rhythm"], 0]]))
    T.append(_code(n + 5, "fizzbuzz", "fizzbuzz(n)", "returns a list of strings for 1..n: 'FizzBuzz' if divisible by 15, 'Fizz' if by 3, 'Buzz' if by 5, else the number as a string.", [[[5], ["1", "2", "Fizz", "4", "Buzz"]], [[15], ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8", "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"]], [[0], []]]))
    return T


BENCHMARK_TASKS: list[Task] = _build()
for _t_ in BENCHMARK_TASKS:
    assert DOMAIN_TO_CATEGORY[_t_.domain]


def category_of(task: Task) -> str:
    return DOMAIN_TO_CATEGORY[task.domain]


def benchmark_version(tasks: list[Task] | None = None) -> str:
    tasks = BENCHMARK_TASKS if tasks is None else tasks
    h = hashlib.sha256("".join(t.content_hash for t in tasks).encode()).hexdigest()[:8]
    return f"v1-{len(tasks)}-{h}"


def benchmark_hashes() -> set[str]:
    return {t.content_hash for t in BENCHMARK_TASKS}


def select_subset(n: int | None) -> list[Task]:
    """Deterministic stratified subset: round-robin over categories in fixed order."""
    if not n or n >= len(BENCHMARK_TASKS):
        return list(BENCHMARK_TASKS)
    by_cat: dict[str, list[Task]] = {}
    for t in BENCHMARK_TASKS:
        by_cat.setdefault(category_of(t), []).append(t)
    out: list[Task] = []
    i = 0
    cats = list(by_cat)
    while len(out) < n:
        for c in cats:
            if i < len(by_cat[c]) and len(out) < n:
                out.append(by_cat[c][i])
        i += 1
    return out
