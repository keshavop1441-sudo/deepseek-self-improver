"""Deterministic expected-value computation per verification method.

Every function recomputes the result from `input_data` only; none of it ever looks at
the model's answer. Exact arithmetic (Fraction/Decimal) is used for decisions.
"""
from __future__ import annotations

import ast
import itertools
import re
import statistics as _st  # noqa: F401  (kept for clarity; we compute exactly below)
from datetime import date
from decimal import Decimal, getcontext
from fractions import Fraction
from math import comb, gcd
from typing import Any

from app.verifier import reference

getcontext().prec = 60


class InvalidTaskInput(Exception):
    """The task's own data is inconsistent (not the model's fault)."""


def F(x: Any) -> Fraction:
    try:
        return Fraction(str(x).strip())
    except (ValueError, ZeroDivisionError) as e:
        raise InvalidTaskInput(f"bad numeric input {x!r}") from e


def _dec(f: Fraction) -> Decimal:
    return Decimal(f.numerator) / Decimal(f.denominator)


def _frac_from_dec(d: Decimal) -> Fraction:
    return Fraction(d)


def _need(d: dict, *keys: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise InvalidTaskInput(f"input_data missing {missing}")


# ----------------------------------------------------------------------------- Euler
def euler(d: dict) -> Fraction:
    _need(d, "problem")
    n = int(d["problem"])
    if n not in reference.EULER:
        raise InvalidTaskInput(f"no reference solver for Euler problem {n}")
    return Fraction(reference.EULER[n][0]())


# ----------------------------------------------------------------------------- statistics
def stats(d: dict) -> Fraction:
    _need(d, "values", "statistic")
    xs = [F(v) for v in d["values"]]
    n = len(xs)
    if n < 2:
        raise InvalidTaskInput("need at least 2 values")
    stat = d["statistic"]
    mean = sum(xs) / n
    if stat == "mean":
        return mean
    if stat == "sum":
        return sum(xs)
    if stat == "range":
        return max(xs) - min(xs)
    if stat == "median":
        s = sorted(xs)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    ss = sum((x - mean) ** 2 for x in xs)
    if stat == "sample_variance":
        return ss / (n - 1)
    if stat == "population_variance":
        return ss / n
    if stat in ("sample_stdev", "population_stdev"):
        var = ss / (n - 1 if stat == "sample_stdev" else n)
        return _frac_from_dec(_dec(var).sqrt())
    if stat == "correlation":
        _need(d, "values_y")
        ys = [F(v) for v in d["values_y"]]
        if len(ys) != n:
            raise InvalidTaskInput("x/y length mismatch")
        my = sum(ys) / n
        sxy = sum((x - mean) * (y - my) for x, y in zip(xs, ys))
        syy = sum((y - my) ** 2 for y in ys)
        if ss == 0 or syy == 0:
            raise InvalidTaskInput("zero variance")
        return _frac_from_dec(_dec(sxy) / (_dec(ss) * _dec(syy)).sqrt())
    raise InvalidTaskInput(f"unknown statistic {stat}")


# ----------------------------------------------------------------------------- probability
def probability(d: dict) -> Fraction:
    _need(d, "kind")
    k = d["kind"]
    if k == "birthday":
        n, days = int(d["n"]), int(d["days"])
        p = Fraction(1)
        for i in range(n):
            p *= Fraction(days - i, days) if days - i > 0 else 0
        return 1 - p
    if k == "binomial_pmf":
        n, kk, p = int(d["n"]), int(d["k"]), F(d["p"])
        return comb(n, kk) * p ** kk * (1 - p) ** (n - kk)
    if k == "binomial_cdf":
        n, kk, p = int(d["n"]), int(d["k"]), F(d["p"])
        return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(kk + 1))
    if k == "dice_sum":
        n, s, t = int(d["dice"]), int(d["sides"]), int(d["target"])
        ways = sum(1 for roll in itertools.product(range(1, s + 1), repeat=n) if sum(roll) == t)
        return Fraction(ways, s ** n)
    if k == "bayes":
        prior, sens, fpr = F(d["prior"]), F(d["sensitivity"]), F(d["false_positive_rate"])
        num = prior * sens
        return num / (num + (1 - prior) * fpr)
    if k == "hypergeometric":
        N, K, n, kk = int(d["N"]), int(d["K"]), int(d["n"]), int(d["k"])
        return Fraction(comb(K, kk) * comb(N - K, n - kk), comb(N, n))
    if k == "expected_value":
        outcomes = d["outcomes"]  # [[value, prob], ...]
        probs = [F(p) for _, p in outcomes]
        if sum(probs) != 1:
            raise InvalidTaskInput("probabilities do not sum to 1")
        return sum(F(v) * F(p) for v, p in outcomes)
    raise InvalidTaskInput(f"unknown probability kind {k}")


# ----------------------------------------------------------------------------- numeric reasoning
def numeric(d: dict) -> Fraction | str:
    _need(d, "kind")
    k = d["kind"]
    if k == "gcd":
        return Fraction(gcd(int(d["a"]), int(d["b"])))
    if k == "lcm":
        a, b = int(d["a"]), int(d["b"])
        return Fraction(a * b // gcd(a, b))
    if k == "modpow":
        return Fraction(pow(int(d["base"]), int(d["exp"]), int(d["mod"])))
    if k == "digit_sum_power":
        return Fraction(sum(int(c) for c in str(int(d["base"]) ** int(d["exp"]))))
    if k == "arith_series_sum":
        a, step, n = F(d["first"]), F(d["step"]), int(d["n"])
        return n * (2 * a + (n - 1) * step) / 2
    if k == "percent_change":
        old, new = F(d["old"]), F(d["new"])
        if old == 0:
            raise InvalidTaskInput("old value is zero")
        return (new - old) / old * 100
    if k == "ratio_split":
        total, parts = F(d["total"]), [F(x) for x in d["ratio"]]
        return total * parts[int(d["index"])] / sum(parts)
    if k == "unit_rate":
        return F(d["distance"]) / F(d["time"]) * F(d.get("factor", 1))
    if k == "to_base":
        digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        n, b = int(d["n"]), int(d["base"])
        if n < 0 or not 2 <= b <= 36:
            raise InvalidTaskInput("bad base conversion input")
        out = ""
        while True:
            n, r = divmod(n, b)
            out = digits[r] + out
            if n == 0:
                return out
    raise InvalidTaskInput(f"unknown numeric kind {k}")


# ----------------------------------------------------------------------------- logic
_ALLOWED = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.Name, ast.Constant, ast.Load)


def eval_formula(formula: str, env: dict[str, bool]) -> bool:
    """Evaluate a propositional formula using and/or/not/True/False only (no eval())."""
    tree = ast.parse(formula, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise InvalidTaskInput(f"disallowed syntax in formula: {type(node).__name__}")

    def ev(n: ast.AST) -> bool:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.BoolOp):
            vals = [ev(v) for v in n.values]
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        if isinstance(n, ast.UnaryOp):
            return not ev(n.operand)
        if isinstance(n, ast.Name):
            if n.id not in env:
                raise InvalidTaskInput(f"unknown variable {n.id}")
            return env[n.id]
        if isinstance(n, ast.Constant) and isinstance(n.value, bool):
            return n.value
        raise InvalidTaskInput("bad formula")
    return ev(tree)


def logic(d: dict) -> Fraction | str:
    _need(d, "kind")
    k = d["kind"]
    if k == "sat_count":
        _need(d, "formula", "variables")
        vs = list(d["variables"])
        return Fraction(sum(eval_formula(d["formula"], dict(zip(vs, vals)))
                            for vals in itertools.product([False, True], repeat=len(vs))))
    if k == "tautology":
        vs = list(d["variables"])
        ok = all(eval_formula(d["formula"], dict(zip(vs, vals)))
                 for vals in itertools.product([False, True], repeat=len(vs)))
        return "yes" if ok else "no"
    if k == "entailment":
        vs = list(d["variables"])
        for vals in itertools.product([False, True], repeat=len(vs)):
            env = dict(zip(vs, vals))
            if all(eval_formula(p, env) for p in d["premises"]) and not eval_formula(d["conclusion"], env):
                return "no"
        return "yes"
    raise InvalidTaskInput(f"unknown logic kind {k}")


# ----------------------------------------------------------------------------- finance
def _check_date(s: str) -> date:
    try:
        return date.fromisoformat(s)
    except (TypeError, ValueError) as e:
        raise InvalidTaskInput(f"invalid date {s!r}") from e


def validate_finance_inputs(d: dict) -> None:
    """Units/date/range validation that makes tasks with bad inputs 'invalid', not 'wrong'."""
    if d.get("unit", "USD") != "USD":
        raise InvalidTaskInput(f"unsupported unit {d.get('unit')!r}")
    for key in ("rate_pct", "annual_rate_pct", "ytm_pct", "coupon_rate_pct"):
        if key in d and not (-100 <= F(d[key]) <= 1000):
            raise InvalidTaskInput(f"{key} out of plausible range")
    for key in ("years", "periods_per_year", "payments_per_year"):
        if key in d and F(d[key]) <= 0:
            raise InvalidTaskInput(f"{key} must be positive")
    if "start_date" in d or "end_date" in d:
        s, e = _check_date(d.get("start_date")), _check_date(d.get("end_date"))
        if e <= s:
            raise InvalidTaskInput("end_date must be after start_date")
    if "record_date" in d:
        _check_date(d["record_date"])


def finance(d: dict) -> Fraction:
    _need(d, "formula")
    validate_finance_inputs(d)
    f = d["formula"]
    if f == "future_value":
        p, r, m, y = F(d["principal"]), F(d["rate_pct"]) / 100, int(d["periods_per_year"]), F(d["years"])
        n = m * y
        if n.denominator != 1:
            raise InvalidTaskInput("periods must be a whole number")
        return p * (1 + r / m) ** int(n)
    if f == "present_value":
        fv, r, y = F(d["future_value"]), F(d["rate_pct"]) / 100, F(d["years"])
        if y.denominator != 1:
            raise InvalidTaskInput("years must be whole")
        return fv / (1 + r) ** int(y)
    if f == "npv":
        r = F(d["rate_pct"]) / 100
        return sum(F(cf) / (1 + r) ** t for t, cf in enumerate(d["cashflows"]))
    if f == "bond_price":
        face, c, y, n = F(d["face"]), F(d["coupon_rate_pct"]) / 100, F(d["ytm_pct"]) / 100, int(d["years"])
        coupon = face * c
        return sum(coupon / (1 + y) ** t for t in range(1, n + 1)) + face / (1 + y) ** n
    if f == "loan_payment":
        p, r, years, ppy = F(d["principal"]), F(d["annual_rate_pct"]) / 100, F(d["years"]), int(d["payments_per_year"])
        n, i = years * ppy, r / ppy
        if n.denominator != 1:
            raise InvalidTaskInput("payment count must be whole")
        n = int(n)
        if i == 0:
            return p / n
        return p * i / (1 - (1 + i) ** (-n))
    if f == "cagr_pct":
        b, e, y = F(d["begin"]), F(d["end"]), F(d["years"])
        if b <= 0 or e <= 0:
            raise InvalidTaskInput("CAGR needs positive begin/end values")
        return (Fraction(((_dec(e / b)) ** (1 / _dec(y)) - 1)) * 100)
    if f == "net_margin_pct":
        ni, rev = F(d["net_income"]), F(d["revenue"])
        if rev <= 0:
            raise InvalidTaskInput("revenue must be positive")
        return ni / rev * 100
    if f == "growth_pct":
        cur, prior = F(d["current"]), F(d["prior"])
        if prior <= 0:
            raise InvalidTaskInput("prior value must be positive")
        return (cur - prior) / prior * 100
    if f == "simple_return_pct":
        b, e = F(d["begin"]), F(d["end"])
        if b <= 0:
            raise InvalidTaskInput("begin must be positive")
        return (e - b) / b * 100
    if f == "effective_annual_rate_pct":
        r, m = F(d["rate_pct"]) / 100, int(d["periods_per_year"])
        return ((1 + r / m) ** m - 1) * 100
    raise InvalidTaskInput(f"unknown finance formula {f}")


METHODS = {
    "euler_reference": euler,
    "stats_computation": stats,
    "probability_exact": probability,
    "numeric_computation": numeric,
    "logic_truth_table": logic,
    "finance_formula": finance,
}
