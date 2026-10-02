"""SymPy-backed verification for symbolic/calculus tasks."""
from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

from app.verifier.compute import InvalidTaskInput

_SAFE_EXPR = re.compile(r"^[0-9x+\-*/^(). ]+$")


def _sympy():
    try:
        import sympy  # noqa: WPS433
        from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application,
                                                parse_expr, standard_transformations)
    except ImportError as e:  # pragma: no cover
        raise InvalidTaskInput("SymPy is not installed (pip install sympy)") from e
    return sympy, parse_expr, standard_transformations + (implicit_multiplication_application, convert_xor)


def parse_answer_expr(text: str):
    """Parse a model-provided expression in x. Whitelisted characters only (no names but x)."""
    sympy, parse_expr, tf = _sympy()
    t = (text or "").strip().replace("−", "-").replace("**", "^")
    t = re.sub(r"^(?:f'?\(x\)|y'?|dy/dx)\s*=\s*", "", t)
    if not t or not _SAFE_EXPR.match(t):
        return None
    try:
        return parse_expr(t, local_dict={"x": sympy.Symbol("x")}, transformations=tf)
    except Exception:  # noqa: BLE001 - any parse failure means "not an expression"
        return None


def expected_expr(d: dict[str, Any]):
    sympy, parse_expr, tf = _sympy()
    x = sympy.Symbol("x")
    if "poly" not in d:
        raise InvalidTaskInput("input_data missing 'poly'")
    poly = parse_expr(str(d["poly"]).replace("^", "**"), local_dict={"x": x})
    kind = d.get("kind")
    if kind == "derivative":
        return sympy.diff(poly, x)
    if kind == "antiderivative":
        return sympy.integrate(poly, x)
    raise InvalidTaskInput(f"unknown symbolic kind {kind}")


def expected_value(d: dict[str, Any]) -> Fraction:
    sympy, parse_expr, tf = _sympy()
    x = sympy.Symbol("x")
    poly = parse_expr(str(d["poly"]).replace("^", "**"), local_dict={"x": x})
    kind = d.get("kind")
    if kind == "definite_integral":
        v = sympy.integrate(poly, (x, sympy.Rational(str(d["a"])), sympy.Rational(str(d["b"]))))
    elif kind == "derivative_at":
        v = sympy.diff(poly, x).subs(x, sympy.Rational(str(d["at"])))
    else:
        raise InvalidTaskInput(f"unknown symbolic kind {kind}")
    v = sympy.nsimplify(v)
    return Fraction(int(v.p), int(v.q))


def expected_roots(d: dict[str, Any]) -> list[Fraction]:
    """Real roots of a polynomial that must all be rational (task generator guarantees)."""
    sympy, parse_expr, tf = _sympy()
    x = sympy.Symbol("x")
    poly = parse_expr(str(d["poly"]).replace("^", "**"), local_dict={"x": x})
    roots = sympy.roots(sympy.Poly(poly, x), filter="R")
    out = []
    for r in roots:
        r = sympy.nsimplify(r)
        if not r.is_rational:
            raise InvalidTaskInput("non-rational root")
        out.append(Fraction(int(r.p), int(r.q)))
    return sorted(set(out))


def equivalent(a, b) -> bool:
    sympy, _, _ = _sympy()
    return bool(sympy.simplify(a - b) == 0)
