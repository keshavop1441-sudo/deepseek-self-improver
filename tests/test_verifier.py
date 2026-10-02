import pytest

from app.ollama.parsing import ModelAnswer
from app.tasks.model import Task
from app.verifier import FAILURE_CATEGORIES, Verifier
from app.verifier.base import VerificationResult
from tests.helpers import CODE, FakeHttp


def T(method, data, domain="mathematics", **kw):
    return Task(domain, 2, "q " + str(data), "https://example.org/s", "src", method, data, **kw)


v = Verifier(FakeHttp())


def test_never_trusts_self_check():
    t = T("numeric_computation", {"kind": "gcd", "a": 12, "b": 18})
    ans = ModelAnswer(answer="7", self_check="verified, 100% confident", method="trust me", parse_ok=True)
    r = v.verify(t, ans)
    assert r.status == "incorrect" and not r.verified


def test_gcd_correct_and_formats():
    t = T("numeric_computation", {"kind": "gcd", "a": 12, "b": 18})
    for a in ["6", "6.0", "The answer is 6", "$6", "12/2"]:
        assert v.verify(t, a).verified, a


def test_ambiguous_answer_is_uncertain_not_verified():
    r = v.verify(T("numeric_computation", {"kind": "gcd", "a": 12, "b": 18}), "6 or 7")
    assert r.status == "uncertain" and not r.verified


def test_empty_answer_missing_data():
    r = v.verify(T("numeric_computation", {"kind": "gcd", "a": 12, "b": 18}), "")
    assert r.status == "incorrect" and r.failure_category == "missing_data"


def test_message_never_leaks_expected():
    r = v.verify(T("numeric_computation", {"kind": "lcm", "a": 12, "b": 18}), "35")
    assert "36" not in r.message and r.details["expected"] == "36.0"


def test_sign_and_unit_categories():
    t = T("finance_formula", {"formula": "net_margin_pct", "net_income": 20, "revenue": 100, "decimals": 2}, "quant_finance")
    assert v.verify(t, "-20").failure_category == "sign_error"
    assert v.verify(t, "0.2").failure_category == "unit_conversion"
    assert v.verify(t, "20").verified
    assert v.verify(t, "20.5").failure_category == "arithmetic"


def test_euler_reference():
    assert v.verify(T("euler_reference", {"problem": 1}), "233168").verified
    assert not v.verify(T("euler_reference", {"problem": 1}), "233167").verified


def test_stats_exact():
    vals = ["2", "4", "4", "4", "5", "5", "7", "9"]
    assert v.verify(T("stats_computation", {"values": vals, "statistic": "mean", "decimals": 4}), "5").verified
    r = v.verify(T("stats_computation", {"values": vals, "statistic": "sample_stdev", "decimals": 4}), "2.1381")
    assert r.verified
    assert not v.verify(T("stats_computation", {"values": vals, "statistic": "sample_stdev", "decimals": 4}), "2.0").verified
    assert v.verify(T("stats_computation", {"values": vals, "statistic": "median", "decimals": 4}), "4.5").verified


def test_probability_exact():
    t = T("probability_exact", {"kind": "birthday", "n": 23, "days": 365, "decimals": 4}, "probability")
    assert v.verify(t, "0.5073").verified and not v.verify(t, "0.5").verified
    t2 = T("probability_exact", {"kind": "dice_sum", "dice": 2, "sides": 6, "target": 7, "decimals": 4}, "probability")
    assert v.verify(t2, "1/6").verified


def test_logic():
    sat = T("logic_truth_table", {"kind": "sat_count", "formula": "(p or q) and not r", "variables": ["p", "q", "r"]}, "logic")
    assert v.verify(sat, "3").verified and not v.verify(sat, "4").verified
    taut = T("logic_truth_table", {"kind": "tautology", "formula": "p or not p", "variables": ["p"]}, "logic")
    assert v.verify(taut, "Yes").verified and not v.verify(taut, "no").verified
    assert v.verify(taut, "perhaps").status == "uncertain"


def test_logic_formula_is_not_eval():
    bad = T("logic_truth_table", {"kind": "sat_count", "formula": "__import__('os').system('echo hi')", "variables": ["p"]}, "logic")
    r = v.verify(bad, "1")
    assert not r.verified and r.task_invalid


def test_symbolic():
    d = T("sympy_symbolic", {"kind": "derivative", "poly": "3*x^3 - 2*x + 5"})
    assert v.verify(d, "9x^2 - 2").verified
    assert v.verify(d, "f'(x) = 9*x**2 - 2").verified
    assert not v.verify(d, "9x^2").verified
    assert v.verify(d, "__import__('os')").failure_category == "interpretation"
    r = T("sympy_symbolic", {"kind": "roots", "poly": "x^2 - 5*x + 6"})
    assert v.verify(r, "x = 3, x = 2").verified and not v.verify(r, "2").verified
    i = T("sympy_symbolic", {"kind": "definite_integral", "poly": "3*x^2 + 2*x", "a": 0, "b": 2, "decimals": 4})
    assert v.verify(i, "12").verified


# ----------------------------------------------------------------- finance
FIN = {"unit": "USD"}


def fin(formula, **kw):
    return T("finance_formula", {"formula": formula, **FIN, **kw}, "financial_calculation")


def test_finance_formulas():
    assert v.verify(fin("future_value", principal=1000, rate_pct="5", periods_per_year=1, years=3, decimals=2), "1157.63").verified
    assert v.verify(fin("present_value", future_value=5000, rate_pct="4", years=5, decimals=2), "4109.64").verified
    assert v.verify(fin("npv", rate_pct="10", cashflows=[-1000, 300, 400, 500], decimals=2), "-21.04").verified
    assert v.verify(fin("bond_price", face=1000, coupon_rate_pct="5", ytm_pct="6", years=3, decimals=2), "973.27").verified
    assert v.verify(fin("loan_payment", principal=20000, annual_rate_pct="6", years=5, payments_per_year=12, decimals=2), "386.66").verified
    assert v.verify(fin("cagr_pct", begin=100, end=200, years=5, decimals=2), "14.87").verified
    assert v.verify(fin("growth_pct", current=110, prior=100, decimals=2), "10").verified
    assert v.verify(fin("effective_annual_rate_pct", rate_pct="12", periods_per_year=12, decimals=4), "12.6825").verified


def test_finance_wrong_answer_detected():
    r = v.verify(fin("future_value", principal=1000, rate_pct="5", periods_per_year=1, years=3, decimals=2), "1150")
    assert r.status == "incorrect" and r.failure_category in FAILURE_CATEGORIES


def test_finance_invalid_inputs_are_task_invalid_not_model_fault():
    assert v.verify(fin("net_margin_pct", net_income=1, revenue=0), "1").task_invalid
    assert v.verify(fin("future_value", principal=1, rate_pct="5", periods_per_year=1, years=0), "1").task_invalid
    bad_unit = T("finance_formula", {"formula": "growth_pct", "current": 1, "prior": 1, "unit": "EUR"})
    assert v.verify(bad_unit, "0").task_invalid
    bad_date = T("finance_formula", {"formula": "growth_pct", "current": 2, "prior": 1, "unit": "USD",
                                     "start_date": "2024-12-31", "end_date": "2024-01-01"})
    r = v.verify(bad_date, "100")
    assert r.task_invalid and not r.verified and "date" in r.message


def test_source_recheck_mismatch_blocks_verification():
    import json
    facts = {"facts": {"us-gaap": {"NetIncomeLoss": {"units": {"USD": [
        {"start": "2023-01-01", "end": "2023-12-31", "accn": "A1", "val": 999, "form": "10-K", "fp": "FY"}]}}}}}
    url = "https://data.sec.gov/api/xbrl/companyfacts/CIK1.json"
    spec = {"kind": "sec_facts", "facts": [{"taxonomy": "us-gaap", "tag": "NetIncomeLoss", "unit": "USD", "start": "2023-01-01",
                                            "end": "2023-12-31", "accn": "A1", "val": 500}]}
    data = {"formula": "net_margin_pct", "net_income": 500, "revenue": 1000, "unit": "USD", "decimals": 2, "recheck": spec}
    t = Task("quant_finance", 3, "q", url, "t", "finance_formula", data)
    r = Verifier(FakeHttp({url: json.dumps(facts)})).verify(t, "50")
    assert not r.verified and r.task_invalid and "source" in r.message.lower()
    spec["facts"][0]["val"] = 999
    t2 = Task("quant_finance", 3, "q2", url, "t", "finance_formula", {**data, "net_income": 999, "recheck": spec})
    r2 = Verifier(FakeHttp({url: json.dumps(facts)})).verify(t2, "99.9")
    assert r2.verified and r2.details["source_recheck"]["state"] == "match"
    assert r2.details["source_recheck"]["source_url"] == url and r2.details["source_recheck"]["checked_at"]


def test_source_unavailable_recorded_not_fatal():
    url = "https://data.sec.gov/x"
    spec = {"kind": "sec_facts", "facts": []}
    t = Task("quant_finance", 3, "q", url, "t", "finance_formula",
             {"formula": "growth_pct", "current": 110, "prior": 100, "unit": "USD", "decimals": 2, "recheck": spec})
    r = Verifier(FakeHttp()).verify(t, "10")
    assert r.verified and r.details["source_recheck"]["state"] == "unavailable"


# ----------------------------------------------------------------- code
def code_task(name="fib", tests=None):
    return T("code_tests", {"function_name": name, "tests": tests or [[[0], 0], [[10], 55], [[20], 6765]]}, "coding")


def test_code_correct_in_fence_and_plain():
    assert v.verify(code_task(), CODE["fib"]).verified
    assert v.verify(code_task(), "Here:\n```python\n" + CODE["fib"] + "```").verified


def test_code_wrong_behavior():
    r = v.verify(code_task(), "def fib(n):\n    return n\n")
    assert r.status == "incorrect" and r.failure_category == "coding_error"


def test_code_syntax_error_and_missing_function():
    assert v.verify(code_task(), "def fib(n) return").failure_category == "coding_error"
    assert v.verify(code_task(), "def other(n):\n    return 1\n").failure_category == "coding_error"
    assert v.verify(code_task(), "").failure_category == "missing_data"


def test_code_timeout_enforced():
    vv = Verifier(code_timeout=1.0)
    r = vv.verify(code_task(), "def fib(n):\n    while True:\n        pass\n")
    assert not r.verified and r.details["timed_out"]


def test_code_isolated_temp_dir(tmp_path):
    marker = tmp_path / "leak.txt"
    code = f"open(r'{marker}', 'w').write('x')\ndef fib(n):\n    return 0\n"
    v.verify(code_task(tests=[[[0], 0]]), code)
    # the code ran with cwd in a throwaway dir that is gone afterwards
    import glob, tempfile, os
    assert not glob.glob(os.path.join(tempfile.gettempdir(), "selfimp_code_*"))


def test_verifier_exception_never_verifies():
    t = T("euler_reference", {"problem": 9999})
    r = v.verify(t, "1")
    assert not r.verified


def test_unknown_method_uncertain():
    assert v.verify(T("nope", {}), "1").status == "uncertain"


def test_result_invariants():
    with pytest.raises(ValueError):
        VerificationResult("bogus", "m")
    assert VerificationResult("uncertain", "m").verified is False
