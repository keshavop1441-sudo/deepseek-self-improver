"""Public-source adapters. Every URL here is a real public endpoint; nothing is invented.

Each adapter retrieves live data through the injected HttpClient and returns candidate Tasks
built from what was retrieved. Failures raise AdapterError so the chain can fall back.
"""
from __future__ import annotations

import json
import random
import re
import urllib.parse
from datetime import date
from typing import Callable

from app.discovery import generators, parsers
from app.discovery.http import FetchError, HttpClient, Response
from app.tasks.model import Task, utc_now
from app.verifier import reference

# recorder(adapter, url, title, status, http_status, error, content_hash)
Recorder = Callable[..., None]


class AdapterError(Exception):
    pass


class Adapter:
    name = "base"
    domains: tuple[str, ...] = ()

    def __init__(self, http: HttpClient, recorder: Recorder | None = None):
        self.http = http
        self._rec = recorder

    def _get(self, url: str, title: str | None = None, headers: dict | None = None) -> Response:
        try:
            resp = self.http.get(url, headers=headers)
        except FetchError as e:
            self._record(url, title, "error", e.status, str(e))
            raise AdapterError(f"{self.name}: {e}") from e
        self._record(url, title, "ok", resp.status, None)
        return resp

    def _record(self, url, title, status, http_status, error) -> None:
        if self._rec:
            self._rec(adapter=self.name, url=url, title=title, status=status, http_status=http_status, error=error)

    def discover(self, rng: random.Random) -> list[Task]:  # pragma: no cover - interface
        raise NotImplementedError


# --------------------------------------------------------------------------- Project Euler
class ProjectEulerAdapter(Adapter):
    name = "project_euler"
    domains = ("mathematics", "numerical_reasoning")
    URL = "https://projecteuler.net/minimal={n}"

    def discover(self, rng: random.Random) -> list[Task]:
        order = list(reference.EULER)
        rng.shuffle(order)
        tasks: list[Task] = []
        errors: list[str] = []
        for n in order[:4]:
            url = self.URL.format(n=n)
            try:
                text = parsers.html_to_text(self._get(url, f"Project Euler Problem {n}").text)
            except AdapterError as e:
                errors.append(str(e))
                continue
            keyword = reference.EULER[n][1]
            if keyword.lower() not in " ".join(text.lower().split()):
                errors.append(f"problem {n}: statement did not contain expected text '{keyword}'; skipped")
                continue
            tasks.append(Task(
                domain="mathematics", difficulty=3 if n < 10 else 4,
                question=("Solve this Project Euler problem. Give the final integer answer only in the `answer` field.\n\n"
                          + text[:1800]),
                source_url=url, source_title=f"Project Euler Problem {n}", verification_method="euler_reference",
                input_data={"problem": n, "recheck": {"kind": "text_contains", "needle": keyword}}))
        if not tasks:
            raise AdapterError("project_euler: " + ("; ".join(errors) or "no tasks"))
        return tasks


# --------------------------------------------------------------------------- NIST StRD
class NistStatsAdapter(Adapter):
    name = "nist_strd"
    domains = ("statistics",)
    BASE = "https://www.itl.nist.gov/div898/strd/univ/data/{name}.dat"
    DATASETS = ("Lew", "Lottery", "Mavro", "Michelso", "PiDigits")
    STATS = ("mean", "median", "sample_stdev", "range", "sample_variance", "population_stdev")

    def discover(self, rng: random.Random) -> list[Task]:
        names = list(self.DATASETS)
        rng.shuffle(names)
        errors = []
        for name in names[:3]:
            url = self.BASE.format(name=name)
            try:
                resp = self._get(url, f"NIST StRD univariate dataset {name}")
            except AdapterError as e:
                errors.append(str(e))
                continue
            vals = parsers.parse_nist_dataset(resp.text)
            if len(vals) < 12:
                errors.append(f"{name}: could not parse dataset ({len(vals)} values)")
                continue
            size = rng.randint(8, 12)
            off = rng.randint(0, len(vals) - size)
            window = vals[off:off + size]
            tasks = []
            for stat in rng.sample(self.STATS, 3):
                label = stat.replace("_", " ")
                tasks.append(Task(
                    domain="statistics", difficulty=2 if stat in ("mean", "range", "median") else 3,
                    question=(f"These {size} values are rows {off + 1}-{off + size} of the NIST StRD dataset '{name}':\n"
                              f"{', '.join(window)}\nCompute the {label} of exactly these values, rounded to 4 decimal places."
                              " Put only the number in the `answer` field."),
                    source_url=url, source_title=f"NIST StRD univariate dataset {name}",
                    verification_method="stats_computation",
                    input_data={"values": window, "statistic": stat, "decimals": 4, "dataset": name, "offset": off,
                                "recheck": {"kind": "nist_dataset", "values": window, "offset": off}}))
            return tasks
        raise AdapterError("nist_strd: " + "; ".join(errors))


# --------------------------------------------------------------------------- US Treasury Fiscal Data
class TreasuryRatesAdapter(Adapter):
    name = "us_treasury"
    domains = ("financial_calculation", "quant_finance")
    API = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v2/accounting/od/avg_interest_rates"

    def _list_url(self) -> str:
        return f"{self.API}?sort=-record_date&page%5Bsize%5D=40&format=json"

    def _record_url(self, rec_date: str, desc: str) -> str:
        flt = urllib.parse.quote(f"record_date:eq:{rec_date},security_desc:eq:{desc}", safe=":,-")
        return f"{self.API}?filter={flt}&format=json"

    def discover(self, rng: random.Random) -> list[Task]:
        url = self._list_url()
        resp = self._get(url, "U.S. Treasury Fiscal Data - Average Interest Rates on Treasury Securities")
        rows = [r for r in parsers.parse_treasury_rates(resp.text) if 0 < float(r["rate_pct"]) < 20]
        if not rows:
            raise AdapterError("us_treasury: response contained no usable rate rows")
        rng.shuffle(rows)
        tasks = []
        for row in rows[:4]:
            r, desc, rec = row["rate_pct"], row["security_desc"], row["record_date"]
            recheck = {"kind": "treasury_rate", "url": self._record_url(rec, desc), "record_date": rec,
                       "security_desc": desc, "rate_pct": r}
            ctx = f"U.S. Treasury Fiscal Data reports an average interest rate of {r}% on '{desc}' as of {rec}. "
            base = {"unit": "USD", "record_date": rec, "recheck": recheck}
            kind = rng.choice(["fv", "pv", "bond", "ear", "loan"])
            if kind == "fv":
                p, y = rng.choice([1000, 2500, 5000, 10000]), rng.randint(2, 10)
                q = ctx + f"Using that rate as an annual rate compounded annually, what is ${p} worth after {y} years? Round to 2 decimals in dollars."
                d = {"formula": "future_value", "principal": p, "rate_pct": r, "periods_per_year": 1, "years": y, "decimals": 2}
            elif kind == "pv":
                fv, y = rng.choice([1000, 5000, 20000]), rng.randint(2, 10)
                q = ctx + f"Using it as the annual discount rate, what is the present value of ${fv} received in {y} years? Round to 2 decimals in dollars."
                d = {"formula": "present_value", "future_value": fv, "rate_pct": r, "years": y, "decimals": 2}
            elif kind == "bond":
                n, ytm = rng.randint(2, 6), f"{float(r) + rng.choice([0.5, 1.0, -0.5]):.2f}"
                q = ctx + (f"A {n}-year bond with face value $1000 pays an annual coupon at that rate. If the yield to maturity is {ytm}% "
                           "(annual compounding), what is its price today? Round to 2 decimals in dollars.")
                d = {"formula": "bond_price", "face": 1000, "coupon_rate_pct": r, "ytm_pct": ytm, "years": n, "decimals": 2}
            elif kind == "ear":
                m = rng.choice([2, 4, 12])
                q = ctx + f"If that nominal annual rate is compounded {m} times per year, what is the effective annual rate in percent? Round to 4 decimals."
                d = {"formula": "effective_annual_rate_pct", "rate_pct": r, "periods_per_year": m, "decimals": 4}
            else:
                p, y = rng.choice([10000, 25000, 200000]), rng.choice([3, 5, 10, 15])
                q = ctx + f"A ${p} loan at that nominal annual rate is repaid in equal monthly payments over {y} years. What is the monthly payment? Round to 2 decimals in dollars."
                d = {"formula": "loan_payment", "principal": p, "annual_rate_pct": r, "years": y, "payments_per_year": 12, "decimals": 2}
            d.update(base)
            tasks.append(Task(domain="financial_calculation", difficulty=3, question=q + " Put only the number in the `answer` field.",
                              source_url=url, source_title="U.S. Treasury Fiscal Data - Average Interest Rates on Treasury Securities",
                              verification_method="finance_formula", input_data=d))
        return tasks


# --------------------------------------------------------------------------- SEC EDGAR
class SecEdgarAdapter(Adapter):
    name = "sec_edgar"
    domains = ("quant_finance", "financial_calculation")
    # Real SEC central index keys of large public companies (verified at retrieval via entityName).
    COMPANIES = {"0000320193": "Apple", "0000789019": "Microsoft", "0001018724": "Amazon", "0001045810": "NVIDIA",
                 "0000200406": "Johnson & Johnson", "0000021344": "Coca-Cola", "0000104169": "Walmart",
                 "0000019617": "JPMorgan Chase"}
    API = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    REVENUE_TAGS = ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet")

    @staticmethod
    def _annual(entries: list[dict]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for e in entries:
            try:
                s, en = date.fromisoformat(e["start"]), date.fromisoformat(e["end"])
            except (KeyError, ValueError):
                continue
            if e.get("form") == "10-K" and e.get("fp") == "FY" and 350 <= (en - s).days <= 380:
                out.setdefault(e["end"], e)
        return out

    def discover(self, rng: random.Random) -> list[Task]:
        ciks = list(self.COMPANIES)
        rng.shuffle(ciks)
        errors = []
        for cik in ciks[:3]:
            url = self.API.format(cik=cik)
            try:
                resp = self._get(url, f"SEC EDGAR company facts (CIK {cik})")
            except AdapterError as e:
                errors.append(str(e))
                continue
            name = parsers.sec_entity_name(resp.text) or self.COMPANIES[cik]
            ni = self._annual(parsers.sec_facts(resp.text, "us-gaap", "NetIncomeLoss"))
            rev, tag = {}, None
            for t in self.REVENUE_TAGS:
                rev = self._annual(parsers.sec_facts(resp.text, "us-gaap", t))
                if rev:
                    tag = t
                    break
            ends = sorted(set(ni) & set(rev))
            if not ends or tag is None:
                errors.append(f"{cik}: no usable annual NetIncomeLoss+revenue facts")
                continue
            tasks = []
            fact = lambda t, e: {"taxonomy": "us-gaap", "tag": t, "unit": "USD", "start": e["start"], "end": e["end"],
                                 "accn": e["accn"], "val": e["val"]}
            title = f"SEC EDGAR XBRL company facts - {name}"
            for end in rng.sample(ends, min(2, len(ends))):
                e_ni, e_rev = ni[end], rev[end]
                if e_rev["val"] <= 0:
                    continue
                tasks.append(Task(
                    domain="quant_finance", difficulty=3,
                    question=(f"{name} reported (SEC EDGAR XBRL, 10-K, fiscal year {e_rev['start']} to {end}) revenue of "
                              f"${e_rev['val']:,} and net income of ${e_ni['val']:,}. What is its net profit margin in percent "
                              "(net income / revenue x 100), rounded to 2 decimals? Put only the number in the `answer` field."),
                    source_url=url, source_title=title, verification_method="finance_formula",
                    input_data={"formula": "net_margin_pct", "net_income": e_ni["val"], "revenue": e_rev["val"], "unit": "USD",
                                "start_date": e_rev["start"], "end_date": end, "decimals": 2, "company": name, "cik": cik,
                                "recheck": {"kind": "sec_facts", "facts": [fact("NetIncomeLoss", e_ni), fact(tag, e_rev)]}}))
                prev = [x for x in ends if x < end]
                if prev:
                    p_end = prev[-1]
                    if 330 <= (date.fromisoformat(end) - date.fromisoformat(p_end)).days <= 400 and rev[p_end]["val"] > 0:
                        tasks.append(Task(
                            domain="quant_finance", difficulty=3,
                            question=(f"{name}'s annual revenue per SEC 10-K filings was ${rev[p_end]['val']:,} for the year ended {p_end} "
                                      f"and ${e_rev['val']:,} for the year ended {end}. What is the year-over-year revenue growth in "
                                      "percent, rounded to 2 decimals? Put only the number in the `answer` field."),
                            source_url=url, source_title=title, verification_method="finance_formula",
                            input_data={"formula": "growth_pct", "current": e_rev["val"], "prior": rev[p_end]["val"], "unit": "USD",
                                        "start_date": rev[p_end]["end"], "end_date": end, "decimals": 2, "company": name, "cik": cik,
                                        "recheck": {"kind": "sec_facts", "facts": [fact(tag, e_rev), fact(tag, rev[p_end])]}}))
            if tasks:
                return tasks
            errors.append(f"{cik}: facts present but unusable")
        raise AdapterError("sec_edgar: " + "; ".join(errors))


# --------------------------------------------------------------------------- Wikipedia concept references
class WikipediaConceptAdapter(Adapter):
    name = "wikipedia_concepts"
    domains = ("probability", "logic", "numerical_reasoning", "mathematics", "coding")
    API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

    def discover(self, rng: random.Random) -> list[Task]:
        titles = list(generators.CONCEPTS)
        rng.shuffle(titles)
        errors = []
        for title in titles[:4]:
            url = self.API.format(title=urllib.parse.quote(title, safe="()_"))
            try:
                resp = self._get(url, title.replace("_", " "))
                page = json.loads(resp.text)
            except AdapterError as e:
                errors.append(str(e))
                continue
            except ValueError:
                errors.append(f"{title}: non-JSON reply")
                continue
            page_url = ((page.get("content_urls") or {}).get("desktop") or {}).get("page")
            page_title = page.get("title")
            if not page_url or not page_title:
                errors.append(f"{title}: reply lacked title/page URL")
                continue
            blurb = re.sub(r"\s+", " ", page.get("extract") or "")[:220]
            tasks = []
            for gen in generators.CONCEPTS[title]:
                for _ in range(3):  # a few parameter variants; the chain de-duplicates
                    domain, diff, q, data, method = gen(rng)
                    data = dict(data, source_role="concept_reference")
                    tasks.append(Task(domain=domain, difficulty=diff,
                                      question=f"(Concept reference: Wikipedia - {page_title}: {blurb})\n{q}",
                                      source_url=page_url, source_title=f"Wikipedia: {page_title}",
                                      verification_method=method, input_data=data, retrieved_at=utc_now()))
            return tasks
        raise AdapterError("wikipedia_concepts: " + "; ".join(errors))


def default_adapters(http: HttpClient, recorder: Recorder | None = None) -> list[Adapter]:
    return [cls(http, recorder) for cls in (WikipediaConceptAdapter, ProjectEulerAdapter, NistStatsAdapter,
                                            TreasuryRatesAdapter, SecEdgarAdapter)]
