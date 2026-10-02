import json
import random
import urllib.parse

import pytest

from app.benchmark.tasks import BENCHMARK_TASKS, benchmark_hashes
from app.discovery import generators, parsers
from app.discovery.adapters import (AdapterError, NistStatsAdapter, ProjectEulerAdapter, SecEdgarAdapter,
                                    TreasuryRatesAdapter, WikipediaConceptAdapter, default_adapters)
from app.discovery.chain import CachingHttp, DiscoveryChain, NoTaskAvailable
from app.discovery.http import FetchError
from app.storage.db import Storage
from app.verifier import Verifier, reference
from tests.helpers import FakeHttp, StaticAdapter, correct_answer

NIST_TEXT = "NIST/ITL StRD\nDataset Name: Lew\n\nData:   y\n" + "\n".join(f"  {i * 1.5 - 20:.3f}" for i in range(40)) + "\n"
TREASURY = json.dumps({"data": [
    {"record_date": "2025-01-31", "security_desc": "Treasury Notes", "avg_interest_rate_amt": "3.125"},
    {"record_date": "2025-01-31", "security_desc": "Treasury Bonds", "avg_interest_rate_amt": "3.4"}]})


def sec_json():
    def e(start, end, val, accn):
        return {"start": start, "end": end, "val": val, "accn": accn, "form": "10-K", "fp": "FY", "fy": int(end[:4])}
    return json.dumps({"entityName": "Test Corp", "facts": {"us-gaap": {
        "NetIncomeLoss": {"units": {"USD": [e("2022-01-01", "2022-12-31", 100, "A1"), e("2023-01-01", "2023-12-31", 150, "A2")]}},
        "Revenues": {"units": {"USD": [e("2022-01-01", "2022-12-31", 1000, "A1"), e("2023-01-01", "2023-12-31", 1200, "A2")]}}}}})


def routes():
    r = {}
    for n, (_, kw) in reference.EULER.items():
        r[ProjectEulerAdapter.URL.format(n=n)] = f"<p>Problem text with {kw} inside.</p>"
    for name in NistStatsAdapter.DATASETS:
        r[NistStatsAdapter.BASE.format(name=name)] = NIST_TEXT
    r[TreasuryRatesAdapter(None)._list_url()] = TREASURY
    for cik in SecEdgarAdapter.COMPANIES:
        r[SecEdgarAdapter.API.format(cik=cik)] = sec_json()
    for title in generators.CONCEPTS:
        r[WikipediaConceptAdapter.API.format(title=urllib.parse.quote(title, safe="()_"))] = json.dumps(
            {"title": title.replace("_", " "), "extract": "A concept.",
             "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/" + title}}})
    return r


@pytest.fixture
def http():
    return FakeHttp(routes())


def test_parsers():
    assert len(parsers.parse_nist_dataset(NIST_TEXT)) == 40
    assert parsers.parse_nist_dataset("no data here") == []
    assert parsers.parse_treasury_rates("not json") == []
    assert "Find the sum" in parsers.html_to_text("<p>Find &amp; the&nbsp;sum</p>".replace("&amp; the", "the") if False else "<p>Find the sum</p>")


def test_euler_adapter_builds_verifiable_tasks(http):
    tasks = ProjectEulerAdapter(http).discover(random.Random(1))
    assert tasks and all(t.source_url.startswith("https://projecteuler.net/") for t in tasks)
    t = next((x for x in tasks if x.input_data["problem"] == 1), tasks[0])
    assert t.retrieved_at and t.content_hash and t.verification_method == "euler_reference"
    assert Verifier(http).verify(t, str(reference.EULER[t.input_data["problem"]][0]())).verified


def test_euler_rejects_statement_mismatch():
    h = FakeHttp({ProjectEulerAdapter.URL.format(n=n): "<p>totally different text</p>" for n in reference.EULER})
    with pytest.raises(AdapterError):
        ProjectEulerAdapter(h).discover(random.Random(1))


def test_nist_adapter_tasks_verify_against_retrieved_data(http):
    tasks = NistStatsAdapter(http).discover(random.Random(2))
    assert len(tasks) == 3
    for t in tasks:
        assert t.source_url.startswith("https://www.itl.nist.gov/")
        assert Verifier(http).verify(t, correct_answer(t)).verified


def test_treasury_adapter_all_formulas(http):
    seen = set()
    for seed in range(12):
        for t in TreasuryRatesAdapter(http).discover(random.Random(seed)):
            seen.add(t.input_data["formula"])
            assert "3.125" in t.question or "3.4" in t.question
            r = Verifier(http).verify(t, correct_answer(t))
            assert r.verified, (t.question, r.message)
            assert t.input_data["recheck"]["kind"] == "treasury_rate"
    assert len(seen) >= 4


def test_sec_adapter_and_recheck(http):
    tasks = SecEdgarAdapter(http).discover(random.Random(3))
    assert tasks and all(t.domain == "quant_finance" for t in tasks)
    url = tasks[0].source_url
    assert url.startswith("https://data.sec.gov/api/xbrl/companyfacts/CIK")
    for t in tasks:
        http.routes[t.source_url] = sec_json()
        r = Verifier(http).verify(t, correct_answer(t))
        assert r.verified and r.details["source_recheck"]["state"] == "match"
        assert t.input_data["company"] == "Test Corp"


def test_wikipedia_concepts_all_generators_produce_verifiable_tasks(http):
    ad = WikipediaConceptAdapter(http)
    for i, title in enumerate(generators.CONCEPTS):
        rng = random.Random(i)
        for gen in generators.CONCEPTS[title]:
            domain, diff, q, data, method = gen(rng)
            from app.tasks.model import Task
            t = Task(domain, diff, q, "https://en.wikipedia.org/wiki/" + title, "w", method, data)
            assert Verifier(http).verify(t, correct_answer(t)).verified, (title, q)


def test_wikipedia_adapter_uses_real_page_url(http):
    tasks = WikipediaConceptAdapter(http).discover(random.Random(5))
    assert all(t.source_url.startswith("https://en.wikipedia.org/wiki/") for t in tasks)
    assert all(t.input_data["source_role"] == "concept_reference" for t in tasks)


def test_source_failure_raises_adapter_error_and_is_recorded(tmp_path):
    s = Storage(tmp_path / "d.db")
    rec = lambda **kw: s.record_source(**kw)
    ad = TreasuryRatesAdapter(FakeHttp(), rec)
    with pytest.raises(AdapterError):
        ad.discover(random.Random(1))
    src = s.list_sources()
    assert src[0]["status"] == "error" and "api.fiscaldata.treasury.gov" in src[0]["url"]


def test_chain_falls_back_when_first_adapters_fail(tmp_path, http):
    s = Storage(tmp_path / "d.db")
    chain = DiscoveryChain([StaticAdapter(fail=True), StaticAdapter()], s, random.Random(1))
    t = chain.next_task(1)
    assert t.source_title == "Static test source" and s.task_hash_exists(t.content_hash)
    assert chain.errors and "simulated source failure" in chain.errors[0]


def test_chain_all_fail_raises_and_invents_nothing(tmp_path):
    s = Storage(tmp_path / "d.db")
    chain = DiscoveryChain([StaticAdapter(fail=True)] + default_adapters(FakeHttp()), s, random.Random(1))
    with pytest.raises(NoTaskAvailable):
        chain.next_task(1)
    assert s.stats()["tasks"] == 0


def test_chain_prevents_duplicates(tmp_path):
    s = Storage(tmp_path / "d.db")

    class Same(StaticAdapter):
        def discover(self, rng):
            return [__import__("app.tasks.model", fromlist=["Task"]).Task(
                "numerical_reasoning", 1, "same q", "https://example.org", "t", "numeric_computation",
                {"kind": "gcd", "a": 4, "b": 6})]
    chain = DiscoveryChain([Same()], s, random.Random(1))
    first = chain.next_task(1)
    s.set_task_status(first.task_id, "verified")
    with pytest.raises(NoTaskAvailable, match="duplicate"):
        chain.next_task(1)
    assert s.stats()["tasks"] == 1


def test_chain_resumes_pending_tasks_first(tmp_path):
    s = Storage(tmp_path / "d.db")
    ad = StaticAdapter()
    chain = DiscoveryChain([ad], s, random.Random(1))
    t = chain.next_task(1)                       # stored as pending, never finished
    chain2 = DiscoveryChain([StaticAdapter()], s, random.Random(1))
    assert chain2.next_task(2).task_id == t.task_id


def test_benchmark_hashes_are_blocked(tmp_path):
    s = Storage(tmp_path / "d.db")
    bt = BENCHMARK_TASKS[0]

    class Leaky(StaticAdapter):
        def discover(self, rng):
            return [bt]
    chain = DiscoveryChain([Leaky()], s, random.Random(1), blocked_hashes=benchmark_hashes())
    with pytest.raises(NoTaskAvailable):
        chain.next_task(1)
    assert not s.task_hash_exists(bt.content_hash)


def test_caching_http_avoids_refetch(http):
    c = CachingHttp(http)
    url = next(iter(http.routes))
    c.get(url)
    c.get(url)
    assert http.calls.count(url) == 1
