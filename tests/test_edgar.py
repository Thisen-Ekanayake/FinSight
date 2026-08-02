# ═══════════════════════════════════════════════════════
# FinSight — Tests: SEC EDGAR Client
# ═══════════════════════════════════════════════════════
# Offline against recorded fixtures. Network tests are marked integration.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

from src.core.errors import DataSourceError, MissingCredentialError
from src.data import edgar
from src.data.config import validate_sec_user_agent

FIXTURES = Path(__file__).parent / "fixtures" / "edgar"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def _patched_fetch(*, tickers=None, submissions=None, companyfacts=None):
    """Route fetch_json to fixtures based on the URL being requested."""

    def fake(provider, url, **kwargs):
        if "company_tickers" in url:
            return tickers if tickers is not None else _fixture("company_tickers.json")
        if "submissions" in url:
            return submissions if submissions is not None else _fixture("submissions_aapl.json")
        if "companyfacts" in url:
            return companyfacts if companyfacts is not None else _fixture("companyfacts_aapl.json")
        raise AssertionError(f"unexpected URL in test: {url}")

    return patch("src.data.edgar.fetch_json", side_effect=fake)


class TestSecUserAgent:
    """The SEC 403s clients without a contact address. Catch it before they do."""

    def test_rejects_empty(self):
        with patch("src.data.config.SEC_USER_AGENT", ""):
            with pytest.raises(MissingCredentialError, match="not set"):
                validate_sec_user_agent()

    def test_rejects_the_shipped_placeholder(self):
        with patch("src.data.config.SEC_USER_AGENT", "FinSight/0.1 (your.email@example.com)"):
            with pytest.raises(MissingCredentialError, match="placeholder"):
                validate_sec_user_agent()

    def test_rejects_value_without_an_email(self):
        with patch("src.data.config.SEC_USER_AGENT", "FinSight/0.1"):
            with pytest.raises(MissingCredentialError, match="contact email"):
                validate_sec_user_agent()

    def test_accepts_a_real_contact(self):
        with patch("src.data.config.SEC_USER_AGENT", "FinSight/0.1 (dev@realdomain.com)"):
            assert "realdomain.com" in validate_sec_user_agent()

    def test_the_configured_value_is_valid(self):
        # Guards against the placeholder surviving into a real .env.
        assert validate_sec_user_agent()


class TestResolveCik:
    def setup_method(self):
        edgar._TICKER_MAP_CACHE = {}

    def teardown_method(self):
        edgar._TICKER_MAP_CACHE = {}

    def test_resolves_a_known_ticker(self):
        with _patched_fetch():
            assert edgar.resolve_cik("AAPL") == "0000320193"

    def test_is_case_insensitive(self):
        with _patched_fetch():
            assert edgar.resolve_cik("aapl") == edgar.resolve_cik("AAPL")

    def test_pads_cik_to_ten_digits(self):
        with _patched_fetch():
            assert len(edgar.resolve_cik("AAPL")) == 10

    def test_unknown_ticker_explains_the_us_only_limitation(self):
        with _patched_fetch():
            with pytest.raises(DataSourceError, match="US-listed"):
                edgar.resolve_cik("NESN")  # Swiss-listed; EDGAR has nothing

    def test_map_is_fetched_once_and_memoised(self):
        with _patched_fetch() as mock:
            edgar.resolve_cik("AAPL")
            edgar.resolve_cik("MSFT")
            assert mock.call_count == 1


class TestGetFilingIndex:
    def setup_method(self):
        edgar._TICKER_MAP_CACHE = {}

    def teardown_method(self):
        edgar._TICKER_MAP_CACHE = {}

    def test_returns_filings(self):
        with _patched_fetch():
            filings = edgar.get_filing_index("AAPL")
        assert filings
        assert all(f["accession_no"] for f in filings)

    def test_form_filter_is_respected(self):
        with _patched_fetch():
            filings = edgar.get_filing_index("AAPL", forms=["10-Q"])
        assert filings
        assert {f["form_type"] for f in filings} == {"10-Q"}

    def test_limit_is_respected(self):
        with _patched_fetch():
            assert len(edgar.get_filing_index("AAPL", limit=3)) <= 3

    def test_since_excludes_older_filings(self):
        cutoff = date(2030, 1, 1)  # far future -> nothing qualifies
        with _patched_fetch():
            assert edgar.get_filing_index("AAPL", since=cutoff) == []

    def test_accession_numbers_match_the_sec_format(self):
        import re

        with _patched_fetch():
            filings = edgar.get_filing_index("AAPL", limit=5)
        # This exact format is asserted by the Phase 5 source_id_validity
        # evaluator, which catches fabricated citation IDs.
        assert all(re.fullmatch(r"\d{10}-\d{2}-\d{6}", f["accession_no"]) for f in filings)

    def test_urls_point_at_sec_archives(self):
        with _patched_fetch():
            filings = edgar.get_filing_index("AAPL", limit=3)
        assert all(f["url"].startswith("https://www.sec.gov/Archives/edgar/data/") for f in filings)

    def test_8k_item_codes_are_parsed(self):
        # Item codes drive severity in Phase 6 — 4.02 is an automatic HIGH.
        with _patched_fetch():
            filings = edgar.get_filing_index("AAPL", forms=["8-K"])
        for filing in filings:
            assert isinstance(filing["items"], list)
            assert all("," not in item for item in filing["items"])

    def test_empty_submissions_yields_no_filings(self):
        with _patched_fetch(submissions={"filings": {}}):
            assert edgar.get_filing_index("AAPL") == []


class TestCompanyFacts:
    """XBRL is the authoritative numeric path — every fact must be citable."""

    def setup_method(self):
        edgar._TICKER_MAP_CACHE = {}

    def teardown_method(self):
        edgar._TICKER_MAP_CACHE = {}

    def test_returns_requested_concepts(self):
        with _patched_fetch():
            facts = edgar.get_company_facts("0000320193", concepts=["NetIncomeLoss"])
        assert "NetIncomeLoss" in facts
        assert facts["NetIncomeLoss"]

    def test_every_fact_carries_an_accession_number(self):
        # A number that cannot be cited has no place in this system.
        with _patched_fetch():
            facts = edgar.get_company_facts("0000320193", concepts=["NetIncomeLoss"])
        assert all(f["accession_no"] for f in facts["NetIncomeLoss"])

    def test_facts_are_sorted_oldest_first(self):
        with _patched_fetch():
            facts = edgar.get_company_facts("0000320193", concepts=["NetIncomeLoss"])
        periods = [f["period_end"] for f in facts["NetIncomeLoss"]]
        assert periods == sorted(periods)

    def test_missing_concepts_are_omitted_not_raised(self):
        # Concept coverage varies by filer; absence is normal, not an error.
        with _patched_fetch():
            facts = edgar.get_company_facts("0000320193", concepts=["NoSuchConceptXYZ"])
        assert facts == {}

    def test_entries_without_accession_are_dropped(self):
        payload = {
            "facts": {
                "us-gaap": {
                    "NetIncomeLoss": {
                        "label": "Net Income",
                        "units": {
                            "USD": [
                                {"val": 100, "end": "2024-01-01", "accn": "0000320193-24-000123", "fy": 2024},
                                {"val": 200, "end": "2024-06-01", "fy": 2024},  # no accn
                            ]
                        },
                    }
                }
            }
        }
        with _patched_fetch(companyfacts=payload):
            facts = edgar.get_company_facts("0000320193", concepts=["NetIncomeLoss"])
        assert len(facts["NetIncomeLoss"]) == 1

    def test_ticker_is_resolved_to_cik_automatically(self):
        with _patched_fetch():
            assert edgar.get_company_facts("AAPL", concepts=["NetIncomeLoss"])

    def test_get_latest_fact_returns_the_newest(self):
        with _patched_fetch():
            facts = edgar.get_company_facts("0000320193", concepts=["NetIncomeLoss"])["NetIncomeLoss"]
            latest = edgar.get_latest_fact("0000320193", "NetIncomeLoss")
        assert latest == facts[-1]

    def test_get_latest_fact_can_filter_to_annual(self):
        with _patched_fetch():
            latest = edgar.get_latest_fact("0000320193", "NetIncomeLoss", form_type="10-K")
        assert latest is None or latest["form_type"] == "10-K"


class TestFilingUrl:
    def test_strips_dashes_and_leading_zeros_correctly(self):
        filing = {
            "cik": "0000320193",
            "accession_no": "0000320193-24-000123",
            "primary_document": "aapl-20240928.htm",
        }
        url = edgar.filing_url(filing)  # type: ignore[arg-type]
        assert url == ("https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm")


@pytest.mark.integration
class TestLiveEdgar:
    """Against the real SEC. Requires network and a valid SEC_USER_AGENT."""

    def test_resolve_apple(self):
        edgar._TICKER_MAP_CACHE = {}
        assert edgar.resolve_cik("AAPL") == "0000320193"

    def test_fetch_recent_10k(self):
        filings = edgar.get_filing_index("AAPL", forms=["10-K"], limit=1)
        assert len(filings) == 1
        assert filings[0]["form_type"] == "10-K"

    def test_revenue_fact_is_plausible(self):
        fact = edgar.get_latest_fact("AAPL", "RevenueFromContractWithCustomerExcludingAssessedTax")
        assert fact is not None
        assert fact["value"] > 1e10  # Apple's revenue is in the hundreds of billions
        assert fact["unit"] == "USD"
