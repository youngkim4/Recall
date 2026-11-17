"""An invented archive for the tests of the report service.

Mina's one-to-one conversation holds four full months of 2023, of about
2,100 words each. Theo, who is not saved, has ten days of June 2024. The
chatter is the one of tests/report_engine/synth.py, pulled into a real archive.

The tests count one token for each word. The models of the registry have a
tokenizer family that is not confirmed, so the engine adds its margin of 1.35
to every count: a month is about 2,850 tokens. With a period budget of 3,400
each month is one period, and no two months fit one period.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from archive import Archive
from recall_services import reports
from recall_services.names import Names
from report_engine.fake_llm import FakeLLM
from tests.integration.report_data import MINA_CONTACT, Pulled
from report_engine import ReportOptions, Runtime
from tests.report_engine.support import LARGE, SMALL, runtime
from tests.report_engine.synth import MINA, THEO, WordCounter, conversation

MODEL = "gpt-6.1-sol"
OPTIONS = ReportOptions(
    period_token_budget=3_400,
    map_max_output_tokens=500,
    reduce_max_output_tokens=800,
    carry_over_token_budget=300,
)
PERIODS = 4
PERIOD_LABELS = ["Jan 2023", "Feb 2023", "Mar 2023", "Apr 2023"]
MINA_MESSAGES = len(LARGE)
THEO_MESSAGES = len(SMALL)


@dataclass
class Desk:
    """One archive, the names, and the places where a run writes."""

    archive: Archive
    names: Names
    places: reports.ReportPlaces
    pulled: Pulled
    # one clock for every run of a test: each run gets the next report id
    clock: Runtime

    def more(self, llm: Any = None) -> dict[str, Any]:
        """The keywords that every call of the service gets in these tests."""
        model = llm if llm is not None else FakeLLM()
        return {
            "archive": self.archive,
            "names": self.names,
            "places": self.places,
            "options": OPTIONS,
            "counter": WordCounter(),
            "llm_factory": lambda _info: model,
            "runtime": self.clock,
        }

    def estimate_more(self) -> dict[str, Any]:
        kept = ("archive", "names", "places", "options", "counter")
        return {name: value for name, value in self.more().items() if name in kept}


def build_desk(folder: Path) -> Desk:
    pulled = Pulled(folder / "world")
    pulled.write(MINA, LARGE)
    pulled.write(THEO, conversation("2024-06-01", days=10, per_day=4, others=(THEO,)))
    pulled.pull([MINA_CONTACT])
    places = reports.ReportPlaces(
        out_dir=folder / "out", store_dir=folder / "saves" / "reports", cache_path=folder / "saves" / "cache.sqlite3"
    )
    return Desk(pulled.open(), Names.from_records([MINA_CONTACT]), places, pulled, runtime())


def close_desk(desk: Desk) -> None:
    desk.archive.close()
    desk.pulled.close()
