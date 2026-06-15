"""
Pytest configuration: shared fixtures + result recorder.
Writes a JSON report to 'test results/' after every session.
"""
import json
import os
import sys
import time
from datetime import datetime

import pytest

# Make project root importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "test results")


# ── Shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def movies_df():
    """Load the movie DataFrame once for the entire test session."""
    from recommender import load_movies
    return load_movies()


@pytest.fixture(scope="session")
def small_candidates(movies_df):
    """Pre-built comedy candidates used across multiple tests."""
    from recommender import get_candidates
    return get_candidates(movies_df, ["comedy", "funny", "light and fun", "skip", "skip"])


# ── Result recorder ────────────────────────────────────────────────────────────

class ResultRecorder:
    def __init__(self):
        self.results = []
        self.session_start = datetime.now().isoformat()

    def record(self, report):
        self.results.append({
            "test": report.nodeid,
            "outcome": report.outcome,       # passed / failed / error
            "duration_s": round(report.duration, 3),
            "message": self._extract_message(report),
        })

    def _extract_message(self, report) -> str:
        if report.outcome == "passed":
            return "OK"
        if hasattr(report, "longreprtext"):
            return report.longreprtext[:500]
        if hasattr(report, "longrepr") and report.longrepr:
            return str(report.longrepr)[:500]
        return ""

    def save(self):
        os.makedirs(RESULTS_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(RESULTS_DIR, f"results_{ts}.json")

        passed = sum(1 for r in self.results if r["outcome"] == "passed")
        failed = sum(1 for r in self.results if r["outcome"] == "failed")
        errored = sum(1 for r in self.results if r["outcome"] == "error")

        report = {
            "session_start": self.session_start,
            "session_end": datetime.now().isoformat(),
            "summary": {
                "total": len(self.results),
                "passed": passed,
                "failed": failed,
                "errored": errored,
            },
            "tests": self.results,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n📄 Test results saved → {path}")
        return path


_recorder = ResultRecorder()


def pytest_runtest_logreport(report):
    if report.when == "call":
        _recorder.record(report)
    elif report.when == "setup" and report.outcome == "error":
        _recorder.record(report)


def pytest_sessionfinish(session, exitstatus):
    _recorder.save()
