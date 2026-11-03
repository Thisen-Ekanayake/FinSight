# ═══════════════════════════════════════════════════════
# FinSight — RL training service
# ═══════════════════════════════════════════════════════
#
# Purpose : Start agent training in the background, write its progress to the
#           database as it goes, and say which agents exist.
#
# Public API:
#   AGENTS, start_portfolio_run(...), load_returns(tickers), TrainingRefused
#
# ══ WHY A THREAD AND NOT A QUEUE ══
#   A training run is a few seconds to a minute of numpy on one core. A worker
#   fleet would be more machinery than the job. The cost is that a restart
#   kills in-flight runs — fail_stale_runs() marks those FAILED at boot so the
#   UI never shows a progress bar that will never move again.
#
# ══ LIMITS ══
#   One active run per account, MAX_CONCURRENT_RUNS across the server, and an
#   episode cap. Training is free to the user but not to the host.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

import numpy as np

from src.finance import repository as repo
from src.rl.budget_agent import BudgetConfig, profile_from_ledger, train_budget_agent
from src.rl.portfolio_agent import TrainConfig, train_portfolio_agent
from src.rl.timing_agent import TimingConfig, train_timing_agent

logger = logging.getLogger(__name__)

MAX_CONCURRENT_RUNS = 2
MAX_EPISODES = 3000
MIN_ASSETS, MAX_ASSETS = 2, 8
HISTORY_PERIOD = "5y"

_slots = threading.BoundedSemaphore(MAX_CONCURRENT_RUNS)

Loader = Callable[[list[str]], tuple[np.ndarray, list[str]]]

# The four agents the platform is built around.
AGENTS: list[dict[str, Any]] = [
    {
        "id": "portfolio",
        "name": "Portfolio allocator",
        "status": "available",
        "summary": "Learns target weights across your tickers and cash, trading return against risk and turnover.",
        "method": "Policy gradient (REINFORCE) over a linear-softmax policy",
    },
    {
        "id": "budget",
        "name": "Budget & savings planner",
        "status": "available",
        "summary": "Learns what share of income to save for your situation, then turns it into limits and goal plans.",
        "method": "Linear Thompson-sampling contextual bandit, trained in a simulator fitted to your ledger",
    },
    {
        "id": "timing",
        "name": "Trade timing",
        "status": "available",
        "summary": "Learns when to be invested or in cash, per ticker. Signals and backtests only — never orders.",
        "method": "Policy gradient over a logistic long/flat policy, weekly decisions",
    },
    {
        "id": "alerts",
        "name": "Alert prioritiser",
        "status": "available",
        "summary": "Learns from your approve / reject decisions and feedback which monitor alerts matter to you.",
        "method": "Beta-Bernoulli Thompson sampling over alert type, severity and ticker",
    },
]


class TrainingRefused(RuntimeError):
    """A run could not start. ``status`` is the HTTP status the API should return."""

    def __init__(self, message: str, status: int = 409) -> None:
        super().__init__(message)
        self.status = status


def load_returns(tickers: list[str]) -> tuple[np.ndarray, list[str]]:
    """Daily simple returns for ``tickers`` on the dates every one of them traded."""
    from src.data.prices import get_prices

    bars = get_prices(tickers, period=HISTORY_PERIOD)
    missing = [t for t in tickers if t not in bars]
    if missing:
        raise ValueError(f"No price history for {', '.join(missing)}")

    closes = [{bar["date"]: bar["close"] for bar in bars[t]} for t in tickers]
    common = sorted(set.intersection(*(set(c) for c in closes)))
    if len(common) < 200:
        raise ValueError(f"Only {len(common)} overlapping trading days — need at least 200")

    matrix = np.array([[c[d] for c in closes] for d in common], dtype=np.float64)
    returns = matrix[1:] / matrix[:-1] - 1.0
    return returns, common[1:]


def _spawn(target: Callable[[], None]) -> None:
    threading.Thread(target=target, name="rl-train", daemon=True).start()


Job = Callable[[Callable[[int, int, float], None], Callable[[str], None]], tuple[dict[str, Any], list[float]]]


def _execute(run_id: str, job: Job, total: int) -> None:
    """
    Run one training job, writing progress to its row.

    ``job(progress, stage)`` does the work: it reports ``progress(done, total,
    reward)`` as it goes and ``stage(text)`` at phase changes, and returns
    ``(result, final_curve)``.
    """
    curve: list[float] = []
    # Roughly a hundred progress writes per run, whatever its length.
    stride = [1]

    def stage(text: str) -> None:
        repo.update_run(run_id, stage=text)

    def progress(done: int, total_: int, reward: float) -> None:
        curve.append(round(reward, 6))
        if len(curve) == 1:
            stride[0] = max(1, total_ // max(1, done) // 100)
        if done == total_ or (len(curve) % stride[0]) == 0:
            repo.update_run(run_id, episodes_done=done, curve=curve[-250:])

    try:
        repo.update_run(run_id, status="RUNNING")
        result, final_curve = job(progress, stage)
        repo.update_run(
            run_id,
            status="DONE",
            stage="Complete",
            episodes_done=total,
            curve=final_curve,
            result=result,
        )
    except Exception as exc:  # noqa: BLE001 - anything that kills a run is reported on the run
        logger.warning("Training run %s failed: %s", run_id, exc)
        repo.update_run(run_id, status="FAILED", stage="Failed", error=str(exc)[:500])
    finally:
        _slots.release()


def _launch(*, subject: str, agent: str, tickers: list[str], params: dict[str, Any], total: int, job: Job) -> str:
    """Enforce the run limits, record the run, and start ``job`` in the background."""
    if repo.count_active_runs(subject=subject):
        raise TrainingRefused("You already have a training run in progress — wait for it to finish.")
    if not _slots.acquire(blocking=False):
        raise TrainingRefused("The training queue is full right now. Try again in a minute.", 503)
    try:
        run_id = repo.create_run(subject=subject, agent=agent, tickers=tickers, params=params, episodes=total)
    except Exception:
        _slots.release()
        raise
    _spawn(lambda: _execute(run_id, job, total))
    return run_id


def _clean_tickers(tickers: list[str]) -> list[str]:
    clean = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    if not MIN_ASSETS <= len(clean) <= MAX_ASSETS:
        raise TrainingRefused(f"Choose between {MIN_ASSETS} and {MAX_ASSETS} tickers.", 422)
    return clean


def start_portfolio_run(
    *,
    subject: str,
    tickers: list[str],
    risk_aversion: float,
    episodes: int,
    cost_bps: float,
    loader: Loader | None = None,
) -> str:
    """Train the portfolio allocator in the background."""
    clean = _clean_tickers(tickers)
    episodes = int(max(50, min(episodes, MAX_EPISODES)))
    config = TrainConfig(episodes=episodes, risk_aversion=risk_aversion, cost_bps=cost_bps)

    def job(
        progress: Callable[[int, int, float], None], stage: Callable[[str], None]
    ) -> tuple[dict[str, Any], list[float]]:
        stage("Downloading price history")
        returns, dates = (loader or load_returns)(clean)
        stage(f"Training on {len(dates)} trading days")
        result = train_portfolio_agent(returns, dates, clean, config, on_progress=progress)
        return result, result["learning_curve"]

    params = {"risk_aversion": risk_aversion, "episodes": episodes, "cost_bps": cost_bps}
    return _launch(subject=subject, agent="portfolio", tickers=clean, params=params, total=episodes, job=job)


def start_timing_run(
    *,
    subject: str,
    tickers: list[str],
    risk_aversion: float,
    episodes: int,
    cost_bps: float,
    loader: Loader | None = None,
) -> str:
    """Train the long/flat timing agent in the background. Signals only — never orders."""
    clean = _clean_tickers(tickers)
    episodes = int(max(50, min(episodes, MAX_EPISODES)))
    config = TimingConfig(episodes=episodes, risk_aversion=risk_aversion, cost_bps=cost_bps)

    def job(
        progress: Callable[[int, int, float], None], stage: Callable[[str], None]
    ) -> tuple[dict[str, Any], list[float]]:
        stage("Downloading price history")
        returns, dates = (loader or load_returns)(clean)
        stage(f"Training on {len(dates)} trading days")
        result = train_timing_agent(returns, dates, clean, config, on_progress=progress)
        return result, result["learning_curve"]

    params = {"risk_aversion": risk_aversion, "episodes": episodes, "cost_bps": cost_bps}
    return _launch(subject=subject, agent="timing", tickers=clean, params=params, total=episodes, job=job)


def start_budget_run(*, subject: str, rounds: int = 3000) -> str:
    """Fit the household simulator to this account's ledger and train the budget bandit on it."""
    rounds = int(max(500, min(rounds, 10_000)))

    def job(
        progress: Callable[[int, int, float], None], stage: Callable[[str], None]
    ) -> tuple[dict[str, Any], list[float]]:
        stage("Reading your ledger")
        profile = profile_from_ledger(
            repo.list_entries(subject=subject, limit=5000),
            repo.list_categories(subject=subject),
            repo.list_goals(subject=subject),
        )
        stage(f"Simulating {rounds:,} months of your household")
        result = train_budget_agent(profile, BudgetConfig(rounds=rounds), on_progress=progress)
        return result, result["learning_curve"]

    return _launch(subject=subject, agent="budget", tickers=[], params={"rounds": rounds}, total=rounds, job=job)
