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
from src.rl.portfolio_agent import TrainConfig, train_portfolio_agent

logger = logging.getLogger(__name__)

MAX_CONCURRENT_RUNS = 2
MAX_EPISODES = 3000
MIN_ASSETS, MAX_ASSETS = 2, 8
HISTORY_PERIOD = "5y"

_slots = threading.BoundedSemaphore(MAX_CONCURRENT_RUNS)

Loader = Callable[[list[str]], tuple[np.ndarray, list[str]]]

# The four agents the platform is built around. Only the portfolio agent trains
# today; the others are listed so the UI can show the roadmap honestly rather
# than pretend they exist.
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
        "status": "planned",
        "summary": "Will learn how to split monthly income across categories, goals and investing.",
        "method": "Planned: contextual bandit over your ledger history",
    },
    {
        "id": "timing",
        "name": "Trade timing",
        "status": "planned",
        "summary": "Will learn buy / hold / sell signals per ticker from price and technical features.",
        "method": "Planned: discrete-action policy, backtest only",
    },
    {
        "id": "alerts",
        "name": "Alert prioritiser",
        "status": "planned",
        "summary": "Will learn from your approve / reject decisions which monitor alerts matter to you.",
        "method": "Planned: Thompson-sampling bandit",
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


def _execute(run_id: str, tickers: list[str], config: TrainConfig, loader: Loader) -> None:
    curve: list[float] = []
    try:
        repo.update_run(run_id, status="RUNNING", stage="Downloading price history")
        returns, dates = loader(tickers)

        repo.update_run(run_id, stage=f"Training on {len(dates)} trading days")
        # Roughly a hundred progress writes per run, whatever its length.
        stride = max(1, -(-config.episodes // config.batch) // 100)

        def progress(done: int, total: int, reward: float) -> None:
            curve.append(round(reward, 6))
            if done == total or (len(curve) % stride) == 0:
                repo.update_run(run_id, episodes_done=done, curve=curve[-250:])

        result = train_portfolio_agent(returns, dates, tickers, config, on_progress=progress)
        repo.update_run(
            run_id,
            status="DONE",
            stage="Backtest complete",
            episodes_done=config.episodes,
            curve=result["learning_curve"],
            result=result,
        )
    except Exception as exc:  # noqa: BLE001 - anything that kills a run is reported on the run
        logger.warning("Training run %s failed: %s", run_id, exc)
        repo.update_run(run_id, status="FAILED", stage="Failed", error=str(exc)[:500])
    finally:
        _slots.release()


def start_portfolio_run(
    *,
    subject: str,
    tickers: list[str],
    risk_aversion: float,
    episodes: int,
    cost_bps: float,
    loader: Loader | None = None,
) -> str:
    """Validate, reserve a slot, record the run and start training it in the background."""
    clean = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    if not MIN_ASSETS <= len(clean) <= MAX_ASSETS:
        raise TrainingRefused(f"Choose between {MIN_ASSETS} and {MAX_ASSETS} tickers.", 422)
    episodes = int(max(50, min(episodes, MAX_EPISODES)))

    if repo.count_active_runs(subject=subject):
        raise TrainingRefused("You already have a training run in progress — wait for it to finish.")
    if not _slots.acquire(blocking=False):
        raise TrainingRefused("The training queue is full right now. Try again in a minute.", 503)

    try:
        params = {"risk_aversion": risk_aversion, "episodes": episodes, "cost_bps": cost_bps}
        run_id = repo.create_run(subject=subject, agent="portfolio", tickers=clean, params=params, episodes=episodes)
    except Exception:
        _slots.release()
        raise

    config = TrainConfig(episodes=episodes, risk_aversion=risk_aversion, cost_bps=cost_bps)
    _spawn(lambda: _execute(run_id, clean, config, loader or load_returns))
    return run_id
