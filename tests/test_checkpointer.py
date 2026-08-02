# ═══════════════════════════════════════════════════════
# FinSight — Tests: Checkpointer
# ═══════════════════════════════════════════════════════
#
# Uses a trivial two-node graph rather than the research graph: the point is
# the checkpointer's behaviour, and a real graph would need an LLM.
# ═══════════════════════════════════════════════════════

from __future__ import annotations

import operator
from typing import Annotated, TypedDict
from unittest.mock import patch

import pytest

from src.persistence import checkpointer as checkpointer_module
from src.persistence.checkpointer import async_checkpointer, sync_checkpointer


class _State(TypedDict, total=False):
    steps: Annotated[list[str], operator.add]


def _tiny_graph(saver):
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(_State)
    graph.add_node("first", lambda s: {"steps": ["first"]})
    graph.add_node("second", lambda s: {"steps": ["second"]})
    graph.add_edge(START, "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)
    return graph.compile(checkpointer=saver)


@pytest.fixture
def temp_checkpoint(tmp_path):
    """Point the checkpointer at a throwaway database file."""
    with patch.object(checkpointer_module, "CHECKPOINT_DB", tmp_path / "checkpoints.sqlite"):
        yield tmp_path / "checkpoints.sqlite"


class TestSyncCheckpointer:
    def test_the_database_file_is_created(self, temp_checkpoint):
        with sync_checkpointer():
            pass
        assert temp_checkpoint.exists()

    def test_wal_is_enabled_on_the_checkpoint_database(self, temp_checkpoint):
        """
        journal_mode is persistent DATABASE state, not connection state — set
        once before the saver connects and every later connection inherits it.
        That is the only reason we can configure a database whose connection
        the saver owns.
        """
        import sqlite3

        with sync_checkpointer():
            pass

        connection = sqlite3.connect(temp_checkpoint)
        try:
            assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        finally:
            connection.close()

    def test_state_survives_the_run(self, temp_checkpoint):
        config = {"configurable": {"thread_id": "t1"}}
        with sync_checkpointer() as saver:
            graph = _tiny_graph(saver)
            graph.invoke({"steps": []}, config=config)
            assert graph.get_state(config).values["steps"] == ["first", "second"]

    def test_the_history_holds_one_entry_per_superstep(self, temp_checkpoint):
        # The audit trail is this: not a summary written afterwards, but the
        # actual intermediate states the graph passed through.
        config = {"configurable": {"thread_id": "t1"}}
        with sync_checkpointer() as saver:
            graph = _tiny_graph(saver)
            graph.invoke({"steps": []}, config=config)
            assert len(list(graph.get_state_history(config))) >= 3

    def test_threads_are_isolated_from_each_other(self, temp_checkpoint):
        with sync_checkpointer() as saver:
            graph = _tiny_graph(saver)
            graph.invoke({"steps": []}, config={"configurable": {"thread_id": "a"}})
            state = graph.get_state({"configurable": {"thread_id": "b"}})
            assert not state.values

    def test_state_outlives_the_process_that_wrote_it(self, temp_checkpoint):
        """
        THE point of a file-backed checkpointer, and what Phase 7's approval
        gate depends on: the API can be killed between the interrupt and the
        approval and the run still resumes.
        """
        config = {"configurable": {"thread_id": "t1"}}

        with sync_checkpointer() as saver:
            _tiny_graph(saver).invoke({"steps": []}, config=config)

        # Everything above is closed — a new saver, as a restarted process
        # would build.
        with sync_checkpointer() as saver:
            assert _tiny_graph(saver).get_state(config).values["steps"] == ["first", "second"]


class TestAsyncCheckpointer:
    async def test_state_survives_the_run(self, temp_checkpoint):
        config = {"configurable": {"thread_id": "t1"}}
        async with async_checkpointer() as saver:
            graph = _tiny_graph(saver)
            await graph.ainvoke({"steps": []}, config=config)
            state = await graph.aget_state(config)
            assert state.values["steps"] == ["first", "second"]

    async def test_it_reads_what_the_sync_saver_wrote(self, temp_checkpoint):
        """
        One database, two access paths. A run started from the CLI has to be
        visible to the API's audit-trail endpoint.
        """
        config = {"configurable": {"thread_id": "shared"}}

        with sync_checkpointer() as saver:
            _tiny_graph(saver).invoke({"steps": []}, config=config)

        async with async_checkpointer() as saver:
            state = await _tiny_graph(saver).aget_state(config)
            assert state.values["steps"] == ["first", "second"]
