from ai_shell import state


def test_task_checkpoint_lifecycle(monkeypatch):
    mem = {}

    def _load():
        return dict(mem)

    def _save(s):
        mem.clear()
        mem.update(dict(s))

    monkeypatch.setattr(state, "load_state", _load)
    monkeypatch.setattr(state, "save_state", _save)

    state.save_task_checkpoint("t1", {"spec": "do thing"})
    got = state.get_task_checkpoint("t1")
    assert got.get("task_id") == "t1"

    rows = state.list_task_checkpoints(limit=10)
    assert rows
    assert rows[0].get("task_id") == "t1"

    state.delete_task_checkpoint("t1")
    assert state.get_task_checkpoint("t1") == {}

