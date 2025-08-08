import time
import requests


def test_backend_stub() -> None:
    # Wait for containerized backend to come up (when running via compose)
    deadline = time.time() + 15
    ok = False
    while time.time() < deadline:
        try:
            r = requests.get("http://localhost:8000/healthz", timeout=1.0)
            ok = r.ok
            if ok:
                break
        except Exception:
            time.sleep(0.5)
    assert ok, "backend healthz failed"

    resp = requests.post(
        "http://localhost:8000/query",
        json={"query": "What is APH-IF?"},
        timeout=3.0,
    )
    assert resp.ok
    data = resp.json()
    assert isinstance(data.get("answer"), str) and "Stub answer" in data["answer"]


