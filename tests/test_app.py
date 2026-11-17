from fastapi.testclient import TestClient

import ragpdp.app as web


def test_rate_limit_per_ip(monkeypatch):
    monkeypatch.setattr(web, "answer", lambda idx, q, history: ([], iter(["ok"])))
    monkeypatch.setattr(web, "_asked", web.defaultdict(web.deque))
    client = TestClient(web.app)
    ask = lambda ip: client.post("/api/ask", json={"question": "tes"}, headers={"x-real-ip": ip})

    per_minute = web.LIMITS[0][1]
    assert all(ask("1.1.1.1").status_code == 200 for _ in range(per_minute))
    assert ask("1.1.1.1").status_code == 429
    assert ask("2.2.2.2").status_code == 200  # other visitors are unaffected
