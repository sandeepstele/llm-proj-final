import os

import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_read_rejects_path_outside_data():
    r = client.get("/read", params={"path": "../../etc/passwd"})
    assert r.status_code == 403


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)
def test_run_smoke_openrouter():
    r = client.post("/run", params={"task": "Say hello in one word."})
    assert r.status_code == 200
    data = r.json()
    assert data.get("status") == "success"
