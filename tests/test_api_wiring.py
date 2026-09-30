"""API wiring smoke tests. Verifies routes register and health responds."""

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient  # noqa: E402


def test_openapi_routes_present():
    from api.main import app

    paths = set(app.openapi().get("paths", {}).keys())
    expected = {
        "/api/health",
        "/api/upload",
        "/api/upload/{scan_id}",
        "/api/predict/{scan_id}",
        "/api/explain/{scan_id}",
        "/api/report/{scan_id}",
        "/api/metrics/benchmarks",
        "/api/metrics/calibration",
        "/api/metrics/conformal",
    }
    missing = expected - paths
    assert not missing, f"missing routes: {missing}"


def test_health_endpoint_responds():
    from api.main import app

    with TestClient(app) as client:
        res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    for key in ("status", "yolo_loaded", "rfdetr_loaded",
                "calibration_ready", "conformal_ready", "device"):
        assert key in body


def test_predict_404_before_upload():
    from api.main import app

    with TestClient(app) as client:
        res = client.post("/api/predict/does-not-exist")
    assert res.status_code in (404, 503)


def test_report_404_when_no_prediction():
    from api.main import app

    with TestClient(app) as client:
        res = client.get("/api/report/does-not-exist")
    assert res.status_code == 404


def test_metrics_benchmarks_reachable():
    """Returns 200 with real data once scripts/regenerate_all_numbers.py has
    run (runs/final_eval/comprehensive_evaluation.json exists), else 404.
    Both are correct depending on repo state — this just asserts the route
    doesn't 500."""
    from pathlib import Path

    from api.main import app

    with TestClient(app) as client:
        res = client.get("/api/metrics/benchmarks")

    report_path = (
        Path(__file__).resolve().parents[1]
        / "runs" / "final_eval" / "comprehensive_evaluation.json"
    )
    expected = 200 if report_path.exists() else 404
    assert res.status_code == expected
