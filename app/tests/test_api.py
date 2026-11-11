"""API-level tests for /analyze."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

SAMPLE = {
    "sources": [
        {"name": "waybills",
         "columns": ["waybill_id", "receiver_zip_old", "city_id", "weight_kg"]},
        {"name": "dim_delivery_zone",
         "columns": ["zone_id", "zip_code", "zone_name"]},
    ],
    "models": [
        {
            "name": "stg_waybill",
            "sql": ("WITH cleaned AS ("
                    " SELECT waybill_id, receiver_zip_old AS zip, city_id, weight_kg"
                    " FROM waybills WHERE receiver_zip_old IS NOT NULL"
                    ") SELECT * FROM cleaned"),
        },
        {
            "name": "dwd_delivery_fact",
            "sql": ("SELECT s.waybill_id, s.zip AS receiver_zip, z.zone_name, s.weight_kg"
                    " FROM stg_waybill s JOIN dim_delivery_zone z ON s.zip = z.zip_code"
                    " WHERE s.zip LIKE '10%'"),
        },
    ],
    "targets": [{"table": "waybills", "column": "receiver_zip_old"}],
}


def test_health():
    assert client.get("/healthz").json() == {"status": "ok"}


def test_sample_end_to_end():
    resp = client.post("/analyze", json=SAMPLE)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    stg = body["models"]["stg_waybill"]
    assert stg["dependent_output_columns"] == [{
        "output_column": "zip",
        "reference_chains": [[
            "stg_waybill.zip", "stg_waybill.cleaned.zip",
            "waybills.receiver_zip_old",
        ]],
    }]
    assert stg["filter_conditions"][0]["type"] == "WHERE"
    fact = body["models"]["dwd_delivery_fact"]
    assert {e["output_column"] for e in fact["dependent_output_columns"]} == {"receiver_zip"}
    assert {c["type"] for c in fact["filter_conditions"]} == {"WHERE", "JOIN ON"}
    # chains cross the model boundary (and retain the CTE hop from the star)
    chain = fact["dependent_output_columns"][0]["reference_chains"][0]
    assert chain == ["dwd_delivery_fact.receiver_zip", "stg_waybill.zip",
                     "stg_waybill.cleaned.zip", "waybills.receiver_zip_old"]


def test_ambiguous_column_returns_400_with_model():
    payload = {
        "sources": SAMPLE["sources"],
        "models": [{
            "name": "bad_model",
            "sql": ("SELECT receiver_zip_old FROM waybills a"
                    " JOIN waybills b ON a.waybill_id = b.waybill_id"),
        }],
        "targets": SAMPLE["targets"],
    }
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert detail[0]["model"] == "bad_model"
    assert "歧义" in detail[0]["message"]


def test_unsupported_aggregate_returns_400():
    payload = {
        "sources": SAMPLE["sources"],
        "models": [{"name": "agg", "sql": "SELECT count(*) FROM waybills"}],
        "targets": SAMPLE["targets"],
    }
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 400
    assert "聚合" in resp.json()["detail"][0]["message"]


def test_unknown_target_returns_400():
    payload = dict(SAMPLE)
    payload["targets"] = [{"table": "waybills", "column": "nope"}]
    resp = client.post("/analyze", json=payload)
    assert resp.status_code == 400


def test_swagger_examples_present():
    schema = client.get("/openapi.json").json()
    example = (schema["components"]["schemas"]["AnalyzeRequest"]
               ["examples"][0])
    assert example["targets"][0]["column"] == "receiver_zip_old"
    assert "/analyze" in schema["paths"]


def test_swagger_example_consecutive_join_using_is_submittable():
    schema = client.get("/openapi.json").json()
    example = (schema["components"]["schemas"]["AnalyzeRequest"]
               ["examples"][0])
    # the built-in sample must demonstrate consecutive JOIN USING
    fact_sql = next(m["sql"] for m in example["models"]
                    if m["name"] == "dwd_delivery_fact")
    assert fact_sql.count("USING") >= 2
    resp = client.post("/analyze", json=example)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    fact = body["models"]["dwd_delivery_fact"]
    # unqualified merged key + WHERE on it resolve through the left side
    assert {c["type"] for c in fact["filter_conditions"]} == {"WHERE", "JOIN USING"}
    chain = fact["dependent_output_columns"][0]["reference_chains"][0]
    assert chain == ["dwd_delivery_fact.receiver_zip", "stg_waybill.zip",
                     "stg_waybill.cleaned.zip", "waybills.receiver_zip_old"]
    # the downstream model keeps the chain intact
    ads = body["models"]["ads_zip_filtered"]
    assert ads["dependent_output_columns"][0]["reference_chains"][0][-1] == \
        "waybills.receiver_zip_old"
