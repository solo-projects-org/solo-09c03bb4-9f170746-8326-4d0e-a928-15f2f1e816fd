"""Tests for the column lineage analyzer (parse-only, no SQL execution)."""
import pytest

from app.lineage import AggregatedAnalysisError, Analyzer

SOURCES = {
    "waybills": ["waybill_id", "receiver_zip_old", "city_id", "weight_kg", "ts"],
    "dim_delivery_zone": ["zone_id", "zip_code", "zone_name"],
}
TARGET = [("waybills", "receiver_zip_old")]


def analyze(models, targets=TARGET, sources=SOURCES):
    return Analyzer(sources=sources, models=models, targets=targets).run()


def chain_sets(result, model, output_column):
    for entry in result[model]["dependent_output_columns"]:
        if entry["output_column"] == output_column:
            return {tuple(c) for c in entry["reference_chains"]}
    return set()


def test_alias_rename_and_where_in_cte():
    result = analyze({
        "stg_waybill": (
            "WITH cleaned AS ("
            " SELECT waybill_id, receiver_zip_old AS zip, city_id"
            " FROM waybills WHERE receiver_zip_old IS NOT NULL"
            ") SELECT waybill_id, zip, city_id FROM cleaned"
        ),
    })
    assert chain_sets(result, "stg_waybill", "zip") == {
        ("stg_waybill.zip", "stg_waybill.cleaned.zip", "waybills.receiver_zip_old"),
    }
    conds = result["stg_waybill"]["filter_conditions"]
    assert len(conds) == 1
    assert conds[0]["type"] == "WHERE"
    assert conds[0]["scope"] == "stg_waybill.cleaned"
    assert conds[0]["reference_chain"][-1] == "waybills.receiver_zip_old"
    # condition dependency must not be counted as an output column
    assert {e["output_column"] for e in result["stg_waybill"]["dependent_output_columns"]} == {"zip"}


def test_cross_model_join_on_and_where():
    result = analyze({
        "stg_waybill": (
            "SELECT waybill_id, receiver_zip_old AS zip, city_id, weight_kg FROM waybills"
        ),
        "dwd_delivery_fact": (
            "SELECT s.waybill_id, s.zip AS receiver_zip, z.zone_name, s.weight_kg"
            " FROM stg_waybill s JOIN dim_delivery_zone z ON s.zip = z.zip_code"
            " WHERE s.zip LIKE '10%'"
        ),
    })
    assert chain_sets(result, "dwd_delivery_fact", "receiver_zip") == {
        ("dwd_delivery_fact.receiver_zip", "stg_waybill.zip",
         "waybills.receiver_zip_old"),
    }
    types = {(c["type"], c["scope"]) for c in result["dwd_delivery_fact"]["filter_conditions"]}
    assert ("WHERE", "dwd_delivery_fact") in types
    assert ("JOIN ON", "dwd_delivery_fact") in types
    # zone_name and weight_kg are independent
    assert chain_sets(result, "dwd_delivery_fact", "zone_name") == set()


def test_plain_star_expansion():
    result = analyze({
        "m": "SELECT * FROM waybills",
    })
    assert chain_sets(result, "m", "receiver_zip_old") == {
        ("m.receiver_zip_old", "waybills.receiver_zip_old"),
    }
    # non-target columns still listed as outputs but without dependency
    assert {e["output_column"] for e in result["m"]["dependent_output_columns"]} == {
        "receiver_zip_old"}


def test_star_exclude_and_qualified_star():
    result = analyze({
        "m": "SELECT w.* EXCLUDE (receiver_zip_old), w.waybill_id AS id FROM waybills w",
    })
    assert result["m"]["dependent_output_columns"] == []


def test_qualified_star_kept_and_alias_chain():
    result = analyze({
        "m": "SELECT w.*, weight_kg * 1000 AS weight_g FROM waybills w",
    })
    assert chain_sets(result, "m", "receiver_zip_old") == {
        ("m.receiver_zip_old", "waybills.receiver_zip_old"),
    }


def test_multiple_reference_chains_via_coalesce():
    result = analyze({
        "m": "SELECT COALESCE(receiver_zip_old, city_id) AS zipish FROM waybills",
    })
    # only one chain reaches the target; city_id path ends elsewhere
    chains = chain_sets(result, "m", "zipish")
    assert chains == {("m.zipish", "waybills.receiver_zip_old")}


def test_self_join_two_chains():
    result = analyze({
        "m": (
            "SELECT a.waybill_id, a.receiver_zip_old AS z1, b.receiver_zip_old AS z2"
            " FROM waybills a JOIN waybills b ON a.waybill_id = b.waybill_id"
        ),
    })
    assert chain_sets(result, "m", "z1") == {
        ("m.z1", "waybills.receiver_zip_old")}
    assert chain_sets(result, "m", "z2") == {
        ("m.z2", "waybills.receiver_zip_old")}


def test_case_insensitive_names():
    result = analyze({
        "M": "select Receiver_Zip_Old as ZIP from WAYBILLS",
    })
    assert "M" in result
    assert chain_sets(result, "M", "ZIP")


def test_unqualified_ambiguous_column_rejected():
    with pytest.raises(AggregatedAnalysisError) as exc:
        analyze({
            "m": (
                "SELECT receiver_zip_old FROM waybills a JOIN waybills b"
                " ON a.waybill_id = b.waybill_id"
            ),
        })
    msg = str(exc.value)
    assert "歧义" in msg and "m" in msg


def test_unknown_qualified_table_rejected():
    with pytest.raises(AggregatedAnalysisError) as exc:
        analyze({"m": "SELECT x.receiver_zip_old FROM waybills w"})
    assert "m" in str(exc.value) and "x" in str(exc.value)


def test_unknown_column_rejected():
    with pytest.raises(AggregatedAnalysisError) as exc:
        analyze({"m": "SELECT not_a_col FROM waybills"})
    assert "找不到列 not_a_col" in str(exc.value)
    assert exc.value.errors[0].model == "m"


def test_unknown_relation_rejected():
    with pytest.raises(AggregatedAnalysisError) as exc:
        analyze({"m": "SELECT 1 FROM ghost_table"})
    assert "找不到表或 CTE ghost_table" in str(exc.value)


def test_aggregation_rejected():
    with pytest.raises(AggregatedAnalysisError) as exc:
        analyze({"m": "SELECT count(*) FROM waybills GROUP BY city_id"})
    msg = str(exc.value)
    assert "聚合" in msg or "GROUP BY" in msg


def test_set_operation_rejected():
    with pytest.raises(AggregatedAnalysisError) as exc:
        analyze({"m": "SELECT receiver_zip_old FROM waybills UNION SELECT receiver_zip_old FROM waybills"})
    assert "集合运算" in str(exc.value)


def test_subquery_in_from_rejected():
    with pytest.raises(AggregatedAnalysisError) as exc:
        analyze({"m": "SELECT x.zip FROM (SELECT receiver_zip_old AS zip FROM waybills) x"})
    assert "子查询" in str(exc.value)


def test_scalar_subquery_rejected():
    with pytest.raises(AggregatedAnalysisError) as exc:
        analyze({"m": "SELECT (SELECT max(waybill_id) FROM waybills) AS v FROM dim_delivery_zone"})
    assert "子查询" in str(exc.value)


def test_recursive_cte_rejected():
    with pytest.raises(AggregatedAnalysisError) as exc:
        analyze({
            "m": (
                "WITH RECURSIVE r AS ("
                " SELECT waybill_id AS n FROM waybills"
                " UNION ALL SELECT n + 1 FROM r WHERE n < 10"
                ") SELECT * FROM r"
            ),
        })
    assert "递归 CTE" in str(exc.value)


def test_model_cycle_rejected():
    with pytest.raises(AggregatedAnalysisError) as exc:
        analyze({
            "a": "SELECT 1 AS x FROM b",
            "b": "SELECT 1 AS x FROM a",
        })
    assert "循环依赖" in str(exc.value)


def test_parse_error_reports_model():
    with pytest.raises(AggregatedAnalysisError) as exc:
        analyze({"m": "SELECT FROM FROM"})
    assert exc.value.errors[0].model == "m"
    assert "解析失败" in str(exc.value)


def test_target_not_in_filter_does_not_report_condition():
    result = analyze({
        "m": "SELECT waybill_id FROM waybills WHERE city_id = 7",
    })
    assert result["m"]["filter_conditions"] == []
    assert result["m"]["dependent_output_columns"] == []


def test_cte_chain_through_multiple_levels():
    result = analyze({
        "m": (
            "WITH c1 AS (SELECT waybill_id, receiver_zip_old AS zip FROM waybills),"
            " c2 AS (SELECT waybill_id, zip AS postcode FROM c1 WHERE zip <> '')"
            " SELECT waybill_id, postcode FROM c2"
        ),
    })
    assert chain_sets(result, "m", "postcode") == {
        ("m.postcode", "m.c2.postcode", "m.c1.zip", "waybills.receiver_zip_old"),
    }
    assert result["m"]["filter_conditions"][0]["scope"] == "m.c2"


def test_star_replace_tracks_expression_lineage():
    result = analyze({
        "m": "SELECT * REPLACE (upper(receiver_zip_old) AS receiver_zip_old) FROM waybills",
    })
    assert chain_sets(result, "m", "receiver_zip_old") == {
        ("m.receiver_zip_old", "waybills.receiver_zip_old"),
    }


def test_multiple_targets():
    result = analyze(
        {
            "m": "SELECT waybill_id, receiver_zip_old, city_id FROM waybills",
        },
        targets=[("waybills", "receiver_zip_old"), ("waybills", "city_id")],
    )
    deps = {e["output_column"] for e in result["m"]["dependent_output_columns"]}
    assert deps == {"receiver_zip_old", "city_id"}


def test_order_by_does_not_create_condition_entry():
    # ORDER BY/LIMIT are presentation clauses, not filters; analysis must still succeed
    result = analyze({
        "m": ("SELECT waybill_id, receiver_zip_old FROM waybills"
              " ORDER BY receiver_zip_old LIMIT 10"),
    })
    assert result["m"]["filter_conditions"] == []
    assert chain_sets(result, "m", "receiver_zip_old")


def test_distinct_and_join_on_with_alias():
    result = analyze({
        "m": (
            "SELECT DISTINCT w.waybill_id, w.receiver_zip_old AS z"
            " FROM waybills w JOIN dim_delivery_zone d ON w.city_id = d.zone_id"
            " WHERE w.receiver_zip_old = d.zip_code"
        ),
    })
    cond_types = {c["type"] for c in result["m"]["filter_conditions"]}
    # JOIN ON only touches city_id/zone_id, so only the WHERE filter is reported
    assert cond_types == {"WHERE"}
    join_entries = [c for c in result["m"]["filter_conditions"] if c["type"] == "JOIN ON"]
    assert join_entries == []


def test_empty_model_result_still_listed():
    result = analyze({
        "unrelated": "SELECT zone_id, zone_name FROM dim_delivery_zone",
        "related": "SELECT receiver_zip_old AS zip FROM waybills",
    })
    assert result["unrelated"] == {
        "dependent_output_columns": [], "filter_conditions": []}
    assert result["related"]["dependent_output_columns"]


def test_using_join_condition_is_tracked():
    sources = dict(SOURCES)
    sources["dim_delivery_zone"] = ["zone_id", "receiver_zip_old", "zone_name"]
    result = analyze({
        "m": ("SELECT w.waybill_id FROM waybills w"
              " JOIN dim_delivery_zone d USING (receiver_zip_old)"),
    }, sources=sources)
    conds = result["m"]["filter_conditions"]
    assert len(conds) == 1
    assert conds[0]["type"] == "JOIN USING"
    assert conds[0]["reference_chain"][-1] == "waybills.receiver_zip_old"


def test_natural_join_rejected():
    with pytest.raises(AggregatedAnalysisError) as exc:
        analyze({
            "m": "SELECT * FROM waybills NATURAL JOIN dim_delivery_zone",
        })
    assert "NATURAL" in str(exc.value)


def test_exists_subquery_rejected_with_clear_message():
    with pytest.raises(AggregatedAnalysisError) as exc:
        analyze({
            "m": ("SELECT receiver_zip_old FROM waybills w WHERE EXISTS"
                  " (SELECT 1 FROM dim_delivery_zone d WHERE d.zip_code = w.receiver_zip_old)"),
        })
    assert "子查询" in str(exc.value)


def test_cte_name_shadows_source_table():
    result = analyze({
        "m": ("WITH waybills AS (SELECT receiver_zip_old AS x FROM waybills)"
              " SELECT x FROM waybills"),
    })
    assert chain_sets(result, "m", "x") == {
        ("m.x", "m.waybills.x", "waybills.receiver_zip_old"),
    }


def test_duplicate_output_name_rejected():
    with pytest.raises(AggregatedAnalysisError) as exc:
        analyze({"m": "SELECT receiver_zip_old, receiver_zip_old FROM waybills"})
    assert "重复" in str(exc.value)
