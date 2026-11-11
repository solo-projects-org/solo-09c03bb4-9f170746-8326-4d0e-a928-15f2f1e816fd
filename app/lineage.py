"""Column-level dependency analysis over DuckDB SELECT models.

Only sqlglot parsing is used; SQL is never executed. The analyzer follows
column references across aliases, plain stars (including ``* EXCLUDE`` /
``* REPLACE``) and non-recursive CTEs, across multiple models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

DIALECT = "duckdb"


class AnalysisError(Exception):
    """A single analysis problem (unsupported SQL or unresolvable reference)."""

    def __init__(self, message: str, model: Optional[str] = None,
                 location: Optional[str] = None):
        self.message = message
        self.model = model
        self.location = location
        prefix = f"[{model}] " if model else ""
        loc = f"{location}: " if location else ""
        super().__init__(f"{prefix}{loc}{message}")


@dataclass
class ColumnNode:
    """One output column of one scope (a model or a CTE)."""

    scope_id: str
    output_name: str
    expression_sql: str
    parents: list["ColumnNode"] = field(default_factory=list)
    is_source: bool = False
    source_table: Optional[str] = None
    source_column: Optional[str] = None

    def label(self) -> str:
        if self.is_source:
            return f"{self.source_table}.{self.source_column}"
        return f"{self.scope_id}.{self.output_name}"

    def chains_to(self, targets: set[tuple[str, str]]) -> list[list[str]]:
        """All simple reference paths from this node down to target source columns."""
        target_tables = {t[0] for t in targets}
        target_columns = {t[1] for t in targets}
        found: list[list[str]] = []

        def walk(node: ColumnNode, seen: set[int]) -> list[list[str]]:
            if node.is_source:
                if _norm(node.source_table) in target_tables and \
                        _norm(node.source_column) in target_columns:
                    return [[node.label()]]
                return []
            paths: list[list[str]] = []
            for parent in node.parents:
                if id(parent) in seen:
                    continue
                for suffix in walk(parent, seen | {id(parent)}):
                    paths.append([node.label(), *suffix])
            return paths

        for chain in walk(self, {id(self)}):
            if chain not in found:
                found.append(chain)
        return found


@dataclass
class Relation:
    """A relation entry visible in a FROM/JOIN scope."""

    binding: str                      # alias if present else relation name (lower-cased)
    display: str                      # original spelling for messages
    columns: dict[str, ColumnNode]    # lower-cased column name -> node


def _norm(value: Optional[str]) -> str:
    return (value or "").lower()


class Analyzer:
    def __init__(self, sources: dict[str, list[str]],
                 models: dict[str, str],
                 targets: list[tuple[str, str]]):
        self.sources = {t.lower(): [c for c in cols] for t, cols in sources.items()}
        self.source_columns = {
            t.lower(): {c.lower(): c for c in cols}
            for t, cols in sources.items()
        }
        self.models = models
        self.targets = [(t.lower(), c.lower()) for t, c in targets]
        # scope_id -> {lower output name -> ColumnNode}
        self.scopes: dict[str, dict[str, ColumnNode]] = {}
        # cached source-column nodes so repeated table references share nodes
        self._source_nodes: dict[str, dict[str, ColumnNode]] = {}
        self.errors: list[AnalysisError] = []

    # ------------------------------------------------------------------ parse
    def parse_all(self) -> dict[str, exp.Select]:
        parsed: dict[str, exp.Select] = {}
        for name, sql in self.models.items():
            try:
                statements = sqlglot.parse(sql, read=DIALECT)
            except (ParseError, TokenError) as exc:
                self.errors.append(AnalysisError(f"SQL 解析失败: {exc}", model=name))
                continue
            real = [s for s in statements if s is not None]
            if len(real) != 1:
                self.errors.append(
                    AnalysisError("每个模型必须是且只能是一条 SELECT 语句", model=name))
                continue
            stmt = real[0]
            if not isinstance(stmt, exp.Select):
                self.errors.append(
                    AnalysisError(f"不支持集合运算（UNION/INTERSECT/EXCEPT），发现 {type(stmt).__name__}",
                                  model=name))
                continue
            parsed[name] = stmt
        return parsed

    # ------------------------------------------------- unsupported construct scan
    def validate(self, parsed: dict[str, exp.Select]) -> None:
        for name, stmt in parsed.items():
            with_node = stmt.args.get("with_")
            if with_node is not None and with_node.args.get("recursive"):
                self.errors.append(
                    AnalysisError("不支持递归 CTE（WITH RECURSIVE）", model=name, location="WITH"))
            for cte in stmt.ctes:
                if not isinstance(cte.this, exp.Select):
                    self.errors.append(
                        AnalysisError(f"CTE {cte.alias} 的主体不是普通 SELECT（可能含集合运算），不支持",
                                      model=name, location=f"CTE {cte.alias}"))
            for node in stmt.walk():
                if isinstance(node, exp.Union):
                    self.errors.append(
                        AnalysisError("不支持集合运算（UNION/INTERSECT/EXCEPT）",
                                      model=name, location=node.sql(DIALECT)[:60]))
                elif isinstance(node, exp.AggFunc):
                    self.errors.append(
                        AnalysisError(f"不支持聚合函数 {node.sql(DIALECT)}",
                                      model=name, location=node.sql(DIALECT)[:60]))
                elif isinstance(node, (exp.Group, exp.Having)):
                    kind = "GROUP BY" if isinstance(node, exp.Group) else "HAVING"
                    self.errors.append(
                        AnalysisError(f"不支持 {kind}", model=name,
                                      location=node.sql(DIALECT)[:60]))
                elif isinstance(node, exp.Subquery):
                    self.errors.append(
                        AnalysisError("不支持 CTE 之外的子查询（FROM/WHERE/标量子查询等）",
                                      model=name, location=node.sql(DIALECT)[:60]))
                elif isinstance(node, exp.Lateral):
                    self.errors.append(
                        AnalysisError("不支持 LATERAL 关联", model=name,
                                      location=node.sql(DIALECT)[:60]))
                elif isinstance(node, (exp.Unnest,)):
                    self.errors.append(
                        AnalysisError("不支持 UNNEST/表函数", model=name,
                                      location=node.sql(DIALECT)[:60]))
                elif isinstance(node, exp.Exists):
                    self.errors.append(
                        AnalysisError("不支持 CTE 之外的子查询（EXISTS 等）",
                                      model=name, location=node.sql(DIALECT)[:60]))
                elif isinstance(node, exp.Join) and node.args.get("method") == "NATURAL":
                    self.errors.append(
                        AnalysisError("不支持 NATURAL JOIN（连接列不明确），请改写为显式 JOIN ON",
                                      model=name, location=node.sql(DIALECT)[:60]))

    # ------------------------------------------------------- cross-model ordering
    def build_order(self, parsed: dict[str, exp.Select]) -> list[str]:
        cte_names = {
            name: {_norm(c.alias) for c in stmt.ctes}
            for name, stmt in parsed.items()
        }
        model_keys = {_norm(n): n for n in parsed}
        deps: dict[str, set[str]] = {n: set() for n in parsed}
        for name, stmt in parsed.items():
            for table in stmt.find_all(exp.Table):
                key = _norm(table.name)
                if key in model_keys and key != _norm(name) and key not in cte_names[name]:
                    deps[name].add(model_keys[key])
        ordered: list[str] = []
        state: dict[str, int] = {}  # 0 unvisited, 1 in progress, 2 done

        def visit(n: str, stack: list[str]) -> None:
            if state.get(n) == 2:
                return
            if state.get(n) == 1:
                cycle = " -> ".join(stack[stack.index(n):] + [n])
                self.errors.append(
                    AnalysisError(f"模型间存在循环依赖: {cycle}", model=n))
                return
            state[n] = 1
            for dep in sorted(deps[n]):
                visit(dep, stack + [n])
            state[n] = 2
            ordered.append(n)

        for n in sorted(parsed):
            visit(n, [])
        return ordered

    # ------------------------------------------------------------- scope helpers
    def _register_scope(self, scope_id: str) -> dict[str, ColumnNode]:
        columns: dict[str, ColumnNode] = {}
        self.scopes[scope_id] = columns
        return columns

    def _source_relation(self, table: exp.Table, model: str) -> Relation:
        tname = table.name
        key = _norm(tname)
        if key not in self.source_columns:
            if key in {_norm(n) for n in self.models}:
                # model reference: handled by caller, defensive guard
                raise AnalysisError(f"模型 {tname} 的引用位置无法解析", model=model,
                                    location=table.sql(DIALECT))
            raise AnalysisError(f"找不到源表 {tname}", model=model,
                                location=table.sql(DIALECT))
        if key in self._source_nodes:
            cols = dict(self._source_nodes[key])
        else:
            cols = {
                _norm(original): ColumnNode(
                    scope_id=tname,
                    output_name=original,
                    expression_sql=original,
                    is_source=True,
                    source_table=tname,
                    source_column=original,
                )
                for original in self.sources[key]
            }
            self._source_nodes[key] = cols
        binding = _norm(table.alias) or key
        return Relation(binding=binding, display=table.sql(DIALECT), columns=cols)

    def _model_relation(self, table: exp.Table, model_name: str, model: str) -> Relation:
        scope = self.scopes.get(_norm(model_name))
        if scope is None:
            raise AnalysisError(
                f"模型 {model_name} 尚未分析（依赖顺序异常）", model=model)
        # share the exact output nodes so reference chains do not duplicate hops
        binding = _norm(table.alias) or _norm(model_name)
        return Relation(binding=binding, display=table.sql(DIALECT),
                        columns=dict(scope))

    def _build_relations(self, scope_id: str, select: exp.Select,
                         cte_scopes: dict[str, tuple[str, dict[str, ColumnNode]]],
                         model: str) -> list[Relation]:
        relations: list[Relation] = []
        bindings: set[str] = set()

        def add_relation(table: exp.Table) -> None:
            tkey = _norm(table.name)
            alias = _norm(table.alias)
            binding = alias or tkey
            if binding in bindings:
                raise AnalysisError(
                    f"关系别名 {binding} 在同一查询中重复", model=model,
                    location=table.sql(DIALECT))
            if not alias and tkey in cte_scopes and tkey in bindings:
                raise AnalysisError(
                    f"CTE {table.name} 被重复引用且未提供别名，无法消歧",
                    model=model, location=table.sql(DIALECT))
            if tkey in cte_scopes:
                cte_scope_id, scope = cte_scopes[tkey]
                # share the CTE output nodes themselves
                relation = Relation(binding=binding, display=table.sql(DIALECT),
                                    columns=dict(scope))
            elif tkey in self.source_columns:
                relation = self._source_relation(table, model)
            elif tkey in {_norm(n) for n in self.models}:
                real_name = next(n for n in self.models if _norm(n) == tkey)
                relation = self._model_relation(table, real_name, model)
            else:
                raise AnalysisError(
                    f"找不到表或 CTE {table.name}", model=model,
                    location=table.sql(DIALECT))
            bindings.add(binding)
            relations.append(relation)

        from_node = select.args.get("from_")
        if from_node is None or not isinstance(from_node.this, exp.Table):
            raise AnalysisError("FROM 子句缺失或不是普通表", model=model)
        add_relation(from_node.this)
        for join in select.args.get("joins") or []:
            target = join.this
            if not isinstance(target, exp.Table):
                raise AnalysisError(
                    "JOIN 目标不是普通表（子查询/LATERAL 等不支持）",
                    model=model, location=target.sql(DIALECT))
            add_relation(target)
        return relations

    # --------------------------------------------------------- column resolution
    def _resolve_column(self, column: exp.Column, relations: list[Relation],
                        model: str) -> ColumnNode:
        col_name = column.name
        col_key = _norm(col_name)
        table_qualifier = _norm(column.table)
        if table_qualifier:
            matches = [r for r in relations if r.binding == table_qualifier]
            if not matches:
                raise AnalysisError(
                    f"列 {column.sql(DIALECT)} 引用的别名/表 {column.table} 不在当前查询中",
                    model=model, location=column.sql(DIALECT))
            relation = matches[0]
            node = relation.columns.get(col_key)
            if node is None:
                available = ", ".join(sorted(n.output_name for n in relation.columns.values()))
                raise AnalysisError(
                    f"关系 {relation.display} 中找不到列 {col_name}，可用列: {available}",
                    model=model, location=column.sql(DIALECT))
            return node
        candidates = [r for r in relations if col_key in r.columns]
        if not candidates:
            raise AnalysisError(
                f"找不到列 {col_name}", model=model, location=column.sql(DIALECT))
        if len(candidates) > 1:
            where = ", ".join(r.display for r in candidates)
            raise AnalysisError(
                f"列 {col_name} 存在歧义，可来自: {where}，请加表别名限定",
                model=model, location=column.sql(DIALECT))
        return candidates[0].columns[col_key]

    def _resolve_expression(self, node: exp.Expression, relations: list[Relation],
                            model: str) -> list[ColumnNode]:
        parents: list[ColumnNode] = []
        for column in node.find_all(exp.Column):
            resolved = self._resolve_column(column, relations, model)
            if resolved not in parents:
                parents.append(resolved)
        return parents

    # -------------------------------------------------------------- star expand
    def _star_spec(self, star: exp.Expression) -> exp.Star:
        """Return the Star node carrying EXCLUDE/REPLACE for both star shapes."""
        if isinstance(star, exp.Star):
            return star
        inner = star.this
        return inner if isinstance(inner, exp.Star) else exp.Star()

    def _expand_star(self, star: exp.Expression, relations: list[Relation],
                     output: dict[str, ColumnNode], ordered_names: list[str],
                     scope_id: str, model: str) -> None:
        spec = self._star_spec(star)
        excludes = {_norm(e.name if isinstance(e, exp.Column) else e.name)
                    for e in (spec.args.get("except_") or [])}
        replaces = spec.args.get("replace") or []
        if isinstance(star, exp.Column) and star.is_star:
            target_relations = [r for r in relations if r.binding == _norm(star.table)]
            if not target_relations:
                raise AnalysisError(
                    f"星号 {star.sql(DIALECT)} 引用的别名 {star.table} 不在当前查询中",
                    model=model, location=star.sql(DIALECT))
        else:
            target_relations = relations
        for relation in target_relations:
            for key, source_node in relation.columns.items():
                if key in excludes:
                    continue
                name = source_node.output_name
                out_key = _norm(name)
                if out_key in output:
                    raise AnalysisError(
                        f"星号展开后输出列 {name} 与已有输出列重名，请显式处理",
                        model=model, location=star.sql(DIALECT))
                replace_match = next(
                    (r for r in replaces
                     if _norm(r.this.name if isinstance(r.this, exp.Column) else r.this.name) == key),
                    None)
                if replace_match is not None:
                    expr = replace_match.this
                    parents = self._resolve_expression(expr, relations, model)
                    alias = replace_match.alias or name
                    node = ColumnNode(
                        scope_id=scope_id, output_name=alias,
                        expression_sql=expr.sql(DIALECT), parents=parents)
                else:
                    node = ColumnNode(
                        scope_id=scope_id, output_name=name,
                        expression_sql=source_node.expression_sql,
                        parents=[source_node])
                output[_norm(node.output_name)] = node
                ordered_names.append(node.output_name)

    # --------------------------------------------------------------- conditions
    def _record_condition(self, kind: str, expression: exp.Expression,
                          relations: list[Relation], scope_id: str,
                          model: str, sink: list[dict]) -> None:
        for column in expression.find_all(exp.Column):
            node = self._resolve_column(column, relations, model)
            chains = node.chains_to(set(self.targets))
            for chain in chains:
                sink.append({
                    "type": kind,
                    "scope": scope_id,
                    "expression": expression.sql(DIALECT),
                    "column": column.sql(DIALECT),
                    "reference_chain": chain,
                })

    # ----------------------------------------------------------------- one select
    def analyze_select(self, select: exp.Select, scope_id: str,
                       cte_scopes: dict[str, tuple[str, dict[str, ColumnNode]]],
                       model: str, condition_sink: list[dict]) -> dict[str, ColumnNode]:
        # CTEs first (non-recursive, lexical, can reference earlier CTEs)
        local_cte = dict(cte_scopes)
        for cte in select.ctes:
            cte_id = f"{scope_id}.{cte.alias}"
            cte_cols = self._register_scope(cte_id)
            cte_output = self.analyze_select(
                cte.this, cte_id, local_cte, model, condition_sink)
            cte_cols.update(cte_output)
            local_cte[_norm(cte.alias)] = (cte_id, cte_cols)

        relations = self._build_relations(scope_id, select, local_cte, model)
        output = self._register_scope(scope_id)
        ordered_names: list[str] = []

        for projection in select.expressions:
            if isinstance(projection, exp.Star) or (
                isinstance(projection, exp.Column) and projection.is_star):
                self._expand_star(projection, relations, output, ordered_names,
                                  scope_id, model)
                continue
            if isinstance(projection, exp.Alias):
                expr = projection.this
                out_name = projection.alias
            else:
                expr = projection
                if isinstance(expr, exp.Column):
                    out_name = expr.name
                else:
                    out_name = expr.sql(DIALECT)
            if _norm(out_name) in output:
                raise AnalysisError(
                    f"输出列 {out_name} 在同一 SELECT 中重复定义",
                    model=model, location=projection.sql(DIALECT)[:60])
            parents = self._resolve_expression(expr, relations, model)
            node = ColumnNode(
                scope_id=scope_id, output_name=out_name,
                expression_sql=expr.sql(DIALECT), parents=parents)
            output[_norm(out_name)] = node
            ordered_names.append(out_name)

        where = select.args.get("where")
        if where is not None:
            self._record_condition(
                "WHERE", where.this, relations, scope_id, model, condition_sink)
        for join in select.args.get("joins") or []:
            on = join.args.get("on")
            if on is not None:
                self._record_condition(
                    "JOIN ON", on, relations, scope_id, model, condition_sink)
            for ident in join.args.get("using") or []:
                # USING(col) equates col on both sides of the join; record each
                # side that actually exposes it.
                col_name = ident.name if isinstance(ident, exp.Identifier) else str(ident)
                for relation in relations:
                    node = relation.columns.get(_norm(col_name))
                    if node is None:
                        continue
                    for chain in node.chains_to(set(self.targets)):
                        condition_sink.append({
                            "type": "JOIN USING",
                            "scope": scope_id,
                            "expression": join.sql(DIALECT),
                            "column": f"{relation.display}.{col_name}",
                            "reference_chain": chain,
                        })
        qualify = select.args.get("qualify")
        if qualify is not None:
            self._record_condition(
                "QUALIFY", qualify.this, relations, scope_id, model, condition_sink)
        return output

    # ---------------------------------------------------------------------- run
    def run(self) -> dict:
        parsed = self.parse_all()
        self.validate(parsed)
        if self.errors:
            raise AggregatedAnalysisError(self.errors)
        order = self.build_order(parsed)
        if self.errors:
            raise AggregatedAnalysisError(self.errors)

        results: dict[str, dict] = {}
        target_set = set(self.targets)
        for model_name in order:
            conditions: list[dict] = []
            try:
                output = self.analyze_select(
                    parsed[model_name], model_name, {}, model_name, conditions)
            except AnalysisError as exc:
                if exc.model is None:
                    exc.model = model_name
                self.errors.append(exc)
                continue
            output_entries = []
            for key, node in output.items():
                chains = node.chains_to(target_set)
                if chains:
                    output_entries.append({
                        "output_column": node.output_name,
                        "reference_chains": chains,
                    })
            # de-duplicate condition entries
            seen: set[tuple] = set()
            cond_entries = []
            for item in conditions:
                fingerprint = (item["type"], item["scope"], item["expression"],
                               item["column"], tuple(map(tuple, item["reference_chain"])))
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    cond_entries.append(item)
            results[model_name] = {
                "dependent_output_columns": output_entries,
                "filter_conditions": cond_entries,
            }
        if self.errors:
            raise AggregatedAnalysisError(self.errors)
        # return in user-supplied model order
        return {name: results[name] for name in self.models if name in results}


class AggregatedAnalysisError(Exception):
    def __init__(self, errors: list[AnalysisError]):
        self.errors = errors
        super().__init__("; ".join(str(e) for e in errors))
