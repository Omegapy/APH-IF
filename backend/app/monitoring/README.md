# Monitoring Module (backend/app/monitoring)

Observability and resilience primitives for the APH-IF backend:
- Performance monitoring and aggregated system metrics
- Hierarchical timing collection with low overhead
- Database (Neo4j) operation metrics and analysis
- Circuit breaker protection and health reporting

Designed to be non-invasive: wrap operations without changing business logic.

---

## Components

### Performance Monitor (`performance_monitor.py`)
Tracks operation timings and aggregates them into system metrics. Produces dashboards and
health summaries for the API layer.

Key types and APIs:
- `PerformanceMonitor`
  - `track_operation(operation: str, metadata: dict | None = None)` (async context manager)
  - `record_metric(operation: str, duration_ms: int, success: bool, metadata: dict | None)`
  - `get_current_stats(time_window_minutes: int = 60) -> SystemMetrics`
  - `get_performance_dashboard() -> dict`
  - `export_metrics(format: str = "json", time_window_hours: int = 24) -> str`
- `SystemMetrics` (typed aggregate of system KPIs)
- Helpers: `safe_mean`, `format_duration_ms`, `mask_sensitive_metadata`
- Health: `performance_health_check()`

### Timing Collector (`timing_collector.py`)
Collects hierarchical timings with parent/child relationships for deeper breakdowns.

Key types and APIs:
- `TimingCollector`
  - `measure(operation_name: str, metadata: dict | None = None)` (async context manager)
  - `get_active_contexts() -> dict`
  - `get_recent_timings(count: int = 100) -> list`
  - `get_timing_breakdown(operation_id: str) -> dict | None`
  - `generate_detailed_breakdown() -> DetailedTimingBreakdown`
  - `get_statistics() -> dict`
  - `clear_completed_timings() -> int`
- Convenience: `get_timing_collector()`, `initialize_timing_collector()`,
  `measure_operation()`, `get_timing_stats()`

### Database Metrics (`database_metrics.py`)
Captures detailed Neo4j operation timings and connection-pool metrics. Supports network
latency sampling and slow-query identification.

Key types and APIs:
- `DatabaseMetricsCollector`
  - `measure_database_operation(operation_type: str, query_text: str | None = None, expected_rows: int | None = None)` (async context manager)
  - `measure_connection_acquisition()` / `measure_query_compilation()`
  - `measure_network_latency(target_uri: str | None = None) -> float`
  - `get_operation_statistics(time_window_minutes: int = 60) -> dict`
  - `get_connection_pool_status() -> dict`
  - `reset_statistics() -> None`
- Models: `DatabaseOperationTiming`, `ConnectionPoolMetrics`
- Helpers: `safe_percentile`, `truncate_query`, `compute_pool_utilization`
- Convenience: `get_database_metrics()`, `initialize_database_metrics()`

### Circuit Breaker (`circuit_breaker.py`)
Protects critical calls with configurable failure detection, slow-call tracking, and half‑open
recovery. Central registry and health report included.

Key types and APIs:
- `CircuitBreaker`
  - `call(func, *args, **kwargs)` (async)
  - `get_metrics() -> dict`
  - `get_recent_events(count: int = 10) -> list`
  - `reset()`
- Models: `CircuitState`, `CircuitBreakerConfig`, `CircuitEvent`, `CallResult`
- Registry: `CircuitBreakerRegistry` with `get_or_create(...)`, `get_all_metrics()`, `reset_all()`, `get_summary()`
- Decorator and accessors: `circuit_breaker(...)`, `get_circuit_breaker(...)`
- Health: `circuit_breaker_health_check()`

---

## Package surface (`app.monitoring`)

```python
from app.monitoring import (
    # Performance
    get_performance_monitor, performance_health_check, initialize_monitor,
    # Timing
    get_timing_collector, initialize_timing_collector, measure_operation, get_timing_stats,
    # Database metrics
    get_database_metrics, initialize_database_metrics,
    # Circuit breaker
    get_circuit_breaker, circuit_breaker, circuit_breaker_health_check,
)
```

---

## Quick start

### Track an operation (performance monitor)
```python
from app.monitoring import get_performance_monitor

monitor = get_performance_monitor()

async def handle_request():
    async with monitor.track_operation("parallel_retrieval") as op:
        # ... do work
        op.add_metadata({"semantic_time_ms": 125, "traversal_time_ms": 210, "result_count": 18})
    stats = monitor.get_current_stats(time_window_minutes=60)
    dashboard = monitor.get_performance_dashboard()
```

### Hierarchical timing (timing collector)
```python
from app.monitoring import get_timing_collector

collector = get_timing_collector()

async def fetch_data():
    async with collector.measure("database_query", {"label": "cypher"}) as t:
        rows = await run_cypher()
        t.add_metadata({"rows_returned": len(rows)})
    stats = collector.get_statistics()
```

### Database operation timing
```python
from app.monitoring import get_database_metrics

db_metrics = get_database_metrics()

async def run_query(cypher: str):
    async with db_metrics.measure_database_operation("query", cypher) as dbt:
        dbt.start_query_timing()
        result = await session.run(cypher)
        dbt.end_query_timing()
        dbt.set_rows_returned(result_count(result))
```

### Circuit breaker
```python
from app.monitoring import circuit_breaker

@circuit_breaker("neo4j")
async def call_graph():
    return await session.run("MATCH (n) RETURN count(n) AS c")
```

### Health checks
```python
from app.monitoring import performance_health_check, circuit_breaker_health_check

perf = await performance_health_check()
circuits = await circuit_breaker_health_check()
```

---

## Integration notes
- Use descriptive operation names (e.g., `parallel_retrieval`, `context_fusion`).
- Attach structured metadata (e.g., `result_count`, `confidence`).
- Mask secrets in metadata using `mask_sensitive_metadata` when exporting/logging.
- Percentile/trend computations need sufficient samples; seed with realistic traffic.
- For DB metrics, prefer lightweight queries for latency checks.

---

## Files
- `__init__.py`: Package surface and re-exports
- `performance_monitor.py`: System metrics and dashboards
- `timing_collector.py`: Hierarchical timing and detailed breakdowns
- `database_metrics.py`: Neo4j operation metrics and analysis
- `circuit_breaker.py`: Circuit breaker, registry, and health

---

## License
Apache-2.0 — © 2025 Alexander Samuel Ricciardi

(Generated by AI)