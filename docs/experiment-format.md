# Experiment recording format

Experiment recordings use UTF-8 JSON Lines with one versioned event per line. Schema version `1`
has four event types: `run_start`, `inference`, `profile_change`, and `run_end`. Event order is
defined by the strictly increasing integer `sequence`; `elapsed_ms` is monotonic time relative to
the beginning of the run.

All events contain `schema_version`, `event_type`, `run_id`, `sequence`, and `elapsed_ms`. JSON
numbers must be finite. Optional measurements use JSON `null`; missing measurements must not be
replaced with zero.

## Run events

`run_start` is sequence zero and records the scheduler, EWMA alpha, source type and basename,
scenario name, initial profile, and model identifier. It never includes an absolute input path,
hostname, username, or environment secrets.

`run_end` contains a `summary` with frame success/failure totals, execution counts by worker,
fallback count, worker switches, mean observed latency when at least one request succeeded, and
measured scenario duration. It is the final event.

## Inference event

An `inference` event contains:

- `frame_index`, `profile`, and `scheduler`.
- `decision.selected_worker`: the scheduler's original choice.
- `decision.executed_worker`: the worker that produced the result after any fallback.
- `decision.reason`: a stable policy reason code.
- `decision.edge_estimate_ms` and `remote_estimate_ms`: scheduler EWMA state, or `null` before an
  estimate exists.
- `timing.observed_request_ms`: scheduler-observed logical request latency. With fallback this
  covers both attempts.
- `timing.model_total_ms` and `model_inference_ms`: timing reported by the executed model worker.
- `timing.round_trip_ms`: HTTP client round-trip measurement for a remote result, otherwise
  `null`.
- `fallback.used` and `fallback.reason`.
- `workers.edge` and `workers.remote`, each containing `available` and `injected_delay_ms` for
  the active controlled profile.
- `detections.count`; full detection payloads are intentionally excluded.
- `error`: an error type/message object for a failed frame, otherwise `null`.

Injected delay is a configured demo condition. It is not model, hardware, transport, or network
latency. It remains under `workers.*.injected_delay_ms`; measured model and request timings remain
under `timing`. The observed request can include injected delay because the scheduler measures
the complete worker call.

## Profile transitions

`profile_change` explicitly records `frame_index`, `from_profile`, and `to_profile`. A transition
is emitted before inference at its configured boundary. Consumers should use these events for
chart annotations instead of deriving transitions by comparing adjacent frames.

The canonical `adaptive-failover` scenario changes at frames 0, 20, 50, and 70. Profiles alter
worker availability or delay only; they never directly choose a route.

## Static replay export

`edgecloud-export-replay INPUT.jsonl OUTPUT.json` validates the complete stream and writes:

```json
{
  "schema_version": 1,
  "run": {},
  "events": []
}
```

`run` is the original `run_start` event. `events` contains every subsequent event in recorded
order, including `run_end`. Export does not rewrite timestamps or measurements.

Validation requires exactly one start and end event, matching run IDs, supported event/schema
types, strictly increasing sequences, non-regressing inference frame indices, known profiles and
workers, non-negative finite timing and delay values, and a structurally complete summary.
