# Scheduler behavior

EdgeCloud Vision keeps simple baselines alongside adaptive policies so experiment results can be
compared against understandable controls. Scheduling remains synchronous: one logical request is
completed before the next frame is dispatched.

## Strategy comparison

| Scheduler | Latency history | Resource pressure | Health | Fallback |
|---|---|---|---|---|
| `edge_only` | No | No | No | No |
| `remote_only` | No | No | No | No |
| `round_robin` | No | No | No | No |
| `latency_aware` | EWMA | No | No | Typed fallback |
| `resource_aware` | EWMA | Yes | No | No |
| `resilient` | EWMA | Yes | Yes | Typed fallback |

## Baselines

`edge_only` and `remote_only` always dispatch to their named worker. `round_robin` begins with
edge and alternates edge, remote, edge, remote. These policies are strict: a selected-worker
failure fails the logical request, and round-robin advances normally after a failure.

## Latency-aware scheduling

Cold start samples edge on the first request and remote on the second. Later requests choose the
lower estimated latency; equal estimates prefer edge. Each worker has an independent exponentially
weighted moving average:

```text
new estimate = alpha * observation + (1 - alpha) * previous estimate
```

The default alpha is `0.3`. A worker's first successful observation initializes its estimate.
Only the worker that successfully executes an attempt is updated. The observation boundary is
the scheduler's elapsed time around the complete synchronous worker call, so a remote observation
includes HTTP serialization and transport performed by `RemoteWorker`.

Recoverable failures trigger at most one attempt on the alternate worker. Eligible failures are:

- local `WorkerExecutionError`;
- remote connection, timeout, or unavailable failures; and
- remote HTTP 429 or 5xx responses.

Malformed responses, invalid input, and programming errors do not trigger fallback. A failed
primary attempt does not update EWMA. A successful fallback updates only the worker that produced
the result. If both attempts fail, `WorkerUnavailableError` describes both failures.

Results distinguish the originally selected worker from the worker that executed successfully.
Fallback status, reason, observed logical latency, and current estimates are serialized.

## Resource-aware scheduling

Resource-aware scheduling retains the edge-then-remote cold start and EWMA estimates. Independent
background threads sample edge and remote telemetry; inference reads cached snapshots rather than
collecting telemetry on the request path. Sampling failure preserves the last valid snapshot
until it becomes stale and records the error.

Pressure is normalized to `0.0`–`1.0`:

```text
CPU worker: max(CPU utilization, system memory utilization) / 100
GPU worker: max(GPU utilization, GPU memory utilization) / 100
```

The execution resource is explicit scheduler configuration. The current CLI configures both
torchvision workers as CPU-backed; NVIDIA telemetry can still be observed but does not affect
their routing.

After cold start:

1. Missing or stale relevant telemetry uses the lower latency estimate, with edge winning a tie.
2. If edge pressure meets the threshold and exceeds remote pressure, select remote.
3. If remote pressure meets the threshold and exceeds edge pressure, select edge.
4. Otherwise use the lower latency estimate, with edge winning a tie.

The pressure threshold is configuration, not a learned value. Inference failures propagate
without fallback.

## Resilient scheduling

The resilient policy uses the resource-aware preference as its primary choice and applies cached
health as a hard routing constraint. A fresh unhealthy worker is not selected. Unknown or stale
health remains eligible. If both workers are freshly unhealthy, the request fails before dispatch.

Remote health comes from `RemoteWorker.health()`; local health is a readiness check. Health
sampling remains outside the inference path. Recoverable failures receive at most one fallback
attempt using the same typed eligibility boundaries described above.

Resilient result metadata records initial and final workers, failure type and code, attempt
latencies, total logical latency, health state, and the decision reason. Summary metrics separate
logical requests from worker attempts and count fallback directions and outcomes.

## Metrics semantics

- Scheduler latency is elapsed time around a complete logical request.
- Remote worker results separately retain measured HTTP round-trip time.
- Failed attempts are never fabricated as successful latency observations.
- Baseline selection counts represent the selected worker.
- Resilient metrics separately report primary selections, attempts, fallbacks, and final workers.
- No policy persists state across process restarts.
