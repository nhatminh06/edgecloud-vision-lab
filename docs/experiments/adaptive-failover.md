# Adaptive failover run

## Purpose

This run checks how the latency-aware scheduler responds to controlled worker changes, a remote
outage, and recovery. The public replay was exported from the recorded JSON Lines stream.

## Configuration

```text
run ID:       portfolio-adaptive-failover-2026-09-30
date:         2026-09-30
model:        SSDLite320 MobileNet V3 Large
backend:      torchvision, CPU
scheduler:    latency_aware
EWMA alpha:   0.3
scenario:     adaptive-failover
frames:       100
confidence:   0.3
```

The input was a 100-frame, 10 FPS excerpt of OpenCV's public `vtest.avi` sample. The excerpt was
stored in `/tmp` for the run and is not committed. Source:
`https://github.com/opencv/opencv/blob/master/samples/data/vtest.avi`.

Both logical workers ran on the same local CPU. The remote path used HTTP over
`127.0.0.1:8000`; it was not a geographically remote cloud worker.

## Controlled phases

```text
frames 0-19   normal       no injected delay; both workers available
frames 20-49  edge_hot     +150 ms controlled delay before edge inference
frames 50-69  cloud_down   remote controlled wrapper unavailable
frames 70-99  recovered    controls reset; both workers available
```

These controls change worker conditions only. They do not select a route or modify scheduler
estimates directly.

## Measurements

- `model_inference_ms` and `model_total_ms` come from the worker's real model execution.
- `round_trip_ms` is measured by the HTTP client for successful remote results.
- `observed_request_ms` is scheduler-observed elapsed time around the logical request and can
  include controlled delay or a failed primary attempt plus fallback.
- Edge and remote latency estimates are EWMA state derived from successful observed attempts.
- Injected delay and forced availability remain separate controlled fields under `workers`.

## Observed behavior

At frame 19, the edge EWMA was 61.76 ms. Frame 20 added the documented 150 ms controlled edge
delay and observed 239.77 ms; the edge EWMA increased to 115.16 ms. At frame 21 the scheduler
selected remote because its 94.88 ms estimate was lower, and later successful remote
observations reduced that estimate to 72.83 ms by frame 49.

At frame 50 the profile changed to `cloud_down`. Remote was still the lower estimated route, so
the scheduler selected remote, received the controlled unavailable failure, and executed edge as
fallback. Frames 50-54 contain five such fallback events. Successful edge fallbacks updated the
edge EWMA until edge became the lower estimate; the remaining outage frames executed directly on
edge.

At frame 70 the remote wrapper became available again. The scheduler continued operating from
its measured state. Edge remained the lower estimate during the remainder of this capture, so
recovery is demonstrated as restored availability and uninterrupted inference, not as a forced
route back to remote.

The run completed all 100 frames successfully: 70 edge executions, 30 remote executions, five
fallbacks, four execution-route switches, and 145 detections across 88 frames.

## Artifacts

- Generated JSONL: `outputs/runs/adaptive-failover-real.jsonl` (untracked)
- Public validated replay: `demo-data/adaptive-failover.json`
- Synthetic regression fixture: `tests/fixtures/replay/sample-synthetic.jsonl`

The public replay was produced with `edgecloud-export-replay`; its numeric values and timestamps
were not edited.

## Limitations

- This is a single-machine local experiment, not a cloud benchmark.
- Local and HTTP workers shared the same CPU and competed for the same physical system.
- HTTP round-trip timing represents loopback transport, not internet or regional network latency.
- The 150 ms edge delay and remote outage are controlled experimental inputs.
- The capture is one run and does not establish general model, device, or cloud performance.
- Routing after recovery depends on retained EWMA state; recovery does not imply an immediate
  route switch.
