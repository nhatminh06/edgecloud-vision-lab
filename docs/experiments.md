# Experiment framework

The experiment CLI runs repeated scheduler workloads. Raw output retains slow and failed
requests.

## Single experiment

```bash
python -m pip install -e '.[experiment]'
edgecloud-experiment run \
  --name latency-delay-50 \
  --scheduler latency_aware \
  --image examples/input.jpg \
  --requests 100 \
  --warmup 10 \
  --repetitions 3 \
  --remote-url http://127.0.0.1:8000 \
  --delay-ms 50 \
  --output results/latency-delay-50
```

Warm-up observations are retained in raw output but excluded from aggregate metrics. Repetitions
receive distinct integer run IDs and are summarized individually and across runs.

`--delay-ms` adds one explicit request-path delay before remote inference through an
experiment-only wrapper. It is not represented as network or model latency.

## Failure and pressure controls

An inclusive request-index failure window controls only calls made by the experiment:

```bash
edgecloud-experiment run \
  --name resilient-outage \
  --scheduler resilient \
  --image examples/input.jpg \
  --requests 20 \
  --failure-start 5 \
  --failure-end 12 \
  --output results/resilient-outage
```

It does not stop an external service. Real process stop/restart validation remains a separate
manual smoke test.

`--resource-scenario edge_cpu_pressure` starts a bounded number of local pressure threads for the
experiment's duration. In local emulation, both logical workers may be affected because they
share physical hardware. Automated tests do not run pressure experiments.

## Matrices and plots

```bash
edgecloud-experiment matrix --config experiment-configs/network-delay.json
edgecloud-experiment plot results/* --output results/plots
```

A matrix file may contain an `experiments` array or compact `schedulers` and `delays_ms` arrays.
Configuration uses the fields serialized in generated `config.json` files, including experiment
name, scheduler, input path, measured request count, delay, and output directory.

Plot generation requires the `experiment` optional dependency. It reads preserved result files;
it does not rerun inference.

## Result files

Each result directory contains:

- `config.json`: resolved experiment configuration;
- `raw.jsonl`: one record per logical request, including warm-up and failures;
- `summary.json`: per-run and cross-run metrics; and
- `summary.csv`: one row per run for analysis tools.

Generated results remain ignored by Git unless a replay is selected for the public viewer.

## Measurement definitions

- Logical latency is scheduler-observed elapsed time for a successful logical request, including
  a failed primary plus fallback when applicable.
- Failed requests retain runner-observed latency in raw output but are excluded from successful
  latency distributions.
- Attempt latency surrounds one worker call. Artificial delay is included in that attempt's
  elapsed time but remains documented as a controlled input.
- Throughput is successful measured requests divided by measured wall-clock duration, not the
  inverse of mean latency.
- p50, p95, and p99 use linear interpolation at rank `(sample_count - 1) * quantile`.
- Success rate is successful measured logical requests divided by all measured logical requests.
- Edge and remote selection percentages use initial choices divided by measured logical requests.
- Fallback rate is fallback attempts divided by measured logical requests; direction counts are
  retained separately.

## Stable replay recording

`edgecloud-run --record` produces the versioned JSON Lines contract used by the static viewer.
The recording workflow, schema, and captured run are documented in:

- [Experiment format](experiment-format.md)
- [Recorded demo guide](demo.md)
- [Adaptive failover capture](experiments/adaptive-failover.md)
