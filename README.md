# EdgeCloud Vision

EdgeCloud Vision measures computer-vision inference across local and HTTP workers. It provides
a local object-detection pipeline for images, video files, and webcams, plus a service boundary
for running the same inference engine remotely.

## Architecture

```text
local:  frame -> EdgeWorker -> inference engine -> WorkerResult

remote: frame -> RemoteWorker -> HTTP API -> EdgeWorker -> inference engine
                     |                                      |
                     +-- client round-trip timing           +-- server phase timings

scheduled:

                  +-------------+
frame ----------->|  Scheduler  |
                  +------+------+ 
                         |
                +--------+--------+
                |                 |
                v                 v
          EdgeWorker        RemoteWorker
                |                 |
                v                 | HTTP
       Inference Engine           v
                            Remote Service
                                  |
                                  v
                           Inference Engine
```

The inference engine owns model phase timing and orchestration. Workers adapt its typed result
without duplicating preprocessing, inference, or postprocessing. HTTP schemas remain in the
transport layer; callers receive the same `WorkerResult` contract from local and remote workers.

## Setup

Python 3.12 or newer is required. The first inference run downloads torchvision's pretrained
SSDLite320 MobileNet V3 weights to the standard PyTorch cache.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install -e '.[dev]'
```

The explicit CPU wheel command avoids installing CUDA runtime packages on machines that do
not use an NVIDIA GPU. For CUDA, install the matching PyTorch build from the official PyTorch
package index before the editable project install.

## Run local inference

Each processed frame writes one JSON object to standard output. It includes detections,
preprocessing/inference/postprocessing/total latency in milliseconds, detection count, and
observed end-to-end stream FPS.

```bash
edgecloud-infer --image examples/input.jpg --output outputs/detected.jpg
edgecloud-infer --video examples/input.mp4 --output outputs/detected.mp4
edgecloud-infer --webcam 0 --output outputs/webcam.mp4
```

Use `--confidence 0.6` to change the detection threshold and `--device cuda` only when a
working CUDA-enabled PyTorch installation is available. Video output currently uses a fixed
30 FPS; inference metrics remain based on measured wall time.

## Run the HTTP worker

Start the service. Model initialization occurs once during service startup, and `/health` does
not report ready unless initialization succeeds.

```bash
edgecloud-serve --host 127.0.0.1 --port 8000
curl http://127.0.0.1:8000/health
```

Send an image through `RemoteWorker` using the remote CLI:

```bash
edgecloud-remote-infer examples/input.jpg \
  --url http://127.0.0.1:8000 \
  --timeout 10
```

The service accepts encoded image bytes at `POST /infer` with an `image/*` or
`application/octet-stream` content type. The response includes detections, server-side phase
timings, worker identity/type, and backend. The remote client adds measured HTTP round-trip
time as `round_trip_ms`; it does not estimate one-way latency.

Service options include `--worker-id`, `--device`, and `--confidence`. Remote client
configuration consists of `--url` and `--timeout`.

## Schedulers

Six deterministic strategies are available:

- `edge_only` always calls the local edge worker.
- `remote_only` always calls the configured HTTP worker.
- `round_robin` starts with edge and then alternates edge, remote, edge, remote.
- `latency_aware` selects the worker with the lower EWMA of observed request latency.
- `resource_aware` combines the same latency history with cached resource pressure.
- `resilient` adds fresh health constraints and at most one typed-failure fallback to the
  resource-aware primary choice.

Run an image or video through a scheduler:

```bash
edgecloud-run --image examples/input.jpg --scheduler edge_only
edgecloud-run --image examples/input.jpg --scheduler remote_only \
  --remote-url http://127.0.0.1:8000
edgecloud-run --video examples/input.mp4 --scheduler round_robin \
  --remote-url http://127.0.0.1:8000
edgecloud-run --video examples/input.mp4 --scheduler latency_aware \
  --remote-url http://127.0.0.1:8000 \
  --latency-alpha 0.3
edgecloud-run --video examples/input.mp4 --scheduler latency_aware \
  --remote-url http://127.0.0.1:8000 \
  --demo-profile cloud_slow
edgecloud-run --video examples/input.mp4 --scheduler resource_aware \
  --remote-url http://127.0.0.1:8000 \
  --telemetry-interval 1 \
  --telemetry-max-age 3 \
  --resource-pressure-threshold 0.85
edgecloud-run --video examples/input.mp4 --scheduler resilient \
  --remote-url http://127.0.0.1:8000 \
  --telemetry-interval 1 \
  --telemetry-max-age 3 \
  --health-interval 1 \
  --health-max-age 3 \
  --resource-pressure-threshold 0.85
```

Output is JSON Lines: one scheduled result per frame followed by a run summary containing
request, selection, success/failure, and mean-latency metrics. Scheduler latency measures the
complete synchronous worker call. Remote results retain their separate HTTP round-trip timing.

`edge_only`, `remote_only`, and `round_robin` remain strict baselines. A selected-worker failure
fails that request; these strategies do not retry or fall back. After a failed round-robin
selection, the next request advances normally.

The latency-aware scheduler samples edge on its first request and remote on its second. Later
requests select the lower estimate; equal estimates select edge. Both estimates use the same
boundary: scheduler-observed elapsed time around the complete synchronous worker call. This
compares edge end-to-end request latency with remote end-to-end request latency, including HTTP
transport and serialization performed by `RemoteWorker`.

For each successful observation, the scheduler updates only the selected worker using:

```text
new estimate = alpha * observation + (1 - alpha) * previous estimate
```

The first observation initializes an estimate. `--latency-alpha` must be greater than zero and
at most one. Recoverable remote transport/service failures and local execution failures trigger
one attempt on the other worker. Failed attempts never update EWMA; a successful fallback
updates only the worker that executed it. Results expose the originally selected worker, the
executed worker, fallback status and reason, current estimates, and total observed request
latency. If both attempts fail, the request raises a `WorkerUnavailableError` describing both
failures. This strategy does not consider utilization or cost.

Optional demo profiles wrap the real workers; they never replace model inference or burn CPU to
simulate load. `normal` and `recovered` add no controls, `edge_hot` adds 150 ms before edge
inference, `cloud_slow` adds 200 ms before remote inference, and `cloud_down` makes the remote
wrapper deterministically unavailable. Injected delay is serialized separately as
`injected_delay_ms` and as the per-worker configured delay, so it is not presented as model,
hardware, or network timing. Without `--demo-profile`, worker behavior is unchanged.

The resource-aware scheduler retains edge-then-remote cold start and EWMA latency updates. Two
background threads refresh edge and remote telemetry independently; routing reads only the
latest cached snapshots. A snapshot older than `--telemetry-max-age` is stale. Sampling failure
preserves the last valid snapshot until it becomes stale and records the error in cache state.

Resource pressure is normalized to `0.0`–`1.0`:

```text
CPU worker: max(CPU utilization, system memory utilization) / 100
GPU worker: max(GPU utilization, GPU memory utilization) / 100
```

Execution resource is explicit scheduler configuration. The current CLI configures both
torchvision workers as CPU-backed, so NVIDIA metrics remain observable but cannot affect its
routing.

After cold start, the exact policy is:

1. Missing or stale relevant telemetry uses the lower latency estimate, with edge winning a
   tie.
2. If edge pressure meets the configured threshold and exceeds remote pressure, select remote.
3. If remote pressure meets the threshold and exceeds edge pressure, select edge.
4. Otherwise select the lower latency estimate, with edge winning a tie.

The pressure threshold is a configured policy value, not a learned parameter. Telemetry refresh
failure never fails inference. Selected-worker inference failure still propagates without retry
or fallback.

The resilient scheduler uses the same cold-start, latency, telemetry, and resource-pressure
rules to choose a primary worker. Independently sampled health is then a hard constraint: a
fresh unhealthy worker is not selected, unknown or stale health remains eligible, and a request
fails before inference when both workers are freshly unhealthy. Local health is a lightweight
readiness check; remote health comes from `RemoteWorker.health()`. `--health-interval` controls
sampling and `--health-max-age` controls freshness, both using local monotonic receipt time.

Only failures for which another worker can plausibly help trigger fallback: remote timeout,
connection, unavailable/not-ready, rate-limit or 5xx service failures, and the explicit local
worker execution failure boundary. Invalid input, malformed remote responses, and programming
errors do not. A logical request makes at most two attempts. Failed primary attempts do not
update latency EWMA; a successful fallback updates only its worker with that attempt's elapsed
time. Result metadata reports the initial and final worker, failure type/code, per-attempt and
total logical latency, health state, and decision reason. Summary metrics distinguish logical
requests from worker attempts and count fallback directions and outcomes.

## Docker

The image installs CPU-only PyTorch and starts the inference service on port 8000. Pretrained
weights are downloaded into the container's PyTorch cache during its first startup.

```bash
docker build -t edgecloud-vision-worker .
docker run --rm -p 8000:8000 edgecloud-vision-worker
curl http://127.0.0.1:8000/health
```

## Experiments

Install the optional plotting dependency and run a single development experiment:

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

`--delay-ms` is total artificial request-path delay added once before each remote inference
call. It is implemented by an experiment-only worker wrapper; production workers and scheduler
policies contain no injected delay. `--resource-scenario edge_cpu_pressure` starts a bounded
number of local pressure threads for the duration of the experiment. In local emulation this
may affect both logical workers because they share physical hardware.

An optional inclusive failure window uses measured request indexes:

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

The failure wrapper only controls calls made by that experiment. It does not stop or kill an
external service. Real process stop/restart validation remains a separate manual smoke test.

Run a matrix from JSON and plot completed result directories:

```bash
edgecloud-experiment matrix --config experiment-configs/network-delay.json
edgecloud-experiment plot results/* --output results/plots
```

The matrix file can contain an `experiments` array or compact `schedulers` and `delays_ms`
arrays. Configuration uses the serialized fields shown in generated `config.json` files,
including `name`, `scheduler`, `input_path`, `measured_requests`, `delay_ms`, and `output_dir`.

Each result directory contains `config.json`, per-logical-request `raw.jsonl`, rich
`summary.json`, and one-row-per-run `summary.csv`. Warm-up observations remain in raw output but
are excluded from aggregates. Repetitions retain distinct integer `run_id` values and are
reported both per run and across runs.

Measurement definitions are:

- Logical latency is scheduler-observed elapsed time for a successful request, including a
  failed primary plus fallback when applicable. Failed requests retain runner-observed elapsed
  time in raw output but are excluded from latency distributions.
- Attempt latency is the experiment wrapper's elapsed time around one worker call. Artificial
  delay is therefore included in remote attempt latency.
- Throughput is successful measured requests divided by measured wall-clock duration, not the
  inverse of mean latency.
- p50, p95, and p99 use linear interpolation at rank `(sample_count - 1) * quantile`.
- Success rate is successful measured logical requests divided by all measured logical
  requests.
- Edge and remote selection percentages use the initial worker choice divided by measured
  logical requests.
- Fallback rate can be derived as fallback attempts divided by measured logical requests;
  counts and directions are retained explicitly.

CPU-pressure experiments are intentionally bounded and do not allocate large amounts of
memory. Automated tests use fake workers and telemetry; real CPU pressure is intended for
manual development experiments. Failed and unfavorable measured requests are retained.

## Single cloud VM deployment

Phase 9 uses the unchanged CPU worker on one provider-neutral Docker VM. The repository does
not provision billable infrastructure. Create a Docker-capable x86_64 VM manually and restrict
inbound TCP port 8000 to the experiment machine's source IP or private CIDR. A reasonable
starting point—not a validated minimum—is 2 vCPUs, 4 GiB RAM, 20 GiB disk, and a current Ubuntu
LTS or equivalent Docker-supported Linux. Initial image construction and model download require
outbound HTTPS.

On the VM:

```bash
chmod +x deploy/cloud/*.sh
deploy/cloud/setup.sh
EDGECLOUD_WORKER_ID=cloud-cpu-1 deploy/cloud/run.sh
docker logs --follow edgecloud-vision-worker
```

Before experiments, verify health, telemetry, and one real inference from the local machine:

```bash
export EDGECLOUD_REMOTE_URL=http://CLOUD_HOST:8000
curl --fail --show-error "$EDGECLOUD_REMOTE_URL/health"
curl --fail --show-error "$EDGECLOUD_REMOTE_URL/telemetry"
edgecloud-remote-infer examples/input.jpg --url "$EDGECLOUD_REMOTE_URL" --timeout 30
```

Copy [the real-cloud template](experiment-configs/real-cloud.template.json) to an untracked
configuration, replace every `replace-before-run` value plus zero vCPU/memory placeholders with
observed VM metadata, and run the initial 6-policy validation matrix without artificial delay:

```bash
cp experiment-configs/real-cloud.template.json /tmp/real-cloud.json
# Edit /tmp/real-cloud.json with actual provider, region, VM type, OS, CPU, vCPU, and RAM.
edgecloud-experiment matrix \
  --config /tmp/real-cloud.json \
  --remote-url "$EDGECLOUD_REMOTE_URL"
```

The template uses 5 warm-up requests, 30 measured requests, and 3 distinct repetitions for
each existing policy. Results go under `results/real-cloud/`, which remains ignored. Cloud
provider, region, and instance type are experiment metadata only; they do not enter inference
or scheduler implementations. See [deploy/cloud/README.md](deploy/cloud/README.md) for
application-level timing, firewall, configuration, failure-test, and cleanup guidance.

Cleanup is mandatory when testing finishes:

```bash
EDGECLOUD_REMOVE_CONTAINER=1 deploy/cloud/stop.sh
```

Remove the firewall rule and terminate/delete the VM through its provider. Stopping the
container does not stop VM billing.

## Telemetry

Collect one local snapshot or emit repeated JSON Lines samples:

```bash
edgecloud-telemetry
edgecloud-telemetry --interval 1 --count 10
curl http://127.0.0.1:8000/telemetry
```

System telemetry contains:

- `cpu_utilization_percent`: non-blocking `psutil.cpu_percent(interval=None)` measurement. It
  covers activity since the previous call; the process's first call covers activity since
  psutil import.
- `memory_used_bytes` and `memory_total_bytes`.
- `memory_utilization_percent`.

Each NVIDIA GPU record contains its device index and name, utilization percentage, memory
used/total in bytes, memory utilization percentage, temperature in Celsius, and power draw in
watts. Unsupported individual metrics are `null`. Multiple GPUs produce multiple records.

`gpu_telemetry_available=false` with an explanatory `gpu_telemetry_error` distinguishes NVML
initialization failure from an NVML-capable machine with zero GPUs. Missing NVIDIA hardware,
drivers, or optional metrics do not prevent system telemetry or inference from working.

Telemetry is available through the standalone CLI, `GET /telemetry`, and
`RemoteWorker.telemetry()`. The resource-aware and resilient policies consume independently
cached samples; telemetry collection remains off the synchronous inference path.

## Validate

```bash
ruff format --check .
ruff check .
pytest
python -m pip check
```

The tests use an injected fake inference backend and do not download model weights or require
a GPU, webcam, or network connection.

## Current limitations

- Pretrained weights must be downloaded before offline inference.
- Video metadata and source frame rate are not yet preserved in rendered output.
- The HTTP worker processes inference in the service process; worker pools and queueing are not
  implemented.
- Resilient fallback is limited to one alternate-worker attempt; there is no retry loop,
  circuit breaker, persistence layer, or multi-worker pool.
- Real-cloud results remain environment-specific; the repository does not provision or retain
  a cloud VM automatically.
