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

## Baseline schedulers

Three deterministic strategies are available:

- `edge_only` always calls the local edge worker.
- `remote_only` always calls the configured HTTP worker.
- `round_robin` starts with edge and then alternates edge, remote, edge, remote.

Run an image or video through a scheduler:

```bash
edgecloud-run --image examples/input.jpg --scheduler edge_only
edgecloud-run --image examples/input.jpg --scheduler remote_only \
  --remote-url http://127.0.0.1:8000
edgecloud-run --video examples/input.mp4 --scheduler round_robin \
  --remote-url http://127.0.0.1:8000
```

Output is JSON Lines: one scheduled result per frame followed by a run summary containing
request, selection, success/failure, and mean-latency metrics. Scheduler latency measures the
complete synchronous worker call. Remote results retain their separate HTTP round-trip timing.

These strategies are strict baselines. A selected-worker failure fails that request. The
scheduler does not retry, fall back, or change round-robin ordering in response to health,
latency, or failure. After a failed round-robin selection, the next request advances normally.

## Docker

The image installs CPU-only PyTorch and starts the inference service on port 8000. Pretrained
weights are downloaded into the container's PyTorch cache during its first startup.

```bash
docker build -t edgecloud-vision-worker .
docker run --rm -p 8000:8000 edgecloud-vision-worker
curl http://127.0.0.1:8000/health
```

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
- No latency-aware scheduler, automatic fallback, telemetry collector, or benchmark harness
  exists yet.
