# Recorded demo guide

This walkthrough connects the real local system to the evidence shown by the static replay
viewer. It deliberately changes worker conditions through safe `ControlledWorker` profiles; it
does not burn CPU, alter machine configuration, stop external services, or expose credentials.

Prepare the Python environment, model weights, input video, and browser tabs before recording.
Use a video with at least 72 frames so every canonical scenario boundary occurs. Do not download
weights, install packages, edit configuration, or enter secrets on camera. Begin the recorded
walkthrough with the public viewer briefly visible, then move to the real system.

The canonical public capture used an uncommitted 100-frame excerpt of OpenCV's public
`vtest.avi` sample. Before recording, reproduce the input outside the repository:

```bash
curl --fail --location --output /tmp/opencv-vtest.avi \
  https://raw.githubusercontent.com/opencv/opencv/master/samples/data/vtest.avi
ffmpeg -hide_banner -loglevel error -y \
  -i /tmp/opencv-vtest.avi -frames:v 100 -c:v mjpeg \
  /tmp/edgecloud-demo-100.avi
```

Verify that OpenCV reports 100 decodable frames. The downloaded and derived videos remain
untracked.

## Preflight

Before the recorded take:

```bash
ffmpeg -hide_banner -loglevel error -y \
  -i /tmp/edgecloud-demo-100.avi -frames:v 1 /tmp/edgecloud-preflight.jpg
edgecloud-infer --image /tmp/edgecloud-preflight.jpg --confidence 0.3 --device cpu
mkdir -p outputs/runs
test -w outputs/runs
edgecloud-run --help | grep adaptive-failover
edgecloud-export-replay --help
```

This verifies input decoding, local model initialization, output permissions, the canonical
scenario, and the installed replay exporter. CUDA may replace CPU only when it is already
configured and verified; never install or change GPU drivers for the recording.

## 1. Start the remote worker

In the first terminal:

```bash
edgecloud-serve --host 127.0.0.1 --port 8000 --confidence 0.3 --device cpu
```

Verify readiness before recording:

```bash
curl --fail http://127.0.0.1:8000/health
```

## 2. Run and record the adaptive scenario

In a second terminal:

```bash
edgecloud-run \
  --video /tmp/edgecloud-demo-100.avi \
  --scheduler latency_aware \
  --remote-url http://127.0.0.1:8000 \
  --scenario adaptive-failover \
  --run-id portfolio-adaptive-failover-2026-09-30 \
  --record outputs/runs/adaptive-failover-real.jsonl \
  --confidence 0.3 \
  --device cpu
```

Explain the observed sequence without predicting machine-dependent routing:

1. Frames 0–19 use normal worker conditions. Edge and remote are each sampled once.
2. At frame 20, `edge_hot` adds 150 ms of explicitly controlled edge delay. The scheduler
   continues choosing from measured EWMA values.
3. At frame 50, `cloud_down` makes the controlled remote wrapper unavailable. If remote is
   selected, show the selected/executed distinction and successful edge fallback.
4. At frame 70, `recovered` restores normal wrapper conditions. Observations continue updating
   naturally.

Do not claim that a particular worker must be fastest. The profile changes conditions; it does
not force a routing decision.

## 3. Validate and export the recording

```bash
edgecloud-export-replay \
  outputs/runs/adaptive-failover-real.jsonl \
  /tmp/adaptive-failover.json
```

The exporter rejects malformed or incomplete runs. Inspect the run label and a few events before
using the file publicly. Never replace measurements with cleaner-looking values.

## 4. Show the public replay

Open:

`https://nhatminh06.github.io/edgecloud-vision-lab/`

Connect the recorded run to the same profile boundaries, decision reasons, fallback events, and
measured timing in the viewer. The website is a static replay; Python and the workers are not
running in GitHub Pages.

## Recording presentation

Keep the system, state changes, scheduler response, fallback, recovery, and exported evidence in
one understandable flow. Minimal captions can identify a profile boundary or fallback. Avoid
fake typing, cinematic transitions, decorative animation, and claims not supported by the
recorded JSON.

Generated recordings under `outputs/` remain untracked. Commit a validated replay export under
`demo-data/` only when it is intentionally selected as the public artifact and clearly labeled
as either a captured run or synthetic fixture.
