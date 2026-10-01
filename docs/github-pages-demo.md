# Static experiment replay

The public replay is hosted at the repository's GitHub Pages URL:

`https://nhatminh06.github.io/edgecloud-vision-lab/`

GitHub Pages serves static replay JSON exported from a local experiment recording. The browser
does not run the Python scheduler, model, or remote worker.

The project has no hosted inference backend and does not use another site host.

The viewer lives in `site/` and uses HTML, CSS, JavaScript, and SVG. The Pages workflow copies the selected file from
`demo-data/` into the deployment artifact. The replay can visualize worker selection and
execution, EWMA estimates, request latency, controlled delay, detection counts, profile changes,
worker state, and fallback events.

The interface uses a compact observability-console layout. Animation is limited to playback,
route changes, profile changes, fallback, and the chart playhead.

The public data is a validated 100-frame local CPU capture recorded on September 30, 2026.
The deterministic synthetic fixture remains under `tests/fixtures/replay/` for schema and viewer
regression tests. See [the demo guide](demo.md) for the reproducible capture flow and
[the experiment note](experiments/adaptive-failover.md) for interpretation and limitations.
