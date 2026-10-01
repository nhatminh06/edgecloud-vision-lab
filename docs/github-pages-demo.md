# Static experiment replay

The eventual public demo will be hosted only at the repository's GitHub Pages URL:

`https://nhatminh06.github.io/edgecloud-vision-lab/`

GitHub Pages will remain a static site. It will consume static replay JSON generated from
captured experiment JSON Lines from real local runs; it will not claim to run the Python
scheduler, model, or remote worker in the browser. The recorded portfolio video will show the
actual live local system.

There will be no Vercel, Netlify, Render, Cloudflare Pages, or hosted backend deployment.

The initial implementation lives in `site/` and uses only HTML, CSS, vanilla JavaScript, SVG,
and a committed static replay asset. The Pages workflow copies the selected file from
`demo-data/` into the deployment artifact. The replay can visualize worker selection and
execution, EWMA estimates, observed request
latency, explicitly injected demo delay, detections, scheduler transitions, worker state, and
fallback events. Replayed data must be labeled as captured experiment data. No dashboard or
deployment is part of the current phase.

The visual direction should resemble an internal observability console: neutral colors,
restrained typography, compact readable data, clear hierarchy, intentional spacing, and minimal
animation. Avoid marketing-page decoration and invented performance claims.

Motion should be reserved for frame progression, route changes, scenario transitions, fallback,
and chart playback. It must not be decorative. The interface should draw from developer tools,
systems research figures, GitHub, Grafana, Chrome DevTools, and restrained observability tools,
not generic SaaS or hackathon templates.

The public artifact is a validated 100-frame local CPU capture recorded on September 30, 2026.
The deterministic synthetic fixture remains under `tests/fixtures/replay/` for schema and viewer
regression tests. See [the demo guide](demo.md) for the reproducible capture flow and
[the experiment note](experiments/adaptive-failover.md) for interpretation and limitations.
