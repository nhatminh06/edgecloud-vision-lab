# Static experiment replay

The eventual public demo will be hosted only at the repository's GitHub Pages URL:

`https://nhatminh06.github.io/edgecloud-vision-lab/`

GitHub Pages will remain a static site. It will load captured JSON Lines or derived static JSON
from real local experiment runs; it will not claim to run the Python scheduler, model, or remote
worker in the browser. The recorded video will show the live local system.

The replay can visualize worker selection and execution, EWMA estimates, observed request
latency, explicitly injected demo delay, detections, scheduler transitions, worker state, and
fallback events. Replayed data must be labeled as captured experiment data. No dashboard or
deployment is part of the current phase.

The visual direction should resemble an internal observability console: neutral colors,
restrained typography, compact readable data, clear hierarchy, intentional spacing, and minimal
animation. Avoid marketing-page decoration and invented performance claims.
