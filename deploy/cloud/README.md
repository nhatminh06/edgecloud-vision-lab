# Single-VM CPU deployment

These scripts deploy the existing `edgecloud-serve` application to one Docker-capable VM.
They do not create cloud resources and contain no provider SDK integration.

Starting recommendation, not a validated minimum:

- x86_64 CPU architecture
- 2 or more vCPUs
- 4 GiB or more RAM
- 20 GiB or more disk
- current Ubuntu LTS or another Docker-supported Linux distribution
- outbound HTTPS during image construction and the initial model download

The recommendation should be validated on the selected VM before benchmark results are used.
The CPU-only image and model cache require several gigabytes of disk. Phase 9 should not use a
GPU instance.

## Network

Allow inbound TCP to `EDGECLOUD_PORT` (default `8000`) only from the experiment machine's
public source IP or private network CIDR. Do not use `0.0.0.0/0` longer than necessary. The
container has no authentication layer, so firewall restriction is required for public VMs.

The bind address defaults to `0.0.0.0` on the VM. It can be restricted with
`EDGECLOUD_BIND_HOST`. No IP address, hostname, region, or credential is stored in the scripts.

## Build and run

Copy or clone the repository onto the VM, then run:

```bash
chmod +x deploy/cloud/*.sh
EDGECLOUD_IMAGE=edgecloud-vision-worker:phase9 deploy/cloud/setup.sh
EDGECLOUD_WORKER_ID=cloud-cpu-1 deploy/cloud/run.sh
docker logs --follow edgecloud-vision-worker
```

The named `edgecloud-model-cache` volume preserves downloaded model weights when the container
is replaced. Configuration variables are:

- `EDGECLOUD_IMAGE`
- `EDGECLOUD_CONTAINER`
- `EDGECLOUD_BIND_HOST`
- `EDGECLOUD_PORT`
- `EDGECLOUD_WORKER_ID`
- `EDGECLOUD_DEVICE` (use `cpu` for Phase 9)
- `EDGECLOUD_CONFIDENCE`

The only inference backend currently implemented is torchvision SSDLite; deployment does not
introduce a cloud-specific backend.

## Verification

From the restricted development machine, set the real URL without committing it:

```bash
export EDGECLOUD_REMOTE_URL=http://CLOUD_HOST:8000
curl --fail --show-error "$EDGECLOUD_REMOTE_URL/health"
curl --fail --show-error "$EDGECLOUD_REMOTE_URL/telemetry"
edgecloud-remote-infer examples/input.jpg --url "$EDGECLOUD_REMOTE_URL" --timeout 30
```

The inference output must identify the configured worker, report the torchvision backend,
include server phase timing, and include client-measured `round_trip_ms`. On a CPU VM, empty or
unavailable NVIDIA telemetry is expected.

For an application-level network baseline, record several health request durations separately
from inference results:

```bash
for index in 1 2 3 4 5; do
  curl --output /dev/null --silent --show-error \
    --write-out 'health_http_seconds=%{time_total}\n' \
    "$EDGECLOUD_REMOTE_URL/health"
done
```

`round_trip_ms` includes encoding, HTTP transport, server handling, response parsing, and the
real network path. Server `timing.total_ms` covers model processing. Their difference must not
be described as pure physical network latency. All client latency remains locally measured
with monotonic clocks and does not require synchronized machines.

## Failure and recovery validation

After the baseline matrix, start a small resilient workload from the development machine. Stop
only the named container created by these scripts:

```bash
docker stop edgecloud-vision-worker
docker start edgecloud-vision-worker
```

Record the failed primary type and latency, fallback latency, total logical latency, requests
routed away while health is unhealthy, and elapsed local monotonic time until recovery is
observed. Do not stop the VM for this test. The first baseline matrix should use normal load and
zero artificial delay; controlled CPU pressure is a separate optional follow-up.

Record the VM instance type, region, and total experiment runtime for approximate cost
reproduction. Storage and network transfer may also be charged. Consult current provider
pricing rather than committing a fixed price to this repository.

## Cleanup

```bash
deploy/cloud/stop.sh
EDGECLOUD_REMOVE_CONTAINER=1 deploy/cloud/stop.sh
docker volume rm edgecloud-model-cache  # only if cached weights are no longer needed
```

Remove the temporary firewall rule after testing. Stop and delete/terminate the VM through the
chosen provider when experiments finish. Stopping or removing the container does not stop VM
billing.
