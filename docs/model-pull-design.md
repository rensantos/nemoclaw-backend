# Model Pull Design

Design for downloading a model onto a Backend Node on operator command.
Written before implementation, per `AGENTS.md`'s rule that phases touching
runtime state, disk and process management need a design document first.

This one earns that rule. UBI's single volume is **99.1% used with ~8 GiB
free**, and most of the other 900 GiB is other researchers' data. A
careless download here does not degrade our service; it fills up someone
else's disk.

## 1. Why this goes through the backend

`OllamaEngine` has deliberately never pulled (`docs/ollama-engine-design.md`):
`load_model()` verifies a tag is present and fails with `ollama pull <tag>`
if not. The frontend can already reach UBI's Ollama daemon directly through
a tunnel, so it *could* call `/api/pull` itself and skip the backend
entirely.

It must not, for a concrete reason: **a directly-pulled model would be
unusable.** `/v1/models` enumerates `config.yaml`'s `model.available`, and
`ModelManager.validate_model()` rejects any id absent from that catalog
*before the engine is consulted*. A tag pulled behind the backend's back
would therefore be invisible to `/v1/models` and rejected by
`/admin/model/switch` with `404 model_not_configured`.

So pulling is model management, the backend owns model management, and the
pull must also register the tag in the catalog. Anything else is
split-brain between what is on disk and what the backend believes exists.

## 2. Ownership

| Concern | Owner |
|---|---|
| "Is there room, and is it safe?" | `HostResourceService` (disk) |
| Talking to the daemon's pull API | `OllamaEngine.pull_model()` |
| Policy: refuse / allow, catalog update | `InferenceService` |
| HTTP surface | `POST /admin/model/pull` |

`TransformersEngine` does **not** get this. `InferenceEngine.supports_pull`
defaults `False`, and the service raises `PullNotSupportedError` → `501`,
mirroring how `supports_runtime_lifecycle` is handled. Implementing a Hub
download would be a second, unrelated download path with its own cache
semantics, and nothing needs it: the deployment is Ollama.

## 3. The disk rule

**Refuse, do not warn.** Everywhere else in this backend a heuristic only
warns, because the cost of being wrong is our own request failing. Here the
cost lands on other people, so the asymmetry flips.

Two gates, because the model's size is not knowable before starting:

1. **Pre-flight.** Read free disk on the daemon's own `OLLAMA_MODELS`
   filesystem. If free space is below `MINIMUM_FREE_MIB` (2048), refuse
   immediately — no useful model fits, and it is not worth touching the
   disk to find out.
2. **In-flight.** Ollama's `/api/pull` streams progress events carrying
   `total` bytes. The first event with a `total` gives the real size. If
   `total + RESERVE_MIB (2048) > free`, **abort the stream** and report the
   measured requirement. The reserve exists because a filesystem at 100%
   breaks everything on the box, not just us.

The in-flight gate is the honest design: the size genuinely is not known in
advance. Ollama's registry manifest could be queried directly to learn it,
but that means hardcoding registry URLs and auth behaviour the daemon
already encapsulates, and it would still be an estimate for a tag whose
layers are partially cached. Reading the daemon's own reported `total` uses
the number the daemon itself is working to.

Aborting mid-pull can leave partial blobs. This is reported plainly rather
than hidden, with the remedy (`ollama rm`), because silently leaving
unexplained disk usage on a shared box is worse than saying so.

**No automatic deletion of other models to make room.** Deleting a model to
free space is destructive, cannot be undone without re-downloading, and on
this box the model being deleted may be what someone else is mid-experiment
with. The operator is told what would fit; the choice stays theirs.

## 4. Catalog registration

On success, the tag is added to `config.yaml`'s `model.available` via a new
`ModelManager.register_model()`, so `/v1/models` lists it and
`/admin/model/switch` accepts it. Pull does **not** switch to the new
model: downloading and serving are separate decisions, and a pull that
silently evicted the running model would be a surprise mid-conversation.

## 5. Progress

`POST /admin/model/pull` stays synchronous and returns its summary when the
pull finishes. A long pull holds the request open; the frontend's own HTTP
timeout governs.

Progress is **polled, not streamed**: `GET /admin/model/pull/status` reports
the current byte count while the POST is still outstanding. Streaming the
progress down the POST's own response was rejected for the reason it always
was - it needs the same eager-validation care as `chat_stream`, where a
refusal must arrive as a real status code and not as a dead connection
mid-stream. A second, separately-addressable GET has neither problem: the
refusal path is untouched, and the poll cannot corrupt a transfer it never
touches.

What made this stop being cosmetic: with no progress at all, a caller cannot
distinguish a running download from a hung one. Observed live - the Telegram
frontend pulling `deepseek-r1:14b` (9GB) held one request for most of an
hour and could show the operator nothing, not even that bytes were moving,
while the daemon had reported the byte count in every event throughout. The
information already existed; only the route out was missing.

Mechanics (`services/pull_progress.py`):

- `OllamaEngine.pull_model` forwards each raw daemon event to an optional
  `on_progress` callback. A callback that raises is dropped with one warning
  and the transfer continues - nine gigabytes are not discarded because a
  status line could not be written.
- `PullProgress` keeps per-digest counts, because the daemon re-reports each
  layer's *cumulative* bytes; summing arrivals would count the same bytes on
  every event.
- On success the layers are snapped to their totals: the last event before
  `success` is routinely a few KiB short, and a final reading of 99% reads
  as a stall.
- In-memory and single-slot. A node pulls one model at a time, and progress
  surviving a restart would describe a download that died with the process.

Still not implemented, and still not faked: no resumable job id, no
cancellation. Those belong with worker supervision, which remains deferred.

## 6. Non-goals

- Pulling on a `TransformersEngine` backend (`501`).
- Deleting models to make room.
- Switching to the pulled model.
- Streaming progress down the POST response, job control, cancellation.
- Any pull path that bypasses the catalog.
