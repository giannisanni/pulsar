# Expert store: the one interface everything hangs off

Status: proposed (written at the end of the bring-up session; review at
the start of the kernel-port session before implementing).

## The problem

ds4 addresses routed experts through global mutable state: a
selected-expert staging cache (`g_stream_selected_cache`), a VRAM decode
cache, a host cache, and kernels that consult whichever is "current" via
slot tables registered as a side effect of the load call. It works, and
it produced every number NeutronStar is proud of - but the milestone-3
OOM (batch kernels silently falling back to whole-tensor model views
when the slot count mismatched) is exactly the failure mode that design
invites: the kernel's data source is decided by distant global state.

pulsar inverts this: **kernels receive explicit device pointers; where
the bytes came from is the host's problem, resolved before launch.**

## Kernel contract

Every routed-expert kernel launch takes a compact pointer array:

```
// per (token, selected expert): gate, up, down device pointers
struct ExpertPtrs { const void *gate, *up, *down; };
// launch args: ExpertPtrs[n_tok * n_used], weights[n_tok * n_used]
```

No slot tables, no globals, no fallback paths. If a pointer is not
resident when the launch is built, the host blocks on (or reorders
around) the fetch - the kernel never improvises.

## Host-side placement

```
enum Placement { Disk, Host(HostSlab), Device(dev_id, DevSlab) }
trait ExpertStore {
    /// Resolve to device pointers for `dev`, fetching/uploading as
    /// needed. Batch call: one union fetch per layer, like ds4's
    /// prepare_selected_batch.
    fn resolve(&mut self, dev: DeviceId, wants: &[ExpertId]) -> ResolvedPtrs;
    /// Hint from the cross-layer prefetcher; never blocks.
    fn prefetch(&mut self, wants: &[ExpertId]);
}
```

- **v1 (single GPU, streaming)**: `StreamingStore` = io_uring fetcher
  (exists, benched) + LFU host cache with persistent warm state (port of
  the proven NeutronStar design) + a fixed device staging arena with
  per-layer compact slots. Prefetch = the measured +21% cross-layer
  router lookahead, as an explicit request channel instead of
  fire-and-forget ring state.
- **v2 (multi-GPU residency)**: `ResidentStore` = static placement map
  (expert -> device) computed at load from VRAM budgets; `resolve`
  becomes a table lookup; misses (experts that fit nowhere) fall back to
  the v1 path. This is the 5060 Ti plan: same trait, new policy - no
  engine changes.

## Ownership rules (the part Rust is here for)

- Host slabs are owned by the cache; `resolve` copies to the device
  arena and returns; eviction after the copy is always safe. No
  cross-thread borrows of cache memory - the warm-loader/get() hazard
  from NeutronStar becomes unrepresentable.
- Device arena slots are owned per layer-step; a slot is reusable only
  after the stream event for its consuming kernel completes (guarded by
  a CUDA event, checked in `resolve`, not assumed).
- The fetcher owns its buffers until completion, then transfers whole
  buffers into the cache (same donation pattern ds4 uses, but as a move,
  not a convention).

## Non-goals for v1

- No expert bundle sidecar (parked in NeutronStar for the same reason:
  post-prefetch economics), though the slab addressing keeps offsets
  abstract so one can be added as a Placement source later.
- No cross-device P2P until v2 hardware exists (the seam is
  `Device(dev_id, ..)` already carrying the device).
