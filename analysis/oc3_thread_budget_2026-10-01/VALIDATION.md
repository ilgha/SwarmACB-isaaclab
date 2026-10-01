# OC3 Native Thread Budget Fix

## Evidence

The cluster's third worker (Foraging) failed before environment creation with:

```text
failed to spawn thread: Os { code: 11, kind: WouldBlock,
message: "Resource temporarily unavailable" }
exitcode=-6 (SIGABRT)
```

The native stack points to Omniverse Client initialization. This proves failure
to create a thread, not the particular job/user/system resource limit involved.
The earlier sampled MaxRSS was about 7.57 GiB against a 48 GiB host RAM request.
There is no evidence here of a training loss, robot controller or reward failure.

## Changes

- Inherit BLAS/OpenMP and default Rayon/Tokio thread budgets before child Python
  imports torch/NumPy while unpickling multiprocessing arguments.
- Apply `worker_threads` (2 by default) to Kit tasking, TBB and physics pools.
- Set PyTorch inter-op parallelism to one; retain its existing intra-op budget.
- Restore the coordinator environment after spawning, including on failure.
- Log effective Kit settings and Linux process thread count / RLIMIT_NPROC.
- Include more worker log lines so the panic is not obscured by crash metadata.

These are per-pool settings, not a hard total OS thread limit. Native components
can have additional threads or explicit pool settings of their own. No learning
objective, network, simulator timestep, reward, Slurm allocation or host limit
was changed. No vendor installation files were edited.

## Verification

69 regression tests ran: 68 passed, one POSIX-only signal test skipped on Windows.
The suite includes a real spawned process checking inherited environment settings
and tests that parent state and pipe cleanup survive a failed spawn.

Two real Isaac Sim 5.1 smoke runs completed on Windows:

| Device | Missions | Decisions per mission | Result |
|---|---|---:|---|
| CPU | XOR, Homing, Foraging, Sheltering | 400 | Startup, shared update, checkpoint, exports |
| CUDA | XOR, Homing, Foraging, Sheltering | 400 | Startup, shared update, checkpoint, exports |

All worker logs reported 2 for each of the three configured Kit pool settings.
Artifacts are under `runs/`, `checkpoints/`, `gpu_runs/`, `gpu_checkpoints/` beside
this file. The Linux container and its resource limits were not available here;
a one-seed cluster pilot remains necessary before a full array relaunch.

## References

- [Isaac Sim 5.1 CPU thread settings](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/reference_material/sim_performance_optimization_handbook.html#cpu-thread-count-optimizations)
- [Rayon default pool configuration](https://docs.rs/rayon-core/latest/rayon_core/struct.ThreadPoolBuilder.html#method.num_threads)
- [Tokio default worker configuration](https://docs.rs/tokio/latest/tokio/runtime/struct.Builder.html#method.worker_threads)
