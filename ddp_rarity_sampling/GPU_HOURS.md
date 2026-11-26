# GPU Hours Record

Contributor: Roshani Patel
Platform: NERSC Perlmutter

| Experiment | Job ID | GPUs Used | GPUs Allocated | Training GPU Hours | Allocated GPU Hours |
|---|---|---:|---:|---:|---:|
| 1-GPU smoke test | 60579782 | 1 | 4 | 0.0012 | 0.0433 |
| 1-GPU full run | 60580474 | 1 | 4 | 0.1355 | 0.6267 |
| 8-GPU smoke test | 61978916 | 8 | 8 | 0.0166 | 0.2089 |
| Linear LR test | 61978990 | 8 | 8 | 0.2050 | 0.3378 |
| Square root LR test | 61978993 | 8 | 8 | 0.2271 | 0.3422 |
| 8-GPU default, seed 0 | 62022882 | 8 | 8 | 0.8400 | 1.1067 |
| 8-GPU rarity, seed 0 | 62023030 | 8 | 8 | 0.9365 | 1.2333 |
| **Total** | | | | **2.3619** | **3.8989** |

## Calculation

Training GPU hours = Training seconds × GPUs used / 3600

Allocated GPU hours = Slurm elapsed seconds × GPUs allocated / 3600

Allocated GPU hours include initialization, validation,
checkpoint saving, and other runtime overhead.

The totals include seven confirmed project jobs.
Unconfirmed interactive sessions are excluded pending
verification of their purpose. Queue waiting time is excluded.