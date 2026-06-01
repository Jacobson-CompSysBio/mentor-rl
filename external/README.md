# External Source Trees

`rwr_hpc/` is a source-only vendored copy of RWR++. Build directories,
Slurm logs, generated results, and bulky example outputs are intentionally
excluded from Git. Refresh it with:

```bash
python scripts/sync_rwr_hpc_source.py --dry-run
python scripts/sync_rwr_hpc_source.py --copy
```
