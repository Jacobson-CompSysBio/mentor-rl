# External Source Trees

`rwr_hpc/` is a source-only vendored copy of RWR++. Build directories,
Slurm logs, generated results, and bulky example outputs are intentionally
excluded from Git. Refresh it with:

```bash
python scripts/sync_rwr_hpc_source.py --dry-run
python scripts/sync_rwr_hpc_source.py --copy
```

`cli11/` is a vendored copy of CLI11 v2.3.2. The Frontier RWR++ build uses it
through CMake `FetchContent`'s local source override so compute-node builds do
not need outbound GitHub access.
