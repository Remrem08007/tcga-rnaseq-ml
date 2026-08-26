# Data directory

Large TCGA source files are intentionally not committed.

List registered official GDC sources:

```bash
python -m tcga_ml.download --list
```

Download one source into `data/raw/`:

```bash
python -m tcga_ml.download --source sample_quality
python -m tcga_ml.download --source expression
```

Each completed download receives a sidecar `*.provenance.json` containing its source URL, retrieval timestamp, size, and SHA-256 digest.
