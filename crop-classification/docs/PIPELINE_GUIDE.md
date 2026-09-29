# Crop classification pipeline map

The classification layout follows the [runbook](RUNBOOK.md). No additional
classification source files were supplied in this transfer; the newly restored
ATS/index/pixel-model code belongs to the yield package.

| Stage | Location | Input/output role |
| --- | --- | --- |
| Source boundaries | `raw_data/` | Original province-wide geometry |
| Polygon selection, extraction, labels and joining | `scripts/00_*` through `04_*` | Source geometry to labeled training rows |
| Tracked pixel preparation | `preprocessing/` | Sampled polygons to downloaded/merged pixel rows, with progress state |
| Training and prediction | `scripts/05_*`, `06_*` | Feature/label tables to models, then predictions |
| Feature comparison | `scripts/07_*` | Compare input families using the common trainer |
| Prepared data | `data/` | Geometry subsets, exports, labels and training/prediction tables |
| Model artifacts | `models/` | Model bundles, metrics and plots |
| Documentation | `docs/` | Runbook, script guide, transfer manifest and next steps |

The main and preprocessing script copies differ in their imports, default paths
and progress tracking. They remain together in their respective folders so those
behaviors stay intact. Both paths use the main training scripts.

The original harvest reports remain at `../../harvest-data/` relative to this
document. Their crop names can provide field reference labels after reviewed
field/year matching. Yield observations primarily belong to the other project.

- [Runbook and commands](RUNBOOK.md)
- [Detailed script process](SCRIPT_GUIDE.md)
- [Next steps](NEXT_STEPS.md)
- [Transfer manifest](FILE_MANIFEST.md)
- [Yield pipeline and restored files](../../crop-yield/docs/PIPELINE_GUIDE.md)
