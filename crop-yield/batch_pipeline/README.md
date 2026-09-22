# Alberta batch pipeline (incomplete)

These are the surviving extraction and crop classification batch scripts from
the yield project's Alberta workflow. They remain here because the yield guides
describe their configuration and downstream use.

This workflow cannot run yet: `quarter_section_batch_utils.py`,
`copernicus_data_space_utils.py`, and `build_copernicus_scene_manifest.py` are
missing. Source geometries, satellite exports, and credentials must also be
supplied. The JSON configuration describes the intended workflow; it does not
establish that its inputs exist.

See [the historical pipeline guide](../docs/PIPELINE_GUIDE.md) for the
specification or [the project README](../README.md) for the runnable benchmark.
