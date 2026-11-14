# LOTSA builders

These builders convert raw source datasets into the LOTSA format. Run them from the repository root.

```bash
python -m amets.data.builder.lotsa_v1 <builder> --storage_path <output_dir>
```

Without `--storage_path`, the builders write to `LOTSA_V1_PATH`. That variable has no default, so set one of the two. The output directory must exist before you run a builder. Pass `--datasets` to build a subset and `--overwrite` to rebuild datasets that already exist.

## ProEnFo

1. Download the data files from the [ProEnFo repository](https://github.com/Leo-VK/ProEnFo). Place each file in a folder named after its dataset.
2. Register the directory that holds these folders by running `echo "PROENFO_PATH=<your_path>" >> .env` from the repository root.
3. Process the datasets by running `python -m amets.data.builder.lotsa_v1 proenfo --storage_path <output_dir>` from the repository root.

Warning: the ProEnFo builder reads the pickle files that the ProEnFo repository provides. Loading a pickle file can run arbitrary code. Load only pickle files from a source you trust.
