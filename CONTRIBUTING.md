# Contributing

This is research code. Bug reports and fixes are welcome.

## Reporting issues

Include your OS, Python version, and the command you ran. For training or evaluation failures, run with `HYDRA_FULL_ERROR=1` and attach the full traceback.

## Pull requests

The repository has no test suite or CI. Describe how you verified your change. At a minimum, confirm that the package imports and that a short training run starts.

```bash
python -c "from amets.model.pretrain import AMETSPretrain"
python scripts/train.py --config-name ame_tiny trainer.devices=1 \
    trainer.max_epochs=1 train_dataloader.num_batches_per_epoch=5 model.num_warmup_steps=1
```

## Conventions

- Strip notebook outputs before committing. Run `pip install nbstripout` and then `nbstripout --install` once inside the repository.
- Do not add absolute paths. Resolve paths from the repository root or from an environment variable listed in `Env.path_vars` in `amets/common/env.py`.
- Keep model weights, datasets, figures, and result files out of the repository.
- Write docstrings in Google style, with `Args:` and `Returns:`. Match the style of the file you edit.

## License headers

Files derived from Salesforce's uni2ts carry an Apache-2.0 header with the Salesforce copyright. Keep that header when you edit such a file. New files must stay compatible with Apache-2.0. See `NOTICE`.
