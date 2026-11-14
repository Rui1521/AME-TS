#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# This file lives at <repo>/amets/common/env.py
REPO_ROOT = Path(__file__).resolve().parents[2]  # <repo>


def get_path_var(var: Optional[str]) -> Optional[Path]:
    if (path := os.getenv(var)) is not None:
        return Path(path)
    return None


class Env:
    """Singleton that loads ``.env`` and exports repo-relative path defaults.

    Path variables (each also readable as a ``Path`` attribute on the class):

    - ``AMETS_ROOT``: the repository root.
    - ``GIFT_EVAL_DATA`` and ``GIFT_EVAL``: GIFT-Eval benchmark data, including the
      val and test regions. Training never reads from this location.
    - ``GIFT_EVAL_REPO``: local clone of the gift-eval code.
    - ``PRETRAIN_DATA``: training pools, read by the data configs through
      ``${oc.env:PRETRAIN_DATA}``.
    - ``REGIME_CACHE_DIR``: label caches written by ``regime_predictor/train.py``.
    - ``LOTSA_V1_PATH``, ``LSF_PATH``, ``CUSTOM_DATA_PATH``, ``HF_CACHE_PATH``: optional
      paths with no default.
    """

    _instance: Optional["Env"] = None
    path_vars: list[str] = [
        "AMETS_ROOT",
        "GIFT_EVAL_DATA",
        "GIFT_EVAL",
        "GIFT_EVAL_REPO",
        "PRETRAIN_DATA",
        "REGIME_CACHE_DIR",
        "LOTSA_V1_PATH",
        "LSF_PATH",
        "CUSTOM_DATA_PATH",
        "HF_CACHE_PATH",
    ]
    # Variables that fall back to a repo-relative location, so a fresh clone
    # works with no configuration. Anything already set in the environment or
    # in .env wins.
    defaults: dict[str, Path] = {
        "AMETS_ROOT": REPO_ROOT,
        "GIFT_EVAL_DATA": REPO_ROOT / "gift-eval-data",
        # The gift_eval package calls Path(os.getenv("GIFT_EVAL")) with no
        # fallback and raises TypeError if the variable is unset. It points to
        # the same location as GIFT_EVAL_DATA.
        "GIFT_EVAL": REPO_ROOT / "gift-eval-data",
        # Clone of https://github.com/SalesforceAIResearch/gift-eval, which
        # provides the `gift_eval` Python package and dataset_properties.json.
        # This repository does not include it. See the README.
        "GIFT_EVAL_REPO": REPO_ROOT / "gift-eval",
        # Training pools, such as the GIFT-Eval train split that
        # scripts/build_train_split.py writes, synthetic data, and LOTSA. This
        # location is separate from GIFT_EVAL_DATA, which holds the full
        # benchmark including the val and test regions.
        "PRETRAIN_DATA": REPO_ROOT / "pretrain-data",
        # Stage-1 label caches that regime_predictor/train.py writes. They are
        # large generated files.
        "REGIME_CACHE_DIR": REPO_ROOT / "regime-label-cache",
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # A .env file is optional because the defaults cover a plain clone.
            load_dotenv()
            cls.apply_defaults()
            cls.monkey_patch_path_vars()
        return cls._instance

    @classmethod
    def apply_defaults(cls):
        """Export defaults into os.environ for any var not already set.

        Hydra configs read these values through ``${oc.env:GIFT_EVAL_DATA}``,
        so the defaults must be in os.environ and not only on the class.
        """
        for var, default in cls.defaults.items():
            if not os.getenv(var):
                os.environ[var] = str(default)

        # GIFT_EVAL_DATA is this project's name for the benchmark data location.
        # GIFT_EVAL is the name the gift_eval package reads. The user may set
        # either one, and this block copies it to the other.
        if os.getenv("GIFT_EVAL_DATA") != os.getenv("GIFT_EVAL"):
            explicit = {
                var for var in ("GIFT_EVAL_DATA", "GIFT_EVAL")
                if os.environ.get(var) != str(cls.defaults[var])
            }
            if explicit == {"GIFT_EVAL_DATA"}:
                os.environ["GIFT_EVAL"] = os.environ["GIFT_EVAL_DATA"]
            elif explicit == {"GIFT_EVAL"}:
                os.environ["GIFT_EVAL_DATA"] = os.environ["GIFT_EVAL"]

    @classmethod
    def monkey_patch_path_vars(cls):
        for var in cls.path_vars:
            setattr(cls, var, get_path_var(var))


_GIFT_EVAL_INSTALL_HINT = (
    "The GIFT-Eval benchmark code is not vendored in this repository. Clone it "
    "and point GIFT_EVAL_REPO at it:\n"
    "    git clone https://github.com/SalesforceAIResearch/gift-eval\n"
    "    export GIFT_EVAL_REPO=$(pwd)/gift-eval\n"
    "See the README for details."
)


def gift_eval_repo() -> Path:
    """Return the path to a local gift-eval checkout.

    Raises:
        FileNotFoundError: If no checkout is found at ``$GIFT_EVAL_REPO``.
    """
    repo = Path(os.environ["GIFT_EVAL_REPO"])
    if not (repo / "src" / "gift_eval").is_dir():
        raise FileNotFoundError(
            f"No gift-eval checkout at {repo}.\n{_GIFT_EVAL_INSTALL_HINT}"
        )
    return repo


def add_gift_eval_to_path() -> Path:
    """Put the ``gift_eval`` package on ``sys.path`` and return the repo path.

    The benchmark is not distributed as a wheel, so importing it means adding its
    ``src/`` directory to the path.
    """
    import sys

    repo = gift_eval_repo()
    src = str(repo / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return repo


def dataset_properties_path() -> Path:
    """Return the path to GIFT-Eval's ``dataset_properties.json``.

    The file maps dataset names to frequency, domain, and variate count. It
    lives in the gift-eval checkout.

    Raises:
        FileNotFoundError: If the file is not present in the checkout.
    """
    path = gift_eval_repo() / "notebooks" / "dataset_properties.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"dataset_properties.json not found at {path}.\n{_GIFT_EVAL_INSTALL_HINT}"
        )
    return path


env = Env()
