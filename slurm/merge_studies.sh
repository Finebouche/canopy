#!/bin/bash
# Merges the parts of the jobs of the studies (outputs/<study>/hpo/parts/<dataset>/<Optuna study>/, slurm/canary.sbatch)
# into the databases of the datasets, outputs/<study>/hpo/marron1_db.sqlite3 (python -m canopy.studies.common merge).
# Run from the root of a checkout of canopy, once the jobs of the studies are done and before the evaluations:
#   bash slurm/merge_studies.sh [studies]   (default: all, phrase_boundaries next_phrase syllables_classification)
set -euo pipefail
cd "$(dirname "$0")/.."
set +u
module load micromamba
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$(micromamba shell hook -s bash)"
micromamba activate hag_env
set -u
PYTHONPATH=$PWD:${HAG_DIR:-$HOME/hag_core} python -m canopy.studies.common merge "$@"
