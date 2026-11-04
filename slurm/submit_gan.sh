#!/bin/bash
# Training of the low-dimensional WaveGAN of the canary syllables (canopy.syllable_generation.wavegan) on a GPU node of
# PlaFRIM: sirocco with a Turing (rtx8000) or Ampere (a100) GPU, the torch of hag_env (CUDA 13) having no kernels for
# the Pascal (p100) and Volta (v100) GPUs; the whole node (no GPU resource in Slurm: the jobs of a node would share its
# GPUs). Results in outputs/canary/syllable_gan/<RUN> (losses.csv, checkpoints every 15 epochs, last.pt), training
# resumed from last.pt: CHAIN jobs one after the other (each resuming the training where the previous one stopped, at
# its time limit; those after the last epoch end at once), or submit again. The syllables
# (canopy.syllable_generation.syllables) are cut at the start of the job from the audio of marron1
# (canopy/datasets/marron1/audio, copied in the checkout: the compute nodes have no internet). Run from the root of a
# checkout of canopy:
#   [RUN=wavegan_0] [EPOCHS=1000] [CHAIN=2] [CONSTRAINT='a100|rtx8000'] [TIME=2-00:00:00] bash slurm/submit_gan.sh
set -euo pipefail
RUN=${RUN:-wavegan_0}
EPOCHS=${EPOCHS:-1000}
CHAIN=${CHAIN:-2}
CONSTRAINT=${CONSTRAINT:-a100|rtx8000}
TIME=${TIME:-2-00:00:00}
cd "$(dirname "$0")/.."
mkdir -p slurm/logs
if [ -z "$(find canopy/datasets/marron1/audio -name '*.wav' -print -quit 2>/dev/null)" ]; then
    echo "No audio in canopy/datasets/marron1/audio: copy it first (the compute nodes have no internet)" >&2
    exit 1
fi
previous=
for _ in $(seq "$CHAIN"); do
previous=$(sbatch --parsable ${previous:+--dependency=afterany:$previous} --job-name="gan-$RUN" \
    --constraint="$CONSTRAINT" --exclusive --nodes=1 --ntasks=1 --time="$TIME" --output="slurm/logs/%x-%j.out" --wrap="
set +u
module load micromamba
export MAMBA_ROOT_PREFIX=\$HOME/micromamba
eval \"\$(micromamba shell hook -s bash)\"
micromamba activate hag_env
set -u
nvidia-smi --query-gpu=name,memory.total --format=csv
export PYTHONPATH=\$PWD NUMBA_CACHE_DIR=/dev/shm/numba_cache_\$SLURM_JOB_ID
python -m canopy.syllable_generation.wavegan train outputs/canary/syllable_gan/$RUN $EPOCHS")
echo "gan-$RUN: job $previous"
done
