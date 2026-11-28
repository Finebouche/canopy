#!/bin/bash
# Control of the next phrase on PlaFRIM: the reservoirs that read the labels of the phrases
# (canopy.studies.next_phrase_labels): one job per study of the pure models at each size (job array), then the merge
# of their parts into outputs/next_phrase_labels/hpo/marron1_db.sqlite3 and the final evaluation on the test songs
# (one job, NB_SEEDS seeds). Run from the root of a checkout of canopy with the MFCC cache and the annotations:
#   [CANOPY_WITH_CALLS=1] [CANOPY_LEAKY_RESERVOIRS=1] [CANOPY_LABELS_WITH_DURATIONS=1] [CANOPY_DATASET=llb3] [PURE="..."]
#       [SIZES="500 1000"] [NB_SEEDS=10] [CONSTRAINT="zonda|suroit|bora"] [BEGIN=20:00] bash slurm/submit_next_phrase_labels.sh
# (CANOPY_DATASET=llb3: the canary llb3, canopy.settings; BEGIN: start of the studies, sbatch --begin, default now)
# The studies are light (one step per phrase: 400 trials in 10 to 60 minutes): 8 cores and 16 GB per job.
set -euo pipefail
NB_SEEDS=${NB_SEEDS:-10}
read -r -a SIZES <<< "${SIZES:-500 1000}"
read -r -a PURE <<< "${PURE:-mean_hag var_hag random_ee random_ei}"
NODES=(--constraint="${CONSTRAINT:-zonda|suroit|bora}")  # CPU nodes, no GPU nor big memory
RESOURCES=(--cpus-per-task=8 --mem=16G --time=04:00:00)
cd "$(dirname "$0")/.."
mkdir -p slurm/logs slurm/tasks

TASKS=slurm/tasks/next_phrase_labels_$(date +%Y%m%d_%H%M%S).txt
for size in "${SIZES[@]}"; do for model in "${PURE[@]}"; do echo "$model $size"; done; done > "$TASKS"
n=$(wc -l < "$TASKS")
studies=$(sbatch --parsable "${NODES[@]}" "${RESOURCES[@]}" --begin="${BEGIN:-now}" --array=0-$((n - 1)) --job-name=labels-studies \
    --export=ALL,CANOPY_TASK=next_phrase_labels,CANOPY_TASK_FILE="$TASKS" slurm/canary.sbatch)
echo "studies: job array $studies ($n studies, $TASKS)"
merge=$(sbatch --parsable "${NODES[@]}" --cpus-per-task=1 --mem=4G --time=00:30:00 --dependency=afterok:$studies \
    --job-name=labels-merge --output=slurm/logs/%x-%j.out --wrap="bash slurm/merge_studies.sh next_phrase_labels")
echo "merge of the studies: job $merge (after $studies)"
evaluation=$(sbatch --parsable "${NODES[@]}" "${RESOURCES[@]}" --dependency=afterok:$merge --job-name=labels-evaluation \
    --export=ALL,CANOPY_JOB_MODE=next_phrase_labels,CANOPY_WORKERS=1 --output=slurm/logs/%x-%j.out \
    slurm/canary.sbatch "$(IFS=,; echo "${SIZES[*]}")" "$NB_SEEDS")
echo "final evaluation: job $evaluation (after $merge)"
