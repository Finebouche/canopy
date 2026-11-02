"""
Conditions of the syllable classification (canopy.syllables_classification.classification): canopy.evaluate_utils.conditions
of the syllable studies (canopy.syllables_classification.hpo.study_name_of), for each SIZE.
"""
from canopy.evaluate_utils import conditions as task_conditions
from canopy.syllables_classification.hpo import study_name_of

SIZES = [250, 500, 1000]
REFERENCE = "random_ee"  # (condition of the paired comparisons)


def conditions(size):
    """canopy.evaluate_utils.conditions of the studies of the task (canopy.syllables_classification.hpo.study_name_of)."""
    return task_conditions(size, study_name_of)
