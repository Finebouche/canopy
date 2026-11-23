import optuna

from canopy.studies.common import completed, copy_into, storage, study_names

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize(path, name, trials):
    study = optuna.create_study(study_name=name, storage=storage(path), direction="maximize", load_if_exists=True)
    study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=trials)


def test_the_studies_of_the_jobs_are_merged_into_one_database(tmp_path):
    """A study of a part replaces that of the database of the dataset only when it has more completed trials."""
    database = tmp_path / "hpo" / "marron1_db.sqlite3"
    assert study_names(database) == set() and not database.exists()  # (no empty database created)
    part = tmp_path / "parts" / "a" / "marron1_db.sqlite3"
    optimize(part, "a", 3)
    assert "3 completed trials copied" in copy_into("a", part, database)
    other = tmp_path / "parts" / "b" / "marron1_db.sqlite3"
    optimize(other, "b", 2)
    copy_into("b", other, database)
    assert study_names(database) == {"a", "b"}
    assert "kept" in copy_into("a", part, database)
    optimize(part, "a", 2)  # (the job resumed: 5 trials)
    copy_into("a", part, database)
    assert completed(optuna.load_study(study_name="a", storage=storage(database))) == 5
    assert completed(optuna.load_study(study_name="b", storage=storage(database))) == 2
