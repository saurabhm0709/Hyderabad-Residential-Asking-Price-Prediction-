"""Load the cleaned listings with the fixed train/test assignment of notebook 16.

Nothing here changes the data. The checks stop immediately if the files are not the ones
the final model was built and evaluated with (notebooks 23-27).
"""

import hashlib
from pathlib import Path

import pandas as pd


# MD5 fingerprints of the files used by notebooks 23-27
EXPECTED_MD5 = {
    "data/processed/cleaned_listings.csv": "3a81a385a9aca28d5b70322e1e143c08",
    "data/processed/train_test_split.csv": "10693e86d1d8a55e606399d718a8f556",
    "data/extracted/localities.csv": "8d9bb54dfc2fb49673e74b976ab73ace",
}
TRAIN_ROWS = 16219
TEST_ROWS = 4081


def file_md5(path):
    # A fingerprint of the file's bytes. If the file changes, the fingerprint changes.
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def check_source_files(project_root):
    # Stop if a source file differs from the one used for the final model
    for relative_path, expected in EXPECTED_MD5.items():
        actual = file_md5(Path(project_root) / relative_path)
        assert actual == expected, f"{relative_path} has changed (MD5 {actual}, expected {expected})"


def load_locality_table(project_root):
    # Only the ID, name and centre of each locality are used; no website price statistic is read
    return pd.read_csv(Path(project_root) / "data" / "extracted" / "localities.csv",
                       usecols=["locality_id", "locality_name", "locality_lat", "locality_long"])


def load_train_test(project_root):
    # Returns (train, test): the cleaned listings split by the saved assignment of notebook 16
    project_root = Path(project_root)
    check_source_files(project_root)

    listings = pd.read_csv(project_root / "data" / "processed" / "cleaned_listings.csv", low_memory=False)
    split_assignment = pd.read_csv(project_root / "data" / "processed" / "train_test_split.csv")
    listings = listings.merge(split_assignment[["property_id", "split"]], on="property_id", validate="one_to_one")

    train = listings[listings["split"] == "train"].copy()
    test = listings[listings["split"] == "test"].copy()

    assert len(train) == TRAIN_ROWS and len(test) == TEST_ROWS
    # Copies of the same property must never be on both sides (docs/duplicate_handling.md)
    assert set(train["property_group_id"]).isdisjoint(test["property_group_id"])
    return train, test
