# Data (not included)

The dataset used in this project is **intentionally not included** in this repository.

It consists of property adverts scraped from a property-listing website on 2 July 2026 (about 20,700 listings). The
adverts contain third-party content: listing text, photos and links, and advertiser names and account identifiers. This
project is released **without the underlying listings**:

- no raw database, no extracted or processed CSV files, and no train/test datasets are published;
- the notebooks keep their saved outputs as a record of the analysis. Advertiser names and advertiser IDs in those
  outputs were redacted before release.

## Files the code expects (if you have the original data)

| Path | Produced by | Used by |
|---|---|---|
| `data/processed/merged_listings.csv` | merge step of the full working project | notebooks 01, 02 |
| `data/extracted/localities.csv` | extraction step of the full working project | notebooks 02–09, `src/` |
| `data/processed/cleaned_listings.csv` | notebook 02 | notebooks 03–09, `src/` |
| `data/processed/train_test_split.csv` | notebook 03 | notebooks 04–09, `src/` |

`src/data/load_data.py` checks the MD5 fingerprints of the three files used by the final model, and stops if they
differ.

## Running the notebooks

The notebooks **cannot be executed from this repository alone**. Their saved outputs are the record of the analysis.
`notebooks/01_eda.ipynb` checks for its input file first and stops with an explanation if it is missing. The other
notebooks stop with a file-not-found error when they try to read the data.

Everything in `data/` except this README is excluded by `.gitignore`.
