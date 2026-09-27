# Hyderabad Asking-Price Prediction

Predicting the **advertised asking price** of homes listed for sale in Hyderabad, with a leakage-aware, group-aware
machine-learning workflow: careful cleaning, duplicate-aware splitting, cross-validated model selection, and **one**
final evaluation on an untouched test set.

> **The scraped dataset is not included in this repository** (see [`data/README.md`](data/README.md)). The notebooks
> keep their saved outputs, so the full analysis can be read without the data. They **cannot be re-run** from this
> repository alone, because they require the original (private) dataset.

---

## 1. Project overview

| | |
|---|---|
| Data | 20,300 cleaned sale listings from one scrape of a property-listing website (2 July 2026) |
| Target | advertised asking price (₹), modelled as log(price) |
| Final model | `HistGradientBoostingRegressor` (scikit-learn), 25 fixed-rule features |
| Test result (4,081 unseen listings, evaluated once) | **MAE ₹57.0 lakh · Median APE 16.8% · R² 0.556 · 66.7% within ±25%** |
| Key limitation | not a random sample of the market: the data was collected through price-range searches with result caps |

(1 lakh = ₹1,00,000; 100 lakh = ₹1 crore.)

## 2. Problem statement

Given the information in a property advert (type, size, rooms, floor, locality, furnishing, possession status, ...),
predict the **asking price** the advertiser states.

The target is the **advertised asking price**. It is **not** a sale or transaction price and **not** a market value.
90.7% of the cleaned listings were posted by owners, so the prices include optimistic and mistyped values.

## 3. Dataset and collection period

| Item | Value |
|---|---|
| Source | one scraped database of property adverts for Hyderabad |
| Scrape date | **2 July 2026**: one snapshot of adverts live on that day |
| Posting dates | mostly May–June 2026 (`listing_date` values from July 2025 to July 2026) |
| Rows | 20,700 extracted → 20,694 Hyderabad sale listings → **20,300 after cleaning** |
| Property types | Apartment, Residential House, Villa, Builder Floor Apartment, Penthouse |
| Asking prices (cleaned) | ₹1.4 lakh to ₹55 Cr; median ₹1.5 Cr |

This is a single snapshot, not a time series. It cannot describe price trends.

## 4. Important sampling limitation

The scraper did not run one "all Hyderabad" search. It ran **9 separate price-range searches**: ₹0–50 L, 50–80 L,
80 L–1.2 Cr, 1.2–1.5 Cr, 1.5–1.75 Cr, 1.75–2 Cr, 2–3 Cr, 3–5 Cr and 5 Cr+. Each search was capped at about 3,000 results,
and five ranges probably hit the cap.

What follows from that:
- **The dataset is not a random or representative sample** of Hyderabad listings, and not of the Hyderabad housing
  market. Price bands, property types and localities are not represented in proportion to the market. The distortion
  cannot be measured or corrected, because the website's true counts were not recorded.
- **No market-share or population-level claims** are made from this data.
- **Scrape time and row order are never used as features.** Both encode the price range, so they would leak the target.
- **Results are reported per price range as well as overall.**

## 5. Data-cleaning methodology

All cleaning rules use only each listing's **own values** and **fixed thresholds written down in advance**, so they learn
nothing from other listings ([`notebooks/02_data_cleaning.ipynb`](notebooks/02_data_cleaning.ipynb)).

| Step | Rows | Decision |
|---|---|---|
| D1: missing price ("Call for Price") | 280 removed | no target |
| R1: impossible price | 114 removed (0.6%) | ₹100 Cr+ ordinary homes (14); a Telecalling batch of houses priced at about ₹3,300/sq-ft (29); a rate below ₹1,000 or above ₹1 lakh per sq-ft with a plausible area (71) |
| R2: unreliable area | 230 kept | area set to missing; never converted or guessed |
| R3: floor above total floors | 3 kept | both floor fields set to missing |
| (0, 0) coordinates | 7,459 | set to missing; no coordinate repaired |
| Transaction type | 0 removed | all listings are sale adverts; the "Rent" labels were shown to be mislabelled sales |
| Genuine luxury homes (₹20 Cr+ at a plausible rate) | kept | extremes are investigated, not removed automatically |

The website's own locality price statistics and every other target-derived column were dropped.

## 6. Leakage prevention and the group-aware split

- **Duplicate properties.** The same property is often advertised several times (reposts, several agents). A
  `property_group_id` groups these copies: 1,534 cleaned listings (7.6%) are in 709 multi-listing groups. Every split
  keeps a group on one side, both train/test and every cross-validation fold. Without this, the model could recall a
  copy's price instead of predicting it, and the scores would be too optimistic.
- **The train/test split.** `GroupShuffleSplit(test_size=0.20, random_state=42)` on `property_group_id` gives **16,219
  training and 4,081 test listings**, with **0 property groups on both sides**
  ([`notebooks/03_train_test_split.ipynb`](notebooks/03_train_test_split.ipynb)).
- **One-time test evaluation.** The test set was removed from memory in every modelling experiment (baselines through
  tuning) and used **once**, after the model was frozen.
- **Learned preprocessing inside a pipeline.** Every learned step (imputation, scaling, one-hot categories,
  rare-locality grouping) sits inside a scikit-learn `Pipeline`, fitted on the training data or training fold only.
- **No target-derived features.** There is no price per sq-ft, no website price statistics, no dates and no identifiers.

## 7. Feature engineering

25 columns, all built by **fixed rules** from the listing's own values: `build_model_features()` in
[`src/features/build_features.py`](src/features/build_features.py).

| Group | Features |
|---|---|
| Size | bedrooms, bathrooms, log(1 + covered area in sq-ft) |
| Floor (apartment-like types only) | floor number, building height |
| Location | locality ID (one-hot, localities with < 10 training listings grouped as "rare"), and **six straight-line (Haversine) distances** from the locality centre to fixed landmarks: city centre, RGIA airport, HITEC City, Gachibowli, Financial District, Secunderabad |
| Flags | area missing, area typed in sq-yd, land-area unit present, prime location, part of a named project |
| Categories | property type, balconies, transaction group, furnishing, possession status, age, facing, posted by |

Two location ideas were tested and handled on the evidence:
- **Distances replaced linear latitude/longitude** (CV MAE −1.26%, better in all 5 folds;
  [`notebooks/05_geographic_features.ipynb`](notebooks/05_geographic_features.ipynb)).
- **Locality target encoding** (a locality price level, fitted with group-aware cross-fitting) improved MAE by only 0.5%
  and was **not** used.

Listing coordinates were rejected as unreliable.

## 8. Models compared

5-fold `GroupKFold(shuffle=True, random_state=42)` on the training set only, with groups = `property_group_id`. Values
are the mean over the folds; MAE and RMSE are in ₹ lakh.

| Model | CV MAE | CV Median APE | CV RMSE | CV R² |
|---|---|---|---|---|
| Dummy (median price) | 115.66 | 40.95% | 255.75 | −0.056 |
| Ridge, log(price), with lat/long | 70.25 | 21.84% | 179.53 | 0.481 |
| Ridge, log(price), with distances (reference) | 69.37 | 21.37% | 177.69 | 0.491 |
| Random Forest | 66.22 | 19.52% | 193.03 | 0.402 |
| HistGradientBoosting (default settings) | 58.35 | 17.04% | 164.76 | 0.561 |
| **HistGradientBoosting, tuned (final)** | **57.53 ± 2.74** | **16.51%** | **161.87** | **0.576** |

How the models differed:
- **HistGradientBoosting** beat the Ridge reference on all four metrics **in all 5 folds**.
- **The Random Forest** was better than Ridge for typical listings, but it could not predict high prices, so its RMSE
  and R² were worse ([`notebooks/06_tree_model_comparison.ipynb`](notebooks/06_tree_model_comparison.ipynb)).
- **Tuning** tested 32 predefined HistGradientBoosting configurations and Ridge alphas 0.01–100
  ([`notebooks/07_model_tuning.ipynb`](notebooks/07_model_tuning.ipynb)).
  - **The selection rule** was fixed in advance: lowest mean CV MAE. Configurations within ₹0.5 lakh of the best count
    as tied, and among tied ones the simpler model is preferred.
  - **The result:** tuning improved HistGradientBoosting's MAE by 1.4% over the default; tuning Ridge brought nothing.
  - **Optimism:** because the configuration was chosen on the same folds, its CV score is slightly optimistic. The
    untouched test set gives the honest estimate.

## 9. Final model

```
Pipeline:
  numeric      median imputation -> StandardScaler
  binary       passthrough
  categorical  OneHotEncoder(handle_unknown="ignore")
  locality     OneHotEncoder(min_frequency=10, handle_unknown="infrequent_if_exist")
  model        TransformedTargetRegressor(np.log / np.exp) around
               HistGradientBoostingRegressor(learning_rate=0.1, max_iter=200, max_leaf_nodes=31,
                                             max_depth=None, min_samples_leaf=40, l2_regularization=1.0,
                                             early_stopping=False, random_state=42)
```

Early stopping is off because its internal validation split is not group-aware. The implementation is in
[`src/models/train_model.py`](src/models/train_model.py).

## 10. Final test results

Fitted once on the 16,219 training listings, then evaluated once on the 4,081 test listings
([`notebooks/08_final_test_evaluation.ipynb`](notebooks/08_final_test_evaluation.ipynb)).

| Metric | CV estimate (used for selection) | **Untouched test set** |
|---|---|---|
| MAE | ₹57.53 lakh | **₹57.01 lakh** (95% group-bootstrap interval ₹52.5–61.8 lakh) |
| Median APE | 16.51% | **16.84%** (interval 16.2–17.6%) |
| RMSE | ₹161.87 lakh | **₹158.34 lakh** |
| R² | 0.576 | **0.556** |
| Within ±10% / ±25% | 32.9% / 66.2% | **31.8% / 66.7%** |
| Off by a factor of 2 or more | – | **5.5%** |

The test result agrees with the cross-validation estimate: every difference is within about one fold standard deviation.

## 11. Error analysis

From [`notebooks/09_final_error_analysis.ipynb`](notebooks/09_final_error_analysis.ipynb), on the test set:

| Group | Listings | Median APE | Median signed error |
|---|---|---|---|
| Apartment | 2,328 | 13.2% | +0.1% |
| Residential House | 1,093 | 25.2% | +2.1% |
| Villa | 503 | 22.3% | −1.0% |
| ₹0–50 L | 311 | 24.9% | +23.2% |
| ₹1.5–1.75 Cr | 458 | 12.4% | +0.5% |
| ₹5 Cr+ band | 228 | 30.1% | −27.5% |
| ₹20 Cr+ | 16 | 58.5% | −58.5% (all under-predicted) |

- **Pull toward the middle.** Cheap listings are predicted too high and expensive ones too low. The highest test
  prediction is ₹14.9 Cr, while the most expensive test listing is priced at ₹44 Cr. Overall there is no bias: the
  median prediction is 1.003 × the asking price.
- **Rupee errors are concentrated at the top.** The 16 listings at ₹20 Cr+ (0.4% of the test set) cause 11.5% of the
  total absolute error. Without them the MAE is ₹50.7 lakh.
- **Houses and villas** have about twice the relative error of apartments. They are 39% of the listings but cause 68% of
  the rupee error. Their "area" can mean plot or built-up area.
- **Missing information makes it worse.** Listings without a usable area (Median APE 26.6%) and listings with an unknown
  or unseen locality (22–23%) are predicted less well.

## 12. Limitations

- **The target is an asking price,** including optimistic prices and probable entry errors that the cleaning rules
  cannot detect, especially when the area is missing.
- **One day's snapshot, collected by price-range searches.** The results describe this collection, not the Hyderabad
  housing market, and they say nothing about other time periods or other websites.
- **Very few luxury listings** (48 training listings at ₹20 Cr+). The model does not predict in that range.
- **Area ambiguity for houses and villas,** and 8.6% of listings have no usable area.
- **About 7% of listings have no usable locality;** their distance features fall back to the training median.
- **`exp()` of a log prediction** is closer to a median than a mean, so rupee predictions are slightly biased low.

**Legitimate claim:** on unseen listings from this collection, the model predicts the **advertised asking price**:
- with an MAE of about ₹57 lakh;
- with a median absolute percentage error of about 17%;
- within ±25% for about two-thirds of listings.

It is most reliable for apartments and for the ₹50 L–3 Cr range.

**It must not be used as:**
- a sale-price, transaction-price or market-value model;
- a description of the Hyderabad housing market, or of market shares or trends;
- a valuation tool for ₹20 Cr+ luxury properties, or for houses/villas without a usable area or listings with an
  unknown locality;
- a model for other time periods or other websites (not evaluated).

## 13. Repository structure

```
├── README.md
├── requirements.txt              pinned versions (Python 3.14.2)
├── data/
│   └── README.md                 why the data is not included; which files the code expects
├── notebooks/
│   ├── 01_eda.ipynb                     exploratory data analysis
│   ├── 02_data_cleaning.ipynb           cleaning rules applied, with an audit table
│   ├── 03_train_test_split.ipynb        group-aware 80/20 split
│   ├── 04_feature_engineering.ipynb     baseline features and preprocessing design
│   ├── 05_geographic_features.ipynb     distance features vs latitude/longitude
│   ├── 06_tree_model_comparison.ipynb   Ridge vs Random Forest vs HistGradientBoosting
│   ├── 07_model_tuning.ipynb            predefined tuning grid and final model selection
│   ├── 08_final_test_evaluation.ipynb   the one-time evaluation on the untouched test set
│   └── 09_final_error_analysis.ipynb    where the final model makes its errors
└── src/
    ├── data/load_data.py            load cleaned data with the fixed train/test assignment (with file checks)
    ├── features/build_features.py   fixed-rule features: build_model_features()
    └── models/
        ├── train_model.py           the frozen final pipeline and the GroupKFold setup
        ├── predict_model.py         predict asking prices with a fitted model
        └── evaluate_model.py        MAE, RMSE, R², Median APE, within ±10% / ±25%, ≥ 2× errors
```

The release notebooks are numbered 01–09 in reading order. They come from a longer working project (notebooks 01–27),
and their own titles still use those **original numbers**:

| Release notebook | Original number |
|---|---|
| 01_eda | 04 |
| 02_data_cleaning | 15 |
| 03_train_test_split | 16 |
| 04_feature_engineering | 18 |
| 05_geographic_features | 22 |
| 06_tree_model_comparison | 24 |
| 07_model_tuning | 25 |
| 08_final_test_evaluation | 26 |
| 09_final_error_analysis | 27 |

The other original notebooks (extraction, investigations and intermediate experiments) and the project's
investigation documents are not included. Their conclusions are summarised in this README. The notebooks' text
sometimes refers to them, and to documents under `docs/`, by name; those references point to the full working
project, which is not published.

## 14. Reproduction

```
python -m venv venv
venv\Scripts\activate            (Windows)   |   source venv/bin/activate   (Linux/macOS)
pip install -r requirements.txt
```

With the original data files in place (see [`data/README.md`](data/README.md)), run from the repository root:

```python
from src.data.load_data import load_train_test, load_locality_table
from src.models.train_model import train_final_model
from src.models.predict_model import predict_asking_price
from src.models.evaluate_model import evaluate_predictions

train, test = load_train_test(".")          # checks the data files' MD5 fingerprints
localities = load_locality_table(".")
model = train_final_model(train, localities)                        # the 16,219 training listings
test_inputs = test.drop(columns=[c for c in test.columns if "price" in c])
predictions = predict_asking_price(model, test_inputs, localities)
print(evaluate_predictions(test["price"], predictions))             # MAE 57.011, Median APE 16.836, ...
```

This reproduces the test metrics of notebook 08 exactly. The seeds are fixed (`random_state=42` for the split, the folds
and the model), and the number of boosting rounds is fixed, so the refit is deterministic. No trained model file is
stored.

## 15. Data availability

**The dataset is not included.** It consists of scraped third-party property adverts, including advertiser names and
account identifiers. The project is released without the underlying listings.

The saved notebook outputs are kept as a record of the analysis. Advertiser names, advertiser IDs and listing IDs in
them were redacted or masked before release. Without the original data, the notebooks and `src/` cannot be run:
`notebooks/01_eda.ipynb` stops with an explanation, and the others stop when they try to read the data. See
[`data/README.md`](data/README.md).
