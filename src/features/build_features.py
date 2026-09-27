"""Fixed-rule feature engineering for the Hyderabad asking-price model.

Every function here uses only a listing's OWN values and rules written down in advance.
Nothing is learned from other listings, so the functions can be applied to the training
set, the test set, or one new advert, and always give the same answer for the same listing.

Learned steps (median imputation, one-hot category lists, rare-locality grouping) are NOT
here. They belong in the scikit-learn preprocessing, fitted on training data (or training
folds) only. See notebook 18, sections 12-13.

The decisions behind each rule are explained in notebooks 17 and 18. The geographic distance
features (build_model_features) come from notebook 22 and are checked in notebook 23.
"""

import numpy as np
import pandas as pd


# Columns the fixed rules are allowed to READ. add_fixed_features() only sees these,
# so a forbidden column (price, identifiers, dates, ...) can never reach the features.
INPUT_COLUMNS = [
    "property_type", "bedrooms", "bathrooms", "balconies", "furnishing", "possession", "age_of_property",
    "facing", "posted_by", "transaction_type", "prime_location",
    "covered_area_sqft_converted", "covered_area_unit", "land_area_unit",
    "floor", "total_floors",
    "locality_id", "recovered_locality_id_candidate", "loc_locality_lat", "loc_locality_long",
    "project_id",   # read ONLY to create the yes/no flag in_named_project; the ID itself is never a feature
]

APARTMENT_LIKE_TYPES = ["Apartment", "Builder Floor Apartment", "Penthouse"]

# Same floor mapping as rule R3 in notebook 14
FLOOR_WORDS = {"Ground": "0", "Upper Basement": "-1", "Lower Basement": "-2"}

# The eight compass directions offered by the website's list (spelling exactly as stored)
DIRECTIONS = ["East", "West", "North", "South", "North - East", "North - West", "South - East", "South -West"]

# The same Hyderabad box as notebook 14 (fixed numbers, written down in advance)
HYDERABAD_LAT_RANGE = (16.8, 18.2)
HYDERABAD_LONG_RANGE = (77.8, 79.2)

# Fixed landmarks for the distance features (notebook 22). Coordinates in degrees, from public maps,
# accurate to about 0.5-1 km. They are constants: never computed from the listings or their prices.
LANDMARKS = {
    "city_centre":        (17.3850, 78.4867),   # standard Hyderabad city coordinate (Abids-Koti area)
    "airport":            (17.2403, 78.4294),   # Rajiv Gandhi International Airport, Shamshabad
    "hitec_city":         (17.4504, 78.3808),   # Cyber Towers, HITEC City
    "gachibowli":         (17.4401, 78.3489),   # Gachibowli
    "financial_district": (17.4170, 78.3400),   # Financial District, Nanakramguda
    "secunderabad":       (17.4337, 78.5016),   # Secunderabad railway station
}
DISTANCE_FEATURES = ["distance_km_" + name for name in LANDMARKS]
EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------- size and area

def rooms_to_number(text_values):
    # Bedrooms/bathrooms are text because the website's last option is "> 10".
    # We store "> 10" as 11, which here means "11 or more" (no listing has exactly 11).
    numbers_as_text = text_values.replace("> 10", "11")
    return pd.to_numeric(numbers_as_text, errors="coerce")


def make_area_features(table):
    # log_area: areas are very right-skewed, the log makes them more even.
    # area_missing: keeps the fact that the area was unknown after imputation fills it.
    area = table["covered_area_sqft_converted"]
    log_area = np.log1p(area)                    # stays missing where the area is missing
    area_missing = area.isna().astype(int)       # 1 = area unknown, 0 = area known
    return log_area, area_missing


def make_area_meaning_flags(table):
    # Hints that the "area" may be a plot area rather than a built-up area (notebook 08)
    area_unit_is_sqyd = (table["covered_area_unit"] == "Sq-yrd").astype(int)
    has_land_area_unit = table["land_area_unit"].notna().astype(int)
    return area_unit_is_sqyd, has_land_area_unit


def make_floor_features(table):
    # Floor and building height only have a clear meaning for apartment-like types
    floor_number = pd.to_numeric(table["floor"].replace(FLOOR_WORDS), errors="coerce")
    building_floors = table["total_floors"].copy()

    # Not applicable for houses and villas -> missing
    is_apartment_like = table["property_type"].isin(APARTMENT_LIKE_TYPES)
    floor_number[~is_apartment_like] = np.nan
    building_floors[~is_apartment_like] = np.nan
    return floor_number, building_floors


# ---------------------------------------------------------------- location

def apply_safe_recovery(table):
    # Safe free-text recovery (level-2 rule, notebook 10; candidates stored by notebook 15).
    # Use the candidate only where the website gave no ID; never overwrite an existing ID.
    locality_id = table["locality_id"].copy()
    use_candidate = locality_id.isna() & table["recovered_locality_id_candidate"].notna()
    locality_id[use_candidate] = table.loc[use_candidate, "recovered_locality_id_candidate"]
    return locality_id


def make_locality_code(locality_id):
    # The locality ID is a category, not a quantity, so it becomes text.
    # Listings without an ID get their own category instead of being dropped or guessed.
    locality_code = pd.Series("unknown locality", index=locality_id.index)
    has_id = locality_id.notna()
    # 84461.0 -> 84461 -> "84461"
    locality_code[has_id] = locality_id[has_id].astype("int64").astype(str)
    return locality_code


def make_centre_lookups(locality_table):
    # Two dictionaries: locality_id -> centre latitude, locality_id -> centre longitude.
    # locality_table is the website's locality table (localities.csv); only ID and centre are used.
    centre_lat_by_id = dict(zip(locality_table["locality_id"], locality_table["locality_lat"]))
    centre_long_by_id = dict(zip(locality_table["locality_id"], locality_table["locality_long"]))
    return centre_lat_by_id, centre_long_by_id


def make_locality_centre(table, locality_id, centre_lat_by_id, centre_long_by_id):
    centre_lat = table["loc_locality_lat"].copy()
    centre_long = table["loc_locality_long"].copy()

    # Recovered listings have an ID but no centre yet: look it up in the website's locality table
    needs_lookup = centre_lat.isna() & locality_id.notna()
    centre_lat[needs_lookup] = locality_id[needs_lookup].map(centre_lat_by_id)
    centre_long[needs_lookup] = locality_id[needs_lookup].map(centre_long_by_id)

    # A centre outside Hyderabad cannot be right for a Hyderabad listing: treat it as unknown
    inside_box = (centre_lat.between(*HYDERABAD_LAT_RANGE) & centre_long.between(*HYDERABAD_LONG_RANGE))
    centre_lat[~inside_box] = np.nan
    centre_long[~inside_box] = np.nan
    return centre_lat, centre_long


# ---------------------------------------------------------------- categorical

def make_balconies_band(balconies):
    # Missing stays "unknown" (not 0): the website never records 0, so we cannot tell
    # "no balcony" from "not stated". Rare large values share the band "4+".
    band = pd.Series("unknown", index=balconies.index)
    band[balconies == "1"] = "1"
    band[balconies == "2"] = "2"
    band[balconies == "3"] = "3"
    four_or_more = ["4", "5", "6", "7", "8", "9", "10", "> 10"]
    band[balconies.isin(four_or_more)] = "4+"
    return band


def make_transaction_group(transaction_type):
    # Every listing is a sale. "Rent" is a known data-entry error (notebook 05), and
    # "Other" and missing are tiny groups, so all three share one group.
    group = pd.Series("other/unknown", index=transaction_type.index)
    group[transaction_type == "Resale"] = "Resale"
    group[transaction_type == "New Property"] = "New Property"
    return group


def possession_to_status(value):
    # One possession value -> "ready", "under construction" or "unknown" (notebook 17, section 7.2)
    if pd.isna(value) or value == "Select Date":
        return "unknown"
    if value in ["Ready to Move", "Immediately"]:
        return "ready"
    if value in ["Under Construction", "Ongoing"]:
        return "under construction"
    # A month such as "Dec '26" contains an apostrophe: an expected possession date in the future
    if "'" in value:
        return "under construction"
    # Any other value is not covered by the documented rule: stop instead of guessing
    raise ValueError(f"Possession value not covered by the rule: {value!r}")


def make_facing_clean(facing):
    # Keep the eight directions; non-directions (landmarks, "Newly Constructed Property") and missing -> "unknown".
    # where() keeps the value if the condition is true, and uses "unknown" otherwise.
    return facing.where(facing.isin(DIRECTIONS), "unknown")


def make_binary_flags(table):
    # prime_location: the website's listing attribute, used as it is (meaning undocumented).
    # in_named_project: 1 if the listing has a project ID (raw_json.psmid). Only the yes/no flag is used.
    # Any prime_location value other than "Y"/"N" would be a new situation: stop instead of guessing.
    assert table["prime_location"].isin(["Y", "N"]).all(), "prime_location has unexpected values"
    prime_location_flag = (table["prime_location"] == "Y").astype(int)
    in_named_project = table["project_id"].notna().astype(int)
    return prime_location_flag, in_named_project


# ---------------------------------------------------------------- all fixed rules together

def add_fixed_features(table, locality_table):
    # Applies every fixed rule above and returns one column per baseline feature.
    # Call it separately on train and test (or on a new advert): it learns nothing.
    table = table[INPUT_COLUMNS]   # the function can only see the allowed input columns
    centre_lat_by_id, centre_long_by_id = make_centre_lookups(locality_table)
    features = pd.DataFrame(index=table.index)

    # numeric
    features["bedrooms"] = rooms_to_number(table["bedrooms"])
    features["bathrooms"] = rooms_to_number(table["bathrooms"])
    features["log_area"], features["area_missing"] = make_area_features(table)
    features["area_unit_is_sqyd"], features["has_land_area_unit"] = make_area_meaning_flags(table)
    features["apartment_floor"], features["apartment_building_floors"] = make_floor_features(table)

    # location
    locality_id = apply_safe_recovery(table)
    features["locality_code"] = make_locality_code(locality_id)
    features["locality_centre_lat"], features["locality_centre_long"] = make_locality_centre(
        table, locality_id, centre_lat_by_id, centre_long_by_id)

    # categorical (missing -> "unknown")
    features["property_type"] = table["property_type"]
    features["balconies_band"] = make_balconies_band(table["balconies"])
    features["transaction_group"] = make_transaction_group(table["transaction_type"])
    features["furnishing"] = table["furnishing"].fillna("unknown")
    features["possession_status"] = table["possession"].apply(possession_to_status)
    features["age_of_property"] = table["age_of_property"].fillna("unknown")
    features["facing_clean"] = make_facing_clean(table["facing"])
    features["posted_by"] = table["posted_by"]

    # binary flags
    features["prime_location_flag"], features["in_named_project"] = make_binary_flags(table)
    return features


# ---------------------------------------------------------------- geographic distances (notebook 22)

def haversine_km(lat_1, long_1, lat_2, long_2):
    # Straight-line ("crow flies") distance in km between two points on the Earth, given in degrees.
    # Works on single numbers and on whole pandas columns; a missing coordinate gives a missing distance.
    lat_1, long_1 = np.radians(lat_1), np.radians(long_1)
    lat_2, long_2 = np.radians(lat_2), np.radians(long_2)
    a = np.sin((lat_2 - lat_1) / 2) ** 2 + np.cos(lat_1) * np.cos(lat_2) * np.sin((long_2 - long_1) / 2) ** 2
    central_angle = 2 * np.arcsin(np.sqrt(a))   # the angle between the two points, seen from the Earth's centre
    return EARTH_RADIUS_KM * central_angle


def make_distance_features(centre_lat, centre_long):
    # One column per landmark: distance from the LOCALITY CENTRE (never the listing's own pin), in km.
    # Listings without a usable centre get NaN; the learned imputer fills it inside each fold.
    distances = pd.DataFrame(index=centre_lat.index)
    for name, (landmark_lat, landmark_long) in LANDMARKS.items():
        distances["distance_km_" + name] = haversine_km(centre_lat, centre_long, landmark_lat, landmark_long)
    return distances


def build_model_features(table, locality_table):
    # The model feature set from notebook 23 on (decision of notebook 22):
    # the 21 baseline features, with the locality-centre latitude/longitude REPLACED by the six distances.
    # The centre coordinates are still built by add_fixed_features(), because the distances need them,
    # but they are dropped from the result. add_fixed_features() itself is unchanged, so notebooks 18-22
    # still reproduce their results.
    features = add_fixed_features(table, locality_table)
    distances = make_distance_features(features["locality_centre_lat"], features["locality_centre_long"])
    features = pd.concat([features, distances], axis=1)
    return features.drop(columns=["locality_centre_lat", "locality_centre_long"])
