"""Tests for AL drinking-status sensitivity helper derivations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.sensitivity.al_drinking_utils import (
    add_refined_drinking_variables,
    alcohol_frequency_days_per_week,
    build_drinking_transition_table,
    clean_quantity,
    raw_current_drink_status,
    weekly_volume,
)


def test_alcohol_frequency_and_quantity_cleaning():
    freq = pd.Series([1, 2, 3, 4, 5, 5, 5, 5, 6, np.nan])
    monthly = pd.Series([np.nan, np.nan, np.nan, np.nan, 1, 2, 3, 4, np.nan, np.nan])
    days = alcohol_frequency_days_per_week(freq, monthly)

    assert days.iloc[0] == 7.0
    assert days.iloc[1] == 5.5
    assert days.iloc[2] == 3.5
    assert days.iloc[3] == 1.5
    assert np.isclose(days.iloc[4], 3.5 / 4.345)
    assert np.isclose(days.iloc[5], 1.5 / 4.345)
    assert np.isclose(days.iloc[6], 0.5 / 4.345)
    assert days.iloc[7] == 0.0
    assert days.iloc[8] == 0.0
    assert pd.isna(days.iloc[9])

    cleaned = clean_quantity(pd.Series([0, 1, 12, 97, 98, 99, -1, np.nan]))
    assert cleaned.iloc[:3].tolist() == [0, 1, 12]
    assert cleaned.iloc[3:].isna().all()


def test_refined_drinking_categories_cover_abstainer_and_current_subtypes():
    raw = pd.DataFrame(
        {
            "B1PA50": [2, 2, 2, 2, 1, 1, 1, np.nan, 2],
            "B1PA51": [np.nan, np.nan, np.nan, np.nan, 5, 4, 1, np.nan, np.nan],
            "B1PA51A": [np.nan, np.nan, np.nan, np.nan, 3, np.nan, np.nan, np.nan, np.nan],
            "B1PA52": [np.nan, np.nan, np.nan, np.nan, 1, 2, 2, np.nan, np.nan],
            "B1PA53": [np.nan, np.nan, np.nan, np.nan, 0, 0, 3, np.nan, np.nan],
            "B1PA54": [6, 5, 4, 2, 5, 4, 1, np.nan, 4],
            "B1PA54A": [np.nan, 3, np.nan, np.nan, 3, np.nan, np.nan, np.nan, np.nan],
            "B1PA55": [0, 1, 2, 2, 1, 2, 2, np.nan, 99],
        }
    )
    current_status = raw_current_drink_status(raw, "B1")
    current_weekly = weekly_volume(raw["B1PA51"], raw["B1PA51A"], raw["B1PA52"])
    max_weekly = weekly_volume(raw["B1PA54"], raw["B1PA54A"], raw["B1PA55"])

    master = pd.DataFrame(
        {
            "M2MRID": range(1, 10),
            "drinkStatus": [
                "Past-month abstainer",
                "Past-month abstainer",
                "Past-month abstainer",
                "Past-month abstainer",
                "Infrequent-non-binge",
                "Frequent-non-binge",
                "Frequent-non-binge",
                np.nan,
                "Past-month abstainer",
            ],
            "female": [0, 0, 0, 1, 0, 0, 1, 0, 0],
            "raw_current_status": current_status,
            "current_weekly_drinks": current_weekly,
            "max_weekly_drinks": max_weekly,
        }
    )

    refined = add_refined_drinking_variables(master, raw=None)
    got_refined = refined["drinkStatus_refined"].astype("object").tolist()
    got_history = refined["drinkHistory6"].astype("object").tolist()

    assert got_refined[:7] == [
        "abstainer_never_or_former_light",
        "abstainer_never_or_former_light",
        "former_moderate",
        "former_heavy",
        "Infrequent-non-binge",
        "Frequent-non-binge",
        "Frequent-non-binge",
    ]
    assert got_history[:7] == [
        "lifelong_or_former_light",
        "lifelong_or_former_light",
        "former_moderate",
        "former_heavy",
        "current_light",
        "current_moderate",
        "current_heavy",
    ]
    assert pd.isna(got_refined[7])
    assert pd.isna(got_history[7])
    assert pd.isna(got_refined[8])
    assert pd.isna(got_history[8])


def test_w1_w2_drinking_transition_construction():
    w1 = pd.DataFrame(
        {
            "M2MRID": [1, 2, 3, 4, 5, 6],
            "drinkStatus": [
                "Infrequent-non-binge",
                "Frequent-non-binge",
                "Past-month abstainer",
                "Past-month abstainer",
                "Binge",
                np.nan,
            ],
            "age_W1": [45, 46, 47, 48, 49, 50],
            "female": [0, 1, 0, 1, 0, 1],
            "education": [12, 14, 16, 12, 18, 10],
            "cohort": ["MIDUS2"] * 6,
        }
    )
    w2 = pd.DataFrame(
        {
            "M2MRID": [1, 2, 3, 4, 5],
            "drinkStatus": [
                "Past-month abstainer",
                "Binge",
                "Past-month abstainer",
                "Infrequent-non-binge",
                np.nan,
            ],
        }
    )

    transitions = build_drinking_transition_table(w1, w2).set_index("M2MRID")

    assert transitions.loc[1, "education"] == 12
    assert transitions.loc[1, "drink_transition_W1_W2"] == "became_past_month_abstainer"
    assert transitions.loc[1, "became_abstainer_W2"] == 1.0
    assert transitions.loc[2, "drink_transition_W1_W2"] == "remained_current_drinker"
    assert transitions.loc[2, "became_abstainer_W2"] == 0.0
    assert transitions.loc[3, "drink_transition_W1_W2"] == "remained_abstainer"
    assert pd.isna(transitions.loc[3, "became_abstainer_W2"])
    assert transitions.loc[4, "drink_transition_W1_W2"] == "initiated_current_drinking"
    assert pd.isna(transitions.loc[5, "drink_transition_W1_W2"])
    assert pd.isna(transitions.loc[6, "drink_transition_W1_W2"])
