import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd 
from pprint import pprint 

def calculate_average(row, columns):
    subset_values = row[columns]
    return subset_values.mean()

def create_model_combo(row):
    model_dict = {
        "llama_2_7b": "Llama 2 7B",
        "llama_2": "Llama 2 7B",
        "tulu_none": "Tulu None",
        "tulu_match": "Tulu Match",

        "coding_20": "Coding 20%",
        "coding_40": "Coding 40%",
        "coding_60": "Coding 60%",
        "coding_80": "Coding 80%",
        "coding_100": "Coding 100%",

        "safety_20": "Safety 20%",
        "safety_40": "Safety 40%",
        "safety_60": "Safety 60%",
        "safety_80": "Safety 80%",
        "safety_100": "Safety 100%",

        "science_100": "Science 100",
        "science_200": "Science 200",
        "science_500": "Science 500",
        "science_1000": "Science 1000",
        "science_2500": "Science 2500",

        "tulu_all": "Tulu",
        "tulu_2_7b": "Tulu 2 7B",

        "tulu_all_with_coding": "Tulu w/ Coding",
        "tulu_2_7b_with_coding": "Tulu 2 7B w/ Coding",
    }

    tokens = row["model_key"].split("-")
    if str(row["merge_method"]) != "nan":
        tokens = tokens[1:]
        tokens[1] = tokens[1]
        base_model_weight = float(tokens[1].split("_")[-1])
        tokens[1] = tokens[1].replace(f"_{base_model_weight}", "")
        tokens[2] = "-".join(tokens[2:])
        tokens = tokens[:3]
        domain_model_weight = float(tokens[2].split("_")[-1])
        tokens[2] = tokens[2].replace(f"_{domain_model_weight}", "")
    base_model = model_dict[tokens[0]]
    tulu_model = model_dict[tokens[1]]
    if len(tokens) == 2:
        domain_model = "None"
    else:
        domain_model = ""
        for token in tokens[2].split("-"):
            if token not in model_dict:
                return ""
            domain_model += " " + model_dict[token]

    if str(row["merge_method"]) == "nan":
        return f"{base_model} ft. on {tulu_model} & {domain_model.strip()}"
    else:
        return f"{domain_model.replace(' 7B Tulu None', '').strip()} - {row['merge_method']}"

def create_row_label(row):
    baseline_keys = {
        "llama_2_7b-tulu_all": "Tulu 2 7B",

        # science baselines:
        "llama_2_7b-tulu_none-science_100": "Llama 2 Task Vector",
        "llama_2_7b-tulu_none-science_200": "Llama 2 Task Vector",
        "llama_2_7b-tulu_none-science_500": "Llama 2 Task Vector",
        "llama_2_7b-tulu_none-science_1000": "Llama 2 Task Vector",
        "llama_2_7b-tulu_none-science_2500": "Llama 2 Task Vector",
        "llama_2_7b-tulu_all-science_100": "Data Merging",
        "llama_2_7b-tulu_all-science_200": "Data Merging",
        "llama_2_7b-tulu_all-science_500": "Data Merging",
        "llama_2_7b-tulu_all-science_1000": "Data Merging",
        "llama_2_7b-tulu_all-science_2500": "Data Merging",
        "tulu_2_7b-tulu_none-science_100": "Tulu 2 Task Vector",
        "tulu_2_7b-tulu_none-science_200": "Tulu 2 Task Vector",
        "tulu_2_7b-tulu_none-science_500": "Tulu 2 Task Vector",
        "tulu_2_7b-tulu_none-science_1000": "Tulu 2 Task Vector",
        "tulu_2_7b-tulu_none-science_2500": "Tulu 2 Task Vector",

        # safety baselines:
        "llama_2_7b-tulu_none-safety_20": "Llama 2 Task Vector",
        "llama_2_7b-tulu_none-safety_40": "Llama 2 Task Vector",
        "llama_2_7b-tulu_none-safety_60": "Llama 2 Task Vector",
        "llama_2_7b-tulu_none-safety_80": "Llama 2 Task Vector",
        "llama_2_7b-tulu_none-safety_100": "Llama 2 Task Vector",
        "llama_2_7b-tulu_all-safety_20": "Data Merging",
        "llama_2_7b-tulu_all-safety_40": "Data Merging",
        "llama_2_7b-tulu_all-safety_60": "Data Merging",
        "llama_2_7b-tulu_all-safety_80": "Data Merging",
        "llama_2_7b-tulu_all-safety_100": "Data Merging",
        "tulu_2_7b-tulu_none-safety_20": "Tulu 2 Task Vector",
        "tulu_2_7b-tulu_none-safety_40": "Tulu 2 Task Vector",
        "tulu_2_7b-tulu_none-safety_60": "Tulu 2 Task Vector",
        "tulu_2_7b-tulu_none-safety_80": "Tulu 2 Task Vector",
        "tulu_2_7b-tulu_none-safety_100": "Tulu 2 Task Vector",

        # coding baselines:
        "llama_2_7b-tulu_none-coding_20": "Llama 2 Task Vector",
        "llama_2_7b-tulu_none-coding_40": "Llama 2 Task Vector",
        "llama_2_7b-tulu_none-coding_60": "Llama 2 Task Vector",
        "llama_2_7b-tulu_none-coding_80": "Llama 2 Task Vector",
        "llama_2_7b-tulu_none-coding_100": "Llama 2 Task Vector",
        "llama_2_7b-tulu_all-coding_20": "Data Merging",
        "llama_2_7b-tulu_all-coding_40": "Data Merging",
        "llama_2_7b-tulu_all-coding_60": "Data Merging",
        "llama_2_7b-tulu_all-coding_80": "Data Merging",
        "llama_2_7b-tulu_all-coding_100": "Data Merging",
        "tulu_2_7b-tulu_none-coding_20": "Tulu 2 Task Vector",
        "tulu_2_7b-tulu_none-coding_40": "Tulu 2 Task Vector",
        "tulu_2_7b-tulu_none-coding_60": "Tulu 2 Task Vector",
        "tulu_2_7b-tulu_none-coding_80": "Tulu 2 Task Vector",
        "tulu_2_7b-tulu_none-coding_100": "Tulu 2 Task Vector",
    }
    
    if row["model_key"] in baseline_keys:
        return baseline_keys[row["model_key"]]

    combo_map = {
        # Science
        "Llama 2 Science 100 - task_arithmetic": "Llama 2 Task Arithmetic",
        "Llama 2 Science 200 - task_arithmetic": "Llama 2 Task Arithmetic",
        "Llama 2 Science 500 - task_arithmetic": "Llama 2 Task Arithmetic",
        "Llama 2 Science 1000 - task_arithmetic": "Llama 2 Task Arithmetic",
        "Llama 2 Science 2500 - task_arithmetic": "Llama 2 Task Arithmetic",

        "Tulu 2 Science 100 - wise-ft": "Tulu 2 Task Arithmetic",
        "Tulu 2 Science 200 - wise-ft": "Tulu 2 Task Arithmetic",
        "Tulu 2 Science 500 - wise-ft": "Tulu 2 Task Arithmetic",
        "Tulu 2 Science 1000 - wise-ft": "Tulu 2 Task Arithmetic",
        "Tulu 2 Science 2500 - wise-ft": "Tulu 2 Task Arithmetic",

        "Llama 2 Science 100 - linear_weighted": "Llama 2 Linear Interpolation",
        "Llama 2 Science 200 - linear_weighted": "Llama 2 Linear Interpolation",
        "Llama 2 Science 500 - linear_weighted": "Llama 2 Linear Interpolation",
        "Llama 2 Science 1000 - linear_weighted": "Llama 2 Linear Interpolation",
        "Llama 2 Science 2500 - linear_weighted": "Llama 2 Linear Interpolation",

        # Safety
        "Llama 2 Safety 20% - task_arithmetic": "Llama 2 Task Arithmetic",
        "Llama 2 Safety 40% - task_arithmetic": "Llama 2 Task Arithmetic",
        "Llama 2 Safety 60% - task_arithmetic": "Llama 2 Task Arithmetic",
        "Llama 2 Safety 80% - task_arithmetic": "Llama 2 Task Arithmetic",
        "Llama 2 Safety 100% - task_arithmetic": "Llama 2 Task Arithmetic",

        "Tulu 2 Safety 20% - wise-ft": "Tulu 2 Task Arithmetic",
        "Tulu 2 Safety 40% - wise-ft": "Tulu 2 Task Arithmetic",
        "Tulu 2 Safety 60% - wise-ft": "Tulu 2 Task Arithmetic",
        "Tulu 2 Safety 80% - wise-ft": "Tulu 2 Task Arithmetic",
        "Tulu 2 Safety 100% - wise-ft": "Tulu 2 Task Arithmetic",

        "Llama 2 Safety 20% - linear_weighted": "Llama 2 Linear Interpolation",
        "Llama 2 Safety 40% - linear_weighted": "Llama 2 Linear Interpolation",
        "Llama 2 Safety 60% - linear_weighted": "Llama 2 Linear Interpolation",
        "Llama 2 Safety 80% - linear_weighted": "Llama 2 Linear Interpolation",
        "Llama 2 Safety 100% - linear_weighted": "Llama 2 Linear Interpolation",

        # Coding
        "Llama 2 Coding 20% - task_arithmetic": "Llama 2 Task Arithmetic",
        "Llama 2 Coding 40% - task_arithmetic": "Llama 2 Task Arithmetic",
        "Llama 2 Coding 60% - task_arithmetic": "Llama 2 Task Arithmetic",
        "Llama 2 Coding 80% - task_arithmetic": "Llama 2 Task Arithmetic",
        "Llama 2 Coding 100% - task_arithmetic": "Llama 2 Task Arithmetic",

        "Tulu 2 Coding 20% - wise-ft": "Tulu 2 Task Arithmetic",
        "Tulu 2 Coding 40% - wise-ft": "Tulu 2 Task Arithmetic",
        "Tulu 2 Coding 60% - wise-ft": "Tulu 2 Task Arithmetic",
        "Tulu 2 Coding 80% - wise-ft": "Tulu 2 Task Arithmetic",
        "Tulu 2 Coding 100% - wise-ft": "Tulu 2 Task Arithmetic",

        "Llama 2 Coding 20% - linear_weighted": "Llama 2 Linear Interpolation",
        "Llama 2 Coding 40% - linear_weighted": "Llama 2 Linear Interpolation",
        "Llama 2 Coding 60% - linear_weighted": "Llama 2 Linear Interpolation",
        "Llama 2 Coding 80% - linear_weighted": "Llama 2 Linear Interpolation",
        "Llama 2 Coding 100% - linear_weighted": "Llama 2 Linear Interpolation",

        # Others (not used, but necessary)
        "Llama 2 7B ft. on Tulu None & Science 2500": "Llama 2 7B ft. on Tulu None & Science 100",
        "Llama 2 7B ft. on Tulu None & Science 2500": "Llama 2 7B ft. on Tulu None & Science 200",
        "Llama 2 7B ft. on Tulu None & Science 2500": "Llama 2 7B ft. on Tulu None & Science 500",
        "Llama 2 7B ft. on Tulu None & Science 2500": "Llama 2 7B ft. on Tulu None & Science 1000",
        "Llama 2 7B ft. on Tulu None & Science 2500": "Llama 2 7B ft. on Tulu None & Science 2500",
    }
    if row["Combo"] not in combo_map:
        return row["Combo"]
    return combo_map[row["Combo"]]

def get_df():
    df = pd.read_csv("results/consistent_mix/results.csv")

    tulu_columns_for_test_average = [
        "mmlu_0shot",
        "gsm_cot",
        "bbh_cot",
        "truthfulqa",
        "alpaca_eval",
    ]

    coding_columns_for_average = [
        "codex_eval_plus_temp_0.8",
        "mbpp_temp_0.8",
    ]

    safety_columns_for_average = [
        "invert_toxigen",
        "normalized_harmbench",
        "invert_unsafe_average",
    ]

    df['invert_toxigen'] = df.apply(lambda row: 1 - row["toxigen"], axis=1)
    df['invert_unsafe_average'] = df.apply(lambda row: 1 - (row["unsafe_average"] / 100), axis=1)
    df["normalized_safe_average"] = df.apply(lambda row: row["safe_average"] / 100, axis=1)
    df["normalized_harmbench"] = df.apply(lambda row: 1 - (row["harmbench"] / 100), axis=1)
    df['invert_toxigen'] = df.apply(lambda row: 1 - row["toxigen"], axis=1)
    df['alpaca_eval'] = df.apply(lambda row: row["alpaca_eval"] / 100, axis=1)
    df['tydiqa_no_context_1shot'] = df.apply(lambda row: row["tydiqa_no_context_1shot"] / 100, axis=1)
    df['tydiqa_goldp_1shot'] = df.apply(lambda row: row["tydiqa_goldp_1shot"] / 100, axis=1)

    # calculate averages
    df['Exaggerated Refusals'] = df.apply(lambda row: row["normalized_safe_average"], axis=1)
    df['Tulu Average'] = df.apply(lambda row: calculate_average(row, tulu_columns_for_test_average), axis=1)
    df['Safety Average'] = df.apply(lambda row: calculate_average(row, safety_columns_for_average), axis=1)
    df["Science Average"] = df.apply(lambda row: row["mean_null"], axis=1)
    df['Coding Average'] = df.apply(lambda row: calculate_average(row, coding_columns_for_average), axis=1)

    df['Combo'] = df.apply(lambda row: create_model_combo(row), axis=1)
    df["Label"] = df.apply(lambda row: create_row_label(row), axis=1)

    return df

def make_plots():
    df = get_df()
    df["Order"] = df["domain_model_weight"]
    df.sort_values(by='Combo', inplace=True)
    df.sort_values(by='Order', inplace=True)

    dataframes = {
        # baselines
        "llama_tulu_all": df[df["model_key"] == "llama_2_7b-tulu_all"],

        # science baselines
        "llama_tulu_none_science_100": df[df["model_key"] == "llama_2_7b-tulu_none-science_100"],
        "llama_tulu_none_science_200": df[df["model_key"] == "llama_2_7b-tulu_none-science_200"],
        "llama_tulu_none_science_500": df[df["model_key"] == "llama_2_7b-tulu_none-science_500"],
        "llama_tulu_none_science_1000": df[df["model_key"] == "llama_2_7b-tulu_none-science_1000"],
        "llama_tulu_none_science_2500": df[df["model_key"] == "llama_2_7b-tulu_none-science_2500"],

        "llama_tulu_all_science_100": df[df["model_key"] == "llama_2_7b-tulu_all-science_100"],
        "llama_tulu_all_science_200": df[df["model_key"] == "llama_2_7b-tulu_all-science_200"],
        "llama_tulu_all_science_500": df[df["model_key"] == "llama_2_7b-tulu_all-science_500"],
        "llama_tulu_all_science_1000": df[df["model_key"] == "llama_2_7b-tulu_all-science_1000"],
        "llama_tulu_all_science_2500": df[df["model_key"] == "llama_2_7b-tulu_all-science_2500"],

        "tulu_2_tulu_none_science_100": df[df["model_key"] == "tulu_2_7b-tulu_none-science_100"],
        "tulu_2_tulu_none_science_200": df[df["model_key"] == "tulu_2_7b-tulu_none-science_200"],
        "tulu_2_tulu_none_science_500": df[df["model_key"] == "tulu_2_7b-tulu_none-science_500"],
        "tulu_2_tulu_none_science_1000": df[df["model_key"] == "tulu_2_7b-tulu_none-science_1000"],
        "tulu_2_tulu_none_science_2500": df[df["model_key"] == "tulu_2_7b-tulu_none-science_2500"],

        "tulu_2_tulu_match_science_100": df[df["model_key"] == "tulu_2_7b-tulu_match-science_100"],
        "tulu_2_tulu_match_science_200": df[df["model_key"] == "tulu_2_7b-tulu_match-science_200"],
        "tulu_2_tulu_match_science_500": df[df["model_key"] == "tulu_2_7b-tulu_match-science_500"],
        "tulu_2_tulu_match_science_1000": df[df["model_key"] == "tulu_2_7b-tulu_match-science_1000"],
        "tulu_2_tulu_match_science_2500": df[df["model_key"] == "tulu_2_7b-tulu_match-science_2500"],

        # safety baselines
        "llama_tulu_none_safety_20": df[df["model_key"] == "llama_2_7b-tulu_none-safety_20"],
        "llama_tulu_none_safety_40": df[df["model_key"] == "llama_2_7b-tulu_none-safety_40"],
        "llama_tulu_none_safety_60": df[df["model_key"] == "llama_2_7b-tulu_none-safety_60"],
        "llama_tulu_none_safety_80": df[df["model_key"] == "llama_2_7b-tulu_none-safety_80"],
        "llama_tulu_none_safety_100": df[df["model_key"] == "llama_2_7b-tulu_none-safety_100"],

        "llama_tulu_all_safety_20": df[df["model_key"] == "llama_2_7b-tulu_all-safety_20"],
        "llama_tulu_all_safety_40": df[df["model_key"] == "llama_2_7b-tulu_all-safety_40"],
        "llama_tulu_all_safety_60": df[df["model_key"] == "llama_2_7b-tulu_all-safety_60"],
        "llama_tulu_all_safety_80": df[df["model_key"] == "llama_2_7b-tulu_all-safety_80"],
        "llama_tulu_all_safety_100": df[df["model_key"] == "llama_2_7b-tulu_all-safety_100"],

        "tulu_2_tulu_none_safety_20": df[df["model_key"] == "tulu_2_7b-tulu_none-safety_20"],
        "tulu_2_tulu_none_safety_40": df[df["model_key"] == "tulu_2_7b-tulu_none-safety_40"],
        "tulu_2_tulu_none_safety_60": df[df["model_key"] == "tulu_2_7b-tulu_none-safety_60"],
        "tulu_2_tulu_none_safety_80": df[df["model_key"] == "tulu_2_7b-tulu_none-safety_80"],
        "tulu_2_tulu_none_safety_100": df[df["model_key"] == "tulu_2_7b-tulu_none-safety_100"],

        "tulu_2_tulu_match_safety_20": df[df["model_key"] == "tulu_2_7b-tulu_match-safety_20"],
        "tulu_2_tulu_match_safety_40": df[df["model_key"] == "tulu_2_7b-tulu_match-safety_40"],
        "tulu_2_tulu_match_safety_60": df[df["model_key"] == "tulu_2_7b-tulu_match-safety_60"],
        "tulu_2_tulu_match_safety_80": df[df["model_key"] == "tulu_2_7b-tulu_match-safety_80"],
        "tulu_2_tulu_match_safety_100": df[df["model_key"] == "tulu_2_7b-tulu_match-safety_100"],

        # coding baselines
        "llama_tulu_none_coding_20": df[df["model_key"] == "llama_2_7b-tulu_none-coding_20"],
        "llama_tulu_none_coding_40": df[df["model_key"] == "llama_2_7b-tulu_none-coding_40"],
        "llama_tulu_none_coding_60": df[df["model_key"] == "llama_2_7b-tulu_none-coding_60"],
        "llama_tulu_none_coding_80": df[df["model_key"] == "llama_2_7b-tulu_none-coding_80"],
        "llama_tulu_none_coding_100": df[df["model_key"] == "llama_2_7b-tulu_none-coding_100"],

        "llama_tulu_all_coding_20": df[df["model_key"] == "llama_2_7b-tulu_all-coding_20"],
        "llama_tulu_all_coding_40": df[df["model_key"] == "llama_2_7b-tulu_all-coding_40"],
        "llama_tulu_all_coding_60": df[df["model_key"] == "llama_2_7b-tulu_all-coding_60"],
        "llama_tulu_all_coding_80": df[df["model_key"] == "llama_2_7b-tulu_all-coding_80"],
        "llama_tulu_all_coding_100": df[df["model_key"] == "llama_2_7b-tulu_all-coding_100"],

        "tulu_2_tulu_none_coding_20": df[df["model_key"] == "tulu_2_7b-tulu_none-coding_20"],
        "tulu_2_tulu_none_coding_40": df[df["model_key"] == "tulu_2_7b-tulu_none-coding_40"],
        "tulu_2_tulu_none_coding_60": df[df["model_key"] == "tulu_2_7b-tulu_none-coding_60"],
        "tulu_2_tulu_none_coding_80": df[df["model_key"] == "tulu_2_7b-tulu_none-coding_80"],
        "tulu_2_tulu_none_coding_100": df[df["model_key"] == "tulu_2_7b-tulu_none-coding_100"],

        "tulu_2_tulu_match_coding_20": df[df["model_key"] == "tulu_2_7b-tulu_match-coding_20"],
        "tulu_2_tulu_match_coding_40": df[df["model_key"] == "tulu_2_7b-tulu_match-coding_40"],
        "tulu_2_tulu_match_coding_60": df[df["model_key"] == "tulu_2_7b-tulu_match-coding_60"],
        "tulu_2_tulu_match_coding_80": df[df["model_key"] == "tulu_2_7b-tulu_match-coding_80"],
        "tulu_2_tulu_match_coding_100": df[df["model_key"] == "tulu_2_7b-tulu_match-coding_100"],

        # Science
        "science_100_llama_ta": df[df["Combo"] == "Llama 2 Science 100 - task_arithmetic"],
        "science_200_llama_ta": df[df["Combo"] == "Llama 2 Science 200 - task_arithmetic"],
        "science_500_llama_ta": df[df["Combo"] == "Llama 2 Science 500 - task_arithmetic"],
        "science_1000_llama_ta": df[df["Combo"] == "Llama 2 Science 1000 - task_arithmetic"],
        "science_2500_llama_ta": df[df["Combo"] == "Llama 2 Science 2500 - task_arithmetic"],

        "science_100_tulu_ta": df[df["Combo"] == "Tulu 2 Science 100 - wise-ft"],
        "science_200_tulu_ta": df[df["Combo"] == "Tulu 2 Science 200 - wise-ft"],
        "science_500_tulu_ta": df[df["Combo"] == "Tulu 2 Science 500 - wise-ft"],
        "science_1000_tulu_ta": df[df["Combo"] == "Tulu 2 Science 1000 - wise-ft"],
        "science_2500_tulu_ta": df[df["Combo"] == "Tulu 2 Science 2500 - wise-ft"],

        "science_100_interp": df[df["Combo"] == "Llama 2 Science 100 - linear_weighted"],
        "science_200_interp": df[df["Combo"] == "Llama 2 Science 200 - linear_weighted"],
        "science_500_interp": df[df["Combo"] == "Llama 2 Science 500 - linear_weighted"],
        "science_1000_interp": df[df["Combo"] == "Llama 2 Science 1000 - linear_weighted"],
        "science_2500_interp": df[df["Combo"] == "Llama 2 Science 2500 - linear_weighted"],

        "science_1000_tulu_match_ta": df[df["Combo"] == "Tulu 2 7B Tulu Match Science 1000 - wise-ft"],
        "science_2500_tulu_match_ta": df[df["Combo"] == "Tulu 2 7B Tulu Match Science 2500 - wise-ft"],

        # Safety
        "safety_20_llama_ta": df[df["Combo"] == "Llama 2 Safety 20% - task_arithmetic"],
        "safety_40_llama_ta": df[df["Combo"] == "Llama 2 Safety 40% - task_arithmetic"],
        "safety_60_llama_ta": df[df["Combo"] == "Llama 2 Safety 60% - task_arithmetic"],
        "safety_80_llama_ta": df[df["Combo"] == "Llama 2 Safety 80% - task_arithmetic"],
        "safety_100_llama_ta": df[df["Combo"] == "Llama 2 Safety 100% - task_arithmetic"],

        "safety_20_tulu_ta": df[df["Combo"] == "Tulu 2 Safety 20% - wise-ft"],
        "safety_40_tulu_ta": df[df["Combo"] == "Tulu 2 Safety 40% - wise-ft"],
        "safety_60_tulu_ta": df[df["Combo"] == "Tulu 2 Safety 60% - wise-ft"],
        "safety_80_tulu_ta": df[df["Combo"] == "Tulu 2 Safety 80% - wise-ft"],
        "safety_100_tulu_ta": df[df["Combo"] == "Tulu 2 Safety 100% - wise-ft"],

        "safety_20_interp": df[df["Combo"] == "Llama 2 Safety 20% - linear_weighted"],
        "safety_40_interp": df[df["Combo"] == "Llama 2 Safety 40% - linear_weighted"],
        "safety_60_interp": df[df["Combo"] == "Llama 2 Safety 60% - linear_weighted"],
        "safety_80_interp": df[df["Combo"] == "Llama 2 Safety 80% - linear_weighted"],
        "safety_100_interp": df[df["Combo"] == "Llama 2 Safety 100% - linear_weighted"],

        # Coding
        "coding_20_llama_ta": df[(df["Combo"] == "Llama 2 Coding 20% - task_arithmetic") & (~df["tulu_model"].str.contains("coding"))],
        "coding_40_llama_ta": df[(df["Combo"] == "Llama 2 Coding 40% - task_arithmetic") & (~df["tulu_model"].str.contains("coding"))],
        "coding_60_llama_ta": df[(df["Combo"] == "Llama 2 Coding 60% - task_arithmetic") & (~df["tulu_model"].str.contains("coding"))],
        "coding_80_llama_ta": df[(df["Combo"] == "Llama 2 Coding 80% - task_arithmetic") & (~df["tulu_model"].str.contains("coding"))],
        "coding_100_llama_ta": df[(df["Combo"] == "Llama 2 Coding 100% - task_arithmetic") & (~df["tulu_model"].str.contains("coding"))],

        "coding_20_tulu_ta": df[(df["Combo"] == "Tulu 2 Coding 20% - wise-ft") & (~df["tulu_model"].str.contains("coding"))],
        "coding_40_tulu_ta": df[(df["Combo"] == "Tulu 2 Coding 40% - wise-ft") & (~df["tulu_model"].str.contains("coding"))],
        "coding_60_tulu_ta": df[(df["Combo"] == "Tulu 2 Coding 60% - wise-ft") & (~df["tulu_model"].str.contains("coding"))],
        "coding_80_tulu_ta": df[(df["Combo"] == "Tulu 2 Coding 80% - wise-ft") & (~df["tulu_model"].str.contains("coding"))],
        "coding_100_tulu_ta": df[(df["Combo"] == "Tulu 2 Coding 100% - wise-ft") & (~df["tulu_model"].str.contains("coding"))],

        "coding_20_interp": df[(df["Combo"] == "Llama 2 Coding 20% - linear_weighted") & (~df["tulu_model"].str.contains("coding"))],
        "coding_40_interp": df[(df["Combo"] == "Llama 2 Coding 40% - linear_weighted") & (~df["tulu_model"].str.contains("coding"))],
        "coding_60_interp": df[(df["Combo"] == "Llama 2 Coding 60% - linear_weighted") & (~df["tulu_model"].str.contains("coding"))],
        "coding_80_interp": df[(df["Combo"] == "Llama 2 Coding 80% - linear_weighted") & (~df["tulu_model"].str.contains("coding"))],
        "coding_100_interp": df[(df["Combo"] == "Llama 2 Coding 100% - linear_weighted") & (~df["tulu_model"].str.contains("coding"))],
    }

    line_width = 3
    markersize = 20
    marker = "*"
    point_size = 300
    def compare_science_curves(amt = "2500"):
        sns.lineplot(
            data=dataframes[f"science_{amt}_interp"],
            x="Tulu Average",
            y="Science Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[0]],
        )
        sns.lineplot(
            data=dataframes[f"science_{amt}_llama_ta"],
            x="Tulu Average",
            y="Science Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[1]],
        )
        sns.lineplot(
            data=dataframes[f"science_{amt}_tulu_ta"],
            x="Tulu Average",
            y="Science Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[2]],
        )
        sns.scatterplot(
            data=dataframes[f"llama_tulu_none_science_{amt}"],
            x="Tulu Average",
            y="Science Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[0]],
        )
        sns.scatterplot(
            data=dataframes[f"tulu_2_tulu_none_science_{amt}"],
            x="Tulu Average",
            y="Science Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[2]],
        )
        sns.scatterplot(
            data=dataframes["llama_tulu_all"],
            x="Tulu Average",
            y="Science Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[3]],
        )
        sns.scatterplot(
            data=dataframes[f"llama_tulu_all_science_{amt}"],
            x="Tulu Average",
            y="Science Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[4]],
        )
        # sns.scatterplot(
        #     data=dataframes[f"tulu_2_tulu_match_science_{amt}"],
        #     x="Tulu Average",
        #     y="Science Average",
        #     hue="Label",
        #     s=point_size,
        #     marker=marker,
        #     palette=[sns.color_palette("colorblind")[6]],
        # )
        plt.legend()
        plt.xlabel("Tulu Average",fontsize=20)
        plt.ylabel("Science Average",fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=11)

        plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        # plt.show()
        plt.savefig(f'results/consistent_mix/plots/science_{amt}.png', dpi=300, bbox_inches='tight')
        plt.clf()

    def compare_safety_curves(amt = "100", x = "Tulu Average", y = "Safety Average"):
        for x, y, label in [
            ("Tulu Average", "Safety Average", "tulu_vs_safety"),
            ("Tulu Average", "Exaggerated Refusals", "tulu_vs_exaggerated_refusals"),
            ("Exaggerated Refusals", "Safety Average", "exaggerated_refusals_vs_safety"),
        ]:
            sns.lineplot(
                data=dataframes[f"safety_{amt}_interp"],
                x=x,
                y=y,
                hue="Label",
                sort=False,
                # marker='X',
                linewidth=line_width,
                markersize=markersize,
                palette=[sns.color_palette("colorblind")[0]],
            )
            sns.lineplot(
                data=dataframes[f"safety_{amt}_llama_ta"],
                x=x,
                y=y,
                hue="Label",
                sort=False,
                # marker='X',
                linewidth=line_width,
                markersize=markersize,
                palette=[sns.color_palette("colorblind")[1]],
            )
            sns.lineplot(
                data=dataframes[f"safety_{amt}_tulu_ta"],
                x=x,
                y=y,
                hue="Label",
                sort=False,
                # marker='X',
                linewidth=line_width,
                markersize=markersize,
                palette=[sns.color_palette("colorblind")[2]],
            )
            sns.scatterplot(
                data=dataframes[f"llama_tulu_none_safety_{amt}"],
                x=x,
                y=y,
                hue="Label",
                s=point_size,
                marker=marker,
                palette=[sns.color_palette("colorblind")[0]],
            )
            sns.scatterplot(
                data=dataframes[f"tulu_2_tulu_none_safety_{amt}"],
                x=x,
                y=y,
                hue="Label",
                s=point_size,
                marker=marker,
                palette=[sns.color_palette("colorblind")[2]],
            )
            sns.scatterplot(
                data=dataframes["llama_tulu_all"],
                x=x,
                y=y,
                hue="Label",
                s=point_size,
                marker=marker,
                palette=[sns.color_palette("colorblind")[3]],
            )
            sns.scatterplot(
                data=dataframes[f"llama_tulu_all_safety_{amt}"],
                x=x,
                y=y,
                hue="Label",
                s=point_size,
                marker=marker,
                palette=[sns.color_palette("colorblind")[4]],
            )
            # sns.scatterplot(
            #     data=dataframes[f"tulu_2_tulu_match_safety_{amt}"],
            #     x=x,
            #     y=y,
            #     hue="Label",
            #     s=point_size,
            #     marker=marker,
            #     palette=[sns.color_palette("colorblind")[6]],
            # )
            plt.legend()
            plt.xlabel(x,fontsize=20)
            plt.ylabel(y,fontsize=20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.legend(fontsize=11)

            plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
            # plt.show()
            plt.savefig(f'results/consistent_mix/plots/safety_{amt}-{label}.png', dpi=300, bbox_inches='tight')
            plt.clf()

    def compare_coding_curves(amt = "100"):
        sns.lineplot(
            data=dataframes[f"coding_{amt}_interp"],
            x="Tulu Average",
            y="Coding Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[0]],
        )
        sns.lineplot(
            data=dataframes[f"coding_{amt}_llama_ta"],
            x="Tulu Average",
            y="Coding Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[1]],
        )
        sns.lineplot(
            data=dataframes[f"coding_{amt}_tulu_ta"],
            x="Tulu Average",
            y="Coding Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[2]],
        )
        sns.scatterplot(
            data=dataframes[f"llama_tulu_none_coding_{amt}"],
            x="Tulu Average",
            y="Coding Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[0]],
        )
        sns.scatterplot(
            data=dataframes[f"tulu_2_tulu_none_coding_{amt}"],
            x="Tulu Average",
            y="Coding Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[2]],
        )
        sns.scatterplot(
            data=dataframes["llama_tulu_all"],
            x="Tulu Average",
            y="Coding Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[3]],
        )
        sns.scatterplot(
            data=dataframes[f"llama_tulu_all_coding_{amt}"],
            x="Tulu Average",
            y="Coding Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[4]],
        )
        # sns.scatterplot(
        #     data=dataframes[f"tulu_2_tulu_match_coding_{amt}"],
        #     x="Tulu Average",
        #     y="Coding Average",
        #     hue="Label",
        #     s=point_size,
        #     marker=marker,
        #     palette=[sns.color_palette("colorblind")[6]],
        # )
        plt.legend()
        plt.xlabel("Tulu Average",fontsize=20)
        plt.ylabel("Coding Average",fontsize=20)
        plt.xticks([0.4, 0.45, 0.5, 0.55, 0.6], fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=11)

        plt.xlim(0.4, 0.6)

        plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        # plt.show()
        plt.savefig(f'results/consistent_mix/plots/coding_{amt}.png', dpi=300, bbox_inches='tight')
        plt.clf()

    def linear_interpolation_100p_curves():
        ticksize=14

        fig, axes = plt.subplots(
            1, 
            3, 
            figsize=(18, 5)
        )

        sns.lineplot(
            data=dataframes["science_2500_interp"],
            x="Tulu Average",
            y="Science Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[0]],
            ax=axes[0],
        )

        axes[0].set_title('Science', fontsize=20)
        axes[0].set(xlabel=None, ylabel=None)
        axes[0].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        axes[0].legend().set_visible(False)
        axes[0].tick_params(axis='both', which='major', labelsize=ticksize)


        sns.lineplot(
            data=dataframes["safety_100_interp"],
            x="Tulu Average",
            y="Safety Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[0]],
            ax=axes[1],
        )

        axes[1].set_title('Safety', fontsize=20)
        axes[1].set(xlabel=None, ylabel=None)
        axes[1].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        axes[1].legend().set_visible(False)
        axes[1].tick_params(axis='both', which='major', labelsize=ticksize)

        sns.lineplot(
            data=dataframes["coding_100_interp"],
            x="Tulu Average",
            y="Coding Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[0]],
            ax=axes[2],
        )

        axes[2].set_title('Coding', fontsize=20)
        axes[2].set(xlabel=None, ylabel=None)
        axes[2].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        axes[2].legend().set_visible(False)
        axes[2].tick_params(axis='both', which='major', labelsize=ticksize)

        # plt.xlabel("Tulu Average",fontsize=20)
        # plt.ylabel("Domain Average",fontsize=20)

        # plt.xticks(
        #     # [0.4, 0.45, 0.5, 0.55, 0.6],
        #     fontsize=16
        # )
        # plt.yticks(fontsize=16)
        # plt.legend(fontsize=11)

        fig.text(0.5, 0.02, 'Tulu Average', ha='center', va='center', fontsize=18)
        fig.text(0.075, 0.5, 'Domain Average', va='center', rotation='vertical', fontsize=18)

        # plt.tight_layout()
        # plt.show()
        plt.savefig(f'results/consistent_mix/plots/linear_interpolation_100p_all_3.png', dpi=300, bbox_inches='tight')
        plt.clf()

    def science_coding_ta_curves():
        ticksize=14

        fig, axes = plt.subplots(
            1, 
            3, 
            figsize=(18, 5)
        )

        # science
        sns.lineplot(
            data=dataframes[f"science_2500_llama_ta"],
            x="Tulu Average",
            y="Science Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[1]],
            ax=axes[0],
        )
        sns.lineplot(
            data=dataframes[f"science_2500_tulu_ta"],
            x="Tulu Average",
            y="Science Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[2]],
            ax=axes[0],
        )

        axes[0].set_title('Science', fontsize=20)
        axes[0].set(xlabel=None, ylabel=None)
        axes[0].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        axes[0].legend(title='')
        axes[0].tick_params(axis='both', which='major', labelsize=ticksize)

        # safety
        sns.lineplot(
            data=dataframes[f"safety_100_llama_ta"],
            x="Tulu Average",
            y="Safety Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[1]],
            ax=axes[1],
        )
        sns.lineplot(
            data=dataframes[f"safety_100_tulu_ta"],
            x="Tulu Average",
            y="Safety Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[2]],
            ax=axes[1],
        )

        axes[1].set_title('Safety', fontsize=20)
        axes[1].set(xlabel=None, ylabel=None)
        axes[1].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        axes[1].legend().set_visible(False)
        axes[1].tick_params(axis='both', which='major', labelsize=ticksize)

        # coding
        sns.lineplot(
            data=dataframes[f"coding_100_llama_ta"],
            x="Tulu Average",
            y="Coding Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[1]],
            ax=axes[2],
        )
        sns.lineplot(
            data=dataframes[f"coding_100_tulu_ta"],
            x="Tulu Average",
            y="Coding Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[2]],
            ax=axes[2],
        )

        axes[2].set_title('Science', fontsize=20)
        axes[2].set(xlabel=None, ylabel=None)
        axes[2].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        axes[2].legend().set_visible(False)
        axes[2].tick_params(axis='both', which='major', labelsize=ticksize)

        # plt.xlabel("Tulu Average",fontsize=20)
        # plt.ylabel("Domain Average",fontsize=20)

        # plt.xticks(
        #     # [0.4, 0.45, 0.5, 0.55, 0.6],
        #     fontsize=16
        # )
        # plt.yticks(fontsize=16)
        # plt.legend(fontsize=11)

        fig.text(0.5, 0.02, 'Tulu Average', ha='center', va='center', fontsize=18)
        fig.text(0.075, 0.5, 'Domain Average', va='center', rotation='vertical', fontsize=18)

        # plt.tight_layout()
        # plt.show()
        plt.savefig(f'results/consistent_mix/plots/task_arithmetic_2_curves_100p_all_3.png', dpi=300, bbox_inches='tight')
        plt.clf()

    def science_interference_curves():
        sns.lineplot(
            data=dataframes["science_2500_llama_ta"],
            x="Tulu Average",
            y="Science Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[1]],
        )
        sns.lineplot(
            data=dataframes["science_2500_tulu_ta"],
            x="Tulu Average",
            y="Science Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[2]],
        )
        sns.lineplot(
            data=dataframes["science_1000_tulu_match_ta"],
            x="Tulu Average",
            y="Science Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[3]],
        )
        sns.lineplot(
            data=dataframes["science_2500_tulu_match_ta"],
            x="Tulu Average",
            y="Science Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[4]],
        )
        plt.legend()
        plt.xlabel("Tulu Average",fontsize=20)
        plt.ylabel("Science Average",fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=11)

        plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        plt.show()
        # plt.savefig(f'results/consistent_mix/plots/science_{amt}.png', dpi=300, bbox_inches='tight')
        plt.clf()

    compare_science_curves()
    for amt in [
        # "20",
        # "40",
        # "60",
        # "80",
        "100",
    ]:
        compare_safety_curves(amt=amt)
    compare_coding_curves()
    linear_interpolation_100p_curves()
    science_coding_ta_curves()
    science_interference_curves()

make_plots()