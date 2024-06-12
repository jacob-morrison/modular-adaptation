import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd 
from pprint import pprint 

def calculate_average(row, columns):
    subset_values = row[columns]
    return 100.0 * subset_values.mean()

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
        "llama_2_7b-tulu_none-science_100": "Domain Vector",
        "llama_2_7b-tulu_none-science_200": "Domain Vector",
        "llama_2_7b-tulu_none-science_500": "Domain Vector",
        "llama_2_7b-tulu_none-science_1000": "Domain Vector",
        "llama_2_7b-tulu_none-science_2500": "Domain Vector",
        "llama_2_7b-tulu_all-science_100": "Retrain With Domain",
        "llama_2_7b-tulu_all-science_200": "Retrain With Domain",
        "llama_2_7b-tulu_all-science_500": "Retrain With Domain",
        "llama_2_7b-tulu_all-science_1000": "Retrain With Domain",
        "llama_2_7b-tulu_all-science_2500": "Retrain With Domain",
        "tulu_2_7b-tulu_none-science_100": "Continued Finetuning",
        "tulu_2_7b-tulu_none-science_200": "Continued Finetuning",
        "tulu_2_7b-tulu_none-science_500": "Continued Finetuning",
        "tulu_2_7b-tulu_none-science_1000": "Continued Finetuning",
        "tulu_2_7b-tulu_none-science_2500": "Continued Finetuning",

        # safety baselines:
        "llama_2_7b-tulu_none-safety_20": "Domain Vector",
        "llama_2_7b-tulu_none-safety_40": "Domain Vector",
        "llama_2_7b-tulu_none-safety_60": "Domain Vector",
        "llama_2_7b-tulu_none-safety_80": "Domain Vector",
        "llama_2_7b-tulu_none-safety_100": "Domain Vector",
        "llama_2_7b-tulu_all-safety_20": "Retrain With Domain",
        "llama_2_7b-tulu_all-safety_40": "Retrain With Domain",
        "llama_2_7b-tulu_all-safety_60": "Retrain With Domain",
        "llama_2_7b-tulu_all-safety_80": "Retrain With Domain",
        "llama_2_7b-tulu_all-safety_100": "Retrain With Domain",
        "tulu_2_7b-tulu_none-safety_20": "Continued Finetuning",
        "tulu_2_7b-tulu_none-safety_40": "Continued Finetuning",
        "tulu_2_7b-tulu_none-safety_60": "Continued Finetuning",
        "tulu_2_7b-tulu_none-safety_80": "Continued Finetuning",
        "tulu_2_7b-tulu_none-safety_100": "Continued Finetuning",

        # coding baselines:
        "llama_2_7b-tulu_none-coding_20": "Domain Vector",
        "llama_2_7b-tulu_none-coding_40": "Domain Vector",
        "llama_2_7b-tulu_none-coding_60": "Domain Vector",
        "llama_2_7b-tulu_none-coding_80": "Domain Vector",
        "llama_2_7b-tulu_none-coding_100": "Domain Vector",
        "llama_2_7b-tulu_all-coding_20": "Retrain With Domain",
        "llama_2_7b-tulu_all-coding_40": "Retrain With Domain",
        "llama_2_7b-tulu_all-coding_60": "Retrain With Domain",
        "llama_2_7b-tulu_all-coding_80": "Retrain With Domain",
        "llama_2_7b-tulu_all-coding_100": "Retrain With Domain",
        "tulu_2_7b-tulu_none-coding_20": "Continued Finetuning",
        "tulu_2_7b-tulu_none-coding_40": "Continued Finetuning",
        "tulu_2_7b-tulu_none-coding_60": "Continued Finetuning",
        "tulu_2_7b-tulu_none-coding_80": "Continued Finetuning",
        "tulu_2_7b-tulu_none-coding_100": "Continued Finetuning",
    }
    
    if row["model_key"] in baseline_keys:
        return baseline_keys[row["model_key"]]

    combo_map = {
        # Science
        "Llama 2 Science 100 - task_arithmetic": "Adding Domain Vector",
        "Llama 2 Science 200 - task_arithmetic": "Adding Domain Vector",
        "Llama 2 Science 500 - task_arithmetic": "Adding Domain Vector",
        "Llama 2 Science 1000 - task_arithmetic": "Adding Domain Vector",
        "Llama 2 Science 2500 - task_arithmetic": "Adding Domain Vector",

        "Tulu 2 Science 100 - wise-ft": "Continued Finetuning",
        "Tulu 2 Science 200 - wise-ft": "Continued Finetuning",
        "Tulu 2 Science 500 - wise-ft": "Continued Finetuning",
        "Tulu 2 Science 1000 - wise-ft": "Continued Finetuning",
        "Tulu 2 Science 2500 - wise-ft": "Continued Finetuning",

        "Llama 2 Science 100 - linear_weighted": "Linear Interpolation",
        "Llama 2 Science 200 - linear_weighted": "Linear Interpolation",
        "Llama 2 Science 500 - linear_weighted": "Linear Interpolation",
        "Llama 2 Science 1000 - linear_weighted": "Linear Interpolation",
        "Llama 2 Science 2500 - linear_weighted": "Linear Interpolation",

        "Tulu 2 7B ft. on Tulu Match & Science 100": "Tulu 2 Task Vector (Ep. Replay)",
        "Tulu 2 7B ft. on Tulu Match & Science 200": "Tulu 2 Task Vector (Ep. Replay)",
        "Tulu 2 7B ft. on Tulu Match & Science 500": "Tulu 2 Task Vector (Ep. Replay)",
        "Tulu 2 7B ft. on Tulu Match & Science 1000": "Tulu 2 Task Vector (Ep. Replay)",
        "Tulu 2 7B ft. on Tulu Match & Science 2500": "Tulu 2 Task Vector (Ep. Replay)",

        "Tulu 2 7B Tulu Match Science 1000 - wise-ft": "Tulu 2 Task Vector (Ep. Replay)",
        "Tulu 2 7B Tulu Match Science 2500 - wise-ft": "Tulu 2 Task Vector (Ep. Replay)",

        # Safety
        "Llama 2 Safety 20% - task_arithmetic": "Adding Domain Vector",
        "Llama 2 Safety 40% - task_arithmetic": "Adding Domain Vector",
        "Llama 2 Safety 60% - task_arithmetic": "Adding Domain Vector",
        "Llama 2 Safety 80% - task_arithmetic": "Adding Domain Vector",
        "Llama 2 Safety 100% - task_arithmetic": "Adding Domain Vector",

        "Tulu 2 Safety 20% - wise-ft": "Continued Finetuning",
        "Tulu 2 Safety 40% - wise-ft": "Continued Finetuning",
        "Tulu 2 Safety 60% - wise-ft": "Continued Finetuning",
        "Tulu 2 Safety 80% - wise-ft": "Continued Finetuning",
        "Tulu 2 Safety 100% - wise-ft": "Continued Finetuning",

        "Llama 2 Safety 20% - linear_weighted": "Linear Interpolation",
        "Llama 2 Safety 40% - linear_weighted": "Linear Interpolation",
        "Llama 2 Safety 60% - linear_weighted": "Linear Interpolation",
        "Llama 2 Safety 80% - linear_weighted": "Linear Interpolation",
        "Llama 2 Safety 100% - linear_weighted": "Linear Interpolation",

        "Tulu 2 7B ft. on Tulu Match & Safety 20%": "Tulu 2 Task Vector (Ep. Replay)",
        "Tulu 2 7B ft. on Tulu Match & Safety 40%": "Tulu 2 Task Vector (Ep. Replay)",
        "Tulu 2 7B ft. on Tulu Match & Safety 60%": "Tulu 2 Task Vector (Ep. Replay)",
        "Tulu 2 7B ft. on Tulu Match & Safety 80%": "Tulu 2 Task Vector (Ep. Replay)",
        "Tulu 2 7B ft. on Tulu Match & Safety 100%": "Tulu 2 Task Vector (Ep. Replay)",

        # Coding
        "Llama 2 Coding 20% - task_arithmetic": "Adding Domain Vector",
        "Llama 2 Coding 40% - task_arithmetic": "Adding Domain Vector",
        "Llama 2 Coding 60% - task_arithmetic": "Adding Domain Vector",
        "Llama 2 Coding 80% - task_arithmetic": "Adding Domain Vector",
        "Llama 2 Coding 100% - task_arithmetic": "Adding Domain Vector",

        "Tulu 2 Coding 20% - wise-ft": "Continued Finetuning",
        "Tulu 2 Coding 40% - wise-ft": "Continued Finetuning",
        "Tulu 2 Coding 60% - wise-ft": "Continued Finetuning",
        "Tulu 2 Coding 80% - wise-ft": "Continued Finetuning",
        "Tulu 2 Coding 100% - wise-ft": "Continued Finetuning",

        "Llama 2 Coding 20% - linear_weighted": "Linear Interpolation",
        "Llama 2 Coding 40% - linear_weighted": "Linear Interpolation",
        "Llama 2 Coding 60% - linear_weighted": "Linear Interpolation",
        "Llama 2 Coding 80% - linear_weighted": "Linear Interpolation",
        "Llama 2 Coding 100% - linear_weighted": "Linear Interpolation",

        "Tulu 2 7B ft. on Tulu Match & Coding 20%": "Tulu 2 Task Vector (Ep. Replay)",
        "Tulu 2 7B ft. on Tulu Match & Coding 40%": "Tulu 2 Task Vector (Ep. Replay)",
        "Tulu 2 7B ft. on Tulu Match & Coding 60%": "Tulu 2 Task Vector (Ep. Replay)",
        "Tulu 2 7B ft. on Tulu Match & Coding 80%": "Tulu 2 Task Vector (Ep. Replay)",
        "Tulu 2 7B ft. on Tulu Match & Coding 100%": "Tulu 2 Task Vector (Ep. Replay)",

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
    df['Exaggerated Refusals'] = df.apply(lambda row: 100 * row["normalized_safe_average"], axis=1)
    df['General Capabilities'] = df.apply(lambda row: calculate_average(row, tulu_columns_for_test_average), axis=1)
    df['Safety Average'] = df.apply(lambda row: calculate_average(row, safety_columns_for_average), axis=1)
    df["Science Average"] = df.apply(lambda row: 100 * row["mean_null"], axis=1)
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
        "llama_tulu_all_with_coding": df[df["model_key"] == "llama_2_7b-tulu_all_with_coding"],

        # heuristic points
        "science_2500_heuristic": df[df["model_key"] == "task_arithmetic-llama_2_7b-tulu_all_1.0-llama_2_7b-tulu_none-science_2500_0.22"],
        "safety_100_heuristic": df[df["model_key"] == "task_arithmetic-llama_2_7b-tulu_all_1.0-llama_2_7b-tulu_none-safety_100_0.24"],
        "coding_100_heuristic": df[df["model_key"] == "task_arithmetic-llama_2_7b-tulu_all_1.0-llama_2_7b-tulu_none-coding_100_0.57"],

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

        # coding with coding
        "llama_tulu_all_with_coding_coding_20": df[df["model_key"] == "llama_2_7b-tulu_all_with_coding-coding_20"],
        "llama_tulu_all_with_coding_coding_40": df[df["model_key"] == "llama_2_7b-tulu_all_with_coding-coding_40"],
        "llama_tulu_all_with_coding_coding_60": df[df["model_key"] == "llama_2_7b-tulu_all_with_coding-coding_60"],
        "llama_tulu_all_with_coding_coding_80": df[df["model_key"] == "llama_2_7b-tulu_all_with_coding-coding_80"],
        "llama_tulu_all_with_coding_coding_100": df[df["model_key"] == "llama_2_7b-tulu_all_with_coding-coding_100"],

        "tulu_2_with_coding_tulu_none_coding_20": df[df["model_key"] == "tulu_2_7b_with_coding-tulu_none-coding_20"],
        "tulu_2_with_coding_tulu_none_coding_40": df[df["model_key"] == "tulu_2_7b_with_coding-tulu_none-coding_40"],
        "tulu_2_with_coding_tulu_none_coding_60": df[df["model_key"] == "tulu_2_7b_with_coding-tulu_none-coding_60"],
        "tulu_2_with_coding_tulu_none_coding_80": df[df["model_key"] == "tulu_2_7b_with_coding-tulu_none-coding_80"],
        "tulu_2_with_coding_tulu_none_coding_100": df[df["model_key"] == "tulu_2_7b_with_coding-tulu_none-coding_100"],

        "tulu_2_with_coding_tulu_match_coding_20": df[df["model_key"] == "tulu_2_7b_with_coding-tulu_match-coding_20"],
        "tulu_2_with_coding_tulu_match_coding_40": df[df["model_key"] == "tulu_2_7b_with_coding-tulu_match-coding_40"],
        "tulu_2_with_coding_tulu_match_coding_60": df[df["model_key"] == "tulu_2_7b_with_coding-tulu_match-coding_60"],
        "tulu_2_with_coding_tulu_match_coding_80": df[df["model_key"] == "tulu_2_7b_with_coding-tulu_match-coding_80"],
        "tulu_2_with_coding_tulu_match_coding_100": df[df["model_key"] == "tulu_2_7b_with_coding-tulu_match-coding_100"],

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

        # Tulu w/ Coding
        "tulu_w_coding_coding_20_llama_ta": df[(df["Combo"] == "Llama 2 Coding 20% - task_arithmetic") & (df["tulu_model"].str.contains("coding"))],
        "tulu_w_coding_coding_40_llama_ta": df[(df["Combo"] == "Llama 2 Coding 40% - task_arithmetic") & (df["tulu_model"].str.contains("coding"))],
        "tulu_w_coding_coding_60_llama_ta": df[(df["Combo"] == "Llama 2 Coding 60% - task_arithmetic") & (df["tulu_model"].str.contains("coding"))],
        "tulu_w_coding_coding_80_llama_ta": df[(df["Combo"] == "Llama 2 Coding 80% - task_arithmetic") & (df["tulu_model"].str.contains("coding"))],
        "tulu_w_coding_coding_100_llama_ta": df[(df["Combo"] == "Llama 2 Coding 100% - task_arithmetic") & (df["tulu_model"].str.contains("coding"))],

        "tulu_w_coding_coding_20_tulu_ta": df[(df["Combo"] == "Tulu 2 7B w/ Coding Tulu None Coding 20% - wise-ft") & (df["tulu_model"].str.contains("coding"))],
        "tulu_w_coding_coding_40_tulu_ta": df[(df["Combo"] == "Tulu 2 7B w/ Coding Tulu None Coding 40% - wise-ft") & (df["tulu_model"].str.contains("coding"))],
        "tulu_w_coding_coding_60_tulu_ta": df[(df["Combo"] == "Tulu 2 7B w/ Coding Tulu None Coding 60% - wise-ft") & (df["tulu_model"].str.contains("coding"))],
        "tulu_w_coding_coding_80_tulu_ta": df[(df["Combo"] == "Tulu 2 7B w/ Coding Tulu None Coding 80% - wise-ft") & (df["tulu_model"].str.contains("coding"))],
        "tulu_w_coding_coding_100_tulu_ta": df[(df["Combo"] == "Tulu 2 7B w/ Coding Tulu None Coding 100% - wise-ft") & (df["tulu_model"].str.contains("coding"))],

        "tulu_w_coding_coding_20_interp": df[(df["Combo"] == "Llama 2 Coding 20% - linear_weighted") & (df["tulu_model"].str.contains("coding"))],
        "tulu_w_coding_coding_40_interp": df[(df["Combo"] == "Llama 2 Coding 40% - linear_weighted") & (df["tulu_model"].str.contains("coding"))],
        "tulu_w_coding_coding_60_interp": df[(df["Combo"] == "Llama 2 Coding 60% - linear_weighted") & (df["tulu_model"].str.contains("coding"))],
        "tulu_w_coding_coding_80_interp": df[(df["Combo"] == "Llama 2 Coding 80% - linear_weighted") & (df["tulu_model"].str.contains("coding"))],
        "tulu_w_coding_coding_100_interp": df[(df["Combo"] == "Llama 2 Coding 100% - linear_weighted") & (df["tulu_model"].str.contains("coding"))],

    }

    # baseline curves - science
    dataframes["science_retrain_ablations"] = pd.concat([
        dataframes["llama_tulu_all_science_100"],
        dataframes["llama_tulu_all_science_200"],
        dataframes["llama_tulu_all_science_500"],
        dataframes["llama_tulu_all_science_2500"],
        dataframes["llama_tulu_all_science_2500"],
    ])
    dataframes["science_cft_ablations"] = pd.concat([
        dataframes["tulu_2_tulu_none_science_100"],
        dataframes["tulu_2_tulu_none_science_200"],
        dataframes["tulu_2_tulu_none_science_500"],
        dataframes["tulu_2_tulu_none_science_1000"],
        dataframes["tulu_2_tulu_none_science_2500"],
    ])
    dataframes["science_cft_match_ablations"] = pd.concat([
        dataframes["tulu_2_tulu_match_science_100"],
        dataframes["tulu_2_tulu_match_science_200"],
        dataframes["tulu_2_tulu_match_science_500"],
        dataframes["tulu_2_tulu_match_science_1000"],
        dataframes["tulu_2_tulu_match_science_2500"],
    ])

    # safety
    dataframes["safety_retrain_ablations"] = pd.concat([
        dataframes["llama_tulu_all_safety_20"],
        dataframes["llama_tulu_all_safety_40"],
        dataframes["llama_tulu_all_safety_60"],
        dataframes["llama_tulu_all_safety_80"],
        dataframes["llama_tulu_all_safety_100"],
    ])
    dataframes["safety_cft_ablations"] = pd.concat([
        dataframes["tulu_2_tulu_none_safety_20"],
        dataframes["tulu_2_tulu_none_safety_40"],
        dataframes["tulu_2_tulu_none_safety_60"],
        dataframes["tulu_2_tulu_none_safety_80"],
        dataframes["tulu_2_tulu_none_safety_100"],
    ])
    dataframes["safety_cft_match_ablations"] = pd.concat([
        dataframes["tulu_2_tulu_match_safety_20"],
        dataframes["tulu_2_tulu_match_safety_40"],
        dataframes["tulu_2_tulu_match_safety_60"],
        dataframes["tulu_2_tulu_match_safety_80"],
        dataframes["tulu_2_tulu_match_safety_100"],
    ])

    # coding
    dataframes["coding_retrain_ablations"] = pd.concat([
        dataframes["llama_tulu_all_coding_20"],
        dataframes["llama_tulu_all_coding_40"],
        dataframes["llama_tulu_all_coding_60"],
        dataframes["llama_tulu_all_coding_80"],
        dataframes["llama_tulu_all_coding_100"],
    ])
    dataframes["coding_cft_ablations"] = pd.concat([
        dataframes["tulu_2_tulu_none_coding_20"],
        dataframes["tulu_2_tulu_none_coding_40"],
        dataframes["tulu_2_tulu_none_coding_60"],
        dataframes["tulu_2_tulu_none_coding_80"],
        dataframes["tulu_2_tulu_none_coding_100"],
    ])
    dataframes["coding_cft_match_ablations"] = pd.concat([
        dataframes["tulu_2_tulu_match_coding_20"],
        dataframes["tulu_2_tulu_match_coding_40"],
        dataframes["tulu_2_tulu_match_coding_60"],
        dataframes["tulu_2_tulu_match_coding_80"],
        dataframes["tulu_2_tulu_match_coding_100"],
    ])

    # manually edit heuristics
    # TODO: update indices if anything changes
    dataframes["science_2500_heuristic"].at[548, "Label"] = "Data-Weighted Merge"
    dataframes["safety_100_heuristic"].loc[533, "Label"] = "Data-Weighted Merge"
    dataframes["coding_100_heuristic"].loc[521, "Label"] = "Data-Weighted Merge"

    even_weights = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    dataframes["science_2500_llama_ta_even"] = dataframes["science_2500_llama_ta"][dataframes["science_2500_llama_ta"]["Order"].isin(even_weights)]
    dataframes["coding_100_llama_ta_even"] = dataframes["coding_100_llama_ta"][dataframes["coding_100_llama_ta"]["Order"].isin(even_weights)]
    dataframes["safety_100_llama_ta_even"] = dataframes["safety_100_llama_ta"][dataframes["safety_100_llama_ta"]["Order"].isin(even_weights)]

    line_width = 3
    markersize = 20
    marker = "*"
    point_size = 500
    def compare_science_curves(amt = "2500"):
        sns.lineplot(
            data=dataframes[f"science_{amt}_interp"],
            x="General Capabilities",
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
            x="General Capabilities",
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
            x="General Capabilities",
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
            x="General Capabilities",
            y="Science Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[0]],
        )
        sns.scatterplot(
            data=dataframes[f"tulu_2_tulu_none_science_{amt}"],
            x="General Capabilities",
            y="Science Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[2]],
        )
        sns.scatterplot(
            data=dataframes["llama_tulu_all"],
            x="General Capabilities",
            y="Science Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[0]],
        )
        sns.scatterplot(
            data=dataframes[f"llama_tulu_all_science_{amt}"],
            x="General Capabilities",
            y="Science Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[4]],
        )
        # sns.scatterplot(
        #     data=dataframes[f"tulu_2_tulu_match_science_{amt}"],
        #     x="General Capabilities",
        #     y="Science Average",
        #     hue="Label",
        #     s=point_size,
        #     marker=marker,
        #     palette=[sns.color_palette("colorblind")[6]],
        # )
        plt.legend()
        plt.xlabel("General Capabilities",fontsize=20)
        plt.ylabel("Science Average",fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=11)

        plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        # plt.show()
        plt.savefig(f'results/consistent_mix/plots/science_{amt}.png', dpi=300, bbox_inches='tight')
        plt.clf()

    def compare_safety_curves(amt = "100", x = "General Capabilities", y = "Safety Average", heuristic=False):
        if heuristic:
            safety_llama_ta = f"safety_{amt}_llama_ta"
        else:
            safety_llama_ta = f"safety_{amt}_llama_ta_even"

        for x, y, label in [
            ("General Capabilities", "Safety Average", "tulu_vs_safety"),
            ("General Capabilities", "Exaggerated Refusals", "tulu_vs_exaggerated_refusals"),
            ("Exaggerated Refusals", "Safety Average", "exaggerated_refusals_vs_safety"),
        ]:
            sns.lineplot(
                data=dataframes[safety_llama_ta],
                x=x,
                y=y,
                hue="Label",
                sort=False,
                # marker='X',
                linewidth=line_width,
                markersize=markersize,
                palette=[sns.color_palette("colorblind")[1]],
            )
            if not heuristic:
                sns.lineplot(
                    data=dataframes["safety_retrain_ablations"],
                    x=x,
                    y=y,
                    hue="Label",
                    sort=False,
                    # marker='X',
                    linewidth=line_width,
                    markersize=markersize,
                    palette=[sns.color_palette("colorblind")[3]],
                )
                sns.lineplot(
                    data=dataframes["safety_cft_ablations"],
                    x=x,
                    y=y,
                    hue="Label",
                    sort=False,
                    # marker='X',
                    linewidth=line_width,
                    markersize=markersize,
                    palette=[sns.color_palette("colorblind")[4]],
                )
                # sns.lineplot(
                #     data=dataframes["safety_cft_match_ablations"],
                #     x=x,
                #     y=y,
                #     hue="Label",
                #     sort=False,
                #     # marker='X',
                #     linewidth=line_width,
                #     markersize=markersize,
                #     palette=[sns.color_palette("colorblind")[2]],
                # )
            else:
                sns.scatterplot(
                    data=dataframes[f"safety_100_heuristic"],
                    x=x,
                    y=y,
                    hue="Label",
                    s=point_size,
                    marker=marker,
                    palette=[sns.color_palette("colorblind")[3]],
                )
            sns.scatterplot(
                data=dataframes["llama_tulu_none_safety_100"],
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
                palette=[sns.color_palette("colorblind")[0]],
            )
            plt.legend()
            plt.xlabel(x,fontsize=20)
            plt.ylabel(y,fontsize=20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.legend(fontsize=11)

            plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
            # plt.show()
            if heuristic:
                plt.savefig(f'results/consistent_mix/plots/safety_{amt}-{label}-heuristic.png', dpi=300, bbox_inches='tight')
            else:
                plt.savefig(f'results/consistent_mix/plots/safety_{amt}-{label}.png', dpi=300, bbox_inches='tight')
            plt.clf()

    def compare_coding_curves(amt = "100"):
        sns.lineplot(
            data=dataframes[f"coding_{amt}_interp"],
            x="General Capabilities",
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
            x="General Capabilities",
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
            x="General Capabilities",
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
            x="General Capabilities",
            y="Coding Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[0]],
        )
        sns.scatterplot(
            data=dataframes[f"tulu_2_tulu_none_coding_{amt}"],
            x="General Capabilities",
            y="Coding Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[2]],
        )
        sns.scatterplot(
            data=dataframes["llama_tulu_all"],
            x="General Capabilities",
            y="Coding Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[0]],
        )
        sns.scatterplot(
            data=dataframes[f"llama_tulu_all_coding_{amt}"],
            x="General Capabilities",
            y="Coding Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[4]],
        )
        # sns.scatterplot(
        #     data=dataframes[f"tulu_2_tulu_match_coding_{amt}"],
        #     x="General Capabilities",
        #     y="Coding Average",
        #     hue="Label",
        #     s=point_size,
        #     marker=marker,
        #     palette=[sns.color_palette("colorblind")[6]],
        # )
        plt.legend()
        plt.xlabel("General Capabilities",fontsize=20)
        plt.ylabel("Coding Average",fontsize=20)
        plt.xticks([40, 45, 50, 55, 60], fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=11)

        plt.xlim(40, 60)

        plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        # plt.show()
        plt.savefig(f'results/consistent_mix/plots/coding_{amt}.png', dpi=300, bbox_inches='tight')
        plt.clf()

    def compare_3_weighting_strategies():
        ticksize=14

        fig, axes = plt.subplots(
            1, 
            3, 
            figsize=(18, 5)
        )

        sns.lineplot(
            data=dataframes["science_2500_interp"],
            x="General Capabilities",
            y="Science Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[0]],
            ax=axes[0],
        )
        sns.lineplot(
            data=dataframes[f"science_2500_llama_ta"],
            x="General Capabilities",
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
            x="General Capabilities",
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


        sns.lineplot(
            data=dataframes["safety_100_interp"],
            x="General Capabilities",
            y="Safety Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[0]],
            ax=axes[1],
        )
        sns.lineplot(
            data=dataframes[f"safety_100_llama_ta"],
            x="General Capabilities",
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
            x="General Capabilities",
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

        sns.lineplot(
            data=dataframes["coding_100_interp"],
            x="General Capabilities",
            y="Coding Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[0]],
            ax=axes[2],
        )
        sns.lineplot(
            data=dataframes[f"coding_100_llama_ta"],
            x="General Capabilities",
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
            x="General Capabilities",
            y="Coding Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[2]],
            ax=axes[2],
        )

        axes[2].set_title('Coding', fontsize=20)
        axes[2].set(xlabel=None, ylabel=None)
        axes[2].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        axes[2].legend().set_visible(False)
        axes[2].tick_params(axis='both', which='major', labelsize=ticksize)

        # plt.xlabel("General Capabilities",fontsize=20)
        # plt.ylabel("Domain Average",fontsize=20)

        # plt.xticks(
        #     # [0.4, 0.45, 0.5, 0.55, 0.6],
        #     fontsize=16
        # )
        # plt.yticks(fontsize=16)
        # plt.legend(fontsize=11)

        fig.text(0.5, 0.02, 'General Capabilities', ha='center', va='center', fontsize=18)
        fig.text(0.075, 0.5, 'Domain Average', va='center', rotation='vertical', fontsize=18)

        # plt.tight_layout()
        # plt.show()
        plt.savefig(f'results/consistent_mix/plots/all_3_weighting_strategies.png', dpi=300, bbox_inches='tight')
        plt.clf()

    def ta_curves_vs_baselines_all_3():
        ticksize=14

        fig, axes = plt.subplots(
            1, 
            2, 
            figsize=(12, 5)
        )

        # # science
        # sns.lineplot(
        #     data=dataframes[f"science_2500_llama_ta"],
        #     x="General Capabilities",
        #     y="Science Average",
        #     hue="Label",
        #     sort=False,
        #     # marker='X',
        #     linewidth=line_width,
        #     markersize=markersize,
        #     palette=[sns.color_palette("colorblind")[1]],
        #     ax=axes[0],
        # )
        # sns.lineplot(
        #     data=dataframes["science_retrain_ablations"],
        #     x="General Capabilities",
        #     y="Science Average",
        #     hue="Label",
        #     sort=False,
        #     # marker='X',
        #     linewidth=line_width,
        #     markersize=markersize,
        #     palette=[sns.color_palette("colorblind")[3]],
        #     ax=axes[0],
        # )
        # sns.lineplot(
        #     data=dataframes["science_cft_ablations"],
        #     x="General Capabilities",
        #     y="Science Average",
        #     hue="Label",
        #     sort=False,
        #     # marker='X',
        #     linewidth=line_width,
        #     markersize=markersize,
        #     palette=[sns.color_palette("colorblind")[4]],
        #     ax=axes[0],
        # )
        # sns.scatterplot(
        #     data=dataframes["llama_tulu_all"],
        #     x="General Capabilities",
        #     y="Science Average",
        #     hue="Label",
        #     s=point_size,
        #     marker=marker,
        #     palette=[sns.color_palette("colorblind")[0]],
        #     ax=axes[0]
        # )
        # # sns.lineplot(
        # #     data=dataframes["science_cft_match_ablations"],
        # #     x="General Capabilities",
        # #     y="Science Average",
        # #     hue="Label",
        # #     sort=False,
        # #     # marker='X',
        # #     linewidth=line_width,
        # #     markersize=markersize,
        # #     palette=[sns.color_palette("colorblind")[2]],
        # #     ax=axes[0],
        # # )

        # axes[0].set_title('Science', fontsize=20)
        # axes[0].set(xlabel=None, ylabel=None)
        # axes[0].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        # axes[0].legend(title='')
        # axes[0].tick_params(axis='both', which='major', labelsize=ticksize)

        # safety
        sns.lineplot(
            data=dataframes[f"safety_100_llama_ta_even"],
            x="General Capabilities",
            y="Safety Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[1]],
            ax=axes[0],
        )
        sns.lineplot(
            data=dataframes["safety_retrain_ablations"],
            x="General Capabilities",
            y="Safety Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[3]],
            ax=axes[0],
        )
        sns.lineplot(
            data=dataframes["safety_cft_ablations"],
            x="General Capabilities",
            y="Safety Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[4]],
            ax=axes[0],
        )
        sns.scatterplot(
            data=dataframes["llama_tulu_all"],
            x="General Capabilities",
            y="Safety Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[0]],
            ax=axes[0]
        )
        # sns.lineplot(
        #     data=dataframes["safety_cft_match_ablations"],
        #     x="General Capabilities",
        #     y="Safety Average",
        #     hue="Label",
        #     sort=False,
        #     # marker='X',
        #     linewidth=line_width,
        #     markersize=markersize,
        #     palette=[sns.color_palette("colorblind")[2]],
        #     ax=axes[0],
        # )

        axes[0].set_title('Safety', fontsize=20)
        axes[0].set(xlabel=None, ylabel=None)
        axes[0].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        axes[0].legend(title='')
        axes[0].tick_params(axis='both', which='major', labelsize=ticksize)

        # coding
        sns.lineplot(
            data=dataframes[f"coding_100_llama_ta_even"],
            x="General Capabilities",
            y="Coding Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[1]],
            ax=axes[1],
        )
        sns.lineplot(
            data=dataframes["coding_retrain_ablations"],
            x="General Capabilities",
            y="Coding Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[3]],
            ax=axes[1],
        )
        sns.lineplot(
            data=dataframes["coding_cft_ablations"],
            x="General Capabilities",
            y="Coding Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[4]],
            ax=axes[1],
        )
        sns.scatterplot(
            data=dataframes["llama_tulu_all"],
            x="General Capabilities",
            y="Coding Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[0]],
            ax=axes[1]
        )
        # sns.lineplot(
        #     data=dataframes["coding_cft_match_ablations"],
        #     x="General Capabilities",
        #     y="Coding Average",
        #     hue="Label",
        #     sort=False,
        #     # marker='X',
        #     linewidth=line_width,
        #     markersize=markersize,
        #     palette=[sns.color_palette("colorblind")[2]],
        #     ax=axes[1],
        # )

        axes[1].set_title('Coding', fontsize=20)
        axes[1].set(xlabel=None, ylabel=None)
        axes[1].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        axes[1].legend().set_visible(False)
        axes[1].tick_params(axis='both', which='major', labelsize=ticksize)

        # plt.xlabel("General Capabilities",fontsize=20)
        # plt.ylabel("Domain Average",fontsize=20)

        # plt.xticks(
        #     # [0.4, 0.45, 0.5, 0.55, 0.6],
        #     fontsize=16
        # )
        # plt.yticks(fontsize=16)
        # plt.legend(fontsize=11)

        fig.text(0.5, 0.02, 'General Capabilities', ha='center', va='center', fontsize=18)
        fig.text(0.075, 0.5, 'Domain Average', va='center', rotation='vertical', fontsize=18)

        # plt.tight_layout()
        # plt.show()
        plt.savefig(f'results/consistent_mix/plots/task_arithmetic_vs_baselines_all_3.png', dpi=300, bbox_inches='tight')
        plt.clf()

    def ta_curves_heuristics():
        ticksize=14

        fig, axes = plt.subplots(
            1, 
            3, 
            figsize=(18, 5)
        )

        # science
        sns.lineplot(
            data=dataframes[f"science_2500_llama_ta"],
            x="General Capabilities",
            y="Science Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[1]],
            ax=axes[0],
        )
        sns.scatterplot(
            data=dataframes[f"science_2500_heuristic"],
            x="General Capabilities",
            y="Science Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[3]],
            ax=axes[0]
        )
        sns.scatterplot(
            data=dataframes["llama_tulu_all"],
            x="General Capabilities",
            y="Science Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[0]],
            ax=axes[0]
        )
        sns.scatterplot(
            data=dataframes["llama_tulu_none_science_2500"],
            x="General Capabilities",
            y="Science Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[2]],
            ax=axes[0]
        )

        axes[0].set_title('Science', fontsize=20)
        axes[0].set(xlabel=None, ylabel=None)
        axes[0].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        axes[0].legend(title='')
        axes[0].tick_params(axis='both', which='major', labelsize=ticksize)

        # safety
        sns.lineplot(
            data=dataframes[f"safety_100_llama_ta"],
            x="General Capabilities",
            y="Safety Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[1]],
            ax=axes[1],
        )
        sns.scatterplot(
            data=dataframes[f"safety_100_heuristic"],
            x="General Capabilities",
            y="Safety Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[3]],
            ax=axes[1]
        )
        sns.scatterplot(
            data=dataframes["llama_tulu_all"],
            x="General Capabilities",
            y="Safety Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[0]],
            ax=axes[1]
        )
        sns.scatterplot(
            data=dataframes["llama_tulu_none_safety_100"],
            x="General Capabilities",
            y="Safety Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[2]],
            ax=axes[1]
        )

        axes[1].set_title('Safety', fontsize=20)
        axes[1].set(xlabel=None, ylabel=None)
        axes[1].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        axes[1].legend().set_visible(False)
        axes[1].tick_params(axis='both', which='major', labelsize=ticksize)

        # coding
        sns.lineplot(
            data=dataframes[f"coding_100_llama_ta"],
            x="General Capabilities",
            y="Coding Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[1]],
            ax=axes[2],
        )
        sns.scatterplot(
            data=dataframes[f"coding_100_heuristic"],
            x="General Capabilities",
            y="Coding Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[3]],
            ax=axes[2]
        )
        sns.scatterplot(
            data=dataframes["llama_tulu_all"],
            x="General Capabilities",
            y="Coding Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[0]],
            ax=axes[2]
        )
        sns.scatterplot(
            data=dataframes["llama_tulu_none_coding_100"],
            x="General Capabilities",
            y="Coding Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[2]],
            ax=axes[2]
        )

        axes[2].set_title('Coding', fontsize=20)
        axes[2].set(xlabel=None, ylabel=None)
        axes[2].grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        axes[2].legend().set_visible(False)
        axes[2].tick_params(axis='both', which='major', labelsize=ticksize)

        # plt.xlabel("General Capabilities",fontsize=20)
        # plt.ylabel("Domain Average",fontsize=20)

        # plt.xticks(
        #     # [0.4, 0.45, 0.5, 0.55, 0.6],
        #     fontsize=16
        # )
        # plt.yticks(fontsize=16)
        # plt.legend(fontsize=11)

        fig.text(0.5, 0.02, 'General Capabilities', ha='center', va='center', fontsize=18)
        fig.text(0.075, 0.5, 'Domain Average', va='center', rotation='vertical', fontsize=18)

        # plt.tight_layout()
        # plt.show()
        plt.savefig(f'results/consistent_mix/plots/task_arithmetic_vs_heuristics_3.png', dpi=300, bbox_inches='tight')
        plt.clf()

    def science_interference_curves():
        sns.lineplot(
            data=dataframes["science_2500_llama_ta"],
            x="General Capabilities",
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
            x="General Capabilities",
            y="Science Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[2]],
        )
        sns.lineplot(
            data=dataframes["science_2500_tulu_match_ta"],
            x="General Capabilities",
            y="Science Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[3]],
        )
        # sns.lineplot(
        #     data=dataframes["science_2500_tulu_match_ta"],
        #     x="General Capabilities",
        #     y="Science Average",
        #     hue="Label",
        #     sort=False,
        #     # marker='X',
        #     linewidth=line_width,
        #     markersize=markersize,
        #     palette=[sns.color_palette("colorblind")[4]],
        # )
        plt.legend()
        plt.xlabel("General Capabilities",fontsize=20)
        plt.ylabel("Science Average",fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=11)

        plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        # plt.show()
        plt.savefig(f'results/consistent_mix/plots/science_interference_curves.png', dpi=300, bbox_inches='tight')
        plt.clf()

    def compare_coding_curves_tulu_with_coding(amt = "100"):
        sns.lineplot(
            data=dataframes[f"tulu_w_coding_coding_{amt}_interp"],
            x="General Capabilities",
            y="Coding Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[0]],
        )
        sns.lineplot(
            data=dataframes[f"tulu_w_coding_coding_{amt}_llama_ta"],
            x="General Capabilities",
            y="Coding Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[1]],
        )
        sns.lineplot(
            data=dataframes[f"tulu_w_coding_coding_{amt}_tulu_ta"],
            x="General Capabilities",
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
            x="General Capabilities",
            y="Coding Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[0]],
        )
        sns.scatterplot(
            data=dataframes[f"tulu_2_with_coding_tulu_none_coding_{amt}"],
            x="General Capabilities",
            y="Coding Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[2]],
        )
        sns.scatterplot(
            data=dataframes["llama_tulu_all_with_coding"],
            x="General Capabilities",
            y="Coding Average",
            hue="Label",
            s=point_size,
            marker=marker,
            palette=[sns.color_palette("colorblind")[3]],
        )
        # sns.scatterplot(
        #     data=dataframes[f"llama_tulu_all_coding_{amt}"],
        #     x="General Capabilities",
        #     y="Coding Average",
        #     hue="Label",
        #     s=point_size,
        #     marker=marker,
        #     palette=[sns.color_palette("colorblind")[4]],
        # )
        # sns.scatterplot(
        #     data=dataframes[f"tulu_2_tulu_match_coding_{amt}"],
        #     x="General Capabilities",
        #     y="Coding Average",
        #     hue="Label",
        #     s=point_size,
        #     marker=marker,
        #     palette=[sns.color_palette("colorblind")[6]],
        # )
        plt.legend()
        plt.xlabel("General Capabilities",fontsize=20)
        plt.ylabel("Coding Average",fontsize=20)
        plt.xticks([40, 45, 50, 55, 60], fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=11)

        plt.xlim(40, 60)

        plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        # plt.show()
        plt.savefig(f'results/consistent_mix/plots/tulu_w_coding_coding_{amt}.png', dpi=300, bbox_inches='tight')
        plt.clf()

    def compare_coding_amount_curves():
        # sns.lineplot(
        #     data=dataframes[f"coding_{amt}_interp"],
        #     x="General Capabilities",
        #     y="Coding Average",
        #     hue="Label",
        #     sort=False,
        #     # marker='X',
        #     linewidth=line_width,
        #     markersize=markersize,
        #     palette=[sns.color_palette("colorblind")[0]],
        # )
        sns.lineplot(
            data=dataframes[f"coding_20_llama_ta"],
            x="General Capabilities",
            y="Coding Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[0]],
        )
        sns.lineplot(
            data=dataframes[f"coding_40_llama_ta"],
            x="General Capabilities",
            y="Coding Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[1]],
        )
        sns.lineplot(
            data=dataframes[f"coding_60_llama_ta"],
            x="General Capabilities",
            y="Coding Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[2]],
        )
        sns.lineplot(
            data=dataframes[f"coding_80_llama_ta"],
            x="General Capabilities",
            y="Coding Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[3]],
        )
        sns.lineplot(
            data=dataframes[f"coding_100_llama_ta"],
            x="General Capabilities",
            y="Coding Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[4]],
        )
        plt.legend()
        plt.xlabel("General Capabilities",fontsize=20)
        plt.ylabel("Coding Average",fontsize=20)
        plt.xticks([40, 45, 50, 55, 60], fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=11)

        plt.xlim(40, 60)

        plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        # plt.show()
        plt.savefig(f'results/consistent_mix/plots/coding_llama_ta_all.png', dpi=300, bbox_inches='tight')
        plt.clf()

    def compare_science_amount_curves():
        sns.lineplot(
            data=dataframes[f"science_100_llama_ta"],
            x="General Capabilities",
            y="Science Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[0]],
        )
        sns.lineplot(
            data=dataframes[f"science_200_llama_ta"],
            x="General Capabilities",
            y="Science Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[1]],
        )
        sns.lineplot(
            data=dataframes[f"science_500_llama_ta"],
            x="General Capabilities",
            y="Science Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[2]],
        )
        sns.lineplot(
            data=dataframes[f"science_1000_llama_ta"],
            x="General Capabilities",
            y="Science Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[3]],
        )
        sns.lineplot(
            data=dataframes[f"science_2500_llama_ta"],
            x="General Capabilities",
            y="Science Average",
            hue="Label",
            sort=False,
            # marker='X',
            linewidth=line_width,
            markersize=markersize,
            palette=[sns.color_palette("colorblind")[4]],
        )
        plt.legend()
        plt.xlabel("General Capabilities",fontsize=20)
        plt.ylabel("Science Average",fontsize=20)
        # plt.xticks([40, 45, 50, 55, 60], fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=11)

        # plt.xlim(40, 60)

        plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        # plt.show()
        plt.savefig(f'results/consistent_mix/plots/science_llama_ta_all.png', dpi=300, bbox_inches='tight')
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
        compare_safety_curves(amt=amt, heuristic=True)
    compare_coding_curves()
    compare_coding_amount_curves()
    compare_science_amount_curves()
    science_interference_curves()
    compare_coding_curves_tulu_with_coding()
    compare_3_weighting_strategies()
    ta_curves_vs_baselines_all_3()
    ta_curves_heuristics()

make_plots()