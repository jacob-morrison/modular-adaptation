import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd 
from pprint import pprint 

def calculate_tulu_average(row, columns):
    subset_values = row[columns]
    return subset_values.mean()

def create_model_combo(row):
    model_dict = {
        "llama_2_7b": "Llama 2 7B",
        "tulu_none": "Tulu None",
        "tulu_match": "Tulu Match",
        # "tulu_all": "Tulu All",
        "coding_none": "Coding None",
        # "coding_50": "Coding 50%",
        "coding_100": "Coding 100%",
        "safety_100": "Safety 100%",
        "science_2500": "Science 2500",
        "tulu_all_no_science_no_safety_no_coding": "Tulu Consistent Mix",
        "tulu_all_no_science_no_safety": "Tulu Consistent Mix w/ Coding",
        "tulu_2_7b": "Tulu 2 7B Consistent Mix",
        "tulu_2_7b_no_science_no_safety_no_coding": "Tulu 2 7B Consistent Mix",
        "tulu_2_7b_with_coding": "Tulu 2 7B Consistent Mix w/ Coding",
        "tulu_2_7b_no_science_no_safety": "Tulu 2 7B Consistent Mix w/ Coding",
        "tulu_consistent_mix": "Tulu Consistent Mix",
        "tulu_consistent_mix_with_coding": "Tulu Consistent Mix w/ Coding",
        "tulu_match_no_science_no_safety_no_coding": "Tulu Match"
        # "tulu_v2_mix": "Full Tulu Mix",
        # "tulu_2_code_none": "Tulu 2 7B w/o Code Alpaca",
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
        print(tokens[2])
        domain_model = ""
        for token in tokens[2].split("-"):
            domain_model += " " + model_dict[token]

    if str(row["merge_method"]) == "nan":
        return f"{base_model} -> {tulu_model} & {domain_model.strip()}"
    else:
        return f"{base_model} -> {tulu_model} merged with {domain_model.strip()}, {row['merge_method']}".replace(" Tulu None", "")
    
models_to_skip = [
    "tulu_2_13b_retrain",
]

def get_df():
    df = pd.read_csv("results/consistent_mix/results.csv")

    tulu_columns_for_test_average = [
        "mmlu_0shot",
        "gsm_cot",
        "bbh_cot",
        "tydiqa_goldp_1shot",
        "truthfulqa",
        "alpaca_eval",
        # "codex_eval_temp_0.8",
        # "invert_toxigen",
    ]

    coding_columns_for_average = [
        # "codex_eval_plus_temp_0.1",
        "codex_eval_plus_temp_0.8",
        # "mbpp_temp_0.1",
        "mbpp_temp_0.8",
    ]

    safety_columns_for_average = [
        "invert_toxigen",
        "normalized_harmbench",
        "invert_unsafe_average",
        # "normalized_safe_average"
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
    df['Tulu Average'] = df.apply(lambda row: calculate_tulu_average(row, tulu_columns_for_test_average), axis=1)
    df['Safety Average'] = df.apply(lambda row: calculate_tulu_average(row, safety_columns_for_average), axis=1)
    df["Science Average"] = df.apply(lambda row: row["mean_null"], axis=1)
    df['Coding Average'] = df.apply(lambda row: calculate_tulu_average(row, coding_columns_for_average), axis=1)

    df['Combo'] = df.apply(lambda row: create_model_combo(row), axis=1)

    return df

def plot_baselines():
    df = get_raw_df()
    sns.scatterplot(data=df, x="Tulu Average (Tulu Subset)", y="Coding Average", hue="Combo", s=100)

    plt.legend()

    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    # plt.ylim(0.1, 0.4)

    plt.show()

def plot_all_curves():
    df = get_raw_df()

    weird_coding_ordering = {
        "coding_50": 1,
        "coding_100": 2,
        "coding_none": 3
    }

    merged_coding_models = {
        "coding_50",
        "coding_100",
    }

    linear_weighted = {
        "linear_weighted",
        # "task_arithmetic",
    }

    task_arithmetic = {
        "task_arithmetic"
    }

    baseline_keys = {
        "llama_2_7b-tulu_all-safety_none",

        "llama_2_7b-tulu_all-coding_50",
        "llama_2_7b-tulu_all-coding_100",

        "tulu_2_7b_continued_ft-tulu_none-coding_50",
        "tulu_2_7b_continued_ft-tulu_none-coding_100",
    }

    continued_ft_keys = {
        "tulu_2_code_none-tulu_none-coding_50",
        "tulu_2_code_none-tulu_none-coding_100",
    }

    continued_ft_mix_keys = {
        "tulu_2_code_none-tulu_match-coding_50",
        "tulu_2_code_none-tulu_match-coding_100",
    }

    # normalize these 4
    df["Order"] = df["domain_model_weight"]

    df_baselines = df[df["model_key"].isin(baseline_keys)]
    df_continued_ft = df[df["model_key"].isin(continued_ft_keys)]
    # df_continued_ft_mix = df[df["model_key"].isin(continued_ft_mix_keys)]

    df_lines = df[df["merge_method"] != "N/A"]
    # print(df_lines["safety_model"])
    df_linear_weighting = df_lines[df_lines["science_model"].isin(merged_coding_models)]
    df_task_arithmetic = df_lines[df_lines["science_model"].isin(merged_coding_models)]
    # print(df_lines)
    df_linear_weighting = df_linear_weighting[df_lines["merge_method"].isin(linear_weighted)]
    df_task_arithmetic = df_task_arithmetic[df_lines["merge_method"].isin(task_arithmetic)]

    df_linear_weighting.sort_values(by='Combo', inplace=True)
    df_linear_weighting.sort_values(by='Order', inplace=True)

    df_task_arithmetic.sort_values(by='Combo', inplace=True)
    df_task_arithmetic.sort_values(by='Order', inplace=True)

    # print(df_lines)

    df_baselines["Order"] = df_baselines.apply(lambda row: weird_coding_ordering[row["science_model"]], axis=1)
    df_baselines.sort_values(by='Order', inplace=True)
    df_continued_ft["Order"] = df_continued_ft.apply(lambda row: weird_coding_ordering[row["science_model"]], axis=1)
    df_continued_ft.sort_values(by='Order', inplace=True)
    # df_continued_ft_mix["Order"] = df_continued_ft_mix.apply(lambda row: weird_coding_ordering[row["science_model"]], axis=1)
    # df_continued_ft_mix.sort_values(by='Order', inplace=True)


    # write to csv
    df.to_csv("results/current/full_results.csv", index=False)

    tulu_subset = "Tulu Average (Tulu Subset)"
    # tulu_subset = "Tulu Average (w/o Toxigen)"

    sns.lineplot(data=df_linear_weighting, x=tulu_subset, y="Coding Average", hue="Combo", sort=False, marker='o', markersize=6)
    sns.lineplot(data=df_task_arithmetic, x=tulu_subset, y="Coding Average", hue="Combo", sort=False, marker='^', markersize=6)
    sns.scatterplot(data=df_baselines, x=tulu_subset, y="Coding Average", hue="Combo", s=100)
    sns.scatterplot(data=df_continued_ft, x=tulu_subset, y="Coding Average", hue="Combo", s=300, marker="*")
    # sns.scatterplot(data=df_continued_ft_mix, x="Tulu Average (Tulu Subset)", y="Coding Average", hue="Combo", s=100, marker="X")

    plt.legend()

    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    # plt.ylim(0.1, 0.4)

    plt.show()

def plot_science_curves():
    df = get_df()
    df["Order"] = df["domain_model_weight"]
    df.sort_values(by='Combo', inplace=True)
    df.sort_values(by='Order', inplace=True)
    df = df[~df["Combo"].str.contains("Safety 100") & ~df["Combo"].str.contains("Coding 100")]

    sns.lineplot(data=df, x="Tulu Average", y="Science Average", hue="Combo", sort=False, marker='o', markersize=6)
    # sns.lineplot(data=df, x="Tulu Average", y="Coding Average", hue="Combo", sort=False, marker='^', markersize=6)
    # sns.scatterplot(data=df, x="Tulu Average", y="Coding Average", hue="Combo", s=100)
    # sns.scatterplot(data=df, x="Tulu Average", y="Coding Average", hue="Combo", s=300, marker="*")
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    plt.show()

def plot_coding_curves():
    df = get_df()
    df["Order"] = df["domain_model_weight"]
    df.sort_values(by='Combo', inplace=True)
    df.sort_values(by='Order', inplace=True)
    df = df[~df["Combo"].str.contains("Science 2500") & ~df["Combo"].str.contains("Safety 100") & ~df["Combo"].str.contains("w/ Coding")]

    sns.lineplot(data=df, x="Tulu Average", y="Coding Average", hue="Combo", sort=False, marker='o', markersize=6)
    # sns.lineplot(data=df, x="Tulu Average", y="Coding Average", hue="Combo", sort=False, marker='^', markersize=6)
    # sns.scatterplot(data=df, x="Tulu Average", y="Coding Average", hue="Combo", s=100)
    # sns.scatterplot(data=df, x="Tulu Average", y="Coding Average", hue="Combo", s=300, marker="*")
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    plt.show()

def plot_safety_curves():
    df = get_df()
    df["Order"] = df["domain_model_weight"]
    df.sort_values(by='Combo', inplace=True)
    df.sort_values(by='Order', inplace=True)
    # df = df[(df["normalized_safe_average"] > 0) | ("safety_100" in df["model_key"])]
    # df = df[~df["domain_model"].isin({
        # "science_2500",
        # "coding_100"
    # })]
    df = df[~df["Combo"].str.contains("Science 2500") & ~df["Combo"].str.contains("Coding 100")]

    pd.set_option('display.max_colwidth', None)
    print(df["model_key"])
    # sns.lineplot(data=df, x="Tulu Average", y="Safety Average", hue="Combo", sort=False, marker='o', markersize=6)
    sns.lineplot(data=df, x="Exaggerated Refusals", y="Safety Average", hue="Combo", sort=False, marker='o', markersize=6)
    # sns.lineplot(data=df, x="Tulu Average", y="Exaggerated Refusals", hue="Combo", sort=False, marker='o', markersize=6)
    plt.legend()
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    plt.show()

# plot_science_curves()
plot_coding_curves()
# plot_safety_curves()