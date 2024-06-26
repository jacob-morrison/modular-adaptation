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
        "tulu_all": "Tulu All",
        "coding_none": "Coding None",
        "coding_50": "Coding 50%",
        "coding_100": "Coding 100%",
        "tulu_v2_mix": "Full Tulu Mix",
        "tulu_2_code_none": "Tulu 2 7B w/o Code Alpaca",
    }

    tokens = row["model_key"].split("-")
    if row["merge_method"] != "N/A":
        tokens = tokens[1:]
        tokens[1] = tokens[1][:-4]
        tokens[2] = tokens[2][:-4]
    if tokens[0] not in model_dict:
        print(tokens)
        print(row)
    if tokens[1][-1] == "_":
        tokens[1] = tokens[1][:-1]
    if tokens[2][-1] == "_":
        tokens[2] = tokens[2][:-1]
    base_model = model_dict[tokens[0]]
    tulu_model = model_dict[tokens[1]]
    coding_model = model_dict[tokens[2]]

    if len(tokens) == 4:
        tulu_model += f" Seed {tokens[3].split('_')[-1]}"

    if row["merge_method"] == "N/A":
        return f"{base_model} -> {tulu_model} & {coding_model}"
    else:
        return f"{base_model} -> {tulu_model} merged with {coding_model}, {row['merge_method']}"
    
models_to_skip = [
    "tulu_2_13b_retrain",
]

def get_raw_df():
    tulu_data = []
    with open("results/third-iteration/coding-results.jsonl") as f_in:
        with open("results/third-iteration/coding-results-manual.jsonl") as f_in2:
            for line in f_in.readlines() + f_in2.readlines():
                data = json.loads(line)
                model_key = data["model_key"]
                if model_key in models_to_skip:
                    continue
                tulu_data.append(data)   
     

    df = pd.DataFrame(tulu_data)

    tulu_columns_for_test_average = [
        "mmlu_0shot",
        "gsm_cot",
        "bbh_cot",
        "tydiqa_goldp_1shot",
        # "codex_eval_temp_0.8",
        "truthfulqa",
        "invert_toxigen",
    ]

    tulu_columns_for_test_average_without_toxigen = [
        "mmlu_0shot",
        "gsm_cot",
        "bbh_cot",
        "tydiqa_goldp_1shot",
        # "codex_eval_temp_0.8",
        "truthfulqa",
        # "invert_toxigen",
    ]

    tulu_columns_for_val_average = [
        "tydiqa_no_context_1shot",
        "mmlu_5shot",
        "bbh_direct",
        # "codex_eval_temp_0.1",
        "gsm_direct",
        "mmlu_5shot",
    ]

    coding_columns_for_average = [
        "codex_eval_plus_temp_0.1",
        "codex_eval_plus_temp_0.8",
        "mbpp_temp_0.1",
        "mbpp_temp_0.8",
    ]

    df['invert_toxigen'] = df.apply(lambda row: 1 - row["toxigen"], axis=1)
    df['alpaca_eval'] = df.apply(lambda row: row["alpaca_eval"] / 100, axis=1)
    df['tydiqa_no_context_1shot'] = df.apply(lambda row: row["tydiqa_no_context_1shot"] / 100, axis=1)
    df['tydiqa_goldp_1shot'] = df.apply(lambda row: row["tydiqa_goldp_1shot"] / 100, axis=1)

    # calculate averages
    df['Tulu Average (Other Evals)'] = df.apply(lambda row: calculate_tulu_average(row, tulu_columns_for_val_average), axis=1)
    df['Tulu Average (Tulu Subset)'] = df.apply(lambda row: calculate_tulu_average(row, tulu_columns_for_test_average), axis=1)
    df['Tulu Average (w/o Toxigen)'] = df.apply(lambda row: calculate_tulu_average(row, tulu_columns_for_test_average_without_toxigen), axis=1)
    df['Coding Average'] = df.apply(lambda row: calculate_tulu_average(row, coding_columns_for_average), axis=1)

    df['Combo'] = df.apply(lambda row: create_model_combo(row), axis=1)

    df.to_csv("results/third-iteration/coding_full_results.csv", index=False)

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
    df["Order"] = df["science_model_weight"]

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
    df.to_csv("results/third-iteration/full_results.csv", index=False)

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


plot_all_curves()