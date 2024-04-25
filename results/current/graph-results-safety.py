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
        "tulu_upsample": "Tulu Upsample",
        "llama_2_70b": "Llama 2 70B",
        "tulu_2_70b_continued_ft": "Tulu 2 70B",
        "llama_2_7b": "Llama 2 7B",
        "tulu_2_7b_continued_ft": "Tulu 2 7B",
        "tulu_none": "Tulu None",
        "tulu_match": "Tulu Match",
        "tulu_all": "Tulu All",
        "safety_v0_100": "Safety V0 100%",
        "safety_none": "Safety None",
        "safety_10": "Safety 10%",
        "safety_20": "Safety 20%",
        "safety_40": "Safety 40%",
        "safety_60": "Safety 60%",
        "safety_80": "Safety 80%",
        "safety_100": "Safety 100%",
        "safety_upsample": "Safety Upsample",
        "tulu_2_7b_uncensored": "Tulu 2 7B Uncensored",
        "tulu_2_7b_uncensored_safety_100": "Tulu 2 7B Uncensored c.t. Safety 100%",
        "tulu_2_7b_continued_ft_lora": "Tulu 2 7B Uncensored c.t. with Lora",
        "tulu": "Tulu All",
        "safet": "Safety X",
        "safety": "Safety X",
    }

    tokens = row["model_key"].split("-")
    if row["merge_method"] == "pareto":
        tokens[1] = "tulu_all"
        tokens[2] = "safety_2500"
    elif row["merge_method"] != "N/A":
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
    safety_model = model_dict[tokens[2]]

    if len(tokens) == 4:
        tulu_model += f" Seed {tokens[3].split('_')[-1]}"

    if row["merge_method"] == "N/A":
        return f"{base_model} -> {tulu_model} & {safety_model}"
    elif row["merge_method"] == "pareto":
        return f"{base_model} -> Tulu All & Safety 2500 Pareto Curve"
    else:
        return f"{base_model} -> {tulu_model} merged with {safety_model}, {row['merge_method']}"
    
models_to_skip = [
    "tulu_2_7b_uncensored-tulu_none-safety_v1_all",
    "tulu_2_7b_uncensored-tulu_match-safety_v0_100",
    "tulu-2-7b-safety-tulu-matched-adapt-3-epochs-real",
    "tulu-2-7b-bottom25-adapt-3-epochs"
    "tulu-2-7b-safety-adapt-1-epochs-real",
    "tulu-2-7b-uncensored-safety-adapt-1-epochs-real",
    "tulu-2-7b-uncensored-safety-v0.1-contrastive-1-epoch",
    "tulu-2-7b-safety-v0.1-contrastive-1-epoch",
    "tulu-2-7b-only-contrastive-1-epoch",
    "tulu-2-7b-uncensored-only-contrastive-1-epoch",
    "tulu-2-7b_safety_adapt_v0.1_contrastive_harmless_epoch1",
    "tulu-2-7b-uncensored_safety_adapt_v0.1_contrastive_combined_epoch1",
    "tulu-2-7b-bottom25-adapt-3-epochs",
    "tulu-2-7b-safety-adapt-1-epochs-real",
    "llama_2_7b-tulu_all-safety_v0_all",
    "llama_2_7b-tulu_all-safety_none_seed_123",
    "llama_2_7b-tulu_all-safety_none_seed_52830",
]

def get_raw_df():
    safety_results = {}
    with open("results/current/safety-evals.csv") as f_in:
        with open("results/current/manual-safety-evals.csv") as f_in2:
            # for baselines:
            merge_method = "N/A"
            i = 0
            for line in f_in.readlines() + f_in2.readlines()[1:]:
                line = line.strip().replace("_4096", "")
                curr_results = {}
                if i == 0: # model_key,base_model,tulu_model,tulu_model_weight,safety_model,safety_model_weight,merge_method,harmbench,safe_average,unsafe_average
                    keys = line.split(',')
                else:
                    curr_data = line.split(',')
                    model = curr_data[0]
                    if model in models_to_skip:
                        continue

                    for key, info in zip(keys, curr_data):
                        if key in [
                            "harmbench",
                            "safe_average",
                            "unsafe_average"
                        ]:
                            curr_results[key] = float(info)

                    safety_results[model] = curr_results
                i += 1

        # pprint(list(safety_results.keys()))
    # pprint(safety_results)
    # quit()

    tulu_data = []
    with open("results/current/tulu-evals-safety.jsonl") as f_in:
        with open("results/current/manual-tulu-evals-safety.jsonl") as f_in2:
            for line in f_in.readlines() + f_in2.readlines():
                data = json.loads(line.replace("_4096", ""))
                model_key = data["model_key"]
                if model_key in models_to_skip:
                    continue
                if model_key not in safety_results:
                    print(f"key not found: {model_key}")
                else:
                    print(f"key found: {model_key}")
                    for key in safety_results[model_key]:
                        data[key] = safety_results[model_key][key]
                tulu_data.append(data)   
     

    df = pd.DataFrame(tulu_data)

    tulu_columns_for_test_average = [
        "mmlu_0shot",
        "gsm_cot",
        "bbh_cot",
        "tydiqa_goldp_1shot",
        "codex_eval_temp_0.8",
        "truthfulqa",
    ]

    tulu_columns_for_val_average = [
        "tydiqa_no_context_1shot",
        "mmlu_5shot",
        "bbh_direct",
        "codex_eval_temp_0.1",
        "gsm_direct",
        "mmlu_5shot",
    ]

    safety_columns_for_average = [
        "invert_toxigen",
        "normalized_harmbench",
        "invert_unsafe_average",
        "normalized_safe_average"
    ]

    safety_columns_for_average_except_exaggerated = [
        "invert_toxigen",
        "normalized_harmbench",
        "invert_unsafe_average",
    ]

    df['invert_toxigen'] = df.apply(lambda row: 1 - row["toxigen"], axis=1)
    df['invert_unsafe_average'] = df.apply(lambda row: 1 - (row["unsafe_average"] / 100), axis=1)
    df["normalized_safe_average"] = df.apply(lambda row: row["safe_average"] / 100, axis=1)
    df["normalized_harmbench"] = df.apply(lambda row: 1 - (row["harmbench"] / 100), axis=1)
    df['alpaca_eval'] = df.apply(lambda row: row["alpaca_eval"] / 100, axis=1)
    df['tydiqa_no_context_1shot'] = df.apply(lambda row: row["tydiqa_no_context_1shot"] / 100, axis=1)
    df['tydiqa_goldp_1shot'] = df.apply(lambda row: row["tydiqa_goldp_1shot"] / 100, axis=1)

    # calculate averages
    df['Tulu Average (Other Evals)'] = df.apply(lambda row: calculate_tulu_average(row, tulu_columns_for_val_average), axis=1)
    df['Tulu Average (Tulu Subset)'] = df.apply(lambda row: calculate_tulu_average(row, tulu_columns_for_test_average), axis=1)
    df['Safety Average'] = df.apply(lambda row: calculate_tulu_average(row, safety_columns_for_average), axis=1)
    df['Safety Average (except exaggerated)'] = df.apply(lambda row: calculate_tulu_average(row, safety_columns_for_average_except_exaggerated), axis=1)

    df['Combo'] = df.apply(lambda row: create_model_combo(row), axis=1)

    df.to_csv("results/current/safety_full_results.csv", index=False)

    return df

def plot_baselines():
    df = get_raw_df()
    sns.scatterplot(data=df, x="Tulu Average (Tulu Subset)", y="Safety Average", hue="Combo", s=100)

    plt.legend()

    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    # plt.ylim(0.1, 0.4)

    plt.show()

def plot_safety_vs_tulu(safety_subset):
    if safety_subset == "Exaggerated Refusals":
        safety_subset = "normalized_safe_average"

    df = get_raw_df()

    weird_safety_ordering = {
        "safety_10": 1,
        "safety_20": 3,
        # "safety_40": 5,
        "safety_60": 2,
        # "safety_80": 6,
        "safety_100": 4,
        # "safety_upsample": 7,
        "safety_none": 5
    }

    merged_safety_models = {
        "safety_10",
        "safety_20",
        "safety_60",
        "safety_100",
    }

    merge_methods = {
        "linear_weighted",
        # "dare_linear",
        # "dare_ties",
        # "ties",
        # "slerp",
        # "pareto",
    }

    baseline_keys = {
        # "llama_2_7b-tulu_none-safety_20",
        # "llama_2_7b-tulu_none-safety_40",
        # "llama_2_7b-tulu_none-safety_60",
        # "llama_2_7b-tulu_none-safety_80",
        # "llama_2_7b-tulu_none-safety_100",
        # "llama_2_7b-tulu_none-safety_upsample",

        "llama_2_7b-tulu_all-safety_none",

        "llama_2_7b-tulu_match-safety_10",
        "llama_2_7b-tulu_match-safety_20",
        # "llama_2_7b-tulu_match-safety_40",
        "llama_2_7b-tulu_match-safety_60",
        # "llama_2_7b-tulu_match-safety_80",
        "llama_2_7b-tulu_match-safety_100",

        "llama_2_7b-tulu_all-safety_10",
        "llama_2_7b-tulu_all-safety_20",
        # "llama_2_7b-tulu_all-safety_40",
        "llama_2_7b-tulu_all-safety_60",
        "llama_2_7b-tulu_all-safety_100",
        # "llama_2_7b-tulu_all-safety_upsample",

        "tulu_2_7b_continued_ft-tulu_none-safety_10",
        "tulu_2_7b_continued_ft-tulu_none-safety_20",
        # "tulu_2_7b_continued_ft-tulu_none-safety_40",
        "tulu_2_7b_continued_ft-tulu_none-safety_60",
        # "tulu_2_7b_continued_ft-tulu_none-safety_80",
        "tulu_2_7b_continued_ft-tulu_none-safety_100",
        # "tulu_2_7b_continued_ft-tulu_none-safety_upsample",

        # "llama_2_7b-tulu_none-safety_1000-seed_123",
        # "llama_2_7b-tulu_none-safety_1000-seed_52830",
        # "llama_2_7b-tulu_all-safety_none-seed_123",
        # "llama_2_7b-tulu_all-safety_none-seed_52830",
    }

    continued_ft_keys = {
        "tulu_2_7b_uncensored-tulu_none-safety_10",
        "tulu_2_7b_uncensored-tulu_none-safety_20",
        # "tulu_2_7b_uncensored-tulu_none-safety_40",
        "tulu_2_7b_uncensored-tulu_none-safety_60",
        # "tulu_2_7b_uncensored-tulu_none-safety_80",
        "tulu_2_7b_uncensored-tulu_none-safety_100",
        # "tulu_2_7b_uncensored-tulu_none-safety_upsample",
    }

    continued_ft_lora_keys = {
        "tulu_2_7b_continued_ft_lora-tulu_none-safety_10",
        "tulu_2_7b_continued_ft_lora-tulu_none-safety_20",
        "tulu_2_7b_continued_ft_lora-tulu_none-safety_60",
        "tulu_2_7b_continued_ft_lora-tulu_none-safety_100",
    }


    continued_ft_mix_keys = {
        "tulu_2_7b_uncensored-tulu_match-safety_10",
        "tulu_2_7b_uncensored-tulu_match-safety_20",
        # "tulu_2_7b_uncensored-tulu_match-safety_40",
        "tulu_2_7b_uncensored-tulu_match-safety_60",
        # "tulu_2_7b_uncensored-tulu_match-safety_80",
        "tulu_2_7b_uncensored-tulu_match-safety_100",
    }

    # normalize these 4
    df["Order"] = df["science_model_weight"]

    df_baselines = df[df["model_key"].isin(baseline_keys)]
    df_continued_ft = df[df["model_key"].isin(continued_ft_keys)]
    df_continued_ft_lora = df[df["model_key"].isin(continued_ft_lora_keys)]
    df_continued_ft_mix = df[df["model_key"].isin(continued_ft_mix_keys)]

    df_lines = df[df["merge_method"] != "N/A"]
    # print(df_lines["safety_model"])
    df_lines = df_lines[df_lines["science_model"].isin(merged_safety_models)]
    # print(df_lines)
    df_lines = df_lines[df_lines["merge_method"].isin(merge_methods)]

    df_lines.sort_values(by='Combo', inplace=True)
    df_lines.sort_values(by='Order', inplace=True)

    print(df_lines)

    df_baselines["Order"] = df_baselines.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_baselines.sort_values(by='Order', inplace=True)
    df_continued_ft["Order"] = df_continued_ft.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_continued_ft.sort_values(by='Order', inplace=True)
    df_continued_ft_lora["Order"] = df_continued_ft_lora.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_continued_ft_lora.sort_values(by='Order', inplace=True)
    df_continued_ft_mix["Order"] = df_continued_ft_mix.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_continued_ft_mix.sort_values(by='Order', inplace=True)


    # write to csv
    df.to_csv("results/current/full_results.csv", index=False)

    sns.lineplot(data=df_lines, x="Tulu Average (Tulu Subset)", y=safety_subset, hue="Combo", sort=False, marker='o', markersize=6)
    sns.scatterplot(data=df_baselines, x="Tulu Average (Tulu Subset)", y=safety_subset, hue="Combo", s=100)
    sns.scatterplot(data=df_continued_ft, x="Tulu Average (Tulu Subset)", y=safety_subset, hue="Combo", s=300, marker="*")
    sns.scatterplot(data=df_continued_ft_lora, x="Tulu Average (Tulu Subset)", y=safety_subset, hue="Combo", s=100, marker="P")
    sns.scatterplot(data=df_continued_ft_mix, x="Tulu Average (Tulu Subset)", y=safety_subset, hue="Combo", s=100, marker="X")

    plt.legend()

    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    # plt.ylim(0.1, 0.4)

    plt.show()

def plot_safety_vs_exaggerated():
    if safety_subset == "Exaggerated Refusals":
        safety_subset = "normalized_safe_average"

    df = get_raw_df()

    weird_safety_ordering = {
        "safety_20": 3,
        "safety_40": 4,
        "safety_60": 2,
        "safety_80": 5,
        "safety_100": 1,
        "safety_upsample": 6,
        "safety_none": 7
    }

    merged_safety_models = {
        "safety_20",
        "safety_60",
        "safety_100",
    }

    merge_methods = {
        "linear_weighted",
        # "dare_linear",
        # "dare_ties",
        # "ties",
        # "slerp",
        # "pareto",
    }

    baseline_keys = {
        # "llama_2_7b-tulu_none-safety_20",
        # "llama_2_7b-tulu_none-safety_40",
        # "llama_2_7b-tulu_none-safety_60",
        # "llama_2_7b-tulu_none-safety_80",
        # "llama_2_7b-tulu_none-safety_100",
        # "llama_2_7b-tulu_none-safety_upsample",

        "llama_2_7b-tulu_all-safety_none",

        "llama_2_7b-tulu_match-safety_20",
        # "llama_2_7b-tulu_match-safety_40",
        "llama_2_7b-tulu_match-safety_60",
        # "llama_2_7b-tulu_match-safety_80",
        "llama_2_7b-tulu_match-safety_100",

        "llama_2_7b-tulu_all-safety_20",
        # "llama_2_7b-tulu_all-safety_40",
        "llama_2_7b-tulu_all-safety_60",
        "llama_2_7b-tulu_all-safety_100",
        # "llama_2_7b-tulu_all-safety_upsample",

        "tulu_2_7b_continued_ft-tulu_none-safety_20",
        # "tulu_2_7b_continued_ft-tulu_none-safety_40",
        "tulu_2_7b_continued_ft-tulu_none-safety_60",
        # "tulu_2_7b_continued_ft-tulu_none-safety_80",
        "tulu_2_7b_continued_ft-tulu_none-safety_100",
        # "tulu_2_7b_continued_ft-tulu_none-safety_upsample",

        # "llama_2_7b-tulu_none-safety_1000-seed_123",
        # "llama_2_7b-tulu_none-safety_1000-seed_52830",
        # "llama_2_7b-tulu_all-safety_none-seed_123",
        # "llama_2_7b-tulu_all-safety_none-seed_52830",
    }

    continued_ft_keys = {
        "tulu_2_7b_uncensored-tulu_none-safety_20",
        # "tulu_2_7b_uncensored-tulu_none-safety_40",
        "tulu_2_7b_uncensored-tulu_none-safety_60",
        # "tulu_2_7b_uncensored-tulu_none-safety_80",
        "tulu_2_7b_uncensored-tulu_none-safety_100",
        # "tulu_2_7b_uncensored-tulu_none-safety_upsample",
    }

    continued_ft_lora_keys = {
        "tulu_2_7b_continued_ft_lora-tulu_none-safety_10",
        "tulu_2_7b_continued_ft_lora-tulu_none-safety_20",
        "tulu_2_7b_continued_ft_lora-tulu_none-safety_60",
        "tulu_2_7b_continued_ft_lora-tulu_none-safety_100",
    }


    continued_ft_mix_keys = {
        "tulu_2_7b_uncensored-tulu_match-safety_20",
        # "tulu_2_7b_uncensored-tulu_match-safety_40",
        "tulu_2_7b_uncensored-tulu_match-safety_60",
        # "tulu_2_7b_uncensored-tulu_match-safety_80",
        "tulu_2_7b_uncensored-tulu_match-safety_100",
    }

    # normalize these 4
    df["Order"] = df["science_model_weight"]

    df_baselines = df[df["model_key"].isin(baseline_keys)]
    df_continued_ft = df[df["model_key"].isin(continued_ft_keys)]
    df_continued_ft_lora = df[df["model_key"].isin(continued_ft_lora_keys)]
    df_continued_ft_mix = df[df["model_key"].isin(continued_ft_mix_keys)]

    df_lines = df[df["merge_method"] != "N/A"]
    # print(df_lines["safety_model"])
    df_lines = df_lines[df_lines["science_model"].isin(merged_safety_models)]
    # print(df_lines)
    df_lines = df_lines[df_lines["merge_method"].isin(merge_methods)]

    df_lines.sort_values(by='Combo', inplace=True)
    df_lines.sort_values(by='Order', inplace=True)

    print(df_lines)

    df_baselines["Order"] = df_baselines.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_baselines.sort_values(by='Order', inplace=True)
    df_continued_ft["Order"] = df_continued_ft.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_continued_ft.sort_values(by='Order', inplace=True)
    df_continued_ft_lora["Order"] = df_continued_ft_lora.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_continued_ft_lora.sort_values(by='Order', inplace=True)
    df_continued_ft_mix["Order"] = df_continued_ft_mix.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_continued_ft_mix.sort_values(by='Order', inplace=True)


    # write to csv
    df.to_csv("results/current/full_results.csv", index=False)

    sns.lineplot(data=df_lines, x="normalized_safe_average", y="Safety Average (except exaggerated)", hue="Combo", sort=False, marker='o', markersize=6)
    sns.scatterplot(data=df_baselines, x="normalized_safe_average", y="Safety Average (except exaggerated)", hue="Combo", s=100)
    sns.scatterplot(data=df_continued_ft, x="normalized_safe_average", y="Safety Average (except exaggerated)", hue="Combo", s=300, marker="*")
    sns.scatterplot(data=df_continued_ft_lora, x="Tulu Average (Tulu Subset)", y=safety_subset, hue="Combo", s=100, marker="P")
    sns.scatterplot(data=df_continued_ft_mix, x="normalized_safe_average", y="Safety Average (except exaggerated)", hue="Combo", s=100, marker="X")

    plt.legend()

    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    # plt.ylim(0.1, 0.4)

    plt.show()

def plot_alpaca_vs_safety(safety_subset):
    if safety_subset == "Exaggerated Refusals":
        safety_subset = "normalized_safe_average"

    df = get_raw_df()

    weird_safety_ordering = {
        "safety_20": 3,
        "safety_40": 4,
        "safety_60": 2,
        "safety_80": 5,
        "safety_100": 1,
        "safety_upsample": 6,
        "safety_none": 7
    }

    merged_safety_models = {
        "safety_20",
        "safety_60",
        "safety_100",
    }

    merge_methods = {
        "linear_weighted",
        # "dare_linear",
        # "dare_ties",
        # "ties",
        # "slerp",
        # "pareto",
    }

    baseline_keys = {
        # "llama_2_7b-tulu_none-safety_20",
        # "llama_2_7b-tulu_none-safety_40",
        # "llama_2_7b-tulu_none-safety_60",
        # "llama_2_7b-tulu_none-safety_80",
        # "llama_2_7b-tulu_none-safety_100",
        # "llama_2_7b-tulu_none-safety_upsample",

        "llama_2_7b-tulu_all-safety_none",

        "llama_2_7b-tulu_match-safety_20",
        # "llama_2_7b-tulu_match-safety_40",
        "llama_2_7b-tulu_match-safety_60",
        # "llama_2_7b-tulu_match-safety_80",
        "llama_2_7b-tulu_match-safety_100",

        "llama_2_7b-tulu_all-safety_20",
        # "llama_2_7b-tulu_all-safety_40",
        "llama_2_7b-tulu_all-safety_60",
        "llama_2_7b-tulu_all-safety_100",
        # "llama_2_7b-tulu_all-safety_upsample",

        "tulu_2_7b_continued_ft-tulu_none-safety_20",
        # "tulu_2_7b_continued_ft-tulu_none-safety_40",
        "tulu_2_7b_continued_ft-tulu_none-safety_60",
        # "tulu_2_7b_continued_ft-tulu_none-safety_80",
        "tulu_2_7b_continued_ft-tulu_none-safety_100",
        # "tulu_2_7b_continued_ft-tulu_none-safety_upsample",

        # "llama_2_7b-tulu_none-safety_1000-seed_123",
        # "llama_2_7b-tulu_none-safety_1000-seed_52830",
        # "llama_2_7b-tulu_all-safety_none-seed_123",
        # "llama_2_7b-tulu_all-safety_none-seed_52830",
    }

    continued_ft_keys = {
        "tulu_2_7b_uncensored-tulu_none-safety_20",
        # "tulu_2_7b_uncensored-tulu_none-safety_40",
        "tulu_2_7b_uncensored-tulu_none-safety_60",
        # "tulu_2_7b_uncensored-tulu_none-safety_80",
        "tulu_2_7b_uncensored-tulu_none-safety_100",
        # "tulu_2_7b_uncensored-tulu_none-safety_upsample",
    }

    continued_ft_lora_keys = {
        "tulu_2_7b_continued_ft_lora-tulu_none-safety_10",
        "tulu_2_7b_continued_ft_lora-tulu_none-safety_20",
        "tulu_2_7b_continued_ft_lora-tulu_none-safety_60",
        "tulu_2_7b_continued_ft_lora-tulu_none-safety_100",
    }


    continued_ft_mix_keys = {
        "tulu_2_7b_uncensored-tulu_match-safety_20",
        # "tulu_2_7b_uncensored-tulu_match-safety_40",
        "tulu_2_7b_uncensored-tulu_match-safety_60",
        # "tulu_2_7b_uncensored-tulu_match-safety_80",
        "tulu_2_7b_uncensored-tulu_match-safety_100",
    }

    # normalize these 4
    df["Order"] = df["science_model_weight"]

    df_baselines = df[df["model_key"].isin(baseline_keys)]
    df_continued_ft = df[df["model_key"].isin(continued_ft_keys)]
    df_continued_ft_lora = df[df["model_key"].isin(continued_ft_lora_keys)]
    df_continued_ft_mix = df[df["model_key"].isin(continued_ft_mix_keys)]

    df_lines = df[df["merge_method"] != "N/A"]
    # print(df_lines["safety_model"])
    df_lines = df_lines[df_lines["science_model"].isin(merged_safety_models)]
    # print(df_lines)
    df_lines = df_lines[df_lines["merge_method"].isin(merge_methods)]

    df_lines.sort_values(by='Combo', inplace=True)
    df_lines.sort_values(by='Order', inplace=True)

    print(df_lines)

    df_baselines["Order"] = df_baselines.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_baselines.sort_values(by='Order', inplace=True)
    df_continued_ft["Order"] = df_continued_ft.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_continued_ft.sort_values(by='Order', inplace=True)
    df_continued_ft_lora["Order"] = df_continued_ft_lora.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_continued_ft_lora.sort_values(by='Order', inplace=True)
    df_continued_ft_mix["Order"] = df_continued_ft_mix.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_continued_ft_mix.sort_values(by='Order', inplace=True)


    # write to csv
    df.to_csv("results/current/full_results.csv", index=False)

    sns.lineplot(data=df_lines, x="alpaca_eval", y=safety_subset, hue="Combo", sort=False, marker='o', markersize=6)
    sns.scatterplot(data=df_baselines, x="alpaca_eval", y=safety_subset, hue="Combo", s=100)
    sns.scatterplot(data=df_continued_ft, x="alpaca_eval", y=safety_subset, hue="Combo", s=300, marker="*")
    sns.scatterplot(data=df_continued_ft_lora, x="Tulu Average (Tulu Subset)", y=safety_subset, hue="Combo", s=100, marker="P")
    sns.scatterplot(data=df_continued_ft_mix, x="alpaca_eval", y=safety_subset, hue="Combo", s=100, marker="X")

    plt.legend()

    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    # plt.ylim(0.1, 0.4)

    plt.show()

def plot_tulu_vs_alpaca_eval():
    if safety_subset == "Exaggerated Refusals":
        safety_subset = "normalized_safe_average"

    df = get_raw_df()

    weird_safety_ordering = {
        "safety_20": 3,
        "safety_40": 4,
        "safety_60": 2,
        "safety_80": 5,
        "safety_100": 1,
        "safety_upsample": 6,
        "safety_none": 7
    }

    merged_safety_models = {
        "safety_20",
        "safety_60",
        "safety_100",
    }

    merge_methods = {
        "linear_weighted",
        # "dare_linear",
        # "dare_ties",
        # "ties",
        # "slerp",
        # "pareto",
    }

    baseline_keys = {
        # "llama_2_7b-tulu_none-safety_20",
        # "llama_2_7b-tulu_none-safety_40",
        # "llama_2_7b-tulu_none-safety_60",
        # "llama_2_7b-tulu_none-safety_80",
        # "llama_2_7b-tulu_none-safety_100",
        # "llama_2_7b-tulu_none-safety_upsample",

        "llama_2_7b-tulu_all-safety_none",

        "llama_2_7b-tulu_match-safety_20",
        # "llama_2_7b-tulu_match-safety_40",
        "llama_2_7b-tulu_match-safety_60",
        # "llama_2_7b-tulu_match-safety_80",
        "llama_2_7b-tulu_match-safety_100",

        "llama_2_7b-tulu_all-safety_20",
        # "llama_2_7b-tulu_all-safety_40",
        "llama_2_7b-tulu_all-safety_60",
        "llama_2_7b-tulu_all-safety_100",
        # "llama_2_7b-tulu_all-safety_upsample",

        "tulu_2_7b_continued_ft-tulu_none-safety_20",
        # "tulu_2_7b_continued_ft-tulu_none-safety_40",
        "tulu_2_7b_continued_ft-tulu_none-safety_60",
        # "tulu_2_7b_continued_ft-tulu_none-safety_80",
        "tulu_2_7b_continued_ft-tulu_none-safety_100",
        # "tulu_2_7b_continued_ft-tulu_none-safety_upsample",

        # "llama_2_7b-tulu_none-safety_1000-seed_123",
        # "llama_2_7b-tulu_none-safety_1000-seed_52830",
        # "llama_2_7b-tulu_all-safety_none-seed_123",
        # "llama_2_7b-tulu_all-safety_none-seed_52830",
    }

    continued_ft_keys = {
        "tulu_2_7b_uncensored-tulu_none-safety_20",
        # "tulu_2_7b_uncensored-tulu_none-safety_40",
        "tulu_2_7b_uncensored-tulu_none-safety_60",
        # "tulu_2_7b_uncensored-tulu_none-safety_80",
        "tulu_2_7b_uncensored-tulu_none-safety_100",
        # "tulu_2_7b_uncensored-tulu_none-safety_upsample",
    }

    continued_ft_lora_keys = {
        "tulu_2_7b_continued_ft_lora-tulu_none-safety_10",
        "tulu_2_7b_continued_ft_lora-tulu_none-safety_20",
        "tulu_2_7b_continued_ft_lora-tulu_none-safety_60",
        "tulu_2_7b_continued_ft_lora-tulu_none-safety_100",
    }

    continued_ft_mix_keys = {
        "tulu_2_7b_uncensored-tulu_match-safety_20",
        # "tulu_2_7b_uncensored-tulu_match-safety_40",
        "tulu_2_7b_uncensored-tulu_match-safety_60",
        # "tulu_2_7b_uncensored-tulu_match-safety_80",
        "tulu_2_7b_uncensored-tulu_match-safety_100",
    }

    # normalize these 4
    df["Order"] = df["science_model_weight"]

    df_baselines = df[df["model_key"].isin(baseline_keys)]
    df_continued_ft = df[df["model_key"].isin(continued_ft_keys)]
    df_continued_ft_lora = df[df["model_key"].isin(continued_ft_lora_keys)]
    df_continued_ft_mix = df[df["model_key"].isin(continued_ft_mix_keys)]

    df_lines = df[df["merge_method"] != "N/A"]
    # print(df_lines["safety_model"])
    df_lines = df_lines[df_lines["science_model"].isin(merged_safety_models)]
    # print(df_lines)
    df_lines = df_lines[df_lines["merge_method"].isin(merge_methods)]

    df_lines.sort_values(by='Combo', inplace=True)
    df_lines.sort_values(by='Order', inplace=True)

    print(df_lines)

    df_baselines["Order"] = df_baselines.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_baselines.sort_values(by='Order', inplace=True)
    df_continued_ft["Order"] = df_continued_ft.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_continued_ft.sort_values(by='Order', inplace=True)
    df_continued_ft_lora["Order"] = df_continued_ft_lora.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_continued_ft_lora.sort_values(by='Order', inplace=True)
    df_continued_ft_mix["Order"] = df_continued_ft_mix.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_continued_ft_mix.sort_values(by='Order', inplace=True)


    # write to csv
    df.to_csv("results/current/full_results.csv", index=False)

    sns.lineplot(data=df_lines, x="Tulu Average (Tulu Subset)", y="alpaca_eval", hue="Combo", sort=False, marker='o', markersize=6)
    sns.scatterplot(data=df_baselines, x="Tulu Average (Tulu Subset)", y="alpaca_eval", hue="Combo", s=100)
    sns.scatterplot(data=df_continued_ft, x="Tulu Average (Tulu Subset)", y="alpaca_eval", hue="Combo", s=300, marker="*")
    sns.scatterplot(data=df_continued_ft_lora, x="Tulu Average (Tulu Subset)", y=safety_subset, hue="Combo", s=100, marker="P")
    sns.scatterplot(data=df_continued_ft_mix, x="Tulu Average (Tulu Subset)", y="alpaca_eval", hue="Combo", s=100, marker="X")

    plt.legend()

    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    # plt.ylim(0.1, 0.4)

    plt.show()

def plot_compare_merge_methods(safety_subset):
    if safety_subset == "Exaggerated Refusals":
        safety_subset = "normalized_safe_average"

    df = get_raw_df()

    weird_safety_ordering = {
        "safety_10": 8,
        "safety_20": 3,
        "safety_40": 4,
        "safety_60": 2,
        "safety_80": 5,
        "safety_100": 1,
        "safety_upsample": 6,
        "safety_none": 7
    }

    merged_safety_models = {
        # "safety_v0_100",
        # "tulu_2_7b_uncensored_safety_100",
        "safety_10",
        # "safety_20",
        # "safety_60",
        "safety_100",
    }

    merge_methods = {
        "linear_weighted",
        "task_arithmetic",
        # "dare_linear",
        # "dare_ties",
        # "ties",
        # "slerp",
        # "pareto",
    }

    baseline_keys = {
        # "llama_2_7b-tulu_none-safety_20",
        # "llama_2_7b-tulu_none-safety_40",
        # "llama_2_7b-tulu_none-safety_60",
        # "llama_2_7b-tulu_none-safety_80",
        # "llama_2_7b-tulu_none-safety_100",
        # "llama_2_7b-tulu_none-safety_upsample",

        "llama_2_7b-tulu_all-safety_none",

        # "llama_2_7b-tulu_match-safety_10",
        # "llama_2_7b-tulu_match-safety_40",
        # "llama_2_7b-tulu_match-safety_60",
        # "llama_2_7b-tulu_match-safety_80",
        "llama_2_7b-tulu_match-safety_100",

        # "llama_2_7b-tulu_all-safety_10",
        # "llama_2_7b-tulu_all-safety_20",
        # "llama_2_7b-tulu_all-safety_40",
        # "llama_2_7b-tulu_all-safety_60",
        "llama_2_7b-tulu_all-safety_100",
        # "llama_2_7b-tulu_all-safety_upsample",

        # "tulu_2_7b_continued_ft-tulu_none-safety_10",
        # "tulu_2_7b_continued_ft-tulu_none-safety_20",
        # "tulu_2_7b_continued_ft-tulu_none-safety_40",
        # "tulu_2_7b_continued_ft-tulu_none-safety_60",
        # "tulu_2_7b_continued_ft-tulu_none-safety_80",
        "tulu_2_7b_continued_ft-tulu_none-safety_100",
        # "tulu_2_7b_continued_ft-tulu_none-safety_upsample",

        # "llama_2_7b-tulu_none-safety_1000-seed_123",
        # "llama_2_7b-tulu_none-safety_1000-seed_52830",
        # "llama_2_7b-tulu_all-safety_none-seed_123",
        # "llama_2_7b-tulu_all-safety_none-seed_52830",
    }

    continued_ft_keys = {
        # "tulu_2_7b_uncensored-tulu_none-safety_10",
        # "tulu_2_7b_uncensored-tulu_none-safety_20",
        # "tulu_2_7b_uncensored-tulu_none-safety_40",
        # "tulu_2_7b_uncensored-tulu_none-safety_60",
        # "tulu_2_7b_uncensored-tulu_none-safety_80",
        "tulu_2_7b_uncensored-tulu_none-safety_100",
        # "tulu_2_7b_uncensored-tulu_none-safety_upsample",
    }

    continued_ft_lora_keys = {
        # "tulu_2_7b_continued_ft_lora-tulu_none-safety_10",
        # "tulu_2_7b_continued_ft_lora-tulu_none-safety_20",
        # "tulu_2_7b_continued_ft_lora-tulu_none-safety_60",
        "tulu_2_7b_continued_ft_lora-tulu_none-safety_100",
    }

    continued_ft_mix_keys = {
        # "tulu_2_7b_uncensored-tulu_match-safety_10",
        # "tulu_2_7b_uncensored-tulu_match-safety_20",
        # "tulu_2_7b_uncensored-tulu_match-safety_40",
        # "tulu_2_7b_uncensored-tulu_match-safety_60",
        # "tulu_2_7b_uncensored-tulu_match-safety_80",
        "tulu_2_7b_uncensored-tulu_match-safety_100",
    }

    merge_data_weighted_linear_keys = {
        # "data_weighted_linear-llama_2_7b-tulu_all-safety_10",
        # "data_weighted_linear-llama_2_7b-tulu_all-safety_20",
        # "data_weighted_linear-llama_2_7b-tulu_all-safety_60",
        "data_weighted_linear-llama_2_7b-tulu_all-safety_100",
    }

    merge_data_weighted_ta_keys = {
        # "data_weighted_task_arithmetic-llama_2_7b-tulu_all-safety_10",
        # "data_weighted_task_arithmetic-llama_2_7b-tulu_all-safety_20",
        # "data_weighted_task_arithmetic-llama_2_7b-tulu_all-safety_60",
        "data_weighted_task_arithmetic-llama_2_7b-tulu_all-safety_100",
    }

    # normalize these 4
    df["Order"] = df["science_model_weight"]

    df_baselines = df[df["model_key"].isin(baseline_keys)]
    df_continued_ft = df[df["model_key"].isin(continued_ft_keys)]
    df_continued_ft_lora = df[df["model_key"].isin(continued_ft_lora_keys)]
    df_continued_ft_mix = df[df["model_key"].isin(continued_ft_mix_keys)]
    df_merge_data_weighted_linear = df[df["model_key"].isin(merge_data_weighted_linear_keys)]
    df_merge_data_weighted_task_arithmetic = df[df["model_key"].isin(merge_data_weighted_ta_keys)]

    df_lines = df[df["merge_method"] != "N/A"]
    # print(df_lines["safety_model"])
    df_lines = df_lines[df_lines["science_model"].isin(merged_safety_models)]
    # print(df_lines)
    df_lines = df_lines[df_lines["merge_method"].isin(merge_methods)]

    df_lines.sort_values(by='Combo', inplace=True)
    df_lines.sort_values(by='Order', inplace=True)

    print(df_lines)

    df_baselines["Order"] = df_baselines.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_baselines.sort_values(by='Order', inplace=True)
    df_continued_ft["Order"] = df_continued_ft.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_continued_ft.sort_values(by='Order', inplace=True)
    df_continued_ft_lora["Order"] = df_continued_ft_lora.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_continued_ft_lora.sort_values(by='Order', inplace=True)
    df_continued_ft_mix["Order"] = df_continued_ft_mix.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_continued_ft_mix.sort_values(by='Order', inplace=True)
    df_merge_data_weighted_linear["Order"] = df_merge_data_weighted_linear.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_merge_data_weighted_linear.sort_values(by='Order', inplace=True)
    df_merge_data_weighted_task_arithmetic["Order"] = df_merge_data_weighted_task_arithmetic.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_merge_data_weighted_task_arithmetic.sort_values(by='Order', inplace=True)


    # write to csv
    df.to_csv("results/current/full_results.csv", index=False)

    sns.lineplot(data=df_lines, x="Tulu Average (Tulu Subset)", y=safety_subset, hue="Combo", sort=False, marker='o', markersize=6)
    sns.scatterplot(data=df_baselines, x="Tulu Average (Tulu Subset)", y=safety_subset, hue="Combo", s=100)
    sns.scatterplot(data=df_continued_ft, x="Tulu Average (Tulu Subset)", y=safety_subset, hue="Combo", s=300, marker="*")
    sns.scatterplot(data=df_continued_ft_lora, x="Tulu Average (Tulu Subset)", y=safety_subset, hue="Combo", s=100, marker="P")
    sns.scatterplot(data=df_continued_ft_mix, x="Tulu Average (Tulu Subset)", y=safety_subset, hue="Combo", s=100, marker="X")
    sns.scatterplot(data=df_merge_data_weighted_linear, x="Tulu Average (Tulu Subset)", y=safety_subset, hue="Combo", s=100, marker="o")
    sns.scatterplot(data=df_merge_data_weighted_task_arithmetic, x="Tulu Average (Tulu Subset)", y=safety_subset, hue="Combo", s=100, marker="o")

    plt.legend()

    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    # plt.ylim(0.1, 0.4)

    plt.show()

def plot_all_curves(safety_subset):
    if safety_subset == "Exaggerated Refusals":
        safety_subset = "normalized_safe_average"

    df = get_raw_df()

    weird_safety_ordering = {
        "safety_10": 8,
        "safety_20": 3,
        "safety_40": 4,
        "safety_60": 2,
        "safety_80": 5,
        "safety_100": 1,
        "safety_upsample": 6,
        "safety_none": 7
    }

    merged_safety_models = {
        "safety_v0_100",
        "tulu_2_7b_uncensored_safety_100",
        "safety_10",
        "safety_20",
        "safety_60",
        "safety_100",
    }

    merge_methods = {
        "linear_weighted",
        "task_arithmetic",
        # "dare_linear",
        # "dare_ties",
        # "ties",
        # "slerp",
        # "pareto",
    }

    baseline_keys = {
        # "llama_2_7b-tulu_none-safety_20",
        # "llama_2_7b-tulu_none-safety_40",
        # "llama_2_7b-tulu_none-safety_60",
        # "llama_2_7b-tulu_none-safety_80",
        # "llama_2_7b-tulu_none-safety_100",
        # "llama_2_7b-tulu_none-safety_upsample",

        "llama_2_7b-tulu_all-safety_none",

        "llama_2_7b-tulu_match-safety_20",
        # "llama_2_7b-tulu_match-safety_40",
        "llama_2_7b-tulu_match-safety_60",
        # "llama_2_7b-tulu_match-safety_80",
        "llama_2_7b-tulu_match-safety_100",

        "llama_2_7b-tulu_all-safety_20",
        # "llama_2_7b-tulu_all-safety_40",
        "llama_2_7b-tulu_all-safety_60",
        "llama_2_7b-tulu_all-safety_100",
        # "llama_2_7b-tulu_all-safety_upsample",

        "tulu_2_7b_continued_ft-tulu_none-safety_20",
        # "tulu_2_7b_continued_ft-tulu_none-safety_40",
        "tulu_2_7b_continued_ft-tulu_none-safety_60",
        # "tulu_2_7b_continued_ft-tulu_none-safety_80",
        "tulu_2_7b_continued_ft-tulu_none-safety_100",
        # "tulu_2_7b_continued_ft-tulu_none-safety_upsample",

        # "llama_2_7b-tulu_none-safety_1000-seed_123",
        # "llama_2_7b-tulu_none-safety_1000-seed_52830",
        # "llama_2_7b-tulu_all-safety_none-seed_123",
        # "llama_2_7b-tulu_all-safety_none-seed_52830",
    }

    continued_ft_keys = {
        "tulu_2_7b_uncensored-tulu_none-safety_20",
        # "tulu_2_7b_uncensored-tulu_none-safety_40",
        "tulu_2_7b_uncensored-tulu_none-safety_60",
        # "tulu_2_7b_uncensored-tulu_none-safety_80",
        "tulu_2_7b_uncensored-tulu_none-safety_100",
        # "tulu_2_7b_uncensored-tulu_none-safety_upsample",
    }

    continued_ft_lora_keys = {
        "tulu_2_7b_continued_ft_lora-tulu_none-safety_10",
        "tulu_2_7b_continued_ft_lora-tulu_none-safety_20",
        "tulu_2_7b_continued_ft_lora-tulu_none-safety_60",
        "tulu_2_7b_continued_ft_lora-tulu_none-safety_100",
    }

    continued_ft_mix_keys = {
        "tulu_2_7b_uncensored-tulu_match-safety_20",
        # "tulu_2_7b_uncensored-tulu_match-safety_40",
        "tulu_2_7b_uncensored-tulu_match-safety_60",
        # "tulu_2_7b_uncensored-tulu_match-safety_80",
        "tulu_2_7b_uncensored-tulu_match-safety_100",
    }

    # normalize these 4
    df["Order"] = df["science_model_weight"]

    df_baselines = df[df["model_key"].isin(baseline_keys)]
    df_continued_ft = df[df["model_key"].isin(continued_ft_keys)]
    df_continued_ft_lora = df[df["model_key"].isin(continued_ft_lora_keys)]
    df_continued_ft_mix = df[df["model_key"].isin(continued_ft_mix_keys)]

    df_lines = df[df["merge_method"] != "N/A"]
    # print(df_lines["safety_model"])
    df_lines = df_lines[df_lines["science_model"].isin(merged_safety_models)]
    # print(df_lines)
    df_lines = df_lines[df_lines["merge_method"].isin(merge_methods)]

    df_lines.sort_values(by='Combo', inplace=True)
    df_lines.sort_values(by='Order', inplace=True)

    print(df_lines)

    df_baselines["Order"] = df_baselines.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_baselines.sort_values(by='Order', inplace=True)
    df_continued_ft["Order"] = df_continued_ft.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_continued_ft.sort_values(by='Order', inplace=True)
    df_continued_ft_lora["Order"] = df_continued_ft_lora.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_continued_ft_lora.sort_values(by='Order', inplace=True)
    df_continued_ft_mix["Order"] = df_continued_ft_mix.apply(lambda row: weird_safety_ordering[row["science_model"]], axis=1)
    df_continued_ft_mix.sort_values(by='Order', inplace=True)


    # write to csv
    df.to_csv("results/current/full_results.csv", index=False)

    sns.lineplot(data=df_lines, x="Tulu Average (Tulu Subset)", y=safety_subset, hue="Combo", sort=False, marker='o', markersize=6)
    # sns.scatterplot(data=df_baselines, x="Tulu Average (Tulu Subset)", y=safety_subset, hue="Combo", s=100)
    # sns.scatterplot(data=df_continued_ft, x="Tulu Average (Tulu Subset)", y=safety_subset, hue="Combo", s=300, marker="*")
    # sns.scatterplot(data=df_continued_ft_lora, x="Tulu Average (Tulu Subset)", y=safety_subset, hue="Combo", s=100, marker="P")
    # sns.scatterplot(data=df_continued_ft_mix, x="Tulu Average (Tulu Subset)", y=safety_subset, hue="Combo", s=100, marker="X")

    plt.legend()

    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    # plt.ylim(0.1, 0.4)

    plt.show()

safety_subset = "Safety Average"
# safety_subset = "Safety Average (except exaggerated)"
# safety_subset = "Exaggerated Refusals"

# plot_baselines()
# plot_safety_vs_tulu(safety_subset)
# plot_alpaca_vs_safety(safety_subset)
# plot_safety_vs_exaggerated()
# plot_tulu_vs_alpaca_eval()
# plot_all_curves(safety_subset)
plot_compare_merge_methods(safety_subset)