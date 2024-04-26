import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd 

def calculate_tulu_average(row, columns):
    subset_values = row[columns]
    return subset_values.mean()

def create_model_combo(row):
    model_dict = {
        "llama_2_70b": "Llama 2 70B",
        "tulu_2_70b_continued_ft": "Tulu 2 70B",
        "llama_2_7b": "Llama 2 7B",
        "tulu_2_7b_continued_ft": "Tulu 2 7B",
        "tulu_none": "Tulu None",
        "tulu_match": "Tulu Match",
        "tulu_all": "Tulu All",
        "science_none": "Science None",
        "science_100": "Science 100",
        "science_200": "Science 200",
        "science_500": "Science 500",
        "science_1000": "Science 1000",
        "science_2500": "Science 2500",
        "science_upsample": "Science Upsample",
        "science_2500_minus_science_2500": "Science 2500 Minus Science 2500",
        "science_2500_minus_tulu": "Science 2500 Minus Tulu",
        "science_2500_": "Science 2500 Data Weighted",
        "tulu_all_": "Tulu All Data Weighted",
        "tulu": "Tulu All",
        "science_": "Science 2500",
        "tulu_2_7b_science_2500": "Tulu 2 7B c.ft. Science 2500"
    }

    tokens = row["model_key"].split("-")
    if row["merge_method"] == "pareto":
        tokens[1] = "tulu_all"
        tokens[2] = "science_2500"
    elif row["merge_method"] != "N/A":
        tokens = tokens[1:]
        tokens[1] = tokens[1][:-4]
        tokens[2] = tokens[2][:-4]
    if tokens[0] not in model_dict:
        print(tokens)
        print(row)
    base_model = model_dict[tokens[0]]
    tulu_model = model_dict[tokens[1]]
    science_model = model_dict[tokens[2]]

    if len(tokens) == 4:
        tulu_model += f" Seed {tokens[3].split('_')[-1]}"

    if row["merge_method"] == "N/A":
        return f"{base_model} -> {tulu_model} & {science_model}"
    elif row["merge_method"] == "pareto":
        return f"{base_model} -> Tulu All & Science 2500 Pareto Curve"
    else:
        return f"{base_model} -> {tulu_model} merged with {science_model}, {row['merge_method']}"

def get_raw_df():
    science_results = {}
    with open("results/current/science-evals.csv") as f_in:
        with open("results/current/manual-science-evals.csv") as f_in2:
            # for baselines:
            merge_method = "N/A"
            i = 0
            for line in f_in.readlines() + f_in2.readlines()[2:]:
                line = line.strip().replace("_4096", "")
                curr_results = {}
                if i == 0: # task,bioasq,biored,discomat,evidence_inference,evidence_inference,evidence_inference,multicite,mup,qasper,scierc,scifact,scifact,scifact,mean,median
                    tasks = line.split(',')[1:]
                elif i == 1: # metric,f1,f1,bleu,f1_exact,f1_overlap,f1_substring,f1,bleu,bleu,f1,f1_evidence_sent,f1_evidence_token,f1_label
                    metrics = line.split(',')[1:]
                else:
                    curr_data = line.split(',')
                    model = curr_data[0]
                    model_tokens = model.split('-')

                    for task, metric, value in zip(tasks, metrics, curr_data[1:]):
                        if value == '':
                            value = 0.0
                        curr_results[task] = {
                            "metric": metric,
                            "value": float(value)
                        }

                    science_results[model] = curr_results
                i += 1

    tulu_data = []
    with open("results/current/tulu-evals.jsonl") as f_in:
        with open("results/current/manual-tulu-evals.jsonl") as f_in2:
            for line in f_in.readlines() + f_in2.readlines():
                data = json.loads(line.replace("_4096", ""))
                model_key = data["model_key"]
                if model_key not in science_results:
                    print(f"key not found: {model_key}")
                else:
                    for task in science_results[model_key]:
                        data[task] = science_results[model_key][task]["value"]
                tulu_data.append(data)        

    df = pd.DataFrame(tulu_data)

    tulu_columns_for_test_average = [
        "mmlu_0shot",
        "gsm_cot",
        "bbh_cot",
        "tydiqa_goldp_1shot",
        "codex_eval_temp_0.8",
        # "alpaca_farm",
        "invert_toxigen",
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

    df['invert_toxigen'] = df.apply(lambda row: 1 - row["toxigen"], axis=1)
    df['alpaca_farm'] = df.apply(lambda row: row["alpaca_farm"] / 100, axis=1)
    df['tydiqa_no_context_1shot'] = df.apply(lambda row: row["tydiqa_no_context_1shot"] / 100, axis=1)
    df['tydiqa_goldp_1shot'] = df.apply(lambda row: row["tydiqa_goldp_1shot"] / 100, axis=1)

    # calculate averages
    df['Tulu Average (Other Evals)'] = df.apply(lambda row: calculate_tulu_average(row, tulu_columns_for_val_average), axis=1)
    df['Tulu Average (Tulu Subset)'] = df.apply(lambda row: calculate_tulu_average(row, tulu_columns_for_test_average), axis=1)
    df['Science Average'] = df['mean']

    df['Combo'] = df.apply(lambda row: create_model_combo(row), axis=1)

    return df

def plot_linear_merge_vs_baselines():
    df = get_raw_df()

    weird_science_ordering = {
        "science_100": 6,
        "science_200": 4,
        "science_500": 2,
        "science_1000": 5,
        "science_2500": 3,
        "science_upsample": 1,
        "science_none": 7,
    }

    merged_science_models = {
        # "science_100",
        # "science_200",
        # "science_500",
        # "science_1000",
        "science_2500",
        # "science_upsample",
        "tulu_2_7b_science_2500",
    }

    merge_methods = {
        "linear_weighted",
        "dare_linear",
        "dare_ties",
        # "ties",
        "slerp",
        # "pareto",
        "task_arithmetic",
    }

    baseline_keys = {
        # "llama_2_7b-tulu_none-science_100",
        # "llama_2_7b-tulu_none-science_200",
        # "llama_2_7b-tulu_none-science_500",
        # "llama_2_7b-tulu_none-science_1000",
        # "llama_2_7b-tulu_none-science_2500",
        # "llama_2_7b-tulu_none-science_upsample",

        # "llama_2_7b-tulu_all-science_none-seed_42",

        # "llama_2_7b-tulu_match-science_100",
        # "llama_2_7b-tulu_match-science_200",
        # "llama_2_7b-tulu_match-science_500",
        # "llama_2_7b-tulu_match-science_1000",
        "llama_2_7b-tulu_match-science_2500",

        # "llama_2_7b-tulu_all-science_100",
        # "llama_2_7b-tulu_all-science_200",
        # "llama_2_7b-tulu_all-science_500",
        # "llama_2_7b-tulu_all-science_1000",
        "llama_2_7b-tulu_all-science_2500",
        # "llama_2_7b-tulu_all-science_upsample",

        # "tulu_2_7b_continued_ft-tulu_none-science_100",
        # "tulu_2_7b_continued_ft-tulu_none-science_200",
        # "tulu_2_7b_continued_ft-tulu_none-science_500",
        # "tulu_2_7b_continued_ft-tulu_none-science_1000",
        # "tulu_2_7b_continued_ft-tulu_none-science_2500",

        # "llama_2_7b-tulu_none-science_1000-seed_123",
        # "llama_2_7b-tulu_none-science_1000-seed_52830",
        # "llama_2_7b-tulu_all-science_none-seed_123",
        # "llama_2_7b-tulu_all-science_none-seed_52830",
    }

    continued_ft_keys = {
        # "tulu_2_7b_continued_ft-tulu_none-science_100",
        # "tulu_2_7b_continued_ft-tulu_none-science_200",
        # "tulu_2_7b_continued_ft-tulu_none-science_500",
        # "tulu_2_7b_continued_ft-tulu_none-science_1000",
        "tulu_2_7b_continued_ft-tulu_none-science_2500",
        # "tulu_2_7b_continued_ft-tulu_none-science_upsample",
    }

    continued_ft_mix_keys = {
        # "tulu_2_7b_continued_ft-tulu_match-science_100",
        # "tulu_2_7b_continued_ft-tulu_match-science_200",
        # "tulu_2_7b_continued_ft-tulu_match-science_500",
        # "tulu_2_7b_continued_ft-tulu_match-science_1000",
        "tulu_2_7b_continued_ft-tulu_match-science_2500",
    }

    merge_data_weighted_linear_keys = {
        "data_weighted_linear-llama_2_7b-tulu_all-science_2500",
    }

    merge_data_weighted_ta_keys = {
        "data_weighted_task_arithmetic-llama_2_7b-tulu_all-science_2500",
    }

    # normalize these 4
    df["Order"] = df["science_model_weight"]

    df_baselines = df[df["model_key"].isin(baseline_keys)]
    df_continued_ft = df[df["model_key"].isin(continued_ft_keys)]
    df_continued_ft_mix = df[df["model_key"].isin(continued_ft_mix_keys)]
    df_merge_data_weighted_linear = df[df["model_key"].isin(merge_data_weighted_linear_keys)]
    df_merge_data_weighted_task_arithmetic = df[df["model_key"].isin(merge_data_weighted_ta_keys)]

    df_lines = df[df["merge_method"] != "N/A"]
    # print(df_lines["science_model"])
    df_lines = df_lines[df_lines["science_model"].isin(merged_science_models)]
    # print(df_lines)
    df_lines = df_lines[df_lines["merge_method"].isin(merge_methods)]

    df_lines.sort_values(by='Combo', inplace=True)
    df_lines.sort_values(by='Order', inplace=True)

    print(df_lines)

    df_baselines["Order"] = df_baselines.apply(lambda row: weird_science_ordering[row["science_model"]], axis=1)
    df_baselines.sort_values(by='Order', inplace=True)
    df_continued_ft["Order"] = df_continued_ft.apply(lambda row: weird_science_ordering[row["science_model"]], axis=1)
    df_continued_ft.sort_values(by='Order', inplace=True)
    df_continued_ft_mix["Order"] = df_continued_ft_mix.apply(lambda row: weird_science_ordering[row["science_model"]], axis=1)
    df_continued_ft_mix.sort_values(by='Order', inplace=True)
    df_merge_data_weighted_linear["Order"] = df_merge_data_weighted_linear.apply(lambda row: weird_science_ordering[row["science_model"]], axis=1)
    df_merge_data_weighted_linear.sort_values(by='Order', inplace=True)
    df_merge_data_weighted_task_arithmetic["Order"] = df_merge_data_weighted_task_arithmetic.apply(lambda row: weird_science_ordering[row["science_model"]], axis=1)
    df_merge_data_weighted_task_arithmetic.sort_values(by='Order', inplace=True)

    # write to csv
    df.to_csv("results/current/full_results.csv", index=False)

    sns.lineplot(data=df_lines, x="Tulu Average (Tulu Subset)", y="Science Average", hue="Combo", sort=False, marker='o', markersize=6)
    sns.scatterplot(data=df_baselines, x="Tulu Average (Tulu Subset)", y="Science Average", hue="Combo", s=100)
    sns.scatterplot(data=df_continued_ft, x="Tulu Average (Tulu Subset)", y="Science Average", hue="Combo", s=300, marker="*")
    sns.scatterplot(data=df_continued_ft_mix, x="Tulu Average (Tulu Subset)", y="Science Average", hue="Combo", s=100, marker="X")
    sns.scatterplot(data=df_merge_data_weighted_linear, x="Tulu Average (Tulu Subset)", y="Science Average", hue="Combo", s=100, marker="o")
    sns.scatterplot(data=df_merge_data_weighted_task_arithmetic, x="Tulu Average (Tulu Subset)", y="Science Average", hue="Combo", s=100, marker="o")

    # sns.scatterplot(data=filtered_df, x="Tulu Average", y="Science Average", hue="Combo", marker="*", s=250, legend=False)

    plt.legend()

    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    # plt.ylim(0.1, 0.4)

    plt.show()

def compare_merge_methods():
    df = get_raw_df()

    merged_science_models = {
        # "science_100",
        # "science_200",
        # "science_500",
        "science_1000",
        # "science_2500",
        # "science_upsample",
    }

    # normalize these 4
    df["Order"] = df["science_model_weight"]

    # df_baselines = df[df["model_key"].isin(baseline_keys)]
    # df_continued_ft = df[df["model_key"].isin(continued_ft_keys)]
    # df_continued_ft_mix = df[df["model_key"].isin(continued_ft_mix_keys)]

    df_subset = df[df["science_model"].isin(merged_science_models)]
    df_subset.sort_values(by='Combo', inplace=True)
    df_subset.sort_values(by='Order', inplace=True)

    df_merges = df_subset[df_subset["merge_method"] != "N/A"]
    df_linear_weighted = df_subset[df_subset["merge_method"] == "linear_weighted"]
    df_dare_linear = df_subset[df_subset["merge_method"] == "dare_linear"]
    df_dare_ties = df_subset[df_subset["merge_method"] == "dare_ties"]
    df_ties = df_subset[df_subset["merge_method"] == "ties"]
    df_slerp = df_subset[df_subset["merge_method"] == "slerp"]


    # df_baselines["Order"] = df_baselines.apply(lambda row: weird_science_ordering[row["science_model"]], axis=1)
    # df_baselines.sort_values(by='Order', inplace=True)
    # df_continued_ft["Order"] = df_continued_ft.apply(lambda row: weird_science_ordering[row["science_model"]], axis=1)
    # df_continued_ft.sort_values(by='Order', inplace=True)
    # df_continued_ft_mix["Order"] = df_continued_ft_mix.apply(lambda row: weird_science_ordering[row["science_model"]], axis=1)
    # df_continued_ft_mix.sort_values(by='Order', inplace=True)


    # write to csv
    df.to_csv("results/current/full_results.csv", index=False)

    sns.lineplot(data=df_merges, x="Tulu Average (Tulu Subset)", y="Science Average", hue="Combo", sort=False, marker='o', markersize=6)
    # sns.lineplot(data=df_linear_weighted, x="Tulu Average (Tulu Subset)", y="Science Average", hue="Combo", sort=False, marker='o', markersize=6)
    # sns.lineplot(data=df_dare_linear, x="Tulu Average (Tulu Subset)", y="Science Average", hue="Combo", sort=False, marker='*', markersize=12)
    # sns.lineplot(data=df_dare_ties, x="Tulu Average (Tulu Subset)", y="Science Average", hue="Combo", sort=False, marker='P', markersize=12)
    # sns.lineplot(data=df_ties, x="Tulu Average (Tulu Subset)", y="Science Average", hue="Combo", sort=False, marker='X', markersize=12)
    # sns.lineplot(data=df_slerp, x="Tulu Average (Tulu Subset)", y="Science Average", hue="Combo", sort=False, marker='>', markersize=12)
    
    # sns.scatterplot(data=df_baselines, x="Tulu Average (Tulu Subset)", y="Science Average", hue="Combo", s=100)
    # sns.scatterplot(data=df_continued_ft, x="Tulu Average (Tulu Subset)", y="Science Average", hue="Combo", s=100, marker="*")
    # sns.scatterplot(data=df_continued_ft_mix, x="Tulu Average (Tulu Subset)", y="Science Average", hue="Combo", s=100, marker="X")

    # sns.scatterplot(data=filtered_df, x="Tulu Average", y="Science Average", hue="Combo", marker="*", s=250, legend=False)

    # print(df_ties)

    plt.legend()

    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    # plt.ylim(0.1, 0.4)

    plt.show()

def plot_single_stage_curves():
    def create_single_stage_combo(row):
        model_dict = {
            "llama_2_70b": "Llama 2 70B",
            "tulu_2_70b_continued_ft": "Tulu 2 70B",
            "llama_2_7b": "Llama 2 7B",
            "tulu_2_7b_continued_ft": "Tulu 2 7B",
            "tulu_none": "Tulu None",
            "tulu_match": "Tulu Match",
            "tulu_all": "Tulu All",
            "science_none": "Science None",
            "science_100": "Science 100",
            "science_200": "Science 200",
            "science_500": "Science 500",
            "science_1000": "Science 1000",
            "science_2500": "Science 2500",
            "science_upsample": "Science Upsample",
        }

        tokens = row["model_key"].split("-")
        if row["merge_method"] != "N/A":
            tokens = tokens[1:]
            tokens[1] = tokens[1][:-4]
            tokens[2] = tokens[2][:-4]
        base_model = model_dict[tokens[0]]
        tulu_model = model_dict[tokens[1]]
        science_model = model_dict[tokens[2]]

        if len(tokens) == 4:
            tulu_model += f" Seed {tokens[3].split('_')[-1]}"

        return f"{base_model} + {tulu_model}"

    def get_science_amount(row):
        science_amount = row["science_model"].split("_")[-1]
        if science_amount == "none":
            return 0
        elif science_amount == "upsample":
            return 318686
        else:
            return int(science_amount)
        
    baseline_keys = {
        "llama_2_7b-tulu_none-science_100",
        "llama_2_7b-tulu_none-science_200",
        "llama_2_7b-tulu_none-science_500",
        "llama_2_7b-tulu_none-science_1000",
        "llama_2_7b-tulu_none-science_2500",
        "llama_2_7b-tulu_none-science_upsample",

        "llama_2_7b-tulu_match-science_100",
        "llama_2_7b-tulu_match-science_200",
        "llama_2_7b-tulu_match-science_500",
        "llama_2_7b-tulu_match-science_1000",
        "llama_2_7b-tulu_match-science_2500",

        "llama_2_7b-tulu_all-science_100",
        "llama_2_7b-tulu_all-science_200",
        "llama_2_7b-tulu_all-science_500",
        "llama_2_7b-tulu_all-science_1000",
        "llama_2_7b-tulu_all-science_2500",
        "llama_2_7b-tulu_all-science_upsample",

        "tulu_2_7b_continued_ft-tulu_match-science_100",
        "tulu_2_7b_continued_ft-tulu_match-science_200",
        "tulu_2_7b_continued_ft-tulu_match-science_500",
        "tulu_2_7b_continued_ft-tulu_match-science_1000",
        "tulu_2_7b_continued_ft-tulu_match-science_2500",

        "tulu_2_7b_continued_ft-tulu_none-science_100",
        "tulu_2_7b_continued_ft-tulu_none-science_200",
        "tulu_2_7b_continued_ft-tulu_none-science_500",
        "tulu_2_7b_continued_ft-tulu_none-science_1000",
        "tulu_2_7b_continued_ft-tulu_none-science_2500",
        "tulu_2_7b_continued_ft-tulu_none-science_upsample",

        # "llama_2_7b-tulu_none-science_1000-seed_123",
        # "llama_2_7b-tulu_none-science_1000-seed_52830",
        # "llama_2_7b-tulu_all-science_none-seed_123",
        # "llama_2_7b-tulu_all-science_none-seed_52830",
    }

    merged_science_models = {
        # "science_100",
        # "science_200",
        "science_500",
        "science_1000",
        "science_2500",
        "science_upsample"
    }

    df_merges = get_raw_df()
    df_merges["Order"] = df_merges["tulu_model_weight"]

    # df_baselines = df[df["model_key"].isin(baseline_keys)]
    # df_continued_ft = df[df["model_key"].isin(continued_ft_keys)]
    # df_continued_ft_mix = df[df["model_key"].isin(continued_ft_mix_keys)]

    df_subset = df_merges[df_merges["science_model"].isin(merged_science_models)]
    df_subset.sort_values(by='Combo', inplace=True)
    df_subset.sort_values(by='Order', inplace=True)

    df_linear_weighted = df_subset[df_subset["merge_method"] == "linear_weighted"]
    df_dare_linear = df_subset[df_subset["merge_method"] == "dare_linear"]
    df_ties = df_subset[df_subset["merge_method"] == "ties"]

    df = get_raw_df()
    df = df[df["merge_method"] == "N/A"]
    df = df[df["model_key"].isin(baseline_keys)]
    df["science_amount"] = df.apply(lambda row: get_science_amount(row), axis=1)
    df["Combo"] = df.apply(lambda row: create_single_stage_combo(row), axis=1)

    df.sort_values(by='science_amount', inplace=True)

    # sns.lineplot(data=df_linear_weighted, x="Tulu Average (Tulu Subset)", y="Science Average", hue="Combo", sort=False, marker='P', markersize=6)
    sns.lineplot(data=df_ties, x="Tulu Average (Tulu Subset)", y="Science Average", hue="Combo", sort=False, marker='*', markersize=12)
    sns.lineplot(data=df, x="Tulu Average (Tulu Subset)", y="Science Average", hue="Combo", sort=False, marker='o', markersize=6)

    plt.legend()

    plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

    plt.show()

plot_linear_merge_vs_baselines()
# compare_merge_methods()
# plot_single_stage_curves()