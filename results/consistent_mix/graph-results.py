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
        if row["merge_method"] == "task_arithmetic" and domain_model_weight == 1.0:
            print(row["model_key"])
    base_model = model_dict[tokens[0]]
    tulu_model = model_dict[tokens[1]]
    if len(tokens) == 2:
        domain_model = "None"
    else:
        # print(tokens[2])
        domain_model = ""
        for token in tokens[2].split("-"):
            if token in model_dict:
                domain_model += " " + model_dict[token]
            else:
                domain_model += f" MISSING {token}"

    if str(row["merge_method"]) == "nan":
        return f"{base_model} ft. on {tulu_model} & {domain_model.strip()}".replace(" Tulu None &", "")
    else:
        # return f"{base_model} ft. on {tulu_model} merged with {domain_model.strip()} ({row['merge_method']})".replace(" Tulu None &", "")
        return f"{domain_model.replace(' 7B Tulu None', '').strip()} - {row['merge_method']}"
    
models_to_skip = [
    "tulu_2_13b_retrain",
]

def get_df():
    df = pd.read_csv("results/consistent_mix/results.csv")

    tulu_columns_for_test_average = [
        "mmlu_0shot",
        "gsm_cot",
        "bbh_cot",
        # "tydiqa_goldp_1shot",
        "truthfulqa",
        "alpaca_eval",
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

    # ("bioasq", "f1"),
    # ("biored", "f1"),
    # ("discomat", "bleu"),
    # ("evidence_inference", "f1_overlap"),
    # ("multicite", "f1"),
    # ("mup", "rouge"),
    # ("qasper", "f1_answer"),
    # ("qasper", "f1_evidence"),
    # ("scierc", "f1"),
    # ("scifact", "f1_label"),
    # ("scifact", "f1_evidence_sent"),
    science_columns_for_average_without_some_evals = [
        "bioasq_f1",
        "biored_f1",
        "discomat_bleu",
        # "evidence_inference_f1_overlap", # problematic
        "multicite_f1",
        "mup_rouge",
        "qasper_f1_answer",
        "qasper_f1_evidence",
        "scierc_f1",
        "scifact_f1_label",
        "scifact_f1_evidence_sent",
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
    # df['Test Science Average'] = df.apply(lambda row: calculate_tulu_average(row, science_columns_for_average_without_some_evals), axis=1)
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

def plot_individual_science_curves():
    science_amounts = [
        # "100",
        # "200",
        # "500",
        # "1000",
        "2500",
    ]
    for amount in science_amounts:
        df = get_df()
        df["Order"] = df["domain_model_weight"]
        df.sort_values(by='Combo', inplace=True)
        df.sort_values(by='Order', inplace=True)
        df = df[
            ~df["Combo"].str.contains("Safety 100") &
            ~df["Combo"].str.contains("Coding 100") &
            (
                # df["Combo"].str.contains("Science 100") |
                # df["Combo"].str.contains("Science 200") |
                # df["Combo"].str.contains("Science 500") |
                # df["Combo"].str.contains("Science 1000") |
                df["Combo"].str.contains(f"cience {amount}")
            )
        ]
        print(df)
        if amount == "100":
            df = df[~df["Combo"].str.contains("Science 1000")]

        sns.lineplot(data=df, x="Tulu Average", y="Science Average", hue="Combo", sort=False, marker='X', linewidth=3, markersize=13)
        plt.legend()
        plt.xlabel("Tulu Average",fontsize=20)
        plt.ylabel("Science Average",fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=11)

        plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        plt.savefig(f'results/consistent_mix/plots/science_{amount}.png', dpi=300, bbox_inches='tight')
        plt.clf()

def plot_individual_coding_curves():
    coding_amounts = [
        "20",
        "40",
        "60",
        "80",
        "100",
    ]
    for amount in coding_amounts:
        df = get_df()
        df["Order"] = df["domain_model_weight"]
        df.sort_values(by='Combo', inplace=True)
        df.sort_values(by='Order', inplace=True)
        df = df[
            ~df["Combo"].str.contains("Science 2500") &
            ~df["Combo"].str.contains("Safety 100") &
            ~df["Combo"].str.contains("w/ Coding") &
            ~df["tulu_model"].str.contains("coding") &
            df["Combo"].str.contains(f"Coding {amount}")
        ]


        print(df)
        sns.lineplot(data=df, x="Tulu Average", y="Coding Average", hue="Combo", sort=False, marker='X', linewidth=3, markersize=13)
        plt.legend()
        plt.xlabel("Tulu Average",fontsize=20)
        plt.ylabel("Coding Average",fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=11)

        plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        plt.savefig(f'results/consistent_mix/plots/coding_{amount}.png', dpi=300, bbox_inches='tight')
        plt.clf()

def plot_individual_coding_curves_tulu_with_coding():
    coding_amounts = [
        "20",
        "40",
        "60",
        "80",
        "100",
    ]
    for amount in coding_amounts:
        df = get_df()
        df["Order"] = df["domain_model_weight"]
        df.sort_values(by='Combo', inplace=True)
        df.sort_values(by='Order', inplace=True)
        df = df[
            ~df["Combo"].str.contains("Science 2500") &
            ~df["Combo"].str.contains("Safety 100") &
            # ~df["Combo"].str.contains("w/ Coding") &
            df["tulu_model"].str.contains("coding") &
            df["Combo"].str.contains(f"Coding {amount}")
        ]


        print(df)
        sns.lineplot(data=df, x="Tulu Average", y="Coding Average", hue="Combo", sort=False, marker='X', linewidth=3, markersize=13)
        plt.legend()
        plt.xlabel("Tulu Average",fontsize=20)
        plt.ylabel("Coding Average",fontsize=20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=11)

        plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        # plt.show()

        plt.savefig(f'results/consistent_mix/plots/tulu_with_coding_{amount}.png', dpi=300, bbox_inches='tight')
        plt.clf()

def plot_individual_safety_curves():
        amount = "100"
        df = get_df()
        df["Order"] = df["domain_model_weight"]
        df.sort_values(by='Combo', inplace=True)
        df.sort_values(by='Order', inplace=True)
        # df = df[(df["normalized_safe_average"] > 0) | ("safety_100" in df["model_key"])]
        # df = df[~df["domain_model"].isin({
            # "science_2500",
            # "coding_100"
        # })]
        df = df[
            ~df["Combo"].str.contains("Science 2500") &
            ~df["Combo"].str.contains("Coding 100") &
            df["Combo"].str.contains(f"Safety {amount}")
        ]

        # pd.set_option('display.max_colwidth', None)

        sns.lineplot(data=df, x="Tulu Average", y="Safety Average", hue="Combo", sort=False, marker='o', linewidth=3, markersize=13)
        plt.legend()
        plt.xlabel("Tulu Average",fontsize=20)
        plt.ylabel("Safety Average",fontsize=20)

        # sns.lineplot(data=df, x="Exaggerated Refusals", y="Safety Average", hue="Combo", sort=False, marker='o', linewidth=3, markersize=13)
        # plt.legend()
        # plt.xlabel("Exaggerated Refusals",fontsize=20)
        # plt.ylabel("Safety Average",fontsize=20)

        # sns.lineplot(data=df, x="Tulu Average", y="Exaggerated Refusals", hue="Combo", sort=False, marker='X', linewidth=3, markersize=13)
        # plt.legend()
        # plt.xlabel("Tulu Average",fontsize=20)
        # plt.ylabel("Exaggerated Refusals",fontsize=20)

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=11)

        plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        plt.savefig(f'results/consistent_mix/plots/safety_{amount}.png', dpi=300, bbox_inches='tight')
        plt.clf()

def plot_individual_safety_vs_exaggerated_refusals():
        amount = "100"
        df = get_df()
        df["Order"] = df["domain_model_weight"]
        df.sort_values(by='Combo', inplace=True)
        df.sort_values(by='Order', inplace=True)
        # df = df[(df["normalized_safe_average"] > 0) | ("safety_100" in df["model_key"])]
        # df = df[~df["domain_model"].isin({
            # "science_2500",
            # "coding_100"
        # })]
        df = df[
            ~df["Combo"].str.contains("cience") &
            ~df["Combo"].str.contains("Coding 100") &
            df["Combo"].str.contains(f"Safety {amount}")
        ]

        # pd.set_option('display.max_colwidth', None)

        # sns.lineplot(data=df, x="Tulu Average", y="Safety Average", hue="Combo", sort=False, marker='o', linewidth=3, markersize=13)
        # plt.legend()
        # plt.xlabel("Tulu Average",fontsize=20)
        # plt.ylabel("Safety Average",fontsize=20)

        sns.lineplot(data=df, x="Exaggerated Refusals", y="Safety Average", hue="Combo", sort=False, marker='o', linewidth=3, markersize=13)
        plt.legend()
        plt.xlabel("Exaggerated Refusals",fontsize=20)
        plt.ylabel("Safety Average",fontsize=20)

        # sns.lineplot(data=df, x="Tulu Average", y="Exaggerated Refusals", hue="Combo", sort=False, marker='X', linewidth=3, markersize=13)
        # plt.legend()
        # plt.xlabel("Tulu Average",fontsize=20)
        # plt.ylabel("Exaggerated Refusals",fontsize=20)

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.legend(fontsize=11)

        plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
        plt.savefig(f'results/consistent_mix/plots/safety_exaggerated_refusals_{amount}.png', dpi=300, bbox_inches='tight')
        plt.clf()


plot_individual_science_curves()
# plot_individual_coding_curves()
# plot_individual_safety_curves()
# plot_individual_coding_curves_tulu_with_coding()
# plot_individual_safety_vs_exaggerated_refusals()