import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd 

science_results = {}
with open("results/current/science-evals.csv") as f_in:
    # for baselines:
    merge_method = "N/A"
    i = 0
    for line in f_in.readlines():
        line = line.strip()
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
                curr_results[task] = {
                    "metric": metric,
                    "value": float(value)
                }

            science_results[model] = curr_results
        i += 1

tulu_data = []
with open("results/current/tulu-evals.jsonl") as f_in:
    for line in f_in.readlines():
        data = json.loads(line)
        model_key = data["model_key"]
        if model_key not in science_results:
            print(f"key not found: {model_key}")
        else:
            for task in science_results[model_key]:
                data[task] = science_results[model_key][task]["value"]
        tulu_data.append(data)        

df = pd.DataFrame(tulu_data)

print(df)

def calculate_tulu_average(row, columns):
    subset_values = row[columns]
    return subset_values.mean()

def create_model_combo(row):
    model_dict = {
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

    if row["merge_method"] == "N/A":
        return f"{base_model} -> {tulu_model} & {science_model}"
    else:
        return f"{base_model} -> {tulu_model} merged with {science_model}, {row['merge_method']}"

baseline_keys = {
    "llama_2_7b-tulu_none-science_100",
    "llama_2_7b-tulu_none-science_200",
    "llama_2_7b-tulu_none-science_500",
    "llama_2_7b-tulu_none-science_1000",
    "llama_2_7b-tulu_none-science_2500",
    "llama_2_7b-tulu_all-science_none-seed_42",
    "llama_2_7b-tulu_match-science_100",
    "llama_2_7b-tulu_match-science_200",
    "tulu_2_7b_continued_ft-tulu_none-science_100",
    "tulu_2_7b_continued_ft-tulu_none-science_200",
    "tulu_2_7b_continued_ft-tulu_none-science_500",
    "tulu_2_7b_continued_ft-tulu_none-science_1000",
    "tulu_2_7b_continued_ft-tulu_match-science_100",
    "tulu_2_7b_continued_ft-tulu_match-science_200",
    "tulu_2_7b_continued_ft-tulu_match-science_2500",

    # "llama_2_7b-tulu_none-science_1000-seed_123",
    # "llama_2_7b-tulu_none-science_1000-seed_52830",
    # "llama_2_7b-tulu_all-science_none-seed_123",
    # "llama_2_7b-tulu_all-science_none-seed_52830",
}

tulu_columns_for_test_average = [
    "mmlu_0shot",
    "gsm_cot",
    "bbh_cot",
    "tydiqa_goldp_1shot",
    "codex_eval_temp_0.8",
    "alpaca_farm",
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

# normalize these 4
df['invert_toxigen'] = df.apply(lambda row: 1 - row["toxigen"], axis=1)
df['alpaca_farm'] = df.apply(lambda row: row["alpaca_farm"] / 100, axis=1)
df['tydiqa_no_context_1shot'] = df.apply(lambda row: row["tydiqa_no_context_1shot"] / 100, axis=1)
df['tydiqa_goldp_1shot'] = df.apply(lambda row: row["tydiqa_goldp_1shot"] / 100, axis=1)

# calculate averages
df['Tulu Average (Other Evals)'] = df.apply(lambda row: calculate_tulu_average(row, tulu_columns_for_val_average), axis=1)
df['Tulu Average (Tulu Subset)'] = df.apply(lambda row: calculate_tulu_average(row, tulu_columns_for_test_average), axis=1)
df['Science Average'] = df['mean']

df['Combo'] = df.apply(lambda row: create_model_combo(row), axis=1)
df["Order"] = df["tulu_model_weight"]

df_baselines = df[df["model_key"].isin(baseline_keys)]
df_lines = df[df["merge_method"] != "N/A"]

df_lines.sort_values(by='Order', inplace=True)
df_baselines.sort_values(by='Combo', inplace=True)

# write to csv
df.to_csv("results/current/full_results.csv", index=False)

sns.lineplot(data=df_lines, x="Tulu Average (Tulu Subset)", y="Science Average", hue="Combo", sort=False, marker='o', markersize=6)

sns.scatterplot(data=df_baselines, x="Tulu Average (Tulu Subset)", y="Science Average", hue="Combo", s=100) #, palette=colors)
# palette=["red", "brown", "cyan"])

# sns.scatterplot(data=filtered_df, x="Tulu Average", y="Science Average", hue="Combo", marker="*", s=250, legend=False)

plt.legend()

plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

# plt.ylim(0.1, 0.4)

plt.show()