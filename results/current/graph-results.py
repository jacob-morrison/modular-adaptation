import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd 

science_results = {}
with open("results/current/baseline-science-evals.csv") as f_in:
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
with open("results/current/baseline-tulu-evals.jsonl") as f_in:
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

tulu_columns_for_test_average = [
    "mmlu_0shot",
    "gsm_cot",
    "bbh_cot",
    "tydiqa_goldp_1shot",
    "codex_eval_temp_0.8",
    "alpaca_farm",
    "invert_toxigen",
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
df['Tulu Average (Validation)'] = df.apply(lambda row: calculate_tulu_average(row, tulu_columns_for_val_average), axis=1)
df['Tulu Average (Test)'] = df.apply(lambda row: calculate_tulu_average(row, tulu_columns_for_test_average), axis=1)
df['Science Average'] = df['mean']

sns.scatterplot(data=df, x="Tulu Average (Test)", y="Science Average", hue="model_key", s=100) #, palette=colors)
# palette=["red", "brown", "cyan"])

# sns.scatterplot(data=filtered_df, x="Tulu Average", y="Science Average", hue="Combo", marker="*", s=250, legend=False)

plt.legend()

plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

# plt.ylim(0.1, 0.4)

plt.show()