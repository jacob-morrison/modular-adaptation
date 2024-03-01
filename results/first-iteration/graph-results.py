import matplotlib.pyplot as plt
import seaborn as sns
import json
import pandas as pd 

science_files = [
    "results/science/baselines.csv",
    "results/science/merges-daves-tulu-model.csv",
    "results/science/merges-my-bad-tulu-model.csv",
    "results/science/merges-another-tulu-model.csv"
]

baselines = [
    "tulu_all_science_1000_eval_no",
    "tulu_all_science_200_eval_no",
    "tulu_all_science_2500_eval_no",
    "my_bad_tulu_all_science_none",
    "another_tulu_all_science_none",
    "tulu_none_science_1000_eval_no",
    "tulu_none_science_200_eval_no",
    "tulu_none_science_2500_eval_no",
    "daves_tulu_all_science_none",
]

other_merge_methods = [
    "dare_linear",
    "dare_ties",
    "slerp",
    "ties",
]

science_results = {}

def get_model_weights(filename):
    if "merged_models" in model:
        pass
    # merged models:
        # tulu_model_weight = float(model[25:28])
        # science_model_weight = float(model[39:42])
    # dave's & my new one:
        # tulu_model_weight = float(model[11:14])
        # science_model_weight = float(model[25:28])
    tokens = filename.split('-')
    science_model_weight = float(tokens[-2])
    tulu_model_weight = float(tokens[-4])
    return tulu_model_weight, science_model_weight

for file in science_files:
    with open(file) as f_in:
        i = 0
        for line in f_in.readlines():
            curr_results = {}
            if i == 0: # task,bioasq,biored,discomat,evidence_inference,evidence_inference,evidence_inference,multicite,mup,qasper,scierc,scifact,scifact,scifact,mean,median
                tasks = line.split(',')[1:]
            elif i == 1: # metric,f1,f1,bleu,f1_exact,f1_overlap,f1_substring,f1,bleu,bleu,f1,f1_evidence_sent,f1_evidence_token,f1_label
                metrics = line.split(',')[1:]
            else:
                curr_data = line.split(',')
                model = curr_data[0]
                merge_method = model.split('-')[0]
                if merge_method == "llama_2_7b" or model in baselines:
                    merge_method = "linear_weighted"
                if "baselines" in file:
                    curr_results["tulu_model"] = model
                    curr_results["science_model"] = model
                    if "tulu_none" in model:
                        tulu_model_weight = 0.0
                    else:
                        tulu_model_weight = 1.0
                else:
                    if "200" in model:
                        science_model = "tulu_none_science_200_eval_no"
                    elif "1000" in model:
                        science_model = "tulu_none_science_1000_eval_no"
                    elif "2500" in model:
                        science_model = "tulu_none_science_2500_eval_no"
                    if "daves" in file:
                        tulu_model = "daves_tulu_all_science_none"
                    elif "my-bad" in file:
                        tulu_model = "my_bad_tulu_all_science_none"
                    elif "another" in file:
                        tulu_model = "another_tulu_all_science_none"
                    curr_results["tulu_model"] = tulu_model
                    curr_results["science_model"] = science_model
                    # tulu_model_weight = float(model[11:14])
                    tulu_model_weight, science_model_weight = get_model_weights(model)
                for task, metric, value in zip(tasks, metrics, curr_data[1:]):
                    curr_results[task] = {
                        "metric": metric,
                        "value": value
                    }
                if merge_method not in science_results:
                    science_results[merge_method] = {}
                if curr_results["tulu_model"] not in science_results[merge_method]:
                    science_results[merge_method][curr_results["tulu_model"]] = {}
                if curr_results["science_model"] not in science_results[merge_method][curr_results["tulu_model"]]:
                    science_results[merge_method][curr_results["tulu_model"]][curr_results["science_model"]] = {}
                science_results[merge_method][curr_results["tulu_model"]][curr_results["science_model"]][tulu_model_weight] = curr_results
            i += 1

data = []

for model in baselines:
    curr_data = science_results["linear_weighted"][model]
    for method in other_merge_methods:
        science_results[method][model] = curr_data


from pprint import pprint
print(science_results.keys())
print(science_results["linear_weighted"].keys())

with open("results/collected-results-2.24.24.jsonl") as f_in:
    for line in f_in.readlines():
        curr_data = json.loads(line)
        data.append(curr_data)

def calculate_tulu_average(row, columns):
    subset_values = row[columns]
    return subset_values.mean()

def get_science_average(row):
    print()
    print(row)
    if row["tulu_model_weight"] == 0.0:
        return float(science_results[row["merge_method"]][row["science_model"]][row["science_model"]][0.0]["mean"]["value"])
    if row["tulu_model_weight"] == 1.0:
        return float(science_results[row["merge_method"]][row["tulu_model"]][row["tulu_model"]][1.0]["mean"]["value"])

    return float(science_results[
            row["merge_method"]
        ][
            row["tulu_model"]
        ][
            row["science_model"]
        ][
            row["tulu_model_weight"]
        ]["mean"]["value"])

tulu_columns_for_average = [
    "bbh_cot",
    "bbh_direct",
    "codex_eval_temp_0.1",
    "codex_eval_temp_0.8",
    "gsm_cot",
    "gsm_direct",
    "mmlu_0shot",
    "mmlu_5shot",
]

df = pd.DataFrame(data)
# print(df.to_string())

# pprint(science_results["my_bad_tulu_all_science_none"]["tulu_none_science_200_eval_no"])
# print(science_results.keys())
# pprint(science_results["tulu_all_science_200_eval_no"]["tulu_all_science_200_eval_no"])

def create_model_combo(tulu_model_name, science_model_name):
    filtered_science_models = {
        "tulu_all_science_200_eval_no",
        "tulu_all_science_1000_eval_no",
        "tulu_all_science_2500_eval_no",
    }

    model_dict = {
        "tulu_all_science_200_eval_no": "Tulu All & Science 200",
        "tulu_all_science_1000_eval_no": "Tulu All & Science 1000",
        "tulu_all_science_2500_eval_no": "Tulu All & Science 2500",
        
        "daves_tulu_all_science_none": "Dave's Tulu All",
        "my_bad_tulu_all_science_none": "Tulu All #3",
        "another_tulu_all_science_none": "Tulu All #4",

        "tulu_none_science_200_eval_no": "Science 200",
        "tulu_none_science_1000_eval_no": "Science 1000",
        "tulu_none_science_2500_eval_no": "Science 2500",
    }

    if science_model_name in filtered_science_models:
        return model_dict[tulu_model_name]
    else:
        return model_dict[tulu_model_name] + " merged with " + model_dict[science_model_name]
    
def create_order_values(row):
    if '200' in row["tulu_model"]:
        return 1
    if '1000' in row["tulu_model"]:
        return 2
    if '2500' in row["tulu_model"]:
        return 3
    return 4
    

df['Tulu Average'] = df.apply(lambda row: calculate_tulu_average(row, tulu_columns_for_average), axis=1)
df['Science Average'] = df.apply(lambda row: get_science_average(row), axis=1)
df['order value'] = df.apply(lambda row: create_order_values(row), axis=1)
df['Combo'] = df.apply(lambda row: create_model_combo(row['tulu_model'], row['science_model']) + ", " + row["merge_method"], axis=1)
print(df)

science_models = [
    # 'tulu_none_science_200_eval_no',
    # 'tulu_none_science_1000_eval_no',
    'tulu_none_science_2500_eval_no'
]

tulu_models = [
    # "tulu_all_science_200_eval_no",
    # "tulu_all_science_1000_eval_no",
    "tulu_all_science_2500_eval_no",
    "daves_tulu_all_science_none",
    # "my_bad_tulu_all_science_none",
    # "another_tulu_all_science_none",
]

merge_methods = [
    "linear_weighted",
    "dare_ties",
    "dare_linear",
    "ties",
    "slerp"
]

df_lines = df[df["tulu_model"].isin(tulu_models)]
df_lines = df_lines[df_lines['science_model'].isin(science_models)]
df_lines = df_lines[df_lines['merge_method'].isin(merge_methods)]
df_lines = df_lines[~df_lines['tulu_model'].isin(science_models)]
df_lines["Order"] = df_lines["tulu_model_weight"]
df_lines.sort_values(by='order value', inplace=True)
df_lines.sort_values(by='Order', inplace=True)

sns.set_palette("colorblind")

print(df_lines)

# sns.lineplot(data=df_lines, x="tulu_model_weight", y="Tulu Average", hue="Combo")
# sns.lineplot(data=df_lines, x="tulu_model_weight", y="Science Average", hue="Combo")

sns.lineplot(data=df_lines, x="Tulu Average", y="Science Average", hue="Combo", sort=False, marker='o', markersize=6)

tulu_with_science_models = [
    'tulu_all_science_200_eval_no',
    'tulu_all_science_1000_eval_no',
    'tulu_all_science_2500_eval_no'
]

df_points = df[df['science_model'].isin(tulu_with_science_models)]
df_points = df_points[df_points["tulu_model"].isin(tulu_models)]

tulu_with_science_tulu_scores = []
tulu_with_science_science_scores = []

for model in tulu_with_science_models:
    tulu_with_science_tulu_scores.append(df[df['science_model'] == model]['Tulu Average'].values[0])
    tulu_with_science_science_scores.append(df[df['science_model'] == model]['Science Average'].values[0])

science_model_weights = {
    'tulu_none_science_200_eval_no': (0.025, 0.975),
    'tulu_none_science_1000_eval_no': (0.1, 0.9),
    'tulu_none_science_2500_eval_no': (0.16, 0.84),
}

filtered_df = pd.DataFrame()
df_stars = df[df['science_model'].isin(science_models)]
df_stars = df_stars[df_stars["tulu_model"].isin(tulu_models)]
df_stars = df_stars[df_stars['merge_method'].isin(merge_methods)]
df_stars.sort_values(by='order value', inplace=True)
for science_model in science_models:
    (science_weight, tulu_weight) = science_model_weights[science_model]
    filtered_df = pd.concat([
        filtered_df,
        df_stars[
            (df_stars['science_model'] == science_model) &
            (df_stars['tulu_model_weight'] == tulu_weight) &
            (df_stars['science_model_weight'] == science_weight)
        ]
    ])
    # print(df_stars[
    #         (df_stars['science_model'] == science_model) &
    #         (df_stars['tulu_model_weight'] == tulu_weight) &
    #         (df_stars['science_model_weight'] == science_weight)
    #     ])

colors = ["red", "brown", "cyan"]

# for model, score, color in zip(tulu_with_science_models, tulu_with_science_models_scores, colors):
#     plt.axvline(x=score, label=model, color=color, linestyle='--')

# plt.scatter(tulu_with_science_tulu_scores, tulu_with_science_science_scores, color=colors, label=tulu_with_science_models)
sns.scatterplot(data=df_points, x="Tulu Average", y="Science Average", hue="Combo", s=100, palette=colors)
# palette=["red", "brown", "cyan"])

sns.scatterplot(data=filtered_df, x="Tulu Average", y="Science Average", hue="Combo", marker="*", s=250, legend=False)

plt.legend()

plt.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5)

plt.ylim(0.1, 0.4)

plt.show()