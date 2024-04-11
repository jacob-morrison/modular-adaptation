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
        "safety_none": "Safety None",
        "safety_20": "Safety 20%",
        "safety_40": "Safety 40%",
        "safety_60": "Safety 60%",
        "safety_80": "Safety 80%",
        "safety_100": "Safety 100%",
        "safety_upsample": "Safety Upsample",
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

def get_raw_df():
    safety_results = {}
    with open("results/current/safety-evals.csv") as f_in:
        # with open("results/current/manual-safety-evals.csv") as f_in2:
        # for baselines:
        merge_method = "N/A"
        i = 0
        for line in f_in.readlines(): # + f_in2.readlines()[2:]:
            line = line.strip().replace("_4096", "")
            curr_results = {}
            if i == 0: # model_key,base_model,tulu_model,tulu_model_weight,safety_model,safety_model_weight,merge_method,harmbench,safe_average,unsafe_average
                keys = line.split(',')
            else:
                curr_data = line.split(',')
                model = curr_data[0]
                model_tokens = model.split('-')

                for key, info in zip(keys, curr_data):
                    if key in [
                        "harmbench",
                        "safe_average",
                        "unsafe_average"
                    ]:
                        curr_results[key] = info

                safety_results[model] = curr_results
            i += 1

    tulu_data = []
    with open("results/current/tulu-evals-safety.jsonl") as f_in:
        # with open("results/current/manual-tulu-evals.jsonl") as f_in2:
            for line in f_in.readlines(): # + f_in2.readlines():
                data = json.loads(line.replace("_4096", ""))
                model_key = data["model_key"]
                if model_key not in safety_results:
                    print(f"key not found: {model_key}")
                else:
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

    safety_columns_for_average = [
        "toxigen",
        "harmbench",
        "invert_unsafe_average",
        "safe_average"
    ]

    df['invert_toxigen'] = df.apply(lambda row: 1 - row["toxigen"], axis=1)
    df['invert_unsafe_average'] = df.apply(lambda row: 1 - row["unsafe_average"], axis=1)
    df['alpaca_farm'] = df.apply(lambda row: row["alpaca_farm"] / 100, axis=1)
    df['tydiqa_no_context_1shot'] = df.apply(lambda row: row["tydiqa_no_context_1shot"] / 100, axis=1)
    df['tydiqa_goldp_1shot'] = df.apply(lambda row: row["tydiqa_goldp_1shot"] / 100, axis=1)

    # calculate averages
    df['Tulu Average (Other Evals)'] = df.apply(lambda row: calculate_tulu_average(row, tulu_columns_for_val_average), axis=1)
    df['Tulu Average (Tulu Subset)'] = df.apply(lambda row: calculate_tulu_average(row, tulu_columns_for_test_average), axis=1)
    df['Safety Average'] = df.apply(lambda row: calculate_tulu_average(row, safety_columns_for_average), axis=1)

    df['Combo'] = df.apply(lambda row: create_model_combo(row), axis=1)

    return df

df = get_raw_df()
print(df.to_string())