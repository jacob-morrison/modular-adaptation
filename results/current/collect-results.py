import json
import os
import pandas as pd

science_path = "/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/science/tulu_evals/"
safety_path = "/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/safety/tulu_evals/"

# print(tulu_evals)

tulu_metrics = [
    "alpaca_farm",
    "bbh_cot",
    "bbh_direct",
    "codex_eval_temp_0.1",
    "codex_eval_temp_0.8",
    "gsm_cot",
    "gsm_direct",
    "mmlu_0shot",
    "mmlu_5shot",
    "toxigen",
    "truthfulqa",
    "tydiqa_goldp_1shot",
    "tydiqa_no_context_1shot",
]

example = {
    "base_model": "llama-2-7b",
    "tulu_model": "tulu_model_name",
    "tulu_model_weight": -0.5,
    "science_model": "science_model_name",
    "science_model_weight": -0.5,

    "bbh_cot": -1,
    "bbh_direct": -1,
    "codex_eval_temp_0.1": -1,
    "codex_eval_temp_0.8": -1,
    "gsm_cot": -1,
    "gsm_direct": -1,
    "mmlu_0shot": -1,
    "mmlu_5shot": -1,
    "toxigen": -1,
    "tydiqa_goldp_1shot": -1,
    "tydiqa_no_context_1shot": -1,
}

def get_model_weights(model_name):
    tokens = model_name.split('-')
    science_model_weight = float(tokens[-1].split('_')[-1])
    tulu_model_weight = float(tokens[-2].split('_')[-1])
    return tulu_model_weight, science_model_weight

def collect_metrics(model_path):
    model_name = model_path.split("/")[-1].replace("_4096", "")
    merged = model_name.split("-")[0] in [
        "linear_weighted",
        "ties",
        "dare_linear",
        "dare_ties",
        "slerp",
    ]
    if merged:
        tokens = model_name.split('-')
        merge_method = tokens[0]
        base_model = tokens[1]
        tulu_model = tokens[2][:-4]
        science_model = tokens[3][:-4]
        tulu_model_weight, science_model_weight = get_model_weights(model_name)
    else:
        merge_method = "N/A"
        tokens = model_name.split('-')
        base_model = tokens[0]
        tulu_model = tokens[1]
        science_model = tokens[2]
        if tulu_model == "tulu_none":
            tulu_model_weight = 0.0
        else:
            tulu_model_weight = 1.0
        if science_model == "science_none":
            science_model_weight = 0.0
        else:
            science_model_weight = 1.0

    model_data = {
        "model_key": model_name,
        "base_model": base_model,
        "tulu_model": tulu_model,
        "tulu_model_weight": tulu_model_weight,
        "science_model": science_model,
        "science_model_weight": science_model_weight,
        "merge_method": merge_method,
    }

    if not os.path.isfile(model_path + f"/bbh_cot/metrics.json"):
        print(f"bbh_cot missing for {model_name}")
        model_data["bbh_cot"] = 0.0
    else:
        with open(model_path + f"/bbh_cot/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["bbh_cot"] = data["average_exact_match"]

    if not os.path.isfile(model_path + f"/bbh_direct/metrics.json"):
        print(f"bbh_direct missing for {model_name}")
        model_data["bbh_direct"] = 0.0
    else:
        with open(model_path + f"/bbh_direct/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["bbh_direct"] = data["average_exact_match"]

    if not os.path.isfile(model_path + f"/codex_eval_temp_0.1/metrics.json"):
        print(f"codex_eval_temp_0.1 missing for {model_name}")
        model_data["codex_eval_temp_0.1"] = 0.0
    else:
        with open(model_path + f"/codex_eval_temp_0.1/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["codex_eval_temp_0.1"] = data["pass@1"]

    if not os.path.isfile(model_path + f"/codex_eval_temp_0.8/metrics.json"):
        print(f"codex_eval_temp_0.8 missing for {model_name}")
        model_data["codex_eval_temp_0.8"] = 0.0
    else:
        with open(model_path + f"/codex_eval_temp_0.8/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["codex_eval_temp_0.8"] = data["pass@10"]

    if not os.path.isfile(model_path + f"/gsm_cot/metrics.json"):
        print(f"gsm_cot missing for {model_name}")
        model_data["gsm_cot"] = 0.0
    else:
        with open(model_path + f"/gsm_cot/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["gsm_cot"] = data["exact_match"]

    if not os.path.isfile(model_path + f"/gsm_direct/metrics.json"):
        print(f"gsm_direct missing for {model_name}")
        model_data["gsm_direct"] = 0.0
    else:
        with open(model_path + f"/gsm_direct/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["gsm_direct"] = data["exact_match"]

    if not os.path.isfile(model_path + f"/mmlu_0shot/metrics.json"):
        print(f"mmlu_0shot missing for {model_name}")
        model_data["mmlu_0shot"] = 0.0
    else:
        with open(model_path + f"/mmlu_0shot/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["mmlu_0shot"] = data["average_acc"]

    if not os.path.isfile(model_path + f"/mmlu_5shot/metrics.json"):
        print(f"mmlu_5shot missing for {model_name}")
        model_data["mmlu_5shot"] = 0.0
    else:
        with open(model_path + f"/mmlu_5shot/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["mmlu_5shot"] = data["average_acc"]

    if not os.path.isfile(model_path + f"/toxigen/metrics.json"):
        print(f"toxigen missing for {model_name}")
        model_data["toxigen"] = 0.0
    else:
        with open(model_path + f"/toxigen/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["toxigen"] = data["overall"]

    if not os.path.isfile(model_path + f"/trutufulqa/metrics.json"):
        print(f"truthfulqa missing for {model_name}")
        model_data["truthfulqa"] = 0.0
    else:
        with open(model_path + f"/trutufulqa/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["truthfulqa"] = data["truth-info acc"]

    if not os.path.isfile(model_path + f"/tydiqa_goldp_1shot/metrics.json"):
        print(f"tydiqa_goldp_1shot missing for {model_name}")
        model_data["tydiqa_goldp_1shot"] = 0.0
    else:
        with open(model_path + f"/tydiqa_goldp_1shot/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["tydiqa_goldp_1shot"] = data["average"]["f1"]

    if not os.path.isfile(model_path + f"/tydiqa_no_context_1shot/metrics.json"):
        print(f"tydiqa_no_context_1shot missing for {model_name}")
        model_data["tydiqa_no_context_1shot"] = 0.0
    else:
        with open(model_path + f"/tydiqa_no_context_1shot/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["tydiqa_no_context_1shot"] = data["average"]["f1"]
            
    if not os.path.isfile(model_path + f"/alpaca_farm/metrics.json"):
        print(f"alpaca_farm missing for {model_name}")
        model_data["alpaca_farm"] = 0.0
    else:
        with open(model_path + f"/alpaca_farm/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["alpaca_farm"] = data["win_rate"]["model-greedy-long"]

    return model_data

science_data = []
print("Starting science models")
for model in os.listdir(science_path):
    model_path = science_path + model
    print(f"Evaluating {model_path}")
    results = collect_metrics(model_path)
    if results != None:
        science_data.append(results)

safety_data = []
print()
print("Starting safety models")
for model in os.listdir(safety_path):
    model_path = safety_path + model
    print(f"Evaluating {model_path}")
    results = collect_metrics(model_path)
    if results != None:
        safety_data.append(results)

df_science = pd.DataFrame(science_data)
df_science.to_csv("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/science/tulu/results.csv", index=False)
with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/science/tulu/results.json", "w") as f_out:
    for blob in science_data:
        f_out.write(json.dumps(blob) + '\n')

df_safety = pd.DataFrame(safety_data)
df_safety.to_csv("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/safety/tulu/results.csv", index=False)
with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/safety/tulu/results.json", "w") as f_out:
    for blob in safety_data:
        f_out.write(json.dumps(blob) + '\n')