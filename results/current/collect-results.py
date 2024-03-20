import json
import os

baselines_path = "/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/fixed_4k/baselines/"
merged_models_path = "/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/fixed_4k/merged_models/"
science_path = "/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/fixed_4k/baselines/"
# tulu_evals = os.listdir(baselines_path) # + os.listdir(merged_models_path)

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

full_data = []
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

def collect_metrics(model_path, merged=False):
    model_name = model_path.split("/")[-1]
    if "llama_2" not in model_name.split("-")[0] and "tulu_2" not in model_name.split("-")[0]:
        merged = True
    if merged:
        tokens = model_name.split('-')
        merge_method = tokens[0]
        base_model = tokens[1]
        tulu_model = tokens[2][:-4]
        science_model = tokens[3][:-4].replace("_4096", "")
        tulu_model_weight, science_model_weight = get_model_weights(model_name)
    else:
        merge_method = "N/A"
        tokens = model_name.split('-')
        base_model = tokens[0]
        tulu_model = tokens[1]
        science_model = tokens[2].replace("_4096", "")
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

    try:

        with open(model_path + f"/bbh_cot/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["bbh_cot"] = data["average_exact_match"]

        with open(model_path + f"/bbh_direct/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["bbh_direct"] = data["average_exact_match"]

        with open(model_path + f"/codex_eval_temp_0.1/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["codex_eval_temp_0.1"] = data["pass@1"]

        with open(model_path + f"/codex_eval_temp_0.8/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["codex_eval_temp_0.8"] = data["pass@10"]

        with open(model_path + f"/gsm_cot/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["gsm_cot"] = data["exact_match"]

        with open(model_path + f"/gsm_direct/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["gsm_direct"] = data["exact_match"]

        with open(model_path + f"/mmlu_0shot/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["mmlu_0shot"] = data["average_acc"]

        with open(model_path + f"/mmlu_5shot/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["mmlu_5shot"] = data["average_acc"]

        with open(model_path + f"/toxigen/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["toxigen"] = data["overall"]

        with open(model_path + f"/truthfulqa/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["truthfulqa"] = data["truth-info acc"]

        with open(model_path + f"/tydiqa_goldp_1shot/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["tydiqa_goldp_1shot"] = data["average"]["f1"]

        with open(model_path + f"/tydiqa_no_context_1shot/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["tydiqa_no_context_1shot"] = data["average"]["f1"]
            
        with open(model_path + f"/alpaca_farm/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["alpaca_farm"] = data["win_rate"]["model-greedy-long"]
    except:
        print(f"Couldn't find metric for {model_path}")
        # return None

    return model_data

print("Starting baseline models")
for model in os.listdir(baselines_path):
    model_path = baselines_path + model
    print(f"Evaluating {model_path}")
    results = collect_metrics(model_path)
    if results != None:
        full_data.append(results)

print()
print("Starting merged models")
for model in os.listdir(merged_models_path):
    model_path = merged_models_path + model
    print(f"Evaluating {model_path}")
    results = collect_metrics(model_path, merged=True)
    if results != None:
        full_data.append(results)

# from pprint import pprint
# pprint(full_data)

with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/fixed_4k/collected/results.json", "w") as f_out:
    for blob in full_data:
        f_out.write(json.dumps(blob) + '\n')