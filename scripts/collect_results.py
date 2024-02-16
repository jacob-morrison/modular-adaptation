import json
import os

domain_adaptation_path = "/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/"
tulu_evals = os.listdir(domain_adaptation_path)
tulu_evals.remove("gsm_cot")
tulu_evals.remove("science")
tulu_evals.remove("collected")

tulu_metrics = [
    "bbh_cot",
    "bbh_direct",
    "codex_eval_temp_0.1",
    "codex_eval_temp_0.8",
    "gsm_cot",
    "gsm_direct",
    "mmlu_0shot",
    "mmlu_5shot",
    "toxigen",
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

for model in tulu_evals:
    model_path = domain_adaptation_path + model
    if "merged_models" in model:
        tulu_model = "tulu_no_science"
        tulu_model_weight = float(model[25:28])
        if "200" in model:
            science_model = "tulu_none_science_200_eval_no"
            science_model_weight = float(model[59:62])
        else:
            science_model = "tulu_none_science_1000_eval_no"
            science_model_weight = float(model[60:63])
    else:
        tulu_model = "tulu_no_science"
        tulu_model_weight = 1.0
        science_model = "N/A"
        science_model_weight = 0.0

    model_data = {
        "base_model": "llama-2-7b",
        "tulu_model": tulu_model,
        "tulu_model_weight": tulu_model_weight,
        "science_model": science_model,
        "science_model_weight": science_model_weight,
    }

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

    with open(model_path + f"/tydiqa_goldp_1shot/metrics.json") as f_in:
        data = json.loads(f_in.read())
        model_data["tydiqa_goldp_1shot"] = data["average"]["f1"]

    with open(model_path + f"/tydiqa_no_context_1shot/metrics.json") as f_in:
        data = json.loads(f_in.read())
        model_data["tydiqa_no_context_1shot"] = data["average"]["f1"]

    # TODO: collect science metrics

    full_data.append(model_data)

from pprint import pprint
pprint(full_data)

with open(domain_adaptation_path + "collected", "w") as f_out:
    for blob in full_data:
        f_out.write(json.dumps(blob))