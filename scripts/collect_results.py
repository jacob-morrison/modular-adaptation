import json
import os

rejected_files = [
    "merged_models-llama_2_7b-0.1-tulu_none_science_200_eval_no-0.9-tulu_no_science",
    "merged_models-llama_2_7b-0.2-tulu_none_science_200_eval_no-0.8-tulu_no_science",
    "merged_models-llama_2_7b-0.3-tulu_none_science_200_eval_no-0.7-tulu_no_science",
    "merged_models-llama_2_7b-0.4-tulu_none_science_200_eval_no-0.6-tulu_no_science",
    "merged_models-llama_2_7b-0.5-tulu_none_science_200_eval_no-0.5-tulu_no_science",
    "merged_models-llama_2_7b-0.6-tulu_none_science_200_eval_no-0.4-tulu_no_science",
    "merged_models-llama_2_7b-0.7-tulu_none_science_200_eval_no-0.3-tulu_no_science",
    "merged_models-llama_2_7b-0.8-tulu_none_science_200_eval_no-0.2-tulu_no_science",
    "merged_models-llama_2_7b-0.9-tulu_none_science_200_eval_no-0.1-tulu_no_science",
]

domain_adaptation_path = "/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/"
tulu_evals_tmp = os.listdir(domain_adaptation_path)
tulu_evals = []
for string in tulu_evals_tmp:
    print(string)
    if string not in rejected_files:
        tulu_evals.append(string)
tulu_evals.remove("llama_2_7b-tulu_no_science") # can't delete this for some reason
tulu_evals.remove("science")
tulu_evals.remove("with_daves_tulu_model")
tulu_evals.remove("another_tulu_only_model")
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

def collect_metrics(model_path):
    pass

for model in tulu_evals:
    model_path = domain_adaptation_path + model
    merge_method = model.split('-')[0]
    if merge_method == "llama-2-7b":
        merge_method = "linear_weighted"
    if "merged_models" in model:
        tulu_model = "tulu_no_science"
        # TODO: fix characters
        # tulu_model_weight = float(model[25:28])
        # science_model_weight = float(model[39:42])
        tulu_model_weight, science_model_weight = get_model_weights(model)
        if "200" in model:
            science_model = "tulu_none_science_200_eval_no"
        elif "1000" in model:
            science_model = "tulu_none_science_1000_eval_no"
        elif "2500" in model:
            science_model = "tulu_none_science_2500_eval_no"
    else:
        if '200' in model or '1000' in model or "2500" in model:
            if 'tulu_all' in model:
                tulu_model = model.replace('llama_2_7b-', '')
                tulu_model_weight = 1.0
                science_model_weight = 0.0
                science_model = "N/A"
            else:
                tulu_model = "N/A"
                tulu_model_weight = 0.0
                science_model_weight = 1.0
                if '200' in model:
                    science_model = "tulu_none_science_200_eval_no"
                elif "1000" in model:
                    science_model = "tulu_none_science_1000_eval_no"
                elif "2500" in model:
                    science_model = "tulu_none_science_2500_eval_no"
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
        "merge_method": merge_method,
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

for model in os.listdir(domain_adaptation_path + "with_daves_tulu_model/"):
    model_path = domain_adaptation_path + "with_daves_tulu_model/" + model
    if "daves_tulu_model" not in model:
        tulu_model = "daves_tulu_no_science"
        # TODO: fix characters
        # tulu_model_weight = float(model[11:14])
        # science_model_weight = float(model[25:28])
        tulu_model_weight, science_model_weight = get_model_weights(model)
        if "200" in model:
            science_model = "tulu_none_science_200_eval_no"
        elif "1000" in model:
            science_model = "tulu_none_science_1000_eval_no"
        elif "2500" in model:
            science_model = "tulu_none_science_2500_eval_no"
    else:
        # if '200' in model or '1000' in model or "2500" in model:
        #     if 'tulu_all' in model:
        #         tulu_model = model.replace('llama_2_7b-', '')
        #         tulu_model_weight = 1.0
        #         science_model_weight = 0.0
        #         science_model = "N/A"
        #     else:
        #         tulu_model = "N/A"
        #         tulu_model_weight = 0.0
        #         science_model_weight = 1.0
        #         if '200' in model:
        #             science_model = "tulu_none_science_200_eval_no"
        #         elif "1000" in model:
        #             science_model = "tulu_none_science_1000_eval_no"
        #         elif "2500" in model:
        #             science_model = "tulu_none_science_2500_eval_no"
        # else:
        tulu_model = "daves_tulu_no_science"
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

for model in os.listdir(domain_adaptation_path + "another_tulu_only_model/"):
    model_path = domain_adaptation_path + "another_tulu_only_model/" + model
    if "llama_2_7b-tulu_all_science_none_eval_no" not in model:
        tulu_model = "another_tulu_all_science_none"
        # TODO: fix characters
        tulu_model_weight, science_model_weight = get_model_weights(model)
        # tulu_model_weight = float(model[11:14])
        # science_model_weight = float(model[25:28])
        if "200" in model:
            science_model = "tulu_none_science_200_eval_no"
        elif "1000" in model:
            science_model = "tulu_none_science_1000_eval_no"
        elif "2500" in model:
            science_model = "tulu_none_science_2500_eval_no"
    else:
        # if '200' in model or '1000' in model or "2500" in model:
        #     if 'tulu_all' in model:
        #         tulu_model = model.replace('llama_2_7b-', '')
        #         tulu_model_weight = 1.0
        #         science_model_weight = 0.0
        #         science_model = "N/A"
        #     else:
        #         tulu_model = "N/A"
        #         tulu_model_weight = 0.0
        #         science_model_weight = 1.0
        #         if '200' in model:
        #             science_model = "tulu_none_science_200_eval_no"
        #         elif "1000" in model:
        #             science_model = "tulu_none_science_1000_eval_no"
        #         elif "2500" in model:
        #             science_model = "tulu_none_science_2500_eval_no"
        # else:
        tulu_model = "another_tulu_all_science_none"
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

with open(domain_adaptation_path + "collected/results.json", "w") as f_out:
    for blob in full_data:
        f_out.write(json.dumps(blob) + '\n')