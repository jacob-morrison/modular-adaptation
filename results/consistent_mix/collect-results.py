import json
import os
import pandas as pd
from pprint import pprint

tulu_eval_path = "/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/consistent_mix/tulu_evals/"

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

def get_model_weights(model_name):
    tokens = model_name.split('-')
    domain_model_weight = float(tokens[-1].split('_')[-1])
    tulu_model_weight = float(tokens[2].split('_')[-1])
    return tulu_model_weight, domain_model_weight

def collect_metrics(model_path):
    model_name = model_path.split("/")[-1]
    tokens = model_name.split("-")
    if tokens[0] in [
        "linear_weighted",
        "ties",
        "dare_linear",
        "dare_ties",
        "slerp",
        "task_arithmetic",
        "dare_task_arithmetic",
        "ties_task_arithmetic",
    ]:
        merge_method = tokens[0]
        tokens = tokens[1:]
        base_model_weight = float(tokens[1].split("_")[-1])
        tokens[1] = tokens[1].replace(f"_{base_model_weight}", "")
        tokens[2] = "-".join(tokens[2:])
        if "tulu_2_7b" in tokens[2]:
            merge_method = "wise-ft"
        tokens = tokens[:3]
        domain_model_weight = float(tokens[2].split("_")[-1])
        tokens[2] = tokens[2].replace(f"_{domain_model_weight}", "")
    else:
        base_model_weight = 1.0
        domain_model_weight = 1.0
        merge_method = "N/A"
    base_model = tokens[0]
    tulu_model = tokens[1]
    if len(tokens) == 2:
        domain_model = "None"
    else:
        domain_model = tokens[2].split("-")[-1]

    # merged = model_name.split("-")[0] in [
    #     "linear_weighted",
    #     "ties",
    #     "dare_linear",
    #     "dare_ties",
    #     "slerp",
    #     "task_arithmetic",
    #     "dare_task_arithmetic",
    # ]
    # if merged:
    #     tokens = model_name.split('-')
    #     merge_method = tokens[0]
    #     base_model = tokens[1]
    #     tulu_model = tokens[2][:-4]
    #     domain_model = tokens[3][:-4]
    #     tulu_model_weight, domain_model_weight = get_model_weights(model_name)
    # else:
    #     merge_method = "N/A"
    #     tokens = model_name.split('-')
    #     if len(tokens) == 1:
    #         base_model = tokens[0]
    #         tulu_model = tokens[0]
    #         domain_model = tokens[0]
    #     elif len(tokens) == 2:
    #         base_model = tokens[0]
    #         tulu_model = tokens[1]
    #         domain_model = "none"
    #     else:
    #         base_model = tokens[0]
    #         tulu_model = tokens[1]
    #         domain_model = tokens[2]
    #     if tulu_model == "tulu_none":
    #         tulu_model_weight = 0.0
    #     else:
    #         tulu_model_weight = 1.0
    #     if domain_model == "science_none":
    #         domain_model_weight = 0.0
    #     else:
    #         domain_model_weight = 1.0

    model_data = {
        "model_key": model_name,
        "base_model": base_model,
        "tulu_model": tulu_model,
        "tulu_model_weight": base_model_weight,
        "domain_model": domain_model,
        "domain_model_weight": domain_model_weight,
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
            
    if not os.path.isfile(model_path + f"/alpaca_eval/metrics.json"):
        print(f"alpaca_eval missing for {model_name}")
        model_data["alpaca_eval"] = 0.0
    else:
        with open(model_path + f"/alpaca_eval/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["alpaca_eval"] = data["win_rate"]["model-greedy-long"]

    # codex_eval_plus_temp_0.1/
    if not os.path.isfile(model_path + f"/codex_eval_plus_temp_0.1/metrics.json"):
        if "coding_" in model_path:
            print(f"codex_eval_plus_temp_0.1 missing for {model_name}")
        model_data["codex_eval_plus_temp_0.1"] = 0.0
    else:
        with open(model_path + f"/codex_eval_plus_temp_0.1/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["codex_eval_plus_temp_0.1"] = data["pass@1"]

    # codex_eval_plus_temp_0.8/
    if not os.path.isfile(model_path + f"/codex_eval_plus_temp_0.8/metrics.json"):
        if "coding_" in model_path:
            print(f"codex_eval_plus_temp_0.8 missing for {model_name}")
        model_data["codex_eval_plus_temp_0.8"] = 0.0
    else:
        with open(model_path + f"/codex_eval_plus_temp_0.8/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["codex_eval_plus_temp_0.8"] = data["pass@10"]

    # mbpp_temp_0.1/
    if not os.path.isfile(model_path + f"/mbpp_temp_0.1/metrics.json"):
        if "coding_" in model_path:
            print(f"mbpp_temp_0.1 missing for {model_name}")
        model_data["mbpp_temp_0.1"] = 0.0
    else:
        with open(model_path + f"/mbpp_temp_0.1/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["mbpp_temp_0.1"] = data["pass@1"]

    # mbpp_temp_0.8/
    if not os.path.isfile(model_path + f"/mbpp_temp_0.8/metrics.json"):
        if "coding_" in model_path:
            print(f"mbpp_temp_0.8 missing for {model_name}")
        model_data["mbpp_temp_0.8"] = 0.0
    else:
        with open(model_path + f"/mbpp_temp_0.8/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["mbpp_temp_0.8"] = data["pass@10"]

    if model_name == "linear_weighted-llama_2_7b-tulu_all_with_coding_0.5-tulu_2_7b_with_coding-tulu_none-coding_80_0.5":
        print(model_data)
        # quit()

    return model_data

tulu_data = []
data_map = {}
print("Starting tulu evals")
for model in os.listdir(tulu_eval_path):
    model_path = tulu_eval_path + model
    print()
    print(f"Evaluating {model_path}")
    results = collect_metrics(model_path)
    tulu_data.append(results)
    data_map[results["model_key"]] = results

# read safety evals
# read science evals

# Read safety eval files
# save to /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/safety/safety/
safety_eval_path = "/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/consistent_mix/safety_evals/"
harmbench = {}
with open(safety_eval_path + "/harmbench_behaviors_text_all/results_harmbench.tsv") as f_in:
    for line in f_in.readlines():
        tokens = line.split("\t")
        if tokens[0] == "Model":
            continue
        harmbench[tokens[0]] = float(tokens[2])
safety_eval_data = []
for model_name in os.listdir(safety_eval_path + "xstest_v2_prompts_annotated"):
    merged = model_name.split("-")[0] in [
        "linear_weighted",
        "ties",
        "dare_linear",
        "dare_ties",
        "slerp",
    ]
    original_model_name = model_name
    model_name = model_name.replace("-4k", "").replace(
            "llama_2_7b-coding_100_0", "llama_2_7b-tulu_none-coding_100_0"
        ).replace(
            "llama_2_7b-safety_100_0", "llama_2_7b-tulu_none-safety_100_0"
        ).replace(
            "llama_2_7b-science_2500_0", "llama_2_7b-tulu_none-science_2500_0"
        )
    if merged:
        tokens = model_name.split('-')
        merge_method = tokens[0]
        base_model = tokens[1]
        tulu_model = tokens[2][:-4]
        safety_model = tokens[3][:-4]
        tulu_model_weight, safety_model_weight = get_model_weights(model_name)
    else:
        merge_method = "N/A"
        tokens = model_name.split('-')
        base_model = tokens[0]
        tulu_model = tokens[1]
        if len(tokens) == 2:
            safety_model = "safety_none"
        else:
            safety_model = tokens[2]
        if tulu_model == "tulu_none":
            tulu_model_weight = 0.0
        else:
            tulu_model_weight = 1.0
        if safety_model == "safety_none":
            safety_model_weight = 0.0
        else:
            safety_model_weight = 1.0
    if original_model_name not in harmbench:
        pprint(harmbench)
    model_data = {
        "model_key": model_name,
        "base_model": base_model,
        "tulu_model": tulu_model,
        "tulu_model_weight": tulu_model_weight,
        "safety_model": safety_model,
        "safety_model_weight": safety_model_weight,
        "merge_method": merge_method,
        "harmbench": harmbench[original_model_name]
    }
    data_map[model_name]["harmbench"] = harmbench[original_model_name]
    
    with open(safety_eval_path + "xstest_v2_prompts_annotated/" + original_model_name + "/compliance_xstest_orig.tsv") as f_in:
        for line in f_in.readlines():
            tokens = line.strip().split('\t')
            if tokens[0] == "safe_average":
                model_data["safe_average"] = float(tokens[1])
                data_map[model_name]["safe_average"] = float(tokens[1])
            elif tokens[0] == "unsafe_average":
                model_data["unsafe_average"] = float(tokens[1])
                data_map[model_name]["unsafe_average"] = float(tokens[1])
        safety_eval_data.append(model_data)

science_eval_data = []
with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/consistent_mix/science/metrics/tables/summary.tsv") as f_in:
    i = 0
    for line in f_in.readlines():
        line = line.replace("\t", ",").strip()
        if i == 0:
            tasks = line.split(",")
            i += 1
        elif i == 1:
            metrics = line.split(",")[:-2]
            metrics.append("null")
            metrics.append("null")
            i += 1
        else:
            tokens = line.split(",")
            model_key = tokens[0]
            curr_data = {
                "model_key": model_key,
            }
            for task, metric, value in zip(tasks[1:], metrics[1:], tokens[1:]):
                if value == '':
                    value = 0.0 # TODO: fix and replace later
                curr_data[f"{task}_{metric}"] = float(value)
                data_map[model_key][f"{task}_{metric}"] = float(value)
            science_eval_data.append(curr_data)

# TODO: collect test results
science_eval_data_test = []
with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/consistent_mix/science-test/metrics/tables/summary.tsv") as f_in:
    i = 0
    for line in f_in.readlines():
        line = line.replace("\t", ",").strip()
        if i == 0:
            tasks = line.split(",")
            i += 1
        elif i == 1:
            metrics = line.split(",")[:-2]
            metrics.append("null")
            metrics.append("null")
            i += 1
        else:
            tokens = line.split(",")
            model_key = tokens[0]
            curr_data = {
                "model_key": model_key,
            }
            for task, metric, value in zip(tasks[1:], metrics[1:], tokens[1:]):
                if value == '':
                    value = 0.0 # TODO: fix and replace later
                curr_data[f"{task}_{metric}_TEST"] = float(value)
                data_map[model_key][f"{task}_{metric}_TEST"] = float(value)
            science_eval_data_test.append(curr_data)

# need to edit a bunch of rows
            
from pprint import pprint
pprint(data_map.keys())
        
for dataset in [
    "coding_20",
    "coding_40",
    "coding_60",
    "coding_80",
    "coding_100",
    "safety_20",
    "safety_40",
    "safety_60",
    "safety_80",
    "safety_100",
    "science_100",
    "science_200",
    "science_500",
    "science_1000",
    "science_2500",
    ]:
    row = data_map[f"llama_2_7b-tulu_none-{dataset}"].copy()
    row["model_key"] = f"linear_weighted-llama_2_7b-tulu_all_0.0-llama_2_7b-tulu_none-{dataset}_1.0"
    row["merge_method"] = "linear_weighted"
    row["tulu_model_weight"] = 0.0
    row["domain_model_weight"] = 1.0
    row["tulu_model"] = "tulu_all"
    row["domain_model"] = dataset
    data_map[f"linear_weighted-llama_2_7b-tulu_all_0.0-llama_2_7b-tulu_none-{dataset}_1.0"] = row

    row2 = data_map["llama_2_7b-tulu_all"].copy()
    row2["model_key"] = f"linear_weighted-llama_2_7b-tulu_all_1.0-llama_2_7b-tulu_none-{dataset}_0.0"
    row2["merge_method"] = "linear_weighted"
    row2["tulu_model_weight"] = 1.0
    row2["domain_model_weight"] = 0.0
    row2["tulu_model"] = "tulu_all"
    row2["domain_model"] = dataset
    data_map[f"linear_weighted-llama_2_7b-tulu_all_1.0-llama_2_7b-tulu_none-{dataset}_0.0"] = row2

    row3 = data_map["llama_2_7b-tulu_all"].copy()
    row3["model_key"] = f"task_arithmetic-llama_2_7b-tulu_all_1.0-llama_2_7b-tulu_none-{dataset}_0.0"
    row3["merge_method"] = "task_arithmetic"
    row3["tulu_model_weight"] = 1.0
    row3["domain_model_weight"] = 0.0
    row3["tulu_model"] = "tulu_all"
    row3["domain_model"] = dataset
    data_map[f"task_arithmetic-llama_2_7b-tulu_all_1.0-llama_2_7b-tulu_none-{dataset}_0.0"] = row3

    row4 = data_map["llama_2_7b-tulu_all"].copy()
    row4["model_key"] = f"linear_weighted-llama_2_7b-tulu_all_1.0-tulu_2_7b-tulu_none-{dataset}_0.0"
    row4["merge_method"] = "wise-ft"
    row4["tulu_model_weight"] = 1.0
    row4["domain_model_weight"] = 0.0
    row4["tulu_model"] = "tulu_all"
    row4["domain_model"] = dataset
    data_map[f"linear_weighted-llama_2_7b-tulu_all_1.0-tulu_2_7b-tulu_none-{dataset}_0.0"] = row4

    row5 = data_map[f"tulu_2_7b-tulu_none-{dataset}"].copy()
    row5["model_key"] = f"linear_weighted-llama_2_7b-tulu_all_0.0-tulu_2_7b-tulu_none-{dataset}_1.0"
    row5["merge_method"] = "wise-ft"
    row5["tulu_model_weight"] = 0.0
    row5["domain_model_weight"] = 1.0
    row5["tulu_model"] = "tulu_all"
    row5["domain_model"] = dataset
    data_map[f"linear_weighted-llama_2_7b-tulu_all_0.0-tulu_2_7b-tulu_none-{dataset}_1.0"] = row5

for dataset in [
    "coding_20",
    "coding_40",
    "coding_60",
    "coding_80",
    "coding_100",
]:
    row = data_map[f"llama_2_7b-tulu_none-{dataset}"].copy()
    row["model_key"] = f"linear_weighted-llama_2_7b-tulu_all_with_coding_0.0-llama_2_7b-tulu_none-{dataset}_1.0"
    row["merge_method"] = "linear_weighted"
    row["tulu_model_weight"] = 0.0
    row["domain_model_weight"] = 1.0
    row["tulu_model"] = "tulu_all_with_coding"
    row["domain_model"] = dataset
    data_map[f"linear_weighted-llama_2_7b-tulu_all_with_coding_0.0-llama_2_7b-tulu_none-{dataset}_1.0"] = row

    row2 = data_map["llama_2_7b-tulu_all_with_coding"].copy()
    row2["model_key"] = f"linear_weighted-llama_2_7b-tulu_all_with_coding_1.0-llama_2_7b-tulu_none-{dataset}_0.0"
    row2["merge_method"] = "linear_weighted"
    row2["tulu_model_weight"] = 1.0
    row2["domain_model_weight"] = 0.0
    row2["tulu_model"] = "tulu_all_with_coding"
    row2["domain_model"] = dataset
    data_map[f"linear_weighted-llama_2_7b-tulu_all_with_coding_1.0-llama_2_7b-tulu_none-{dataset}_0.0"] = row2

    row3 = data_map["llama_2_7b-tulu_all_with_coding"].copy()
    row3["model_key"] = f"task_arithmetic-llama_2_7b-tulu_all_with_coding_1.0-llama_2_7b-tulu_none-{dataset}_0.0"
    row3["merge_method"] = "task_arithmetic"
    row3["tulu_model_weight"] = 1.0
    row3["domain_model_weight"] = 0.0
    row3["tulu_model"] = "tulu_all_with_coding"
    row3["domain_model"] = dataset
    data_map[f"task_arithmetic-llama_2_7b-tulu_all_with_coding_1.0-llama_2_7b-tulu_none-{dataset}_0.0"] = row3

    row4 = data_map["llama_2_7b-tulu_all_with_coding"].copy()
    row4["model_key"] = f"linear_weighted-llama_2_7b-tulu_all_with_coding_1.0-tulu_2_7b_with_coding-tulu_none-{dataset}_0.0"
    row4["merge_method"] = "wise-ft"
    row4["tulu_model_weight"] = 1.0
    row4["domain_model_weight"] = 0.0
    row4["tulu_model"] = "tulu_all_with_coding"
    row4["domain_model"] = dataset
    data_map[f"linear_weighted-llama_2_7b-tulu_all_with_coding_1.0-tulu_2_7b_with_coding-tulu_none-{dataset}_0.0"] = row4

    row5 = data_map[f"tulu_2_7b-tulu_none-{dataset}"].copy()
    row5["model_key"] = f"linear_weighted-llama_2_7b-tulu_all_with_coding_0.0-tulu_2_7b_with_coding-tulu_none-{dataset}_1.0"
    row5["merge_method"] = "wise-ft"
    row5["tulu_model_weight"] = 0.0
    row5["domain_model_weight"] = 1.0
    row5["tulu_model"] = "tulu_all_with_coding"
    row5["domain_model"] = dataset
    data_map[f"linear_weighted-llama_2_7b-tulu_all_with_coding_0.0-tulu_2_7b_with_coding-tulu_none-{dataset}_1.0"] = row5

for dataset in [
    "science_500",
    "science_1000",
    "science_2500",
]:
    row4 = data_map["llama_2_7b-tulu_all"].copy()
    row4["model_key"] = f"linear_weighted-llama_2_7b-tulu_all_1.0-tulu_match-{dataset}_0.0"
    row4["merge_method"] = "wise-ft"
    row4["tulu_model_weight"] = 1.0
    row4["domain_model_weight"] = 0.0
    row4["tulu_model"] = "tulu_all"
    row4["domain_model"] = dataset
    data_map[f"linear_weighted-llama_2_7b-tulu_all_1.0-tulu_2_7b-tulu_match-{dataset}_0.0"] = row4

    row5 = data_map[f"tulu_2_7b-tulu_match-{dataset}"].copy()
    row5["model_key"] = f"linear_weighted-llama_2_7b-tulu_all_0.0-tulu_2_7b-tulu_match-{dataset}_1.0"
    row5["merge_method"] = "wise-ft"
    row5["tulu_model_weight"] = 0.0
    row5["domain_model_weight"] = 1.0
    row5["tulu_model"] = "tulu_all"
    row5["domain_model"] = dataset
    data_map[f"linear_weighted-llama_2_7b-tulu_all_0.0-tulu_2_7b-tulu_match-{dataset}_1.0"] = row5

for dataset in [
    "science_2500",
    "coding_100",
]:
    row2 = data_map["llama_2_7b-tulu_all"].copy()
    row2["model_key"] = f"dare_task_arithmetic-llama_2_7b-tulu_all_1.0-llama_2_7b-tulu_none-{dataset}_0.0"
    row2["merge_method"] = "dare_task_arithmetic"
    row2["tulu_model_weight"] = 1.0
    row2["domain_model_weight"] = 0.0
    row2["tulu_model"] = "tulu_all"
    row2["domain_model"] = dataset
    data_map[f"dare_task_arithmetic-llama_2_7b-tulu_all_1.0-llama_2_7b-tulu_none-{dataset}_0.0"] = row2

    row2 = data_map["llama_2_7b-tulu_all"].copy()
    row2["model_key"] = f"ties_task_arithmetic-llama_2_7b-tulu_all_1.0-llama_2_7b-tulu_none-{dataset}_0.0"
    row2["merge_method"] = "ties_task_arithmetic"
    row2["tulu_model_weight"] = 1.0
    row2["domain_model_weight"] = 0.0
    row2["tulu_model"] = "tulu_all"
    row2["domain_model"] = dataset
    data_map[f"ties_task_arithmetic-llama_2_7b-tulu_all_1.0-llama_2_7b-tulu_none-{dataset}_0.0"] = row2

df = pd.DataFrame(data_map.values())
print(df)
df.to_csv("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/consistent_mix/results.csv", index=False)
df.to_csv("/net/nfs.cirrascale/allennlp/jacobm/modular-adaptation/results/consistent_mix/results.csv", index=False)

# df_safety_evals = pd.DataFrame(safety_eval_data)
# df_safety_evals.to_csv("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/safety/safety/results.csv", index=False)
# with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/safety/safety/results.jsonl", "w") as f_out:
#     for blob in safety_eval_data:
#         f_out.write(json.dumps(blob) + "\n")

# df_science = pd.DataFrame(science_data)
# df_science.to_csv("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/science/tulu/results.csv", index=False)
# with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/science/tulu/results.jsonl", "w") as f_out:
#     for blob in science_data:
#         f_out.write(json.dumps(blob) + '\n')

# df_safety = pd.DataFrame(safety_data)
# df_safety.to_csv("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/safety/tulu/results.csv", index=False)
# with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/safety/tulu/results.jsonl", "w") as f_out:
#     for blob in safety_data:
#         f_out.write(json.dumps(blob) + '\n')

# df_coding = pd.DataFrame(coding_data)
# df_coding.to_csv("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/coding/tulu/results.csv", index=False)
# with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/coding/tulu/results.jsonl", "w") as f_out:
#     for blob in coding_data:
#         f_out.write(json.dumps(blob) + '\n')