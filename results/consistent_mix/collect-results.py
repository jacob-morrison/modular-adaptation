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
    model_name = model_path.split("/")[-1].replace("-4k", "")
    tokens = model_name.split("-")
    if tokens[0] in [
        "linear_weighted",
        "ties",
        "dare_linear",
        "dare_ties",
        "slerp",
        "task_arithmetic",
        "dare_task_arithmetic",
    ]:
        merge_method = tokens[0]
        tokens = tokens[1:]
        tokens[1] = tokens[1]
        base_model_weight = float(tokens[1].split("_")[-1])
        tokens[1] = tokens[1].replace(f"_{base_model_weight}", "")
        tokens[2] = "-".join(tokens[2:])
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
        if "coding_100" in model_path:
            print(f"codex_eval_plus_temp_0.1 missing for {model_name}")
        model_data["codex_eval_plus_temp_0.1"] = 0.0
    else:
        with open(model_path + f"/codex_eval_plus_temp_0.1/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["codex_eval_plus_temp_0.1"] = data["pass@1"]

    # codex_eval_plus_temp_0.8/
    if not os.path.isfile(model_path + f"/codex_eval_plus_temp_0.8/metrics.json"):
        if "coding_100" in model_path:
            print(f"codex_eval_plus_temp_0.8 missing for {model_name}")
        model_data["codex_eval_plus_temp_0.8"] = 0.0
    else:
        with open(model_path + f"/codex_eval_plus_temp_0.8/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["codex_eval_plus_temp_0.8"] = data["pass@10"]

    # mbpp_temp_0.1/
    if not os.path.isfile(model_path + f"/mbpp_temp_0.1/metrics.json"):
        if "coding_100" in model_path:
            print(f"mbpp_temp_0.1 missing for {model_name}")
        model_data["mbpp_temp_0.1"] = 0.0
    else:
        with open(model_path + f"/mbpp_temp_0.1/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["mbpp_temp_0.1"] = data["pass@1"]

    # mbpp_temp_0.8/
    if not os.path.isfile(model_path + f"/mbpp_temp_0.8/metrics.json"):
        if "coding_100" in model_path:
            print(f"mbpp_temp_0.8 missing for {model_name}")
        model_data["mbpp_temp_0.8"] = 0.0
    else:
        with open(model_path + f"/mbpp_temp_0.8/metrics.json") as f_in:
            data = json.loads(f_in.read())
            model_data["mbpp_temp_0.8"] = data["pass@10"]

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
    model_name = model_name.replace("-4k", "")
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
    for line in f_in.readlines("\t"):
        if i == 0:
            tasks = line.split("\t")
            i += 1
        elif i == 1:
            metrics = line.split("\t")
            metrics.append("null")
            metrics.append("null")
            i += 1
        else:
            tokens = line.split()
            model_key = tokens[0].replace("-4k", "")
            curr_data = {
                "model_key": model_key,
            }
            for task, metric, value in zip(tasks[1:], metrics[1:], tokens[1:]):
                curr_data[f"{task}_{metric}"] = float(value)
                data_map[model_key][f"{task}_{metric}"] = float(value)
            science_eval_data.append(curr_data)

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