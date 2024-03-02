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
        curr_results = {}
        if i == 0: # task,bioasq,biored,discomat,evidence_inference,evidence_inference,evidence_inference,multicite,mup,qasper,scierc,scifact,scifact,scifact,mean,median
            tasks = line.split(',')[1:]
        elif i == 1: # metric,f1,f1,bleu,f1_exact,f1_overlap,f1_substring,f1,bleu,bleu,f1,f1_evidence_sent,f1_evidence_token,f1_label
            metrics = line.split(',')[1:]
        else:
            curr_data = line.split(',')
            model = curr_data[0]
            model_tokens = model.split('-')
            base_model = model_tokens[0]
            tulu_model = model_tokens[1]
            science_model = model_tokens[2]
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

tulu_data = []
with open("results/current/baseline-tulu-evals.jsonl") as f_in:
    for line in f_in.readlines():
        data = json.loads(line)
        tulu_data.append(data)

df = pd.DataFrame(tulu_data)

print(df)