import copy
import subprocess
import yaml
import random
import re
import itertools
from datetime import date

today = date.today().strftime("%m%d%Y")

with open("beaker_configs/default_eval.yaml", 'r') as f:
    default_yaml = f.read()
d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)

# cluster = "ai2/general-cirrascale"
# cluster = "ai2/allennlp-cirrascale"
# cluster = "ai2/s2-cirrascale-l40"
# cluster = "ai2/mosaic-cirrascale-a100",
cluster = "ai2/general-cirrascale-a100-80g-ib"
# cluster = "ai2/prior-elanding"
num_gpus = 1
# d1['tasks'][0]['context']['cluster'] = cluster
d1['tasks'][0]['context']['priority'] = "low"
# d1['tasks'][0]['context']['priority'] = "preemptible"
d1['tasks'][0]['resources']['gpuCount'] = num_gpus

merge_models = False

# modify here for different set of experiments
experiment_groups = [
    "mmlu_0shot",
    "mmlu_5shot",
    "gsm_direct",
    "gsm_cot",
    "bbh_direct",
    "bbh_cot",
    "tydiqa_goldp_1shot",
    "tydiqa_no_context_1shot",
    "toxigen",
    "codex_eval_temp_0.1",
    "codex_eval_temp_0.8",
    "truthfulqa",
    "alpaca_farm",
]

lora = False

datasets = [
    # ("llama_2_7b-tulu_all-safety_none", "/model", "safety"),
    # ("tulu_2_7b_uncensored-tulu_none-safety_v1_all", "/model", "safety"),

    # ("01HRMHV81KCJT3FDWTQFGN85CG", "tulu-2-7b-safety-tulu-matched-adapt-3-epochs-real", "/model", "safety-for-faeze"),
    # ("01HRG4TSBFWGRCJR3VF9XDCWG5", "tulu-2-7b-bottom25-adapt-3-epochs", "/model", "safety-for-faeze"),
    # ("01HRTBDJCE7SBBF41ZCCY07E4Y", "tulu-2-7b-safety-adapt-1-epochs-real", "/model", "safety-for-faeze"),
    # ("01HRWX16NN9JPQ6Y6HQRW5FASE", "tulu-2-7b-uncensored-safety-adapt-1-epochs-real", "/model", "safety-for-faeze"),
    # ("01HS2C5BWB75WD1NBCEDZHH7BG", "tulu-2-7b-uncensored-safety-v0.1-contrastive-1-epoch", "/model", "safety-for-faeze"),
    # ("01HS6PEC7M8J8RTRSEVZ9PMAGH", "tulu-2-7b-uncensored-safety-v0.1-contrastive-2-epochs", "/model", "safety-for-faeze"),
    # ("01HS2C5C0TCD7CFBP5F8QF9YVF", "tulu-2-7b-uncensored-safety-v0.1-contrastive-3-epochs", "/model", "safety-for-faeze"),
    # ("01HS2BE0GH4G91H0T3CSXPQ2FY", "tulu-2-7b-safety-v0.1-contrastive-1-epoch", "/model", "safety-for-faeze"),
    # ("01HS2BE0NWMBYEGQ3EZHSVG1K5", "tulu-2-7b-uncensored-safety-v0.1-contrastive-2-epochs", "/model", "safety-for-faeze"),
    # ("01HS2BE0TRQJ81ZFNCFG7KZNXM", "tulu-2-7b-uncensored-safety-v0.1-contrastive-3-epochs", "/model", "safety-for-faeze"),
    # ("01HS93ZEYKGG0TD0XRB184JTMC", "tulu-2-7b-uncensored-only-contrastive-1-epoch", "/model", "safety-for-faeze"),
    # ("01HS93ZEMKA9MBWNAJ4JPV9H11", "tulu-2-7b-only-contrastive-1-epoch", "/model", "safety-for-faeze"),
    
    ("", "tulu_2_7b_uncensored-tulu_none-safety_all", "/model", "safety"),

]

# model to evaluate, each in the followng format: model name, their beaker id, checkpoint subfolder
models = [    
    # llama2 models
    ("llama2-7B", "01HCJYBBWA629B8GJTHPT496TT", None, "vanilla_lm"),
    # ("llama2-13B", "01HCJZQBM2KGQZSZRPF4HKVBZX", None, "vanilla_lm"),
    # ("llama2-70B", "01HCK281AFAXV2Y7T54NMNSC55", None, "vanilla_lm"),
    # ("llama2-chat-7B", "01HCT5D48MSRF0PCNAWNSJDN54", None, "tuned_lm"),
    # ("llama2-chat-13B", "01HCT5Q7A6FE8RZKY8TYN64ZW2", None, "tuned_lm"),
    # ("llama2-chat-70B", "01HCT63DVK7YPT6P9SN35XH417", None, "tuned_lm"),
]

#--------------- experiments about number of supervision tasks -------------------------

if not merge_models:
    # for experiment_group, model_info in itertools.product(experiment_groups, models):
    # for dataset, model_info, experiment_group in itertools.product(pairwise_trained_datasets, models, experiment_groups):
    for (beaker, model_name, model_path, save_subdir), model_info, experiment_group in itertools.product(datasets, models, experiment_groups):
        if beaker == "":
            beaker = "jacobm/" + model_name
        # if dataset != "/model":
        #     model_path = f'/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/checkpoints/domain_addition/{dataset}/'
        # else:
        #     model_path = "/model"
        # print(f"Submitting {experiment_group} for model: {dataset}")
        d = copy.deepcopy(d1)

        # model_name = model_info[0] + f"_{model_info[2]}" if model_info[2] is not None else model_info[0]
        # if lora:
            # name = f"open_instruct_eval_{experiment_group}_{model_name}_{dataset}_{today}".replace('/', '-')
        # else:
        name = f"oi_eval_{experiment_group}_{model_name}"
        # shorter_name = name.replace('llama_2_7b-', '')

        # if dataset == "/model":
        d['tasks'][0]['datasets'][1]['source']['beaker'] = beaker # "jacobm/" + model_name
        d['description'] = name
        d['tasks'][0]['name'] = name

        # if "dave" in dataset or "another" in dataset:
        #     save_dir = f"/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/{dataset}/{experiment_group}/"
        # else:
        #     save_dir = f"/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/{dataset.replace('/', '-')}/{experiment_group}/"
        # save_dir = f"/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/with_daves_tulu_model/daves_tulu_model/{experiment_group}/"
        # save_dir = "/output/"
        save_dir = f"/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/new_baselines_results/{save_subdir}/{model_name}/{experiment_group}/"

        if experiment_group == "mmlu_0shot":
            d['tasks'][0]['arguments'][0] = f'''
                python -m eval.mmlu.run_eval \
                --ntrain 0 \
                --data_dir /data/mmlu/ \
                --save_dir {save_dir} \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path} \
                --eval_batch_size 4 \
                --load_in_8bit \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
            '''
        elif experiment_group == "mmlu_5shot":
            d['tasks'][0]['arguments'][0] = f'''
                python -m eval.mmlu.run_eval \
                --ntrain 5 \
                --data_dir /data/mmlu/ \
                --save_dir {save_dir} \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path} \
                --eval_batch_size 4 \
                --load_in_8bit \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
            '''
        elif experiment_group == "bbh_direct":
            d['tasks'][0]['arguments'][0] = f'''
                python -m eval.bbh.run_eval \
                --data_dir /data/bbh \
                --save_dir {save_dir} \
                --use_vllm \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path} \
                --max_num_examples_per_task 40 \
                --no_cot \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
            '''
        elif experiment_group == "bbh_cot":
            d['tasks'][0]['arguments'][0] = f'''
                python -m eval.bbh.run_eval \
                --data_dir /data/bbh \
                --save_dir {save_dir} \
                --use_vllm \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path} \
                --max_num_examples_per_task 40 \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
            '''
        elif experiment_group == "gsm_direct":
            d['tasks'][0]['arguments'][0] = f'''
                python -m eval.gsm.run_eval \
                --data_dir /data/gsm/ \
                --max_num_examples 200 \
                --save_dir {save_dir} \
                --use_vllm \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path} \
                --n_shot 8 \
                --no_cot \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
            '''
        elif experiment_group == "gsm_cot":
            d['tasks'][0]['arguments'][0] = f'''
                python -m eval.gsm.run_eval \
                --data_dir /data/gsm/ \
                --max_num_examples 200 \
                --save_dir {save_dir} \
                --use_vllm \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path} \
                --n_shot 8 \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
            ''' 
        elif experiment_group == "tydiqa_goldp_1shot":
            d["tasks"][0]["arguments"][0] = f'''
                python -m eval.tydiqa.run_eval \
                --data_dir /data/tydiqa/ \
                --n_shot 1 \
                --max_num_examples_per_lang 100 \
                --max_context_length 512 \
                --save_dir {save_dir} \
                --use_vllm \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path} \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
            '''
        elif experiment_group == "tydiqa_no_context_1shot":
            d["tasks"][0]["arguments"][0] = f'''
                python -m eval.tydiqa.run_eval \
                --data_dir /data/tydiqa/ \
                --no_context \
                --n_shot 1 \
                --max_num_examples_per_lang 100 \
                --max_context_length 512 \
                --save_dir {save_dir} \
                --use_vllm \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path} \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
            '''
        elif experiment_group == "codex_eval_temp_0.1":
            d['tasks'][0]['arguments'][0] = f'''
                python -m eval.codex_humaneval.run_eval \
                --data_file /data/codex_humaneval/HumanEval.jsonl.gz \
                --eval_pass_at_ks 1 5 10 20 \
                --unbiased_sampling_size_n 20 \
                --temperature 0.1 \
                --save_dir {save_dir} \
                --use_vllm \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path}
            '''
        elif experiment_group == "codex_eval_temp_0.8":
            d['tasks'][0]['arguments'][0] = f'''
                python -m eval.codex_humaneval.run_eval \
                --data_file /data/codex_humaneval/HumanEval.jsonl.gz \
                --eval_pass_at_ks 1 5 10 20 \
                --unbiased_sampling_size_n 20 \
                --temperature 0.8 \
                --save_dir {save_dir} \
                --use_vllm \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path}
            '''
        elif experiment_group == "truthfulqa":
            d['tasks'][0]['arguments'][0] = f'''
            python -m eval.truthfulqa.run_eval \
                --data_dir /data/truthfulqa \
                --save_dir {save_dir} \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path} \
                --metrics truth info mc \
                --preset qa \
                --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
                --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
                --eval_batch_size 20 \
                --load_in_8bit \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
            '''
        elif experiment_group == "toxigen":
            d['tasks'][0]['arguments'][0] = f'''
            python -m eval.toxigen.run_eval \
                --data_dir /data/toxigen/ \
                --save_dir {save_dir} \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path} \
                --eval_batch_size 32 \
                --use_vllm \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
            '''
        elif experiment_group == "alpaca_farm":
            d['tasks'][0]['arguments'][0] = f'''
            python -m eval.alpaca_farm.run_eval \
                --use_vllm \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path} \
                --save_dir {save_dir} \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
            '''
        else:
            raise ValueError("experiment_group not supported")
        
        # TODO: fix if I use lora ever
        if lora:
            d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].strip() + f' --lora_weight_path /net/nfs.cirrascale/allennlp/jacobm/tulu_7B_lora_exp/pairwise_experts/{dataset}/'

        # if a specific checkpoint is specified, load model from that checkpoint
        if model_info[2] is not None:
            assert "--model_name_or_path /model" in d['tasks'][0]['arguments'][0]
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--model_name_or_path /model", "--model_name_or_path /model/"+model_info[2])]
            assert "--tokenizer_name_or_path /model" in d['tasks'][0]['arguments'][0]
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--tokenizer_name_or_path /model", "--tokenizer_name_or_path /model/"+model_info[2])]

        # for vanilla_lm, remove the chat formatting function
        if model_info[3] == "vanilla_lm" and not lora:
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--use_chat_format", "")]

        if "13B" in model_info[0]:
            # find the batch size argument, and reduce by 4x
            if "--eval_batch_size" in d['tasks'][0]['arguments'][0]:
                original_batch_size = re.search("--eval_batch_size (\d+)", d['tasks'][0]['arguments'][0]).group(1)
                new_batch_size = max(1, int(original_batch_size) // 2)
                d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--eval_batch_size {}".format(original_batch_size), "--eval_batch_size {}".format(new_batch_size))]


        if "30B" in model_info[0] or "34B" in model_info[0]:
            # find the batch size argument, and reduce by 4x
            if "--eval_batch_size" in d['tasks'][0]['arguments'][0]:
                original_batch_size = re.search("--eval_batch_size (\d+)", d['tasks'][0]['arguments'][0]).group(1)
                new_batch_size = max(1, int(original_batch_size) // 4)
                d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--eval_batch_size {}".format(original_batch_size), "--eval_batch_size {}".format(new_batch_size))]

            if "codex_eval" in experiment_group:
                # request 2x more GPUs
                d['tasks'][0]['resources']['gpuCount'] = 2 * d['tasks'][0]['resources']['gpuCount']
        
        elif "70B" in model_info[0] or "65B" in model_info[0] or "40B" in model_info[0]:
            # find the batch size argument, and reduce by 4x
            if "--eval_batch_size" in d['tasks'][0]['arguments'][0]:
                original_batch_size = re.search("--eval_batch_size (\d+)", d['tasks'][0]['arguments'][0]).group(1)
                new_batch_size = max(1, int(original_batch_size) // 4)
                d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--eval_batch_size {}".format(original_batch_size), "--eval_batch_size {}".format(new_batch_size))]

            if "codex_eval" in experiment_group:
                # request 4x more GPUs
                d['tasks'][0]['resources']['gpuCount'] = 4 * d['tasks'][0]['resources']['gpuCount']
            else:
                # request 2x more GPUs
                d['tasks'][0]['resources']['gpuCount'] = 2 * d['tasks'][0]['resources']['gpuCount']

        if model_info[0].startswith("hf-"):  # if it's a huggingface model, load it from the model hub
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--use_chat_format", "")]
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--model_name_or_path /model", "--model_name_or_path "+model_info[1])]
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--tokenizer_name_or_path /model", "--model_name_or_path "+model_info[1])]
        # else:  # if it's a beaker model, mount the beaker dataset to `/model`
            # d['tasks'][0]['datasets'][1]['source']['beaker'] = 01HKG46RNVAP3NSHNDH019R5KB # model_info[1]

        if "llama2-chat" in model_info[0]:
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace(
                "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format", 
                "--chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format")
            ]
        elif "code_llama_instruct" in model_info[0]:
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace(
                "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format", 
                "--chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format")
            ] 

        # TODO: vllm doesn't support lora yet
        if any([x in model_info[0] for x in ["opt", "pythia", "falcon"]]) or lora:
            if "--use_vllm" in d['tasks'][0]['arguments'][0]:
                print(f"Removing --use_vllm for {model_info[0]}")
                d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--use_vllm", "")] 

        # print(d)

        fn = "beaker_configs/auto_created/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/modular-adaptation-safety".format(fn)
        subprocess.Popen(cmd, shell=True)

# merge and eval models
else:
    # for experiment_group, model_info in itertools.product(experiment_groups, models):
    for i in range(1, len(datasets) - 1):
        for j in range(i + 1, len(datasets)):
            target_dataset_1 = datasets[i]
            target_dataset_2 = datasets[j]
            for model_info, experiment_group in itertools.product(models, experiment_groups):
                print(f"Submitting {experiment_group} for model: {model_info[0]}")
                print(f'merging datasets {target_dataset_1} and {target_dataset_2}')
                d = copy.deepcopy(d1)

                model_name = model_info[0] + f"_{model_info[2]}" if model_info[2] is not None else model_info[0]
                name = f"open_instruct_eval_{experiment_group}_merge_{target_dataset_1}_and_{target_dataset_2}_{today}"
                d['description'] = name
                d['tasks'][0]['name'] = name

                d['tasks'][0]['arguments'][0] = f'''
                python -u -m eval.merge_models     \
                    --base_model /model \
                    --target_lora_modules {target_dataset_1}  {target_dataset_2} \
                    --results_dir /output/ \
                    --task {experiment_group}
                '''        
                # if lora:
                    # d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].strip() + f' --lora_weight_path /net/nfs.cirrascale/allennlp/jacobm/tulu_7B_lora_exp/{dataset}/'

                if "13B" in model_info[0]:
                    # find the batch size argument, and reduce by 4x
                    if "--eval_batch_size" in d['tasks'][0]['arguments'][0]:
                        original_batch_size = re.search("--eval_batch_size (\d+)", d['tasks'][0]['arguments'][0]).group(1)
                        new_batch_size = max(1, int(original_batch_size) // 2)
                        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--eval_batch_size {}".format(original_batch_size), "--eval_batch_size {}".format(new_batch_size))]


                if "30B" in model_info[0] or "34B" in model_info[0]:
                    # find the batch size argument, and reduce by 4x
                    if "--eval_batch_size" in d['tasks'][0]['arguments'][0]:
                        original_batch_size = re.search("--eval_batch_size (\d+)", d['tasks'][0]['arguments'][0]).group(1)
                        new_batch_size = max(1, int(original_batch_size) // 4)
                        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--eval_batch_size {}".format(original_batch_size), "--eval_batch_size {}".format(new_batch_size))]

                    if "codex_eval" in experiment_group:
                        # request 2x more GPUs
                        d['tasks'][0]['resources']['gpuCount'] = 2 * d['tasks'][0]['resources']['gpuCount']
                
                elif "70B" in model_info[0] or "65B" in model_info[0] or "40B" in model_info[0]:
                    # find the batch size argument, and reduce by 4x
                    if "--eval_batch_size" in d['tasks'][0]['arguments'][0]:
                        original_batch_size = re.search("--eval_batch_size (\d+)", d['tasks'][0]['arguments'][0]).group(1)
                        new_batch_size = max(1, int(original_batch_size) // 4)
                        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--eval_batch_size {}".format(original_batch_size), "--eval_batch_size {}".format(new_batch_size))]

                    if "codex_eval" in experiment_group:
                        # request 4x more GPUs
                        d['tasks'][0]['resources']['gpuCount'] = 4 * d['tasks'][0]['resources']['gpuCount']
                    else:
                        # request 2x more GPUs
                        d['tasks'][0]['resources']['gpuCount'] = 2 * d['tasks'][0]['resources']['gpuCount']

                if model_info[0].startswith("hf-"):  # if it's a huggingface model, load it from the model hub
                    d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--use_chat_format", "")]
                    d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--model_name_or_path /model", "--model_name_or_path "+model_info[1])]
                    d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--tokenizer_name_or_path /model", "--model_name_or_path "+model_info[1])]
                else:  # if it's a beaker model, mount the beaker dataset to `/model`
                    d['tasks'][0]['datasets'][1]['source']['beaker'] = model_info[1]

                fn = "beaker_configs/auto_created/{}.yaml".format(name)
                file = open(fn, "w")
                yaml.dump(d, file, default_flow_style=True)
                file.close()

                cmd = "beaker experiment create {} --workspace ai2/lora-instruct".format(fn)
                subprocess.Popen(cmd, shell=True)
