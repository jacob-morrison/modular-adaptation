


import copy
import os
import subprocess
import yaml

# beaker session create --gpus 1 --budget ai2/oe-adapt  \
#     --mount beaker://jacobm/llama_2_7b-tulu_all_no_science_no_safety_no_coding=/llama_2_7b-tulu_consistent_mix \
#     --mount beaker://jacobm/tulu_2_7b_no_science_no_safety_no_coding-tulu_none-coding_100=/tulu_2_7b-tulu_none-coding_100 \
#     --mount beaker://jacobm/tulu_2_7b_no_science_no_safety_no_coding-tulu_none-safety_100=/tulu_2_7b-tulu_none-safety_100 \
#     --mount beaker://jacobm/tulu_2_7b_no_science_no_safety_no_coding-tulu_none-science_2500=/tulu_2_7b-tulu_none-science_2500

# beaker session create --gpus 1 --budget ai2/oe-adapt  \
#     --mount beaker://jacobm/llama_2_7b-tulu_all_no_science_no_safety_no_coding=/llama_2_7b-tulu_consistent_mix \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-coding_100-4k=/llama_2_7b-coding_100 \
#     --mount beaker://jacobm/tulu_2_7b_no_science_no_safety_no_coding-tulu_none-coding_100=/tulu_2_7b-tulu_none-coding_100 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-safety_100-4k=/llama_2_7b-safety_100 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-science_2500-4k=/llama_2_7b-science_2500

# beaker session create --gpus 1 --budget ai2/oe-adapt  \
#     --mount beaker://jacobm/llama_2_7b-tulu_all_no_science_no_safety=/llama_2_7b-tulu_consistent_mix_with_coding \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-coding_100-4k=/llama_2_7b-coding_100 \
#     --mount beaker://jacobm/tulu_2_7b_no_science_no_safety-tulu_none-coding_100=/tulu_2_7b_with_coding-tulu_none-coding_100

# beaker session create --gpus 1 --budget ai2/oe-adapt  \
#     --mount beaker://jacobm/llama_2_7b-tulu_all=/llama_2_7b-tulu_all \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-safety_100	=/llama_2_7b-tulu_none-safety_100 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-coding_100	=/llama_2_7b-tulu_none-coding_100 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-science_2500=/llama_2_7b-tulu_none-science_2500 

# beaker session create --gpus 1 --budget ai2/oe-adapt  \
#     --mount beaker://jacobm/llama_2_7b-tulu_all=/llama_2_7b-tulu_all \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-coding_100=/tulu_2_7b-tulu_none-coding_100 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-safety_100=/tulu_2_7b-tulu_none-safety_100 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-science_2500=/tulu_2_7b-tulu_none-science_2500 

# beaker session create --gpus 1 --budget ai2/oe-adapt  \
#     --mount beaker://jacobm/llama_2_7b-tulu_all=/llama_2_7b-tulu_all \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-science_100=/llama_2_7b-tulu_none-science_100 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-science_200=/llama_2_7b-tulu_none-science_200 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-science_500=/llama_2_7b-tulu_none-science_500 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-science_1000=/llama_2_7b-tulu_none-science_1000 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-safety_20=/llama_2_7b-tulu_none-safety_20 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-safety_40=/llama_2_7b-tulu_none-safety_40 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-safety_60=/llama_2_7b-tulu_none-safety_60 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-safety_80=/llama_2_7b-tulu_none-safety_80

# beaker session create --gpus 1 --budget ai2/oe-adapt  \
#     --mount beaker://faezeb/tulu-2-mix_llmbar_llama-2-13b-chat_rm=/tulu_2_mix_llmbar_llama_2_13b_chat_rm

weights = [
    # (0.1, 0.9),
    (0.2, 0.8),
    # (0.3, 0.7),
    (0.4, 0.6),
    (0.5, 0.5),
    (0.6, 0.4),
    # (0.7, 0.3),
    (0.8, 0.2),
    # (0.9, 0.1),
]

domain_models = {
   "faeze_dpo": "/tulu_2_mix_llmbar_llama_2_13b_chat_rm"
}

merge_methods = [
    "linear_weighted",
    # "task_arithmetic",
    # "dare_task_arithmetic",
    # "dare_linear",
    # "dare_ties",
    # "ties",
    # "slerp",
]

tulu_file = "allenai/tulu-2-dpo-13b"

def print_and_run(cmd):
    print(cmd)
    subprocess.run(cmd, shell=True)

for model_tag in domain_models:
    for merge_method in merge_methods:
        for (tuluWeight, domainWeight) in weights:
            # Copy yaml
            base_yaml = f"scripts/merge_models/merge-{merge_method}-base.yml"
            with open(base_yaml, 'r') as f:
                d1 = yaml.load(f.read(), Loader=yaml.FullLoader)
            d = copy.deepcopy(d1)
            if merge_method == "task_arithmetic" or merge_method == "dare_task_arithmetic":
                if tuluWeight == 0.65:
                    domainWeight = 0.53
                tuluWeight = 1.0
            if merge_method == "linear_weighted" or merge_method == "task_arithmetic":
                # Set merge-specific parameters
                d["models"][0]["model"] = tulu_file
                d["models"][0]["parameters"]["weight"] = tuluWeight
                d["models"][1]["model"] = domain_models[model_tag]
                d["models"][1]["parameters"]["weight"] = domainWeight
            elif merge_method in ["dare_linear", "dare_ties", "ties", "dare_task_arithmetic"]:
                # Set merge-specific parameters
                d["models"][1]["model"] = tulu_file
                d["models"][1]["parameters"]["weight"] = tuluWeight
                d["models"][2]["model"] = domain_models[model_tag]
                d["models"][2]["parameters"]["weight"] = domainWeight
            elif merge_method == "slerp":
                # Set merge-specific parameters
                d["slices"][0]["sources"][0]["model"] = tulu_file
                d["slices"][0]["sources"][1]["model"] = domain_models[model_tag]
                d["parameters"]["t"][0]["value"] = domainWeight
            else:
                raise Exception

            # Create folders and files
            print_and_run("mkdir tmp-dpo")
            file = open("tmp-dpo/merge-config.yaml", "w")
            yaml.dump(d, file, default_flow_style=True)
            file.close()

            # Merge model
            print_and_run(f"mergekit-yaml tmp-dpo/merge-config.yaml tmp-dpo/ --cuda")

            # Upload model
            # model_name = f"{merge_method}-llama_2_7b-tulu_all_{tuluWeight}-{science_files[science_amount][1:]}_{scienceWeight}"
            model_name = f"{merge_method}-{tulu_file.replace('allenai/', '')}_{tuluWeight}-{domain_models[model_tag][1:]}_{domainWeight}"
            print_and_run(f"beaker dataset create tmp-dpo/ --name {model_name} --workspace ai2/synth-data")

            # Cleanup
            print_and_run("rm -rf tmp-dpo/")