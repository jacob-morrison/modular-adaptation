


import copy
import os
import subprocess
import yaml

# beaker session create --gpus 1 --budget ai2/oe-adapt \
#     --mount beaker://jacobm/llama_2_7b-tulu_all-coding_none=/tulu_all \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-coding_50=/coding_50 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-coding_100=/coding_100


weights = [
    (0.1, 0.9),
    (0.2, 0.8),
    (0.3, 0.7),
    (0.4, 0.6),
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
    (0.8, 0.2),
    (0.9, 0.1),
]

# weights = [
#     (1.0, 1.0),
#     (1.0, 0.9),
#     (1.0, 0.8),
#     (1.0, 0.7),
#     (1.0, 0.6),
#     (1.0, 0.5),
#     (1.0, 0.4),
#     (1.0, 0.3),
#     # (1.0, 0.), # coding 100
#     # (1.0, 0.), # coding 100
#     (1.0, 0.2),
#     (1.0, 0.1),
# ]

domain_models = {
    "safety_100": "/llama_2_7b-tulu_none-safety_100-4k",
    "coding_100": "/llama_2_7b-tulu_none-coding_100-4k",
    "science_2500": "/llama_2_7b-tulu_none-science_2500-4k",
}

merge_methods = [
    "linear_weighted",
    "task_arithmetic",
]

tulu_file = "/llama_2_7b-tulu_all_no_science_no_safety_no_coding"

def print_and_run(cmd):
    print(cmd)
    subprocess.run(cmd, shell=True)

for merge_method in merge_methods:
    # for science_amount in science_files:
    for model_tag in domain_models:
        # for (tuluWeight, scienceWeight) in weights:
        for (tuluWeight, domainWeight) in weights:
            # Copy yaml
            base_yaml = f"scripts/merge_models/merge-{merge_method}-base.yml"
            with open(base_yaml, 'r') as f:
                d1 = yaml.load(f.read(), Loader=yaml.FullLoader)
            d = copy.deepcopy(d1)
            if merge_method == "task_arithmetic":
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
            print_and_run("mkdir tmp-4k")
            file = open("tmp-4k/merge-config.yaml", "w")
            yaml.dump(d, file, default_flow_style=True)
            file.close()

            # Merge model
            print_and_run(f"mergekit-yaml tmp-4k/merge-config.yaml tmp-4k/ --cuda")

            # Upload model
            # model_name = f"{merge_method}-llama_2_7b-tulu_all_{tuluWeight}-{science_files[science_amount][1:]}_{scienceWeight}"
            model_name = f"{merge_method}-{tulu_file[1:]}_{tuluWeight}-{domain_models[model_tag][1:]}_{domainWeight}"
            print_and_run(f"beaker dataset create tmp-4k/ --name {model_name} --workspace ai2/modular_adaptation")

            # Cleanup
            print_and_run("rm -rf tmp-4k/")