import copy
import os
import subprocess
import yaml

# beaker session create --gpus 1 --budget ai2/oe-adapt \
#     --mount beaker://jacobm/llama_2_7b-tulu_all-coding_none=/tulu_all \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-coding_50=/coding_50 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-coding_100=/coding_100


# weights = [
#     (0.1, 0.9),
#     (0.2, 0.8),
#     (0.3, 0.7),
#     (0.4, 0.6),
#     (0.5, 0.5),
#     (0.6, 0.4),
#     (0.7, 0.3),
#     (0.8, 0.2),
#     (0.9, 0.1),
# ]

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

weights = [
    # linear weighted
    # (0.80, 0.20), # code 50
    # (0.66, 0.34), # code 100

    # task arithmetic
    (1.0, 0.26), # code 50
    # (1.0, 0.51), # code 100
]

coding_files = {
    "50": "/coding_50",
    # "100": "/coding_100",
}

merge_methods = [
    # "linear_weighted",
    "task_arithmetic",
]

tulu_file = "/tulu_all"

def print_and_run(cmd):
    print(cmd)
    subprocess.run(cmd, shell=True)

for merge_method in merge_methods:
    # for science_amount in science_files:
    for coding_amount in coding_files:
        # for (tuluWeight, scienceWeight) in weights:
        for (tuluWeight, codingWeight) in weights:
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
                d["models"][1]["model"] = coding_files[coding_amount]
                d["models"][1]["parameters"]["weight"] = codingWeight
            elif merge_method in ["dare_linear", "dare_ties", "ties", "dare_task_arithmetic"]:
                # Set merge-specific parameters
                d["models"][1]["model"] = tulu_file
                d["models"][1]["parameters"]["weight"] = tuluWeight
                d["models"][2]["model"] = coding_files[coding_amount]
                d["models"][2]["parameters"]["weight"] = codingWeight
            elif merge_method == "slerp":
                # Set merge-specific parameters
                d["slices"][0]["sources"][0]["model"] = tulu_file
                d["slices"][0]["sources"][1]["model"] = coding_files[coding_amount]
                d["parameters"]["t"][0]["value"] = codingWeight
            else:
                raise Exception

            # Create folders and files
            print_and_run("mkdir tmp-coding")
            file = open("tmp-coding/merge-config.yaml", "w")
            yaml.dump(d, file, default_flow_style=True)
            file.close()

            # Merge model
            print_and_run(f"mergekit-yaml tmp-coding/merge-config.yaml tmp-coding/ --cuda")

            # Upload model
            # model_name = f"{merge_method}-llama_2_7b-tulu_all_{tuluWeight}-{science_files[science_amount][1:]}_{scienceWeight}"
            model_name = f"{merge_method}-llama_2_7b-tulu_all_{tuluWeight}-{coding_files[coding_amount][1:]}_{codingWeight}"
            print_and_run(f"beaker dataset create tmp-coding/ --name {model_name} --workspace ai2/modular-adaptation-coding")

            # Cleanup
            print_and_run("rm -rf tmp-coding/")