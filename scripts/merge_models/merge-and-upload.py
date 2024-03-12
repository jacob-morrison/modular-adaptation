import copy
import os
import subprocess
import yaml

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

science_files = {
    "500": "/science_500",
    "100": "/science_100",
    "200": "/science_200",
    # "1000": "/science_1000",
    # "2500": "/science_2500",
    # "upsample": "/science_upsample"
}

merge_methods = [
    # "linear_weighted",
    # "dare_linear",
    "dare_ties",
    "ties",
    "slerp",
]

tulu_file = "/tulu_all"

def print_and_run(cmd):
    print(cmd)
    subprocess.run(cmd, shell=True)

for merge_method in merge_methods:
    for science_amount in science_files:
        for (tuluWeight, scienceWeight) in weights:
            # Copy yaml
            base_yaml = f"scripts/merge_models/merge-{merge_method}-base.yml"
            with open(base_yaml, 'r') as f:
                d1 = yaml.load(f.read(), Loader=yaml.FullLoader)
            d = copy.deepcopy(d1)
            if merge_method == "linear_weighted":
                # Set merge-specific parameters
                d["models"][0]["model"] = tulu_file
                d["models"][0]["parameters"]["weight"] = tuluWeight
                d["models"][1]["model"] = science_files[science_amount]
                d["models"][1]["parameters"]["weight"] = scienceWeight
            elif merge_method in ["dare_linear", "dare_ties", "ties"]:
                # Set merge-specific parameters
                d["models"][1]["model"] = tulu_file
                d["models"][1]["parameters"]["weight"] = tuluWeight
                d["models"][2]["model"] = science_files[science_amount]
                d["models"][2]["parameters"]["weight"] = scienceWeight
            elif merge_method == "slerp":
                # Set merge-specific parameters
                d["slices"][0]["sources"][0]["model"] = tulu_file
                d["slices"][0]["sources"][1]["model"] = science_files[science_amount]
                d["parameters"]["t"][0]["value"] = scienceWeight
            else:
                raise Exception

            # Create folders and files
            print_and_run("mkdir tmp")
            file = open("tmp/merge-config.yaml", "w")
            yaml.dump(d, file, default_flow_style=True)
            file.close()

            # Merge model
            print_and_run(f"mergekit-yaml tmp/merge-config.yaml tmp/ --cuda")

            # Upload model
            model_name = f"{merge_method}-llama_2_7b-tulu_all_{tuluWeight}-{science_files[science_amount][1:]}_{scienceWeight}"
            print_and_run(f"beaker dataset create tmp --name {model_name} --workspace ai2/modular-adaptation-science")

            # Cleanup
            print_and_run("rm -rf tmp")