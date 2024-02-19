import copy
import os
import subprocess
import yaml

weights = [
    # (0.0, 1.0),
    (0.1, 0.9),
    (0.2, 0.8),
    (0.3, 0.7),
    (0.4, 0.6),
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
    (0.8, 0.2),
    (0.9, 0.1),
    # (1.0, 0.0),
]

yaml_files = [
    "scripts/mergekit-configs/merge-tulu-and-science-200-linear-weighted.yml",
    "scripts/mergekit-configs/merge-tulu-and-science-1000-linear-weighted.yml",
    "scripts/mergekit-configs/merge-tulu-and-science-2500-linear-weighted.yml"
]

output_dir = "/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/checkpoints/domain_addition/merged_models"

for yaml_file in yaml_files:
    if '200' in yaml_file:
        num_science = '200'
    elif '2500' in yaml_file:
        num_science = '2500'
    else:
        num_science = '1000'
    with open(yaml_file, 'r') as f:
        default_yaml = f.read()
    d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)
    for (scienceWeight, tuluWeight) in weights:
        d = copy.deepcopy(d1)

        # create yaml
        d["models"][0]["parameters"]["weight"] = scienceWeight
        d["models"][1]["parameters"]["weight"] = tuluWeight

        # merge models
        name = f"merge-tulu-{tuluWeight}-science-{num_science}-{scienceWeight}"
        fn = f"scripts/mergekit-configs/auto_created/{name}.yaml"
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/modular_adaptation".format(fn)
        cmd = (f"mergekit-yaml scripts/mergekit-configs/auto_created/{name}.yaml "
                f"{output_dir}/llama_2_7b-{tuluWeight}-tulu_only-{scienceWeight}-science_{num_science} "
                "--cuda")
        print(cmd)
        # subprocess.Popen(cmd, shell=True)

        # tulu evals

        # science evals