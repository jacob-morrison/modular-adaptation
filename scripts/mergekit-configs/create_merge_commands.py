import copy
import os
import subprocess
import yaml

# math for dataset-sized weighting:
# Tulu-only: 318686
# Science 200: 8193
# Science 1000: 35357
# Science 2500: 61349

science_model_weights = {
    '200': [(0.025, 0.975)],
    '1000': [], # [(0.1, 0.9)], # unnecessary
    '2500': [(0.16, 0.84)],
}

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

yaml_files = [
    # "scripts/mergekit-configs/slerp_merges/merge-daves-tulu-and-science-200-slerp-weighted.yml",
    # "scripts/mergekit-configs/slerp_merges/merge-daves-tulu-and-science-1000-slerp-weighted.yml",
    "scripts/mergekit-configs/slerp_merges/merge-daves-tulu-and-science-2500-slerp-weighted.yml"
]

output_dir = "/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/checkpoints/domain_addition/with_daves_tulu_model"

for yaml_file in yaml_files:
    if '200' in yaml_file:
        num_science = '200'
    elif '1000' in yaml_file:
        num_science = '1000'
    elif '2500' in yaml_file:
        num_science = '2500'
    else:
        print('what')
        quit()
    with open(yaml_file, 'r') as f:
        default_yaml = f.read()
    d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)
    for (scienceWeight, tuluWeight) in weights + science_model_weights[num_science]:
        d = copy.deepcopy(d1)

        # create yaml
        # d["models"][0]["parameters"]["weight"] = scienceWeight
        # d["models"][1]["parameters"]["weight"] = tuluWeight
        d["parameters"]["t"][0]["value"] = tuluWeight

        # merge models
        name = f"slerp-daves-tulu-{tuluWeight}-science-{num_science}-{scienceWeight}"
        fn = f"scripts/mergekit-configs/slerp_merges/auto_created/{name}.yaml"
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/modular_adaptation".format(fn)
        cmd = (f"mergekit-yaml scripts/mergekit-configs/slerp_merges/auto_created/{name}.yaml "
                f"{output_dir}/slerp-{tuluWeight}-tulu_only-{scienceWeight}-science_{num_science} "
                "--cuda")
        print(cmd)
        # subprocess.Popen(cmd, shell=True)

        # tulu evals

        # science evals