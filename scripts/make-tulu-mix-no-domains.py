import json
import random

tulu_no_science_no_safety_no_coding = []
with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/consistent_mix/tulu_all_no_science_no_safety_no_coding.jsonl") as f_in:
    for line in f_in.readlines():
        tulu_no_science_no_safety_no_coding.append(json.loads(line)) 

science_amounts = [
    "science_100",
    "science_200",
    "science_500",
    "science_1000",    
]

for amt in science_amounts:
    science_examples = []
    with open(f"/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/science/tulu_none_{amt}_eval_no.jsonl") as f_in:
        for line in f_in.readlines():
            science_examples.append(json.loads(line))
    
    all_combined = science_examples + tulu_no_science_no_safety_no_coding
    random.shuffle(all_combined)
    with open(f"/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/consistent_mix/tulu_all_no_science_no_safety_no_coding-{amt}.jsonl", "w") as f_out:
        for elem in all_combined:
            print(json.dumps(elem), file=f_out)

    match_combined = science_examples + tulu_no_science_no_safety_no_coding[:len(science_examples)]
    random.shuffle(match_combined)
    with open(f"/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/consistent_mix/tulu_match_no_science_no_safety_no_coding-{amt}.jsonl", "w") as f_out:
        for elem in match_combined:
            print(json.dumps(elem), file=f_out)

safety_amounts = [
    "safety_20",
    "safety_40",
    "safety_60",
    "safety_80",
]

for amt in safety_amounts:
    science_examples = []
    with open(f"/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/safety/tulu_none-{amt}.jsonl") as f_in:
        for line in f_in.readlines():
            science_examples.append(json.loads(line))
    
    all_combined = science_examples + tulu_no_science_no_safety_no_coding
    random.shuffle(all_combined)
    with open(f"/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/consistent_mix/tulu_all_no_science_no_safety_no_coding-{amt}.jsonl", "w") as f_out:
        for elem in all_combined:
            print(json.dumps(elem), file=f_out)

    match_combined = science_examples + tulu_no_science_no_safety_no_coding[:len(science_examples)]
    random.shuffle(match_combined)
    with open(f"/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/consistent_mix/tulu_match_no_science_no_safety_no_coding-{amt}.jsonl", "w") as f_out:
        for elem in match_combined:
            print(json.dumps(elem), file=f_out)

coding_amounts = [
    "coding_20",
    "coding_40",
    "coding_60",
    "coding_80",
]

for amt in coding_amounts:
    science_examples = []
    with open(f"/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/coding/tulu_none-{amt}.jsonl") as f_in:
        for line in f_in.readlines():
            science_examples.append(json.loads(line))
    
    all_combined = science_examples + tulu_no_science_no_safety_no_coding
    random.shuffle(all_combined)
    with open(f"/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/consistent_mix/tulu_all_no_science_no_safety_no_coding-{amt}.jsonl", "w") as f_out:
        for elem in all_combined:
            print(json.dumps(elem), file=f_out)

    match_combined = science_examples + tulu_no_science_no_safety_no_coding[:len(science_examples)]
    random.shuffle(match_combined)
    with open(f"/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/consistent_mix/tulu_match_no_science_no_safety_no_coding-{amt}.jsonl", "w") as f_out:
        for elem in match_combined:
            print(json.dumps(elem), file=f_out)