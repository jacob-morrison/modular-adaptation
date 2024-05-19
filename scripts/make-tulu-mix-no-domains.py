import json
import random

tulu_no_science_no_safety_no_coding = []
with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/consistent_mix/tulu_all_no_science_no_safety.jsonl") as f_in:
    for line in f_in.readlines():
        tulu_no_science_no_safety_no_coding.append(json.loads(line)) 

science_amounts = [
    "science_100",
    "science_200",
    "science_500",
    "science_1000",    
]

safety_amounts = [
    "safety_20",
    "safety_40",
    "safety_60",
    "safety_80",
]

coding_amounts = [
    "coding_20",
    "coding_40",
    "coding_60",
    "coding_80",
]

science_only_path = "/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/science/tulu_none_science_2500_eval_no.jsonl"
safety_only_path = "/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/safety/tulu_none-safety_100.jsonl"
coding_only_path = "/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/coding/tulu_none-coding_100.jsonl"

science_only_examples = []
with open(science_only_path) as f_in:
    for line in f_in.readlines():
        science_only_examples.append(json.loads(line))

safety_only_examples = []
with open(safety_only_path) as f_in:
    for line in f_in.readlines():
        safety_only_examples.append(json.loads(line))

coding_only_examples = []
with open(coding_only_path) as f_in:
    for line in f_in.readlines():
        coding_only_examples.append(json.loads(line))

random.shuffle(tulu_no_science_no_safety)
random.shuffle(tulu_no_science_no_safety_no_coding)

tulu_1_and_science = science_only_examples + tulu_no_science_no_safety[:len(science_only_examples)]
random.shuffle(tulu_1_and_science)
with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/consistent_mix/tulu_match_no_science_no_safety-science_2500.jsonl", "w") as f_out:
    for elem in tulu_1_and_science:
        print(json.dumps(elem), file=f_out)

tulu_2_and_science = science_only_examples + tulu_no_science_no_safety_no_coding[:len(science_only_examples)]
random.shuffle(tulu_2_and_science)
with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/consistent_mix/tulu_match_no_science_no_safety_no_coding-science_2500.jsonl", "w") as f_out:
    for elem in tulu_2_and_science:
        print(json.dumps(elem), file=f_out)

tulu_1_and_safety = safety_only_examples + tulu_no_science_no_safety[:len(safety_only_examples)]
random.shuffle(tulu_1_and_safety)
with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/consistent_mix/tulu_match_no_science_no_safety-safety_100.jsonl", "w") as f_out:
    for elem in tulu_1_and_safety:
        print(json.dumps(elem), file=f_out)

tulu_2_and_safety = safety_only_examples + tulu_no_science_no_safety_no_coding[:len(safety_only_examples)]
random.shuffle(tulu_2_and_safety)
with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/consistent_mix/tulu_match_no_science_no_safety_no_coding-safety_100.jsonl", "w") as f_out:
    for elem in tulu_2_and_safety:
        print(json.dumps(elem), file=f_out)

tulu_1_and_coding = coding_only_examples + tulu_no_science_no_safety[:len(coding_only_examples)]
random.shuffle(tulu_1_and_coding)
with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/consistent_mix/tulu_match_no_science_no_safety-coding_100.jsonl", "w") as f_out:
    for elem in tulu_1_and_coding:
        print(json.dumps(elem), file=f_out)
        
tulu_2_and_coding = coding_only_examples + tulu_no_science_no_safety_no_coding[:len(coding_only_examples)]
random.shuffle(tulu_2_and_coding)
with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/consistent_mix/tulu_match_no_science_no_safety_no_coding-coding_100.jsonl", "w") as f_out:
    for elem in tulu_2_and_coding:
        print(json.dumps(elem), file=f_out)