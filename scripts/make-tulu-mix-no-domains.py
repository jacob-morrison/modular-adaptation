import json
import random

coding_path = "/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/coding/tulu_all-coding_none.jsonl"
science_path = "/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/science/tulu_all_science_none_eval_no.jsonl"
safety_path = "/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/safety/tulu_all-safety_none.jsonl"

safety_examples = set()
science_examples = set()
coding_examples = set()

tulu_no_science_no_safety = []
tulu_no_science_no_safety_no_coding = []

with open(safety_path) as f_in:
    for line in f_in.readlines():
        safety_examples.add(line)

with open(coding_path) as f_in:
    for line in f_in.readlines():
        coding_examples.add(line)

with open(science_path) as f_in:
    for line in f_in.readlines():
        if line in safety_examples:
            tulu_no_science_no_safety.append(json.loads(line))
            if line in coding_examples:
                tulu_no_science_no_safety_no_coding.append(json.loads(line))

# with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/consistent_mix/tulu_all_no_science_no_safety.jsonl", "w") as f_out:
#     for elem in tulu_no_science_no_safety:
#         print(json.dumps(elem), file=f_out)

# with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/consistent_mix/tulu_all_no_science_no_safety_no_coding.jsonl", "w") as f_out:
#     for elem in tulu_no_science_no_safety_no_coding:
#         print(json.dumps(elem), file=f_out)
                
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