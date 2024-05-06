import json

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

with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/tulu_all_no_science_no_safety.jsonl", "w") as f_out:
    for elem in tulu_no_science_no_safety:
        print(json.dumps(elem), file=f_out)

with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/tulu_no_science_no_safety_no_coding.jsonl", "w") as f_out:
    for elem in tulu_no_science_no_safety_no_coding:
        print(json.dumps(elem), file=f_out)