import json
import random

coding_only_path = "/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/coding/tulu_none-coding_100.jsonl"

coding_examples = []
with open(coding_only_path) as f_in:
    for line in f_in.readlines():
        coding_examples.append(json.loads(line)) 

random.shuffle(coding_examples)

amounts = [
    20,
    40,
    60,
    80,
]

for amt in amounts:
    current_coding_examples = coding_examples[:int(len(coding_examples) * amt / 100.)]
    with open(f"/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/coding/tulu_none-coding_{amt}.jsonl", "w") as f_out:
        for elem in current_coding_examples:
            print(json.dumps(elem), file=f_out)