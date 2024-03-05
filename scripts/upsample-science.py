import json
import random

tulu_data = []
with open("/net/nfs.cirrascale/allennlp/davidw/proj/science-instruct/science-adapt/data/training_mixtures/4096/tulu_all_science_0_eval_no.jsonl") as f_in:
    for line in f_in.readlines():
        data = json.loads(line)
        tulu_data.append(data)

science_data = []
with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/tulu_all_science_2500_eval_no.jsonl") as f_in:
    for line in f_in.readlines():
        data = json.loads(line)
        science_data.append(data)

print(len(science_data))
science_data = science_data * 6
print(len(science_data))
science_data = science_data[:len(tulu_data)]
print(len(science_data))
full_data = tulu_data + science_data
random.shuffle(full_data)

with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/tulu_all_science_upsample_eval_no.jsonl", "w") as f:
    for x in full_data:
        print(json.dumps(x), file=f)