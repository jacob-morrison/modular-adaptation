import json
import random

tulu_data = []
with open("/net/nfs.cirrascale/allennlp/davidw/proj/science-instruct/science-adapt/data/training_mixtures/4096/tulu_all_science_0_eval_no.jsonl") as f_in:
    for line in f_in.readlines():
        data = json.loads(line)
        tulu_data.append(data)

science_data = []
with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/tulu_none_science_2500_eval_no.jsonl") as f_in:
    for line in f_in.readlines():
        data = json.loads(line)
        science_data.append(data)

random.shuffle(tulu_data)
random.shuffle(science_data)

total_number_of_examples = len(tulu_data) + len(science_data)
print(f"Total number of examples: {total_number_of_examples}\n")

science_data = science_data * 7
tulu_data = tulu_data * 2

mix_weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for science_weight in mix_weights:
    num_science_examples = total_number_of_examples * science_weight
    print(f"Science weight: {science_weight}")
    print(f"Number of science examples: {num_science_examples}")
    print(f"Number of Tulu examples: {total_number_of_examples - num_science_examples}\n")
    full_data = tulu_data[:(total_number_of_examples - num_science_examples)] + science_data[:num_science_examples]
    random.shuffle(full_data)

    with open(f"/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/pareto_data/tulu_{total_number_of_examples - num_science_examples}-science_{num_science_examples}.jsonl", "w") as f:
        for x in full_data:
            print(json.dumps(x), file=f)