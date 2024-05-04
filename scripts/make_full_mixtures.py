import json
import random

tulu_examples = []
with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/tulu_v2_mix.jsonl") as f_in:
    for line in f_in.readlines():
        tulu_examples.append(line)

science_examples = []
with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/science/tulu_none_science_2500_eval_no.jsonl") as f_in:
    for line in f_in.readlines():
        science_examples.append(line)
science_examples = science_examples + tulu_examples[:len(science_examples)]
random.shuffle(science_examples)

with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/full_tulu_mixtures/tulu_mix_v2_match-science_2500.jsonl", "w") as f_out:
    for elem in science_examples:
        print(json.dumps(elem), file=f_out)

safety_examples = []
with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/safety/tulu_none-safety_100.jsonl") as f_in:
    for line in f_in.readlines():
        safety_examples.append(line)
safety_examples = safety_examples + tulu_examples[:len(safety_examples)]
random.shuffle(safety_examples)

with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/full_tulu_mixtures/tulu_mix_v2_match-safety_100.jsonl", "w") as f_out:
    for elem in safety_examples:
        print(json.dumps(elem), file=f_out)

code_examples_50p = []
with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/coding/tulu_none-coding_50.jsonl") as f_in:
    for line in f_in.readlines():
        code_examples_50p.append(line)
code_examples_50p = code_examples_50p + tulu_examples[:len(code_examples_50p)]
random.shuffle(code_examples_50p)

with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/full_tulu_mixtures/tulu_mix_v2_match-coding_50.jsonl", "w") as f_out:
    for elem in code_examples_50p:
        print(json.dumps(elem), file=f_out)

code_examples = []
with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/coding/tulu_none-coding_100.jsonl") as f_in:
    for line in f_in.readlines():
        code_examples.append(line)
code_examples = code_examples + tulu_examples[:len(code_examples)]
random.shuffle(code_examples)

with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/full_tulu_mixtures/tulu_mix_v2_match-coding_100.jsonl", "w") as f_out:
    for elem in code_examples:
        print(json.dumps(elem), file=f_out)