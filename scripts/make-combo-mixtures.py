import json
import random

tulu_all_no_science_no_safety_no_coding = []
with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/consistent_mix/tulu_all_no_science_no_safety_no_coding.jsonl") as f_in:
    for line in f_in.readlines():
        tulu_all_no_science_no_safety_no_coding.append(json.loads(line)) 
random.shuffle(tulu_all_no_science_no_safety_no_coding)

science_examples = []
with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/science/tulu_none_science_2500_eval_no.jsonl") as f_in:
    for line in f_in.readlines():
        science_examples.append(json.loads(line))

safety_examples = []
with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/safety/tulu_none-safety_100.jsonl") as f_in:
    for line in f_in.readlines():
        safety_examples.append(json.loads(line))

code_examples = []
with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/coding/tulu_none-coding_100.jsonl") as f_in:
    for line in f_in.readlines():
        code_examples.append(json.loads(line))

three_domains = science_examples + safety_examples + code_examples
random.shuffle(three_domains)
# with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/consistent_mix/tulu_none-science_2500-safety_100-coding_100.jsonl", "w") as f_out:
#     for elem in three_domains:
#         print(json.dumps(elem), file=f_out)

double_tulu = tulu_all_no_science_no_safety_no_coding * 2

tulu_and_three_domains = double_tulu[:len(three_domains)] + three_domains
random.shuffle(tulu_and_three_domains)
with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/consistent_mix/tulu_match_no_science_no_safety_no_coding-science_2500-safety_100-coding_100.jsonl", "w") as f_out:
    for elem in tulu_and_three_domains:
        print(json.dumps(elem), file=f_out)