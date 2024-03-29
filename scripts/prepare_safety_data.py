import json
import random

# with open("/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/train_data/tulu_none_science_2500_eval_no.jsonl") as f_in:
science_amounts = [
    # "safety_10",
    # "safety_20",
    # "safety_30",
    # "safety_40",
    # "safety_50",
    # "safety_60",
    # "safety_70",
    # "safety_80",
    # "safety_90",
    # "safety_100",
    "safety_upsample",
    # "safety_v0_all"
]
tulu_amounts = [
    # "tulu_all",
    # "tulu_match",
    "tulu_none",
]
for tulu_amount in tulu_amounts:
    for safety_amount in science_amounts:
        if safety_amount == "safety_upsample" and tulu_amount == "tulu_match":
            continue
        science_data = []
        print(safety_amount)
        with open(f"training_data/safety-mixtures/tulu_none-safety_100.jsonl") as f_in:
            for line in f_in.readlines():
                data = json.loads(line)
                science_data.append(data)

        tulu_data = []
        with open("training_data/safety-mixtures/tulu_all-safety_none.jsonl") as f_in:
            for line in f_in.readlines():
                data = json.loads(line)
                tulu_data.append(data)
        
        if tulu_amount == "tulu_match":
            random.shuffle(tulu_data)
            tulu_data = tulu_data[:len(science_data)]

        print(len(science_data))
        if safety_amount == "safety_upsample":
            science_data = science_data * 6
            print(len(science_data))
            science_data = science_data[:len(tulu_data)]
            print(len(science_data))
        print(len(tulu_data))
        full_data = tulu_data + science_data
        random.shuffle(full_data)
        print(len(full_data))

        with open("training_data/safety-mixtures/tulu_none-safety_upsample.jsonl", "w") as f:
            for x in science_data:
                print(json.dumps(x), file=f)

        with open(f"training_data/safety-mixtures/tulu_all-{safety_amount}.jsonl", "w") as f:
            for x in full_data:
                print(json.dumps(x), file=f)

        print()