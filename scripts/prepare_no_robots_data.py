from datasets import load_dataset
import json

def process_example(example):

    return {
        "dataset": "no_robots",
        "id": example['prompt_id'],
        "messages": example['messages'],
        "category": example['category'],
    }

dataset = load_dataset("HuggingFaceH4/no_robots")

data = {}

for datapoint in dataset['train_sft']:
    category = datapoint['category']
    if category not in data:
        data[category] = []
    data[category].append(datapoint)

# Training files to create:
    # All
    # Per category
    # Splitting into n experts

num_experts = 3
expert_data = {
    0: [],
    1: [],
    2: [],
    # 3: [],
}

i = 0

with open('training_data/no_robots.jsonl', 'w') as all_data:
    for category in data:
        with open(f'training_data/no_robots-{category}.jsonl', 'w') as category_data:
            for elem in data[category]:
                all_data.write(json.dumps(elem) + '\n')
                category_data.write(json.dumps(elem) + '\n')
                expert_data[i % num_experts].append(elem)
                i += 1
        # print(category)
        # print(len(data[category]))
        # print()

for expert_num in range(num_experts):
    with open(f'training_data/no_robots-expert_{expert_num}.jsonl', 'w') as expert_file:
        for elem in expert_data[expert_num]:
            expert_file.write(json.dumps(elem) + '\n')