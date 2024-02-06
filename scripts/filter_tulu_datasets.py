

base_path = '/net/nfs.cirrascale/allennlp/jacobm/tulu_data/tulu-v2/'

# tulu_no_code_or_wizardlm_datasets = [
#     'flan_v2_filtered_data.jsonl',
#     'hard_coded_filtered_data.jsonl',
#     'open_orca_filtered_data.jsonl',
#     'sharegpt_filtered_data.jsonl',
#     'cot_filtered_data.jsonl',
#     'gpt4_alpaca_filtered_data.jsonl',
#     'lima_filtered_data.jsonl',
#     'oasst1_filtered_data.jsonl',

#     'science_filtered_data.jsonl',

#     # 'code_alpaca_filtered_data.jsonl',
#     # 'wizardlm_filtered_data.jsonl',

#     # 'tulu_v2_filtered_data.jsonl',
#     # 'tulu_v2_data.jsonl',
# ]

# with open(f'{base_path}tulu_v2_filtered_minus_wizardlm_and_code_alpaca.jsonl', 'w') as f_out:
#     for dataset in tulu_no_code_or_wizardlm_datasets:
#         with open(f'{base_path}{dataset}') as f_in:
#             for line in f_in.readlines():
#                 f_out.write(line)

# tulu_no_science_datasets = [
#     'flan_v2_filtered_data.jsonl',
#     'hard_coded_filtered_data.jsonl',
#     'open_orca_filtered_data.jsonl',
#     'sharegpt_filtered_data.jsonl',
#     'cot_filtered_data.jsonl',
#     'gpt4_alpaca_filtered_data.jsonl',
#     'lima_filtered_data.jsonl',
#     'oasst1_filtered_data.jsonl',
#     'code_alpaca_filtered_data.jsonl',
#     'wizardlm_filtered_data.jsonl',

#     # 'science_filtered_data.jsonl',

#     # 'tulu_v2_filtered_data.jsonl',
#     # 'tulu_v2_data.jsonl',
# ]

# with open(f'{base_path}tulu_v2_filtered_minus_science.jsonl', 'w') as f_out:
#     for dataset in tulu_no_science_datasets:
#         with open(f'{base_path}{dataset}') as f_in:
#             for line in f_in.readlines():
#                 f_out.write(line)

tulu_only_code_and_wizardlm_datasets = [
    # 'flan_v2_filtered_data.jsonl',
    # 'hard_coded_filtered_data.jsonl',
    # 'open_orca_filtered_data.jsonl',
    # 'sharegpt_filtered_data.jsonl',
    # 'cot_filtered_data.jsonl',
    # 'gpt4_alpaca_filtered_data.jsonl',
    # 'lima_filtered_data.jsonl',
    # 'oasst1_filtered_data.jsonl',
    'code_alpaca_filtered_data.jsonl',
    'wizardlm_filtered_data.jsonl',

    # 'science_filtered_data.jsonl',

    # 'tulu_v2_filtered_data.jsonl',
    # 'tulu_v2_data.jsonl',
]

with open(f'{base_path}tulu_v2_filtered_only_code_alpaca_and_wizardlm.jsonl', 'w') as f_out:
    for dataset in tulu_only_code_and_wizardlm_datasets:
        with open(f'{base_path}{dataset}') as f_in:
            for line in f_in.readlines():
                f_out.write(line)
