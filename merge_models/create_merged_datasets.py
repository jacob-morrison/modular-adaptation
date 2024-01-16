datasets =  [
    'hard_coded_filtered',
    'oasst1_filtered',
    'science_filtered',
    'wizardlm_filtered',
    'cot_filtered',
    'gpt4_alpaca_filtered',
    'lima_filtered',
    'open_orca_filtered',
    'sharegpt_filtered',
    'code_alpaca_filtered',
    'flan_v2_filtered',
]

# model_out_dir = '/net/nfs.cirrascale/allennlp/jacobm/tulu_7B_lora_exp/pairwise_experts'
out_dir = '/net/nfs.cirrascale/allennlp/jacobm/tulu_data/tulu-v2/merged_datasets/'

file_count = 0
for i in range(len(datasets) - 1):
    for j in range(i + 1, len(datasets)):
        file_path1 = f'/net/nfs.cirrascale/allennlp/jacobm/tulu_data/tulu-v2/{datasets[i]}_data.jsonl'
        file_path2 = f'/net/nfs.cirrascale/allennlp/jacobm/tulu_data/tulu-v2/{datasets[j]}_data.jsonl'
        with open(file_path1) as f_in1, open(file_path2) as f_in2, open(out_dir + f'{datasets[i]}-{datasets[j]}.jsonl', 'w') as f_out:
            for line in f_in1.readlines():
                f_out.write(line)
            for line in f_in2.readlines():
                f_out.write(line)
        print(f'Done with {file_path1} and {file_path2}')
        file_count += 1
print(f'Total number of files: {file_count}')
