import subprocess

def print_and_run(cmd):
    print(cmd)
    subprocess.run(cmd, shell=True)

baselines = [
    "llama_2_7b-tulu_all-science_none-seed_123",
    "llama_2_7b-tulu_match-science_500",
    "llama_2_7b-tulu_none-science_upsample",
    "tulu_2_7b_continued_ft-tulu_none-science_100",
    "llama_2_7b-tulu_all-science_none-seed_42",
    "llama_2_7b-tulu_none-science_100",
    "tulu_2_7b_continued_ft-tulu_none-science_1000",
    "llama_2_7b-tulu_all-science_none-seed_52830",
    "llama_2_7b-tulu_none-science_1000",
    "tulu_2_7b_continued_ft-tulu_none-science_200",
    "llama_2_7b-tulu_all-science_100",
    "llama_2_7b-tulu_all-science_upsample",
    "llama_2_7b-tulu_none-science_1000-seed_123",
    "tulu_2_7b_continued_ft-tulu_match-science_100",
    "tulu_2_7b_continued_ft-tulu_none-science_2500",
    "llama_2_7b-tulu_all-science_1000",
    "llama_2_7b-tulu_match-science_100",
    "llama_2_7b-tulu_none-science_1000-seed_52830",
    "tulu_2_7b_continued_ft-tulu_match-science_1000",
    "tulu_2_7b_continued_ft-tulu_none-science_500",
    "llama_2_7b-tulu_all-science_200",
    "llama_2_7b-tulu_match-science_1000",
    "llama_2_7b-tulu_none-science_200",
    "tulu_2_7b_continued_ft-tulu_match-science_200",
    "tulu_2_7b_continued_ft-tulu_none-science_upsample",
    "llama_2_7b-tulu_all-science_2500",
    "llama_2_7b-tulu_match-science_200",
    "llama_2_7b-tulu_none-science_2500",
    "tulu_2_7b_continued_ft-tulu_match-science_2500",
    "llama_2_7b-tulu_all-science_500",
    "llama_2_7b-tulu_match-science_2500",
    "llama_2_7b-tulu_none-science_500",
    "tulu_2_7b_continued_ft-tulu_match-science_500",
]

big_baselines = [
    "tulu_2_70b_continued_ft-tulu_none-science_1000",
    "llama_2_70b-tulu_none-science_1000",
    "tulu_2_70b_continued_ft-tulu_match-science_1000",
    "llama_2_70b-tulu_none-science_100",
    "llama_2_70b-tulu_all-science_none",
]

merged_models = [
    # put these here
]

for model in baselines:
    beaker_name = "jacobm/" + model
    new_beaker_name = f"{model}_4096"
    print_and_run("mkdir tmp_model_directory")
    print_and_run(f"beaker dataset fetch {beaker_name} -o tmp_model_directory/")
    print_and_run("rm tmp_model_directory/config.json tmp_model_directory/generation_config.json tmp_model_directory/special_tokens_map.json tmp_model_directory/tokenizer.model tmp_model_directory/tokenizer_config.json")
    print_and_run("ls tmp_model_directory/")
    print_and_run("cp llama_tokenizer_4k/* tmp_model_directory/")
    print_and_run("ls tmp_model_directory/")
    print_and_run(f"beaker dataset create tmp_model_directory/ -n {new_beaker_name} -w ai2/modular-adaptation-science")
    print_and_run("rm -rf tmp_model_directory")