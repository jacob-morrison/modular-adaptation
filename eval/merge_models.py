from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
from peft import PeftModel, PeftConfig
from collections import OrderedDict
import argparse
from peft import LoraConfig, PeftModel, PeftConfig, TaskType, get_peft_model, set_peft_model_state_dict
import os

def get_parser(task):
    if task == 'mmlu_0shot' or task == 'mmlu_5shot':
        mmlu_parser = argparse.ArgumentParser()
        mmlu_parser.add_argument(
            "--ntrain",
            type=int,
            default=5
        )
        mmlu_parser.add_argument(
            "--data_dir",
            type=str,
            default="data/mmlu"
        )
        mmlu_parser.add_argument(
            "--save_dir",
            type=str,
            default="results/mmlu/llama-7B/"
        )
        mmlu_parser.add_argument(
            "--model_name_or_path",
            type=str,
            default=None,
            help="if specified, we will load the model to generate the predictions."
        )
        mmlu_parser.add_argument(
            "--tokenizer_name_or_path",
            type=str,
            default=None,
            help="if specified, we will load the tokenizer from here."
        )
        mmlu_parser.add_argument(
            "--eval_batch_size",
            type=int,
            default=1,
            help="batch size for evaluation."
        )
        mmlu_parser.add_argument(
            "--use_chat_format", 
            action="store_true", 
            help="If given, we will use the chat format for the prompts."
        )
        mmlu_parser.add_argument(
            "--chat_formatting_function", 
            type=str, 
            default="eval.templates.create_prompt_with_tulu_chat_format", 
            help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
        )
        mmlu_parser.add_argument(
            "--lora_weight_path",
            help="If given, we load lora weights."
        )

        mmlu_parser.add_argument(
            "--use_slow_tokenizer",
            action="store_true",
            help="If given, we will use the slow tokenizer."
        )
        mmlu_parser.add_argument(
            "--openai_engine",
            type=str,
            default=None,
            help="if specified, we will use the OpenAI API to generate the predictions."
        )
        mmlu_parser.add_argument(
            "--subjects",
            nargs="*",
            help="which subjects to evaluate. If not specified, all the 57 subjects will be evaluated."
        )
        mmlu_parser.add_argument(
            "--n_instances",
            type=int,
            help="if specified, a maximum of n_instances per subject will be used for the evaluation."
        )
        mmlu_parser.add_argument(
            "--load_in_8bit",
            action="store_true",
            help="load model in 8bit mode, which will reduce memory and speed up inference."
        )
        mmlu_parser.add_argument(
            "--gptq",
            action="store_true",
            help="If given, we're evaluating a 4-bit quantized GPTQ model."
        )
        return mmlu_parser
    elif task == 'gsm_direct' or task == 'gsm_cot':
        gsm_parser = argparse.ArgumentParser()
        gsm_parser.add_argument(
            "--data_dir", 
            type=str, 
            default="data/gsm"
        )
        gsm_parser.add_argument(
            "--max_num_examples", 
            type=int, 
            default=None, 
            help="maximum number of examples to evaluate."
        )
        gsm_parser.add_argument(
            "--save_dir", 
            type=str, 
            default="results/gsm"
        )
        gsm_parser.add_argument(
            "--model_name_or_path", 
            type=str, 
            default=None, 
            help="if specified, we will load the model to generate the predictions."
        )
        gsm_parser.add_argument(
            "--tokenizer_name_or_path", 
            type=str, 
            default=None, 
            help="if specified, we will load the tokenizer from here."
        )
        gsm_parser.add_argument(
            "--use_slow_tokenizer",
            action="store_true",
            help="If given, we will use the slow tokenizer."
        )
        gsm_parser.add_argument(
            "--openai_engine", 
            type=str, 
            default=None, help="if specified, we will use the OpenAI API to generate the predictions."
        )
        gsm_parser.add_argument(
            "--n_shot", 
            type=int, 
            default=8, 
            help="max number of examples to use for demonstration."
        )
        gsm_parser.add_argument(
            "--no_cot", 
            action="store_true", 
            help="If given, we're evaluating a model without chain-of-thought."
        )
        gsm_parser.add_argument(
            "--eval_batch_size", 
            type=int, 
            default=1, 
            help="batch size for evaluation."
        )
        gsm_parser.add_argument(
            "--load_in_8bit", 
            action="store_true", 
            help="load model in 8bit mode, which will reduce memory and speed up inference."
        )
        gsm_parser.add_argument(
            "--gptq", 
            action="store_true", 
            help="If given, we're evaluating a 4-bit quantized GPTQ model."
        )
        gsm_parser.add_argument(
            "--use_vllm",
            action="store_true", 
            help="If given, we will use the vllm library, which will likely increase the inference throughput."
        )
        gsm_parser.add_argument(
            "--use_chat_format", 
            action="store_true", 
            help="If given, we will use the chat format for the prompts."
        )
        gsm_parser.add_argument(
            "--chat_formatting_function", 
            type=str, 
            default="eval.templates.create_prompt_with_tulu_chat_format", 
            help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
        )
        gsm_parser.add_argument(
            "--lora_weight_path",
            help="If given, we load lora weights."
        )
        return gsm_parser
    elif task == 'bbh_direct' or task == 'bbh_cot':
        bbh_parser = argparse.ArgumentParser()
        bbh_parser.add_argument(
            "--data_dir", 
            type=str, 
            default="data/bbh"
        )
        bbh_parser.add_argument(
            "--save_dir", 
            type=str, 
            default="results/bbh"
        )
        bbh_parser.add_argument(
            "--model_name_or_path", 
            type=str, 
            default=None, 
            help="if specified, we will load the model to generate the predictions."
        )
        bbh_parser.add_argument(
            "--tokenizer_name_or_path", 
            type=str, 
            default=None, 
            help="if specified, we will load the tokenizer from here."
        )
        bbh_parser.add_argument(
            "--use_slow_tokenizer",
            action="store_true",
            help="If given, we will use the slow tokenizer."
        )
        bbh_parser.add_argument(
            "--openai_engine", 
            type=str, 
            default=None, 
            help="if specified, we will use the OpenAI API to generate the predictions."
        )
        bbh_parser.add_argument(
            "--no_cot", 
            action="store_true", 
            help="if specified, chain of thoughts will be removed from the prompts."
        )
        bbh_parser.add_argument(
            "--max_num_examples_per_task", 
            type=int, 
            default=None, 
            help="maximum number of examples to evaluate per task."
        )
        bbh_parser.add_argument(
            "--eval_batch_size", 
            type=int, 
            default=1, 
            help="batch size for evaluation."
        )
        bbh_parser.add_argument(
            "--load_in_8bit", 
            action="store_true", 
            help="load model in 8bit mode, which will reduce memory and speed up inference."
        )
        bbh_parser.add_argument(
            "--gptq", 
            action="store_true", 
            help="If given, we're evaluating a 4-bit quantized GPTQ model."
        )
        bbh_parser.add_argument(
            "--use_vllm",
            action="store_true", 
            help="If given, we will use the vllm library, which will likely increase the inference throughput."
        )
        bbh_parser.add_argument(
            "--use_chat_format", 
            action="store_true", 
            help="If given, we will use the chat format for the prompts."
        )
        bbh_parser.add_argument(
            "--chat_formatting_function", 
            type=str, 
            default="eval.templates.create_prompt_with_tulu_chat_format", 
            help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
        )
        bbh_parser.add_argument(
            "--lora_weight_path",
            help="If given, we load lora weights."
        )
        return bbh_parser
    elif task == 'tydiqa_goldp_1shot' or task == 'tydiqa_no_context_1shot':
        tydiqa_parser = argparse.ArgumentParser()
        tydiqa_parser.add_argument(
            "--data_dir",
            type=str,
            default="data/xorqa/"
        )
        tydiqa_parser.add_argument(
            "--max_num_examples_per_lang",
            type=int,
            default=None,
            help="maximum number of examples per language to evaluate."
        )
        tydiqa_parser.add_argument(
            "--n_shot",
            type=int,
            default=1,
            help="number of examples to use for few-shot evaluation."
        )
        tydiqa_parser.add_argument(
            "--no_context",
            action="store_true",
            help="If given, we're evaluating a model without the gold context passage."
        )
        tydiqa_parser.add_argument(
            "--max_context_length",
            type=int,
            default=512,
            help="maximum number of tokens in the context passage."
        )
        tydiqa_parser.add_argument(
            "--save_dir",
            type=str,
            default="results/tydiqa/"
        )
        tydiqa_parser.add_argument(
            "--model_name_or_path",
            type=str,
            default=None,
            help="if specified, we will load the model to generate the predictions."
        )
        tydiqa_parser.add_argument(
            "--tokenizer_name_or_path",
            type=str,
            default=None,
            help="if specified, we will load the tokenizer from here."
        )
        tydiqa_parser.add_argument(
            "--use_slow_tokenizer",
            action="store_true",
            help="If given, we will use the slow tokenizer."
        )
        tydiqa_parser.add_argument(
            "--openai_engine",
            type=str,
            default=None,
            help="if specified, we will use the OpenAI API to generate the predictions."
        )
        tydiqa_parser.add_argument(
            "--eval_batch_size",
            type=int,
            default=1,
            help="batch size for evaluation."
        )
        tydiqa_parser.add_argument(
            "--load_in_8bit",
            action="store_true",
            help="load model in 8bit mode, which will reduce memory and speed up inference."
        )
        tydiqa_parser.add_argument(
            "--gptq",
            action="store_true",
            help="If given, we're evaluating a 4-bit quantized GPTQ model."
        )
        tydiqa_parser.add_argument(
            "--use_vllm",
            action="store_true", 
            help="If given, we will use the vllm library, which will likely increase the inference throughput."
        )
        tydiqa_parser.add_argument(
            "--use_chat_format", 
            action="store_true", 
            help="If given, we will use the chat format for the prompts."
        )
        tydiqa_parser.add_argument(
            "--chat_formatting_function", 
            type=str, 
            default="eval.templates.create_prompt_with_tulu_chat_format", 
            help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
        )
        tydiqa_parser.add_argument(
            "--lora_weight_path",
            help="If given, we load lora weights."
        )
        return tydiqa_parser
    elif task == 'toxigen':
        toxigen_parser = argparse.ArgumentParser()
        toxigen_parser.add_argument(
            "--data_dir", 
            type=str, 
            default="data/eval/toxigen"
        )
        toxigen_parser.add_argument(
            "--save_dir", 
            type=str, 
            default="results/toxigen"
        )
        toxigen_parser.add_argument(
            "--model_name_or_path",
            type=str,
            default=None,
            help="if specified, we will load the model to generate the predictions.",
        )
        toxigen_parser.add_argument(
            "--tokenizer_name_or_path",
            type=str,
            default=None,
            help="if specified, we will load the tokenizer from here.",
        )
        toxigen_parser.add_argument(
            "--use_slow_tokenizer",
            action="store_true",
            help="If given, we will use the slow tokenizer."
        )
        toxigen_parser.add_argument(
            "--openai_engine",
            type=str,
            default=None,
            help="if specified, we will use the OpenAI API to generate the predictions.",
        )
        toxigen_parser.add_argument(
            "--eval_batch_size", type=int, default=1, help="batch size for evaluation."
        )
        toxigen_parser.add_argument(
            "--classifier_batch_size",
            type=int,
            default=32,
            help="batch size to use for toxicity classifier.",
        )
        toxigen_parser.add_argument(
            "--classifier_device",
            type=str,
            default="cuda",
            help="device to use for toxicity classifier.",
        )
        toxigen_parser.add_argument(
            "--load_in_8bit",
            action="store_true",
            help="load model in 8bit mode, which will reduce memory and speed up inference.",
        )
        toxigen_parser.add_argument(
            "--gptq",
            action="store_true",
            help="If given, we're evaluating a 4-bit quantized GPTQ model.",
        )
        toxigen_parser.add_argument(
            "--use_chat_format", 
            action="store_true", 
            help="If given, we will use the chat format for the prompts."
        )
        toxigen_parser.add_argument(
            "--chat_formatting_function", 
            type=str, 
            default="eval.templates.create_prompt_with_tulu_chat_format", 
            help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
        )
        toxigen_parser.add_argument(
            "--use_vllm",
            action="store_true",
            help="If given, we will use vLLM to generate the predictions - much faster.",
        )
        toxigen_parser.add_argument(
            "--max_prompts_per_group",
            type=int,
            default=500,
            help="If given, we will only use this many prompts per group. Default to 500 (half the available prompts).",
        )
        toxigen_parser.add_argument(
            "--lora_weight_path",
            help="If given, we load lora weights."
        )
        return toxigen_parser
    else:
        print('unsupported task!!')
        print(task)

base_dir = '/net/nfs.cirrascale/allennlp/jacobm/tulu_7B_lora_exp/'

tasks = [
    "mmlu_0shot",
    "mmlu_5shot",
    "gsm_direct",
    "gsm_cot",
    "bbh_direct",
    "bbh_cot",
    "tydiqa_goldp_1shot",
    "tydiqa_no_context_1shot",
    "toxigen",

    ### Need an OpenAI API Key
    # "codex_eval_temp_0.1",
    # "codex_eval_temp_0.8",
    # "trutufulqa",
    # "alpaca_farm",
]

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--base_model", help = "Which base model", type = str)
parser.add_argument("-l", "--base_lora_path", help = "Which base model", type = str)
parser.add_argument("-t", "--task", help = "Which eval task", type = str)
parser.add_argument('--target_lora_modules', nargs='+')
parser.add_argument('--results_dir', nargs='?', default='/results')
args = parser.parse_args()

if args.task not in tasks:
    print('Invalid task! ')
    print(args)
    quit()

if args.base_lora_path:
    base_dir = args.base_lora_path

print(int(len(args.target_lora_modules)))
print(args)

# load the base model
base_state_dict = OrderedDict()
base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
tokenizer = AutoTokenizer.from_pretrained(args.base_model)
base_model.resize_token_embeddings(len(tokenizer))

for key in base_model.state_dict():
    base_state_dict[key] = base_model.state_dict()[key]

new_state_dict = OrderedDict()

lora_base_models = []
lora_models = []
model_weights = []

for lora_module in args.target_lora_modules:
    lora_base_models.append(AutoModelForCausalLM.from_pretrained(args.base_model))
    lora_base_models[-1].resize_token_embeddings(len(tokenizer))
    lora_models.append(PeftModel.from_pretrained(lora_base_models[-1], os.path.join(base_dir, lora_module)))
    model_weights.append(len(args.target_lora_modules))

for i in range(len(lora_base_models)):
    lora_model = lora_models[i]
    for key in lora_model.state_dict():
        if key in base_state_dict:
            new_state_dict[key] = lora_model.state_dict()[key]
        else:
            if key not in new_state_dict:
                new_state_dict[key] = torch.div(lora_model.state_dict()[key], model_weights[i])
            else:
                new_state_dict[key] += torch.div(lora_model.state_dict()[key], model_weights[i])

# load a single model to test
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=256, 
    lora_alpha=256, 
    lora_dropout=0.05,
    target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]
)
base_lora_model = AutoModelForCausalLM.from_pretrained(args.base_model)
base_lora_model.resize_token_embeddings(len(tokenizer))
final_model = get_peft_model(base_lora_model, peft_config)
# print(new_state_dict.keys())
# print(final_model.state_dict().keys())
# merged_model = set_peft_model_state_dict(model, new_state_dict)
merge_result = final_model.load_state_dict(new_state_dict)
print(merge_result)

if len(merge_result.missing_keys) == 0 and len(merge_result.unexpected_keys) == 0:
    print("This worked!")

    path_to_write = args.base_model.replace('/', '-')
    out_dir = os.path.join(args.results_dir, 'merged-lora-weights/')
    print("Writing to " + out_dir)
    final_model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
else:
    print("Why didn't merging work??")

# deleting things
del base_lora_model
del final_model
del lora_base_models
del lora_models
del base_model

import sys

print(sys.path)

if args.task == 'mmlu_0shot':
    from eval.mmlu.run_eval import main

    mmlu_parser = get_parser(args.task)
    mmlu_args = [
        "--ntrain", '0',
        "--data_dir", "/data/mmlu/",
        "--save_dir", args.results_dir,
        "--model_name_or_path", "/model",
        "--tokenizer_name_or_path", "/model",
        "--eval_batch_size", '4',
        "--load_in_8bit",
        "--use_chat_format",
        "--chat_formatting_function", "eval.templates.create_prompt_with_tulu_chat_format",
        "--lora_weight_path", out_dir,
    ]
    parsed_args = mmlu_parser.parse_args(mmlu_args)
    print('MMLU Args: ')
    print(parsed_args)
    main(parsed_args)

elif args.task == 'mmlu_5shot':
    from eval.mmlu.run_eval import main

    mmlu_parser = get_parser(args.task)
    mmlu_args = [
        "--ntrain", '5',
        "--data_dir", "/data/mmlu/",
        "--save_dir", args.results_dir,
        "--model_name_or_path", "/model",
        "--tokenizer_name_or_path", "/model",
        "--eval_batch_size", '4',
        "--load_in_8bit",
        "--use_chat_format",
        "--chat_formatting_function", "eval.templates.create_prompt_with_tulu_chat_format",
        "--lora_weight_path", out_dir,
    ]
    parsed_args = mmlu_parser.parse_args(mmlu_args)
    print('MMLU Args: ')
    print(parsed_args)
    main(parsed_args)

elif args.task == 'gsm_direct':
    from eval.gsm.run_eval import main

    gsm_parser = get_parser(args.task)
    gsm_args = [
        "--data_dir", "/data/gsm/",
        "--max_num_examples", "200",
        "--save_dir", args.results_dir,
        "--model_name_or_path", "/model",
        "--tokenizer_name_or_path", "/model",
        "--n_shot", "8",
        "--no_cot",
        "--use_chat_format",
        "--chat_formatting_function", "eval.templates.create_prompt_with_tulu_chat_format",
        "--lora_weight_path", out_dir,
    ]
    parsed_args = gsm_parser.parse_args(gsm_args)
    print('GSM Args: ')
    print(parsed_args)
    main(parsed_args)

elif args.task == 'gsm_cot':
    from eval.gsm.run_eval import main

    gsm_parser = get_parser(args.task)
    gsm_args = [
        "--data_dir", "/data/gsm/",
        "--max_num_examples", "200",
        "--save_dir", args.results_dir,
        "--model_name_or_path", "/model",
        "--tokenizer_name_or_path", "/model",
        "--n_shot", "8",
        "--use_chat_format",
        "--chat_formatting_function", "eval.templates.create_prompt_with_tulu_chat_format",
        "--lora_weight_path", out_dir,
    ]
    parsed_args = gsm_parser.parse_args(gsm_args)
    print('GSM Args: ')
    print(parsed_args)
    main(parsed_args)

elif args.task == 'bbh_direct':
    from eval.bbh.run_eval import main

    bbh_parser = get_parser(args.task)

    bbh_args = [
        "--data_dir", "/data/bbh/",
        "--save_dir", args.results_dir,
        "--model_name_or_path", "/model",
        "--tokenizer_name_or_path", "/model",
        "--max_num_examples_per_task", "40",
        "--no_cot",
        "--use_chat_format",
        "--chat_formatting_function", "eval.templates.create_prompt_with_tulu_chat_format",
        "--lora_weight_path", out_dir,
    ]
    parsed_args = bbh_parser.parse_args(bbh_args)
    print('BBH Args: ')
    print(parsed_args)
    main(parsed_args)

elif args.task == 'bbh_cot':
    from eval.bbh.run_eval import main

    bbh_parser = get_parser(args.task)

    bbh_args = [
        "--data_dir", "/data/bbh/",
        "--save_dir", args.results_dir,
        "--model_name_or_path", "/model",
        "--tokenizer_name_or_path", "/model",
        "--max_num_examples_per_task", "40",
        "--use_chat_format",
        "--chat_formatting_function", "eval.templates.create_prompt_with_tulu_chat_format",
        "--lora_weight_path", out_dir,
    ]
    parsed_args = bbh_parser.parse_args(bbh_args)
    print('BBH Args: ')
    print(parsed_args)
    main(parsed_args)

elif args.task == 'tydiqa_goldp_1shot':
    pass
    from eval.tydiqa.run_eval import main

    tydiqa_parser = get_parser(args.task)

    tydiqa_args = [
        "--data_dir", "/data/tydiqa/",
        "--n_shot", "1",
        "--max_num_examples_per_lang", "100",
        "--max_context_length", "512",
        "--save_dir", args.results_dir,
        "--model_name_or_path", "/model",
        "--tokenizer_name_or_path", "/model",
        "--use_chat_format",
        "--chat_formatting_function", "eval.templates.create_prompt_with_tulu_chat_format",
        "--lora_weight_path", out_dir,
    ]
    parsed_args = tydiqa_parser.parse_args(tydiqa_args)
    print('TydiQA Args: ')
    print(parsed_args)
    main(parsed_args)


elif args.task == 'tydiqa_no_context_1shot':
    pass
    from eval.tydiqa.run_eval import main

    tydiqa_parser = get_parser(args.task)

    tydiqa_args = [
        "--data_dir", "/data/tydiqa/",
        "--no_context",
        "--n_shot", "1",
        "--max_num_examples_per_lang", "100",
        "--max_context_length", "512",
        "--save_dir", args.results_dir,
        "--model_name_or_path", "/model",
        "--tokenizer_name_or_path", "/model",
        "--use_chat_format",
        "--chat_formatting_function", "eval.templates.create_prompt_with_tulu_chat_format",
        "--lora_weight_path", out_dir,
    ]
    parsed_args = tydiqa_parser.parse_args(tydiqa_args)
    print('TydiQA Args: ')
    print(parsed_args)
    main(parsed_args)

elif args.task == 'toxigen':
    pass
    from eval.toxigen.run_eval import main

    toxigen_parser = get_parser(args.task)

    toxigen_args = [
        "--data_dir", "/data/toxigen/",
        "--save_dir", args.results_dir,
        "--model_name_or_path", "/model",
        "--tokenizer_name_or_path", "/model",
        "--eval_batch_size", "32",
        "--use_chat_format",
        "--chat_formatting_function", "eval.templates.create_prompt_with_tulu_chat_format",
        "--lora_weight_path", out_dir,
    ]
    parsed_args = toxigen_parser.parse_args(toxigen_args)
    print('Toxigen Args: ')
    print(parsed_args)
    main(parsed_args)

else:
    print('Invalid task! ')
    print(args)
    quit()

print('done!')