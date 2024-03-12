'''
Adapted from https://github.com/bigcode-project/octopack/blob/main/dataset/oasst/filter_oasst.py
Filter out refusals from a dataset. Mainly keyword based.
'''
from datasets import load_dataset
import argparse
import json

UNWANTED_WORDS = [
    # checked or generally okay
    "Open Assistant",
    "OpenAssistant",
    "LAION",
    "*This chat conversation is shared from",
    "*This conversation is shared from",
    "Kilcher",
    "Koepf",
    "As an AI language model, I cannot",
    "As an AI language model, I do not",
    "As an AI language model, I am not able",
    "As an AI language model, I don't have personal",
    "As an AI language model, I don't have",
    "As an AI language model, I am only able",
    "AI language model and I do not",
    "As an AI language model, I cannot modify",
    "As an AI language model, I do not",
    "I am an AI language model and do not",
    "as a language model",
    "an AI language",
    "as an AI language model",
    "I'm sorry, but as an AI language model",
    "I'm sorry, but",
    "As an AI language model, I don't have",
    "Unfortunately, I cannot provide",
    "I'm sorry, I cannot",
    "I'm sorry, I cannot generate",
    "I don't have the ability",
    "my knowledge cutoff",
    "my knowledge cut off",
    "text-based AI language model",
    "I cannot fulfill your request",
    "I cannot assist",
    "I apologize, but",
    "I'm an AI",
    "I am an AI",
    "As a large language model",
    "As an AI",
    "against my programming",
    "I'm afraid I cannot create",
    "I cannot",
    "Lo siento, pero no puedo",
    "Lo siento, pero como modelo de lenguaje, no puedo proporcionar",
    "Lo siento, como modelo de lenguaje, no tengo",
    "Lo siento, debe haber habido una confusi\u00f3n",
    "Lo siento, como modelo de lenguaje, no puedo realizar",
    "Lo siento, soy un modelo de lenguaje y no tengo la capacidad de generar",
    "como modelo de lenguaje AI",
    "Lo siento, como modelo de lenguaje",
    "no puedo proporcionar",
    "pero debido a mi capacidad para generar c\u00f3digos complejos y completos es limitado",
    "Lamento no poder proporcionarte el c\u00f3digo",
    "Desculpe-me, mas a linguagem vulgar e ofensiva",
    "apropriada em nenhum contexto",
    "Como modelo de linguagem",
    "Como um modelo de linguagem, n\u00e3o tenho a capacidade de",
    "I am an AI",
    "as an AI language model, you cannot",
    "As a machine",
    "I'm sorry,",
    "However, it is important to use any code or information provided responsibly and within legal and ethical boundaries.",
    "September 2021",
    "It is not possible",
    "it is not appropriate",
    "it's not appropriate",
    "cannot provide guidance",
    "cannot provide information",
    "cannot provide any information",
    "cannot engage in discussions",
    "programming prohibits",
    "cannot support or promote",
    "not within the scope of my training data",
    "I am not sentient and cannot provide",
    "I am a machine learning model and not a human",
    "I can't provide"
    "my main goal",
    "my purpose is to ",
    "as a language AI",
    "as a text-based language AI",
    "As an artificial intelligence",
    "I am an artificial intelligence",
    "it is never acceptable",
    "It is illegal to",
    "please refrain",
    "it is not appropriate",
    "it is never okay",
    "it is never acceptable",
    "I am not able to",
    "it is not acceptable",
    "it is never safe",
    "I am a bot",
    "it is not safe",
    "I am not able to answer",
    "I am a machine learning model",
    "I am not able to provide",
    "As a language model,",
    "I do not have access",
    "I am unable to",
    "legal",
    "suicide"
]
# consider: legal, since this will filter legality refusals
# consider: suicide, rape, since this will filter a chunk

# Taken from: https://huggingface.co/datasets/ehartford/WizardLM_alpaca_evol_instruct_70k_unfiltered/raw/main/wizardlm_clean.py
# Added references to LAION, Open Assistant, Yannic Kilcher etc to debias entirely
def contains_unwanted_words(text):
    for word in UNWANTED_WORDS:
        if word.lower() in text.lower():
            return True
    return False

# very simple joining since the filter is simple.
def create_dialogue(example):
    example["dialogue"] = " ".join(x["content"] for x in example["messages"])
    return example

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='allenai/tulu-v2-sft-mixture')
    parser.add_argument('--output', type=str, default='tulu-v2-sft-mixture-refusals-filtered')
    parser.add_argument('--analyze_word_counts', action='store_true')
    parser.add_argument('--analyze_dataset_counts', action='store_true')
    args = parser.parse_args()

    ds = load_dataset(args.dataset)
    ds = ds.map(create_dialogue, num_proc=6)
    # only examine distilled datasets in Tulu 2 that may contain examples of refusals
    filter_datasets = ['oasst1', 'gpt4_alpaca', 'code_alpaca', 'sharegpt', 'wizardlm', 'open_orca']
    ds = ds.filter(lambda x: x['dataset'] not in filter_datasets or not contains_unwanted_words(x["dialogue"]), num_proc=32)
    ds = ds.remove_columns(["dialogue"])
    ds.save_to_disk(args.output)
    # save sample
    with open(args.output + '.jsonl', 'w') as w:
        for sample in ds['train']:
            w.write(json.dumps(sample) + '\n')
    # do some analysis
    if args.analyze_dataset_counts:
        all_filtered = load_dataset(args.dataset).map(create_dialogue, num_proc=6).filter(lambda x: x['dataset'] in filter_datasets and contains_unwanted_words(x["dialogue"]))
        print("Filtered out {} examples".format(len(all_filtered['train'])))
        # grab the dataset counts
        print("Dataset counts:")
        from collections import Counter, defaultdict
        counts = Counter()
        for sample in all_filtered['train']:
            counts[sample['dataset']] += 1
        for key in counts:
            print("{}: {}".format(key, counts[key]))
    # grab matched word counts
    if args.analyze_word_counts:
        counts = Counter()
        words_to_samples = defaultdict(list)
        for sample in all_filtered['train']:
            for word in UNWANTED_WORDS:
                if word.lower() in sample['dialogue'].lower():
                    counts[word] += 1
                    words_to_samples[word].append(sample)
        print("Word counts:")
        for key in counts:
            print("{}: {}".format(key, counts[key]))