import copy
import subprocess
import yaml
import random
from datetime import date
import time

today = date.today().strftime("%m%d%Y")

with open("beaker_configs/default_experiment.yaml", 'r') as f:
    default_yaml = f.read()
d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)

# cluster = "ai2/allennlp-cirrascale"
cluster = "ai2/allennlp-elanding-a100-40g"

def set_argument_value(arguments, name, value):
    if name not in arguments:
        raise ValueError(f"{name} not in arguments.")
    idx = arguments.index(name)
    assert not (isinstance(arguments[idx+1], str) and arguments[idx+1].startswith("-")) # make sure the next argument is the value
    arguments[idx+1] = value
    return arguments

experiment_group = "merge_models"

#--------------- experiments about model variants -------------------------

if experiment_group == "merge_models":

    model_to_dataset = {
        "google/t5-small-lm-adapt" : {
            "eqa": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            },
            "nli": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            },
            "sa": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            }
        },
        "google/t5-base-lm-adapt" : {
            "eqa": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            },
            "nli": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            },
            "sa": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            }
        }, 
        "google/t5-large-lm-adapt" : {
            "eqa": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            },
            "nli": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            },
            "sa": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            }
        },
        "google/t5-xl-lm-adapt" : {
            "eqa": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            },
            "nli": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            },
            "sa": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            }
        },
        "allenai/tk-instruct-small-def-pos" : {
            "eqa": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            },
            "nli": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            },
            "sa": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            }
        },
        "allenai/tk-instruct-base-def-pos" : {
            "eqa": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            },
            "nli": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            },
            "sa": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            }
        }, 
        "allenai/tk-instruct-large-def-pos" : {
            "eqa": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            },
            "nli": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            },
            "sa": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            }
        },
        "allenai/tk-instruct-3b-def-pos" : {
            "eqa": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            },
            "nli": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            },
            "sa": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            }
        },
        "bigscience/T0_3B" : {
            "eqa": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            },
            "nli": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            },
        },
        "google/flan-t5-small" : {
            "eqa": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            },
        },
        "google/flan-t5-base" : {
            "eqa": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            },
        }, 
        "google/flan-t5-large" : {
            "eqa": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            },
        },
        "google/flan-t5-xl" : {
            "eqa": {
                "low data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
                "large data": {
                    "full finetuning": '',
                    "LoRA": '',
                },
            },
        },
    }
        
    for model_name in model_to_dataset:
        d = copy.deepcopy(d1)

        # d['tasks'][0]['image']['beaker'] = 'jacobm/train-with-lora-no-delete'
        d['tasks'][0]['image']['beaker'] = 'jacobm/retrain_tk_sentiment_analysis'
        d['tasks'][0]['context']['cluster'] = cluster
        d['tasks'][0]['envVars'][3]['value'] = 'placeholder'

        set_argument_value(d['tasks'][0]['command'], "--base_model", model_name)
        if low_data:
            # set_argument_value(d['tasks'][0]['command'], "--checkpoint", 5000)
            d['tasks'][0]['datasets'][0]['subPath'] = 'placeholder!!!!' # TODO fix this
        else:
            # per size
            pass

        set_argument_value(d['tasks'][0]['command'], "--lora", False)
        if lora:
            set_argument_value(d['tasks'][0]['command'], "--lora", True)

        d['tasks'][0]['datasets'][0]['source']['beaker'] = 'jacobm/retrain-tk'
        d['tasks'][0]['datasets'][0]['subPath'] = 'placeholder!!!!' # TODO fix this
        set_argument_value(d['tasks'][0]['command'], "--data_dir", '/data/splits/default')
        # name = f"ni_training_{experiment_group}_{model_name.split('/')[-1]}_{str(100)}_{today}"
        name = f"retrain_tk-instruct_lora_{model_name.split('/')[-1]}"
        d['description'] = name
        d['tasks'][0]['name'] = name
        set_argument_value(d['tasks'][0]['command'], "--run_name", name)
        d['tasks'][0]['resources']['gpuCount'] = 1
        d['tasks'][0]['context']['cluster'] = 'ai2/aristo-cirrascale'
        
        print(d)

        fn = "beaker_configs/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/robustness-interpolations".format(fn)
        subprocess.Popen(cmd, shell=True)
        time.sleep(3)
