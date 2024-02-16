import json
import os

domain_adaptation_path = "/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/"
tulu_evals = os.listdir(domain_adaptation_path)#.remove("gsm_cot").remove("science")
print(tulu_evals)