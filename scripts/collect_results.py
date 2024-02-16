import json
import os

domain_adaptation_path = "/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/"
tulu_evals = os.listdir(domain_adaptation_path)
tulu_evals.remove("gsm_cot")
tulu_evals.remove("science")
print(tulu_evals)