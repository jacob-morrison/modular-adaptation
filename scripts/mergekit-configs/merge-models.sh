mergekit-yaml scripts/mergekit-configs/auto_created/merge-my-new-tulu-0.975-science-200-0.025.yaml /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/checkpoints/domain_addition/another_tulu_only_model/llama_2_7b-0.975-tulu_only-0.025-science_200 --cuda &&
mergekit-yaml scripts/mergekit-configs/auto_created/merge-my-new-tulu-0.84-science-2500-0.16.yaml /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/checkpoints/domain_addition/another_tulu_only_model/llama_2_7b-0.84-tulu_only-0.16-science_2500 --cuda &&
mergekit-yaml scripts/mergekit-configs/auto_created/merge-tulu-0.975-science-200-0.025.yaml /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/checkpoints/domain_addition/merged_models/llama_2_7b-0.975-tulu_only-0.025-science_200 --cuda &&
mergekit-yaml scripts/mergekit-configs/auto_created/merge-tulu-0.84-science-2500-0.16.yaml /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/checkpoints/domain_addition/merged_models/llama_2_7b-0.84-tulu_only-0.16-science_2500 --cuda &&
mergekit-yaml scripts/mergekit-configs/auto_created/merge-daves-tulu-0.84-science-2500-0.16.yaml /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/checkpoints/domain_addition/with_daves_tulu_model/llama_2_7b-0.84-tulu_only-0.16-science_2500 --cuda &&
mergekit-yaml scripts/mergekit-configs/auto_created/merge-daves-tulu-0.975-science-200-0.025.yaml /net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/checkpoints/domain_addition/with_daves_tulu_model/llama_2_7b-0.975-tulu_only-0.025-science_200 --cuda