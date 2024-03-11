beaker session create --gpus 1 -d --budget ai2/allennlp \
    --mount beaker://jacobm/llama_2_7b-tulu_all-science_none-seed_42=/tulu_all \
    --mount beaker://jacobm/llama_2_7b-tulu_none-science_100=/science_100 \
    --mount beaker://jacobm/llama_2_7b-tulu_none-science_200=/science_200 \
    --mount beaker://jacobm/llama_2_7b-tulu_none-science_500=/science_500 \
    --mount beaker://jacobm/llama_2_7b-tulu_none-science_1000=/science_1000 \
    --mount beaker://jacobm/llama_2_7b-tulu_none-science_2500=/science_2500 \
    --mount beaker://jacobm/llama_2_7b-tulu_none-science_upsample=/science_upsample
