beaker session create --gpus 1 --budget ai2/oe-adapt \
    --mount beaker://jacobm/llama_2_7b-tulu_all-science_none-seed_42_4096=/tulu_all \
    --mount beaker://jacobm/llama_2_7b-tulu_none-science_100_4096=/science_100 \
    --mount beaker://jacobm/llama_2_7b-tulu_none-science_200_4096=/science_200 \
    --mount beaker://jacobm/llama_2_7b-tulu_none-science_500_4096=/science_500 \
    --mount beaker://jacobm/llama_2_7b-tulu_none-science_1000_4096=/science_1000 \
    --mount beaker://jacobm/llama_2_7b-tulu_none-science_2500_4096=/science_2500 \
    --mount beaker://jacobm/llama_2_7b-tulu_none-science_upsample_4096=/science_upsample

beaker session create --gpus 1 --budget ai2/oe-adapt \
    --mount beaker://jacobm/llama_2_7b-tulu_all-science_none-seed_42_4096=/tulu_all \
    --mount beaker://jacobm/llama_2_7b-tulu_none-science_2500_4096=/science_2500 \

beaker session create --gpus 1 --budget ai2/oe-adapt \
    --mount beaker://jacobm/llama_2_7b-tulu_all-safety_none=/tulu_all \
    --mount beaker://jacobm/llama_2_7b-tulu_none-safety_10=/safety_10 \
    --mount beaker://jacobm/llama_2_7b-tulu_none-safety_20=/safety_20 \
    --mount beaker://jacobm/llama_2_7b-tulu_none-safety_60=/safety_60 \
    --mount beaker://jacobm/llama_2_7b-tulu_none-safety_100=/safety_100

beaker session create --gpus 1 --budget ai2/oe-adapt \
    --mount beaker://jacobm/llama_2_7b-tulu_all-safety_none=/tulu_all \
    --mount beaker://jacobm/llama_2_7b-tulu_none-safety_100=/safety_100

beaker session create --gpus 1 --budget ai2/oe-adapt \
    --mount beaker://jacobm/llama_2_7b-tulu_all-safety_none=/tulu_all \
    --mount beaker://jacobm/llama_2_7b-tulu_none-safety_10=/safety_10

beaker session create --gpus 1 --budget ai2/oe-adapt \
    --mount beaker://jacobm/llama_2_7b-tulu_all-safety_none=/tulu_all \
    --mount beaker://jacobm/llama_2_7b-tulu_none-safety_10=/safety_10
    --mount beaker://jacobm/llama_2_7b-tulu_none-safety_v0_100=/safety_v0_100

beaker session create --gpus 1 --budget ai2/oe-adapt \
    --mount beaker://jacobm/llama_2_7b-tulu_all-safety_none=/tulu_all \
    --mount beaker://jacobm/tulu_2_7b_uncensored-tulu_none-safety_100=/tulu_2_7b_uncensored_safety_100


beaker session create --gpus 1 --budget ai2/oe-adapt \
    --mount beaker://jacobm/llama_2_7b-tulu_all-science_none-seed_42_4096=/tulu_all \
    --mount beaker://jacobm/tulu_2_7b_continued_ft-tulu_none-science_2500_4096=/tulu_2_7b_science_2500

beaker session create --gpus 1 --budget ai2/oe-adapt \
    --mount beaker://jacobm/llama_2_7b-tulu_all-science_none-seed_42_4096=/tulu_all \
    --mount beaker://jacobm/llama_2_7b-tulu_none-science_2500_4096=/science_2500