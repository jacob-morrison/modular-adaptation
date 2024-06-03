


import copy
import os
import subprocess
import yaml

# beaker session create --gpus 1 --budget ai2/oe-adapt  \
#     --mount beaker://jacobm/llama_2_7b-tulu_all_no_science_no_safety_no_coding=/llama_2_7b-tulu_consistent_mix \
#     --mount beaker://jacobm/tulu_2_7b_no_science_no_safety_no_coding-tulu_none-coding_100=/tulu_2_7b-tulu_none-coding_100 \
#     --mount beaker://jacobm/tulu_2_7b_no_science_no_safety_no_coding-tulu_none-safety_100=/tulu_2_7b-tulu_none-safety_100 \
#     --mount beaker://jacobm/tulu_2_7b_no_science_no_safety_no_coding-tulu_none-science_2500=/tulu_2_7b-tulu_none-science_2500

# beaker session create --gpus 1 --budget ai2/oe-adapt  \
#     --mount beaker://jacobm/llama_2_7b-tulu_all_no_science_no_safety_no_coding=/llama_2_7b-tulu_consistent_mix \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-coding_100-4k=/llama_2_7b-coding_100 \
#     --mount beaker://jacobm/tulu_2_7b_no_science_no_safety_no_coding-tulu_none-coding_100=/tulu_2_7b-tulu_none-coding_100 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-safety_100-4k=/llama_2_7b-safety_100 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-science_2500-4k=/llama_2_7b-science_2500

# beaker session create --gpus 1 --budget ai2/oe-adapt  \
#     --mount beaker://jacobm/llama_2_7b-tulu_all_no_science_no_safety=/llama_2_7b-tulu_consistent_mix_with_coding \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-coding_100-4k=/llama_2_7b-coding_100 \
#     --mount beaker://jacobm/tulu_2_7b_no_science_no_safety-tulu_none-coding_100=/tulu_2_7b_with_coding-tulu_none-coding_100

# beaker session create --gpus 1 --budget ai2/oe-adapt  \
#     --mount beaker://jacobm/llama_2_7b-tulu_all=/llama_2_7b-tulu_all \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-safety_100	=/llama_2_7b-tulu_none-safety_100 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-coding_100	=/llama_2_7b-tulu_none-coding_100 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-science_2500=/llama_2_7b-tulu_none-science_2500 

# beaker session create --gpus 1 --budget ai2/oe-adapt  \
#     --mount beaker://jacobm/llama_2_7b-tulu_all=/llama_2_7b-tulu_all \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-coding_100=/tulu_2_7b-tulu_none-coding_100 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-safety_100=/tulu_2_7b-tulu_none-safety_100 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-science_2500=/tulu_2_7b-tulu_none-science_2500 

# beaker session create --gpus 1 --budget ai2/oe-adapt  \
#     --mount beaker://jacobm/llama_2_7b-tulu_all=/llama_2_7b-tulu_all \
    # --mount beaker://jacobm/llama_2_7b-tulu_none-coding_20=/llama_2_7b-tulu_none-coding_20 \
    # --mount beaker://jacobm/llama_2_7b-tulu_none-coding_40=/llama_2_7b-tulu_none-coding_40 \
    # --mount beaker://jacobm/llama_2_7b-tulu_none-coding_60=/llama_2_7b-tulu_none-coding_60 \
    # --mount beaker://jacobm/llama_2_7b-tulu_none-coding_80=/llama_2_7b-tulu_none-coding_80 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-safety_20=/llama_2_7b-tulu_none-safety_20 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-safety_40=/llama_2_7b-tulu_none-safety_40 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-safety_60=/llama_2_7b-tulu_none-safety_60 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-safety_80=/llama_2_7b-tulu_none-safety_80 \
    # --mount beaker://jacobm/llama_2_7b-tulu_none-science_100=/llama_2_7b-tulu_none-science_100 \
    # --mount beaker://jacobm/llama_2_7b-tulu_none-science_200=/llama_2_7b-tulu_none-science_200 \
    # --mount beaker://jacobm/llama_2_7b-tulu_none-science_500=/llama_2_7b-tulu_none-science_500 \
    # --mount beaker://jacobm/llama_2_7b-tulu_none-science_1000=/llama_2_7b-tulu_none-science_1000 \

# beaker session create --gpus 1 --budget ai2/oe-adapt  \
#     --mount beaker://jacobm/llama_2_7b-tulu_all=/llama_2_7b-tulu_all \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-coding_20=/tulu_2_7b-tulu_none-coding_20 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-coding_40=/tulu_2_7b-tulu_none-coding_40 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-coding_60=/tulu_2_7b-tulu_none-coding_60 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-coding_80=/tulu_2_7b-tulu_none-coding_80 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-safety_20=/tulu_2_7b-tulu_none-safety_20 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-safety_40=/tulu_2_7b-tulu_none-safety_40 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-safety_60=/tulu_2_7b-tulu_none-safety_60 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-safety_80=/tulu_2_7b-tulu_none-safety_80

# beaker session create --gpus 1 --budget ai2/oe-adapt  \
#     --mount beaker://jacobm/llama_2_7b-tulu_all_with_coding=/llama_2_7b-tulu_all_with_coding \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-coding_20=/llama_2_7b-tulu_none-coding_20 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-coding_40=/llama_2_7b-tulu_none-coding_40 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-coding_60=/llama_2_7b-tulu_none-coding_60 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-coding_80=/llama_2_7b-tulu_none-coding_80 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-coding_100=/llama_2_7b-tulu_none-coding_100 \
#     --mount beaker://jacobm/tulu_2_7b_with_coding-tulu_none-coding_20=/tulu_2_7b_with_coding-tulu_none-coding_20 \
#     --mount beaker://jacobm/tulu_2_7b_with_coding-tulu_none-coding_40=/tulu_2_7b_with_coding-tulu_none-coding_40 \
#     --mount beaker://jacobm/tulu_2_7b_with_coding-tulu_none-coding_60=/tulu_2_7b_with_coding-tulu_none-coding_60 \
#     --mount beaker://jacobm/tulu_2_7b_with_coding-tulu_none-coding_80=/tulu_2_7b_with_coding-tulu_none-coding_80 \
#     --mount beaker://jacobm/tulu_2_7b_with_coding-tulu_none-coding_100=/tulu_2_7b_with_coding-tulu_none-coding_100 

# beaker session create --gpus 1 --budget ai2/oe-adapt  \
#     --mount beaker://jacobm/llama_2_7b-tulu_all=/llama_2_7b-tulu_all \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-safety_20=/llama_2_7b-tulu_none-safety_20 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-safety_40=/llama_2_7b-tulu_none-safety_40 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-safety_60=/llama_2_7b-tulu_none-safety_60 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-safety_80=/llama_2_7b-tulu_none-safety_80 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-safety_100=/llama_2_7b-tulu_none-safety_100 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-safety_20=/tulu_2_7b-tulu_none-safety_20 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-safety_40=/tulu_2_7b-tulu_none-safety_40 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-safety_60=/tulu_2_7b-tulu_none-safety_60 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-safety_80=/tulu_2_7b-tulu_none-safety_80 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-safety_100=/tulu_2_7b-tulu_none-safety_100 

# beaker session create --gpus 1 --budget ai2/oe-adapt  \
#     --mount beaker://jacobm/llama_2_7b-tulu_all=/llama_2_7b-tulu_all \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-science_100=/llama_2_7b-tulu_none-science_100 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-science_200=/llama_2_7b-tulu_none-science_200 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-science_500=/llama_2_7b-tulu_none-science_500 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-science_1000=/llama_2_7b-tulu_none-science_1000 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-science_2500=/llama_2_7b-tulu_none-science_2500 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-science_100=/tulu_2_7b-tulu_none-science_100 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-science_200=/tulu_2_7b-tulu_none-science_200 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-science_500=/tulu_2_7b-tulu_none-science_500 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-science_1000=/tulu_2_7b-tulu_none-science_1000 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-science_2500=/tulu_2_7b-tulu_none-science_2500 

# beaker session create --gpus 1 --budget ai2/oe-adapt  \
#     --mount beaker://jacobm/llama_2_7b-tulu_all_with_coding=/llama_2_7b-tulu_all_with_coding \
#     --mount beaker://jacobm/tulu_2_7b_with_coding-tulu_none-coding_20=/tulu_2_7b_with_coding-tulu_none-coding_20 \
#     --mount beaker://jacobm/tulu_2_7b_with_coding-tulu_none-coding_40=/tulu_2_7b_with_coding-tulu_none-coding_40 \
#     --mount beaker://jacobm/tulu_2_7b_with_coding-tulu_none-coding_60=/tulu_2_7b_with_coding-tulu_none-coding_60 \
#     --mount beaker://jacobm/tulu_2_7b_with_coding-tulu_none-coding_80=/tulu_2_7b_with_coding-tulu_none-coding_80 \
#     --mount beaker://jacobm/tulu_2_7b_with_coding-tulu_none-coding_100=/tulu_2_7b_with_coding-tulu_none-coding_100 

# beaker session create --gpus 1 --budget ai2/oe-adapt  \
#     --mount beaker://jacobm/llama_2_7b-tulu_all_with_coding=/llama_2_7b-tulu_all_with_coding \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-coding_100=/llama_2_7b-tulu_none-coding_100

# beaker session create --gpus 1 --budget ai2/oe-adapt  \
#     --mount beaker://jacobm/llama_2_7b-tulu_all=/llama_2_7b-tulu_all \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-safety_20=/llama_2_7b-tulu_none-safety_20 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-safety_40=/llama_2_7b-tulu_none-safety_40 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-safety_60=/llama_2_7b-tulu_none-safety_60 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-safety_80=/llama_2_7b-tulu_none-safety_80 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-safety_100=/llama_2_7b-tulu_none-safety_100 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-coding_20=/llama_2_7b-tulu_none-coding_20 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-coding_40=/llama_2_7b-tulu_none-coding_40 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-coding_60=/llama_2_7b-tulu_none-coding_60 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-coding_80=/llama_2_7b-tulu_none-coding_80 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-coding_100=/llama_2_7b-tulu_none-coding_100 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-science_100=/llama_2_7b-tulu_none-science_100 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-science_200=/llama_2_7b-tulu_none-science_200 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-science_500=/llama_2_7b-tulu_none-science_500 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-science_1000=/llama_2_7b-tulu_none-science_1000 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-science_2500=/llama_2_7b-tulu_none-science_2500

# beaker session create --gpus 1 --budget ai2/oe-adapt  \
#     --mount beaker://jacobm/llama_2_7b-tulu_all=/llama_2_7b-tulu_all \
    # --mount beaker://jacobm/llama_2_7b-tulu_none-science_2500=/llama_2_7b-tulu_none-science_2500 \
    # --mount beaker://jacobm/llama_2_7b-tulu_none-safety_100=/llama_2_7b-tulu_none-safety_100 \
    # --mount beaker://jacobm/llama_2_7b-tulu_none-coding_100=/llama_2_7b-tulu_none-coding_100

# beaker session create --gpus 1 --budget ai2/oe-adapt  \
#     --mount beaker://jacobm/llama_2_7b-tulu_all=/llama_2_7b-tulu_all \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-science_2500=/tulu_2_7b-tulu_none-science_2500 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-coding_100=/tulu_2_7b-tulu_none-coding_100 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_none-safety_100=/tulu_2_7b-tulu_none-safety_100 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-science_2500=/llama_2_7b-tulu_none-science_2500 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-safety_100=/llama_2_7b-tulu_none-safety_100 \
#     --mount beaker://jacobm/llama_2_7b-tulu_none-coding_100=/llama_2_7b-tulu_none-coding_100

# beaker session create --gpus 1 --budget ai2/oe-adapt  \
#     --mount beaker://jacobm/llama_2_7b-tulu_all=/llama_2_7b-tulu_all \
#     --mount beaker://jacobm/tulu_2_7b-tulu_match-science_100=/tulu_2_7b-tulu_match-science_100 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_match-science_200=/tulu_2_7b-tulu_match-science_200 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_match-science_500=/tulu_2_7b-tulu_match-science_500 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_match-science_1000=/tulu_2_7b-tulu_match-science_1000 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_match-science_2500=/tulu_2_7b-tulu_match-science_2500

# beaker session create --gpus 1 --budget ai2/oe-adapt  \
#     --mount beaker://jacobm/llama_2_7b-tulu_all=/llama_2_7b-tulu_all \
#     --mount beaker://jacobm/tulu_2_7b-tulu_match-coding_100=/tulu_2_7b-tulu_match-coding_100 \
#     --mount beaker://jacobm/tulu_2_7b-tulu_match-safety_100=/tulu_2_7b-tulu_match-safety_100


weights = [
    (0.1, 0.9),
    (0.2, 0.8),
    (0.3, 0.7),
    (0.4, 0.6),
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
    (0.8, 0.2),
    (0.9, 0.1),
    # (1.0, 1.0),
]

data_weighted_coefficients = {
    # # science 100
    # (
    #     "/llama_2_7b-tulu_all_with_coding",
    #     "/llama_2_7b-tulu_none-coding_20",
    #     "linear_weighted",
    # ) :
    # [
    #     (0.90, 0.10),
    # ],
    # # science 200
    # (
    #     "/llama_2_7b-tulu_all_with_coding",
    #     "/llama_2_7b-tulu_none-coding_40",
    #     "linear_weighted",
    # ) :
    # [
    #     (0.83, 0.17),
    # ],
    # # science 500
    # (
    #     "/llama_2_7b-tulu_all_with_coding",
    #     "/llama_2_7b-tulu_none-coding_60",
    #     "linear_weighted",
    # ) :
    # [
    #     (0.76, 0.24),
    # ],
    # science 1000
    # (
    #     "/llama_2_7b-tulu_all_with_coding",
    #     "/llama_2_7b-tulu_none-coding_80",
    #     "linear_weighted",
    # ) :
    # [
    #     (0.70, 0.30),
    # ],
    # # science 2500
    # (
    #     "/llama_2_7b-tulu_all_with_coding",
    #     "/llama_2_7b-tulu_none-coding_100",
    #     "linear_weighted",
    # ) :
    # [
    #     (0.65, 0.35),
    # ],
    # science 100
    # (
    #     "/llama_2_7b-tulu_all_with_coding",
    #     "/tulu_2_7b_with_coding-tulu_none-coding_20",
    #     "linear_weighted",
    # ) :
    # [
    #     (0.90, 0.10),
    # ],
    # # science 200
    # (
    #     "/llama_2_7b-tulu_all_with_coding",
    #     "/tulu_2_7b_with_coding-tulu_none-coding_40",
    #     "linear_weighted",
    # ) :
    # [
    #     (0.83, 0.17),
    # ],
    # # science 500
    # (
    #     "/llama_2_7b-tulu_all_with_coding",
    #     "/tulu_2_7b_with_coding-tulu_none-coding_60",
    #     "linear_weighted",
    # ) :
    # [
    #     (0.76, 0.24),
    # ],
    # science 1000
    # (
    #     "/llama_2_7b-tulu_all_with_coding",
    #     "/tulu_2_7b_with_coding-tulu_none-coding_80",
    #     "linear_weighted",
    # ) :
    # [
    #     (0.70, 0.30),
    # ],
    # # science 2500
    # (
    #     "/llama_2_7b-tulu_all_with_coding",
    #     "/tulu_2_7b_with_coding-tulu_none-coding_100",
    #     "linear_weighted",
    # ) :
    # [
    #     (0.65, 0.35),
    # ],
    # science 100
    (
        "/llama_2_7b-tulu_all_with_coding",
        "/llama_2_7b-tulu_none-coding_20",
        "task_arithmetic",
    ) :
    [
        (1.0, 0.11),
    ],
    # science 200
    (
        "/llama_2_7b-tulu_all_with_coding",
        "/llama_2_7b-tulu_none-coding_40",
        "task_arithmetic",
    ) :
    [
        (1.0, 0.21),
    ],
    # science 500
    (
        "/llama_2_7b-tulu_all_with_coding",
        "/llama_2_7b-tulu_none-coding_60",
        "task_arithmetic",
    ) :
    [
        (1.0, 0.32),
    ],
    # science 1000
    (
        "/llama_2_7b-tulu_all_with_coding",
        "/llama_2_7b-tulu_none-coding_80",
        "task_arithmetic",
    ) :
    [
        (1.0, 0.42),
    ],
    # science 2500
    (
        "/llama_2_7b-tulu_all_with_coding",
        "/llama_2_7b-tulu_none-coding_100",
        "task_arithmetic",
    ) :
    [
        (1.0, 0.53),
    ],
    # # science 100
    # (
    #     "/llama_2_7b-tulu_all_with_coding",
    #     "/llama_2_7b-tulu_none-coding_20",
    #     "task_arithmetic",
    # ) :
    # [
    #     (1.0, 1.0),
    # ],
    # # science 200
    # (
    #     "/llama_2_7b-tulu_all_with_coding",
    #     "/llama_2_7b-tulu_none-coding_40",
    #     "task_arithmetic",
    # ) :
    # [
    #     (1.0, 1.0),
    # ],
    # # science 500
    # (
    #     "/llama_2_7b-tulu_all_with_coding",
    #     "/llama_2_7b-tulu_none-coding_60",
    #     "task_arithmetic",
    # ) :
    # [
    #     (1.0, 1.0),
    # ],
    # # science 1000
    # (
    #     "/llama_2_7b-tulu_all_with_coding",
    #     "/llama_2_7b-tulu_none-coding_80",
    #     "task_arithmetic",
    # ) :
    # [
    #     (1.0, 1.0),
    # ],
    # # science 2500
    # (
    #     "/llama_2_7b-tulu_all_with_coding",
    #     "/llama_2_7b-tulu_none-coding_100",
    #     "task_arithmetic",
    # ) :
    # [
    #     (1.0, 1.0),
    # ],
}


domain_models = {
    # "safety_100": "/llama_2_7b-tulu_none-safety_100",
    # "coding_100": "/llama_2_7b-tulu_none-coding_100",

    # "safety_100": "/tulu_2_7b-tulu_none-safety_100",
    # "coding_100": "/tulu_2_7b-tulu_none-coding_100",
    # "science_2500": "/tulu_2_7b-tulu_none-science_2500",

    # "science_100": "/llama_2_7b-tulu_none-science_100",
    # "science_200": "/llama_2_7b-tulu_none-science_200",
    # "science_500": "/llama_2_7b-tulu_none-science_500",
    # "science_1000": "/llama_2_7b-tulu_none-science_1000",
    # "science_2500": "/llama_2_7b-tulu_none-science_2500",

    # "safety_20": "/llama_2_7b-tulu_none-safety_20",
    # "safety_40": "/llama_2_7b-tulu_none-safety_40",
    # "safety_60": "/llama_2_7b-tulu_none-safety_60",
    # "safety_80": "/llama_2_7b-tulu_none-safety_80",
    # "safety_100": "/llama_2_7b-tulu_none-safety_100",

    # "coding_20": "/llama_2_7b-tulu_none-coding_20",
    # "coding_40": "/llama_2_7b-tulu_none-coding_40",
    # "coding_60": "/llama_2_7b-tulu_none-coding_60",
    # "coding_80": "/llama_2_7b-tulu_none-coding_80",
    # "coding_100": "/llama_2_7b-tulu_none-coding_100",

    # "science_100": "/tulu_2_7b-tulu_none-science_100",
    # "science_200": "/tulu_2_7b-tulu_none-science_200",
    # "science_500": "/tulu_2_7b-tulu_none-science_500",
    # "science_1000": "/tulu_2_7b-tulu_none-science_1000",

    # "safety_20": "/tulu_2_7b-tulu_none-safety_20",
    # "safety_40": "/tulu_2_7b-tulu_none-safety_40",
    # "safety_60": "/tulu_2_7b-tulu_none-safety_60",
    # "safety_80": "/tulu_2_7b-tulu_none-safety_80",

    # "coding_20": "/tulu_2_7b-tulu_none-coding_20",
    # "coding_40": "/tulu_2_7b-tulu_none-coding_40",
    # "coding_60": "/tulu_2_7b-tulu_none-coding_60",
    # "coding_80": "/tulu_2_7b-tulu_none-coding_80",

    # "coding_20": "/tulu_2_7b_with_coding-tulu_none-coding_20",
    # "coding_40": "/tulu_2_7b_with_coding-tulu_none-coding_40",
    # "coding_60": "/tulu_2_7b_with_coding-tulu_none-coding_60",
    # "coding_80": "/tulu_2_7b_with_coding-tulu_none-coding_80",
    # "coding_100": "/tulu_2_7b_with_coding-tulu_none-coding_100",

    # "science_100": "/tulu_2_7b-tulu_match-science_100",
    # "science_200": "/tulu_2_7b-tulu_match-science_200",
    # "science_1000": "/tulu_2_7b-tulu_match-science_1000",
    # "science_500": "/tulu_2_7b-tulu_match-science_500",
    # "science_2500": "/tulu_2_7b-tulu_match-science_2500",

    "coding_100": "/tulu_2_7b-tulu_match-coding_100",
    "safety_100": "/tulu_2_7b-tulu_match-safety_100",
}

merge_methods = [
    "linear_weighted",
    # "task_arithmetic",
    # "dare_task_arithmetic",
    # "dare_linear",
    # "dare_ties",
    # "ties",
    # "slerp",
]

tulu_file = "/llama_2_7b-tulu_all"
# tulu_file = "/llama_2_7b-tulu_all_with_coding"

def print_and_run(cmd):
    print(cmd)
    subprocess.run(cmd, shell=True)

for model_tag in domain_models:
    for merge_method in merge_methods:
        for (tuluWeight, domainWeight) in weights:
# for base_model, domain_model, merge_method in data_weighted_coefficients:
        # for (tuluWeight, domainWeight) in data_weighted_coefficients[(base_model, domain_model, merge_method)]:
            # Copy yaml
            base_yaml = f"scripts/merge_models/merge-{merge_method}-base.yml"
            with open(base_yaml, 'r') as f:
                d1 = yaml.load(f.read(), Loader=yaml.FullLoader)
            d = copy.deepcopy(d1)
            if merge_method == "task_arithmetic" or merge_method == "dare_task_arithmetic":
                tuluWeight = 1.0
            if merge_method == "linear_weighted" or merge_method == "task_arithmetic":
                # Set merge-specific parameters
                d["models"][0]["model"] = tulu_file
                d["models"][0]["parameters"]["weight"] = tuluWeight
                d["models"][1]["model"] = domain_models[model_tag]
                # d["models"][1]["model"] = domain_model
                d["models"][1]["parameters"]["weight"] = domainWeight
            elif merge_method in ["dare_linear", "dare_ties", "ties", "dare_task_arithmetic"]:
                # Set merge-specific parameters
                d["models"][1]["model"] = tulu_file
                d["models"][1]["parameters"]["weight"] = tuluWeight
                d["models"][2]["model"] = domain_models[model_tag]
                # d["models"][2]["model"] = domain_model
                d["models"][2]["parameters"]["weight"] = domainWeight
            elif merge_method == "slerp":
                # Set merge-specific parameters
                d["slices"][0]["sources"][0]["model"] = tulu_file
                d["slices"][0]["sources"][1]["model"] = domain_models[model_tag]
                # d["slices"][0]["sources"][1]["model"] = domain_model
                d["parameters"]["t"][0]["value"] = domainWeight
            else:
                raise Exception

            # Create folders and files
            print_and_run("mkdir tmp-4k")
            file = open("tmp-4k/merge-config.yaml", "w")
            yaml.dump(d, file, default_flow_style=True)
            file.close()

            # Merge model
            print_and_run(f"mergekit-yaml tmp-4k/merge-config.yaml tmp-4k/ --cuda")

            # Upload model
            model_name = f"{merge_method}-{tulu_file[1:]}_{tuluWeight}-{domain_models[model_tag][1:]}_{domainWeight}"
            # model_name = f"{merge_method}-{base_model[1:]}_{tuluWeight}-{domain_model[1:]}_{domainWeight}"
            print_and_run(f"beaker dataset create tmp-4k/ --name {model_name} --workspace ai2/modular_adaptation")

            # Cleanup
            print_and_run("rm -rf tmp-4k/")