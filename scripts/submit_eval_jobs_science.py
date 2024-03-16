import copy
import subprocess
import yaml
import random
import re
import itertools
from datetime import date

today = date.today().strftime("%m%d%Y")

with open("beaker_configs/default_eval.yaml", 'r') as f:
    default_yaml = f.read()
d1 = yaml.load(default_yaml, Loader=yaml.FullLoader)

# cluster = "ai2/general-cirrascale"
# cluster = "ai2/allennlp-cirrascale"
# cluster = "ai2/s2-cirrascale-l40"
# cluster = "ai2/mosaic-cirrascale-a100",
cluster = "ai2/general-cirrascale-a100-80g-ib"
# cluster = "ai2/prior-elanding"
num_gpus = 1
d1['tasks'][0]['context']['cluster'] = cluster
d1['tasks'][0]['context']['priority'] = "high"
# d1['tasks'][0]['context']['priority'] = "preemptible"
d1['tasks'][0]['resources']['gpuCount'] = num_gpus

merge_models = False

# modify here for different set of experiments
experiment_groups = [
    "mmlu_0shot",
    "mmlu_5shot",
    "gsm_direct",
    "gsm_cot",
    "bbh_direct",
    "bbh_cot",
    "tydiqa_goldp_1shot",
    "tydiqa_no_context_1shot",
    "toxigen",
    "codex_eval_temp_0.1",
    "codex_eval_temp_0.8",
    "truthfulqa",
    "alpaca_farm",
]

lora = False

datasets = [
    # 70b
    # ("01HR8E8SNZKVEKK46S650M4Q42", "llama_2_70b-tulu_all-science_none", "/model", "baselines"),
    # ("01HRNSDNB34WY6NXJR80707V2Z", "llama_2_70b-tulu_none-science_100", "/model", "baselines"),
    # ("01HRNGGG1VJS6C52164ESYYJBY", "llama_2_70b-tulu_none-science_1000", "/model", "baselines"),
    # ("01HRAKXBWVW1YYA9T56FFVFVTP", "tulu_2_70b_continued_ft-tulu_match-science_1000", "/model", "baselines"),
    # ("01HREEEJ5YRJEX16RGA9F5CZV9", "tulu_2_70b_continued_ft-tulu_none-science_1000", "/model", "baselines"),


    # ("01HQVW2761BAF4JMD31YESTZWX", "llama_2_7b-tulu_none-science_100", "/model", "baselines"),
    # ("01HQW32DZGYQTWPX7Q8XYJS4CB", "llama_2_7b-tulu_none-science_200", "/model", "baselines"),
    # ("01HQW98WC34DAXR596V7WDZGTT", "llama_2_7b-tulu_none-science_500", "/model", "baselines"),
    # ("01HQVZSGJRJTY2GQD6N7RMK5G4", "llama_2_7b-tulu_none-science_1000", "/model", "baselines"),
    # ("01HQXNRAHA3Z08504TGNMP6VD5", "llama_2_7b-tulu_none-science_2500", "/model", "baselines"),
    # ("01HRK8G8ZXN5YRTSGRFM18QF58", "llama_2_7b-tulu_none-science_upsample", "/model", "baselines"),

    # ("01HR0C4NR36RF6YG3PTYK21H0X", "llama_2_7b-tulu_all-science_none-seed_42", "/model", "baselines"),
    # ("01HR0DJCFPX98BHS4RSD46MGDQ", "llama_2_7b-tulu_all-science_none-seed_123", "/model", "baselines"),
    # ("01HR2Z38CNQX3ADW6HXQV1DV0B", "llama_2_7b-tulu_all-science_none-seed_52830", "/model", "baselines"),
    # ("01HR0QTFF2QMFHB160MWJ8CFS9", "llama_2_7b-tulu_none-science_1000-seed_123", "/model", "baselines"),
    # ("01HR13H0FV4B77MKQJPW9CB7BD", "llama_2_7b-tulu_none-science_1000-seed_52830", "/model", "baselines"),

    # continued finetuning:
    # ("01HR1640TEQAHJMSWCQJW4FF6J", "tulu_2_7b_continued_ft-tulu_none-science_100", "/model", "baselines"),
    # ("01HR18P3ZY4EHHV5KEXSS5CFWV", "tulu_2_7b_continued_ft-tulu_none-science_200", "/model", "baselines"),
    # ("01HR1EEPG578W05YHV6MRN1TMV", "tulu_2_7b_continued_ft-tulu_none-science_500", "/model", "baselines"),
    # ("01HR2WRZ7FY8XBV7218XKR96PD", "tulu_2_7b_continued_ft-tulu_none-science_1000", "/model", "baselines"),
    # ("01HR3VZM057Z8HKBF9N9VC9K88", "tulu_2_7b_continued_ft-tulu_none-science_2500", "/model", "baselines"),
    # ("01HRQ7H1VDHWG7EAH5373G8Z18", "tulu_2_7b_continued_ft-tulu_none-science_upsample", "/model", "baselines"),
    # ("01HR324VPMHA9BJGVXY0TW3XDT", "tulu_2_7b_continued_ft-tulu_match-science_100", "/model", "baselines"),
    # ("01HR343V9HQFH71XH4RHMJRQJ7", "tulu_2_7b_continued_ft-tulu_match-science_200", "/model", "baselines"),
    # ("01HR5C00ZNXQPDBY3T7GBCBFCA", "tulu_2_7b_continued_ft-tulu_match-science_500", "/model", "baselines"),
    # ("01HR609Z6CXES73B3C3VSKPE28", "tulu_2_7b_continued_ft-tulu_match-science_1000", "/model", "baselines"),
    # ("01HR2Y2P2YBFJXPVRMDGNB2TH4", "tulu_2_7b_continued_ft-tulu_match-science_2500", "/model", "baselines"),



    # Sequential baselines
    # ("01HR318SEPKWWRTVR4WR16SP87", "llama_2_7b-tulu_match-science_100", "/model", "baselines"),
    # ("01HR353BWT0Q62NXQA6CCF7QF5", "llama_2_7b-tulu_match-science_200", "/model", "baselines"),
    # ("01HR69T26E9N17966FBM55R1N2", "llama_2_7b-tulu_match-science_500", "/model", "baselines"),
    # ("01HR8NN6DZ6BCV0SN4QWJ5YEKA", "llama_2_7b-tulu_match-science_1000", "/model", "baselines"),
    # ("01HR43C2AA7VSE3573TH0SSP7Y", "llama_2_7b-tulu_match-science_2500", "/model", "baselines"),
    # ("01HRB0ZR866EGXB506YSS08MH2", "llama_2_7b-tulu_all-science_100", "/model", "baselines"),
    # ("01HRCZ875RX615Z455QR9Z8480", "llama_2_7b-tulu_all-science_200", "/model", "baselines"), # need to run
    # ("01HRDA7S6C1QP8B1A3WBFVWPJD", "llama_2_7b-tulu_all-science_500", "/model", "baselines"), # need to run
    # ("01HRD0PSPQP1C2F458RS59X7TQ", "llama_2_7b-tulu_all-science_1000", "/model", "baselines"), # need to run
    # ("01HR6M9TYWFXFKRX1NBJFT4YNH", "llama_2_7b-tulu_all-science_2500", "/model", "baselines"),
    # ("01HRB2999DT5MKJKMS2FZ2SWY1", "llama_2_7b-tulu_all-science_upsample", "/model", "baselines"),

    # Merged models:
    # ("01HRRG91TNWC1YHGCH3KN1Q2D7", "dare_ties-llama_2_7b-tulu_all_0.1-science_100_0.9", "/model", "merged_models"),
    # ("01HRRGM04F6D3KTXGQ38Q5M0QW", "dare_ties-llama_2_7b-tulu_all_0.2-science_100_0.8", "/model", "merged_models"),
    # ("01HRRGZWFWAZ4X0Y1T417805D6", "dare_ties-llama_2_7b-tulu_all_0.3-science_100_0.7", "/model", "merged_models"),
    # ("01HRRHCT125CPZ1RP0ADGBNF5Z", "dare_ties-llama_2_7b-tulu_all_0.4-science_100_0.6", "/model", "merged_models"),
    # ("01HRRHSYZSS63CXMKH663J5XEG", "dare_ties-llama_2_7b-tulu_all_0.5-science_100_0.5", "/model", "merged_models"),
    # ("01HRRJ6APX5J94ZRV9KRCFYCA4", "dare_ties-llama_2_7b-tulu_all_0.6-science_100_0.4", "/model", "merged_models"),
    # ("01HRRJJHCSPFX9X7FVK55W7SVF", "dare_ties-llama_2_7b-tulu_all_0.7-science_100_0.3", "/model", "merged_models"),
    # ("01HRRJZ0JTVN0VYPXHKZY6YQGP", "dare_ties-llama_2_7b-tulu_all_0.8-science_100_0.2", "/model", "merged_models"),
    # ("01HRRKBCJJ5W52F4SPV9AXM548", "dare_ties-llama_2_7b-tulu_all_0.9-science_100_0.1", "/model", "merged_models"),

    # ("01HRRKPQV1B5BY6QQKDZN1HK8V", "dare_ties-llama_2_7b-tulu_all_0.1-science_200_0.9", "/model", "merged_models"),
    # ("01HRRM2W3FR4M0AHAFEP4BQBHC", "dare_ties-llama_2_7b-tulu_all_0.2-science_200_0.8", "/model", "merged_models"),
    # ("01HRRME5P69KJVKVMC1DPPZ2NS", "dare_ties-llama_2_7b-tulu_all_0.3-science_200_0.7", "/model", "merged_models"),
    # ("01HRRMT7G8M4SKNXF3VWHNQXBW", "dare_ties-llama_2_7b-tulu_all_0.4-science_200_0.6", "/model", "merged_models"),
    # ("01HRRN6HY6B8JDS1JV23STMB0F", "dare_ties-llama_2_7b-tulu_all_0.5-science_200_0.5", "/model", "merged_models"),
    # ("01HRRNJTYEHJBXDDZK518827MW", "dare_ties-llama_2_7b-tulu_all_0.6-science_200_0.4", "/model", "merged_models"),
    # ("01HRRNY1HZKCJRY3F2VGPK4N2D", "dare_ties-llama_2_7b-tulu_all_0.7-science_200_0.3", "/model", "merged_models"),
    # ("01HRRPA14QGJHVCXMEMS0GHQK0", "dare_ties-llama_2_7b-tulu_all_0.8-science_200_0.2", "/model", "merged_models"),
    # ("01HRRPNGR3R13JZ1CRQ4CQ7RFT", "dare_ties-llama_2_7b-tulu_all_0.9-science_200_0.1", "/model", "merged_models"),

    # ("01HRRD1YVBW5516ZYJ5P2G4JTR", "dare_ties-llama_2_7b-tulu_all_0.1-science_500_0.9", "/model", "merged_models"),
    # ("01HRRDDZQM632J78CQGA22N1BY", "dare_ties-llama_2_7b-tulu_all_0.2-science_500_0.8", "/model", "merged_models"),
    # ("01HRRDSG3F3VW0JV26J762R73E", "dare_ties-llama_2_7b-tulu_all_0.3-science_500_0.7", "/model", "merged_models"),
    # ("01HRRE528YCPX0H646GERS1MDB", "dare_ties-llama_2_7b-tulu_all_0.4-science_500_0.6", "/model", "merged_models"),
    # ("01HRREH4TR7DSZ41S2CVYTBDW1", "dare_ties-llama_2_7b-tulu_all_0.5-science_500_0.5", "/model", "merged_models"),
    # ("01HRREWYETKKJ50FZGH5AGMHMX", "dare_ties-llama_2_7b-tulu_all_0.6-science_500_0.4", "/model", "merged_models"),
    # ("01HRRF7M33WH1NWQK4PNRAXQSX", "dare_ties-llama_2_7b-tulu_all_0.7-science_500_0.3", "/model", "merged_models"),
    # ("01HRRFJHE09ETZZTD2G7H24JCZ", "dare_ties-llama_2_7b-tulu_all_0.8-science_500_0.2", "/model", "merged_models"),
    # ("01HRRFXFQF59HZXK4JF1F8104N", "dare_ties-llama_2_7b-tulu_all_0.9-science_500_0.1", "/model", "merged_models"),

    # ("01HRRD782H490HWK88G7FJ2E34", "dare_ties-llama_2_7b-tulu_all_0.1-science_1000_0.9", "/model", "merged_models"),
    # ("01HRRB50QPE4G6Y04Z47QB6BHV", "dare_ties-llama_2_7b-tulu_all_0.2-science_1000_0.8", "/model", "merged_models"),
    # ("01HRRDKW6NZ2EE59SBPP1KFNEM", "dare_ties-llama_2_7b-tulu_all_0.3-science_1000_0.7", "/model", "merged_models"),
    # ("01HRRDZH45HRTMYVWQS08SNS6C", "dare_ties-llama_2_7b-tulu_all_0.4-science_1000_0.6", "/model", "merged_models"),
    # ("01HRREB2HF1P5GVP69D5M0S5S8", "dare_ties-llama_2_7b-tulu_all_0.5-science_1000_0.5", "/model", "merged_models"),
    # ("01HRREP0AJ1NEGKVTEY55D6P25", "dare_ties-llama_2_7b-tulu_all_0.6-science_1000_0.4", "/model", "merged_models"),
    # ("01HRRF10M9WX6P6KARKX6F6M4R", "dare_ties-llama_2_7b-tulu_all_0.7-science_1000_0.3", "/model", "merged_models"),
    # ("01HRRFBXP47HGW0NA39BY4HACR", "dare_ties-llama_2_7b-tulu_all_0.8-science_1000_0.2", "/model", "merged_models"),
    # ("01HRRFQPTABRCSSK8GX6H01NFS", "dare_ties-llama_2_7b-tulu_all_0.9-science_1000_0.1", "/model", "merged_models"),

    # ("01HRRG3JAYZ7QWGYCYWP4WZS14", "dare_ties-llama_2_7b-tulu_all_0.1-science_2500_0.9", "/model", "merged_models"),
    # ("01HRRGFCGBBRCQ8VZ23SZG3TRA", "dare_ties-llama_2_7b-tulu_all_0.2-science_2500_0.8", "/model", "merged_models"),
    # ("01HRRGVG2AX9D7G6BCTDSAS1RK", "dare_ties-llama_2_7b-tulu_all_0.3-science_2500_0.7", "/model", "merged_models"),
    # ("01HRRH703J74N0TNF0T5J84Q8X", "dare_ties-llama_2_7b-tulu_all_0.4-science_2500_0.6", "/model", "merged_models"),
    # ("01HRRHK5XG8GY79PFE20P0XNRV", "dare_ties-llama_2_7b-tulu_all_0.5-science_2500_0.5", "/model", "merged_models"),
    # ("01HRRHYATV6BQZBCBF6RB1EMZF", "dare_ties-llama_2_7b-tulu_all_0.6-science_2500_0.4", "/model", "merged_models"),
    # ("01HRRJA7WRV7FJCA26G3SE4GC4", "dare_ties-llama_2_7b-tulu_all_0.7-science_2500_0.3", "/model", "merged_models"),
    # ("01HRRJPDJ6PHCD5DHDV97MMN66", "dare_ties-llama_2_7b-tulu_all_0.8-science_2500_0.2", "/model", "merged_models"),
    # ("01HRRK2BCHAQ5YCM9C31XY9XD6", "dare_ties-llama_2_7b-tulu_all_0.9-science_2500_0.1", "/model", "merged_models"),

    # ("01HRRKDKZ9N7XTQYKFW2RS0F08", "dare_ties-llama_2_7b-tulu_all_0.1-science_upsample_0.9", "/model", "merged_models"),
    # ("01HRRKSPWAQF5JMMY76VWZ7X91", "dare_ties-llama_2_7b-tulu_all_0.2-science_upsample_0.8", "/model", "merged_models"),
    # ("01HRRM5SWHM9R77ARCVSR3NJF7", "dare_ties-llama_2_7b-tulu_all_0.3-science_upsample_0.7", "/model", "merged_models"),
    # ("01HRRMJ7AYHW5SBB0B0QSF3KAC", "dare_ties-llama_2_7b-tulu_all_0.4-science_upsample_0.6", "/model", "merged_models"),
    # ("01HRRMY8RQKV96PBCF61PYHAJV", "dare_ties-llama_2_7b-tulu_all_0.5-science_upsample_0.5", "/model", "merged_models"),
    # ("01HRRN9HV9WWB2RSCG9M24JG2V", "dare_ties-llama_2_7b-tulu_all_0.6-science_upsample_0.4", "/model", "merged_models"),
    # ("01HRRNNWE8MR2NH1KR9A7JPYCN", "dare_ties-llama_2_7b-tulu_all_0.7-science_upsample_0.3", "/model", "merged_models"),
    # ("01HRRP1XMV0KRTTKHZTGQAYM6F", "dare_ties-llama_2_7b-tulu_all_0.8-science_upsample_0.2", "/model", "merged_models"),
    # ("01HRRPDC72KFPYKM3838JDX2TN", "dare_ties-llama_2_7b-tulu_all_0.9-science_upsample_0.1", "/model", "merged_models"),

    # ties:
    # ("01HRRSZ892N6B6VTD7JH1AEVM6", "ties-llama_2_7b-tulu_all_0.1-science_100_0.9", "/model", "merged_models"),
    # ("", "ties-llama_2_7b-tulu_all_0.2-science_100_0.8", "/model", "merged_models"),
    # ("01HRRTNTF418K1N837WS3JQZYG", "ties-llama_2_7b-tulu_all_0.3-science_100_0.7", "/model", "merged_models"),
    # ("01HRRV2CDFDTVW5G0W68AW2WXP", "ties-llama_2_7b-tulu_all_0.4-science_100_0.6", "/model", "merged_models"),
    # ("", "ties-llama_2_7b-tulu_all_0.5-science_100_0.5", "/model", "merged_models"),
    # ("", "ties-llama_2_7b-tulu_all_0.6-science_100_0.4", "/model", "merged_models"),
    # ("01HRRW7GBBJKYZE0NSWD428AVT", "ties-llama_2_7b-tulu_all_0.7-science_100_0.3", "/model", "merged_models"),
    # ("", "ties-llama_2_7b-tulu_all_0.8-science_100_0.2", "/model", "merged_models"),
    # ("01HRRX9QFYEWE9YSRCWZ2PSGT8", "ties-llama_2_7b-tulu_all_0.9-science_100_0.1", "/model", "merged_models"),

    # ("01HRRXNJKXPK27DSV89TGHRZ7K", "ties-llama_2_7b-tulu_all_0.1-science_200_0.9", "/model", "merged_models"),
    # ("01HRRY1N2F6EASZAHG7Q3CDTT6", "ties-llama_2_7b-tulu_all_0.2-science_200_0.8", "/model", "merged_models"),
    # ("01HRRYE12NYSJ6TVW3Z5HC4AN5", "ties-llama_2_7b-tulu_all_0.3-science_200_0.7", "/model", "merged_models"),
    # ("01HRRYSRB124GG81DAN6RNG3S2", "ties-llama_2_7b-tulu_all_0.4-science_200_0.6", "/model", "merged_models"),
    # ("01HRRZ67VAC4KE2WH4VCG33N0Q", "ties-llama_2_7b-tulu_all_0.5-science_200_0.5", "/model", "merged_models"),
    # ("01HRRZJ211NC1FDV80V01FTRKP", "ties-llama_2_7b-tulu_all_0.6-science_200_0.4", "/model", "merged_models"),
    # ("01HRRZX5VGWT2FWNG2HR9XKN30", "ties-llama_2_7b-tulu_all_0.7-science_200_0.3", "/model", "merged_models"),
    # ("01HRS09BS6JR5596M6XA2MJ88R", "ties-llama_2_7b-tulu_all_0.8-science_200_0.2", "/model", "merged_models"),
    # ("01HRS0PSD4TWA2XVYGSN1CVH4Z", "ties-llama_2_7b-tulu_all_0.9-science_200_0.1", "/model", "merged_models"),

    # ("", "ties-llama_2_7b-tulu_all_0.1-science_500_0.9", "/model", "merged_models"),
    # ("", "ties-llama_2_7b-tulu_all_0.2-science_500_0.8", "/model", "merged_models"),
    # ("01HRRQED6D7WVFD9SZJMAX8VEP", "ties-llama_2_7b-tulu_all_0.3-science_500_0.7", "/model", "merged_models"),
    # ("01HRRQT4HKCPKZAQGF61B4N48Y", "ties-llama_2_7b-tulu_all_0.4-science_500_0.6", "/model", "merged_models"),
    # ("01HRRR5779DFW4VS4JC4KAWNA1", "ties-llama_2_7b-tulu_all_0.5-science_500_0.5", "/model", "merged_models"),
    # ("01HRRRHP8WN0W8Z4B4372JP8WM", "ties-llama_2_7b-tulu_all_0.6-science_500_0.4", "/model", "merged_models"),
    # ("01HRRRWWA4Q8WJ5Z67REX3PPSN", "ties-llama_2_7b-tulu_all_0.7-science_500_0.3", "/model", "merged_models"),
    # ("01HRRS857XHWY1WSSP4AANTE39", "ties-llama_2_7b-tulu_all_0.8-science_500_0.2", "/model", "merged_models"),
    # ("01HRRSKBK3C2CBM33DGB4NFN44", "ties-llama_2_7b-tulu_all_0.9-science_500_0.1", "/model", "merged_models"),

    # ("01HRRPQWZNRE8RSVEQDX6A0AZK", "ties-llama_2_7b-tulu_all_0.1-science_1000_0.9", "/model", "merged_models"),
    # ("", "ties-llama_2_7b-tulu_all_0.2-science_1000_0.8", "/model", "merged_models"),
    # ("", "ties-llama_2_7b-tulu_all_0.3-science_1000_0.7", "/model", "merged_models"),
    # ("01HRRQHMSBGV7YJB7BX74DFG3Z", "ties-llama_2_7b-tulu_all_0.4-science_1000_0.6", "/model", "merged_models"),
    # ("01HRRQXKBDD4H3E6X3YDJ7CVEE", "ties-llama_2_7b-tulu_all_0.5-science_1000_0.5", "/model", "merged_models"),
    # ("01HRRR8TGMYVJ8TJFBD0RRF6Z9", "ties-llama_2_7b-tulu_all_0.6-science_1000_0.4", "/model", "merged_models"),
    # ("01HRRRMATPHV98FHM6N49PGF31", "ties-llama_2_7b-tulu_all_0.7-science_1000_0.3", "/model", "merged_models"),
    # ("01HRRRZ9Z6VRDQSW0T7DJ8GXBY", "ties-llama_2_7b-tulu_all_0.8-science_1000_0.2", "/model", "merged_models"),
    # ("01HRRSA7SVF5N595ADKHVZKDMJ", "ties-llama_2_7b-tulu_all_0.9-science_1000_0.1", "/model", "merged_models"),

    # ("01HRRSP1V1WZW8FE01Y7RC7MVT", "ties-llama_2_7b-tulu_all_0.1-science_2500_0.9", "/model", "merged_models"),
    # ("01HRRT1D8FVW87MMNASWJRS7Z7", "ties-llama_2_7b-tulu_all_0.2-science_2500_0.8", "/model", "merged_models"),
    # ("01HRRTCZ5DFXBQ3186M8FFPR85", "ties-llama_2_7b-tulu_all_0.3-science_2500_0.7", "/model", "merged_models"),
    # ("01HRRTRDJVAFM4YV8K0A84HP4X", "ties-llama_2_7b-tulu_all_0.4-science_2500_0.6", "/model", "merged_models"),
    # ("01HRRV4Z3JFJWJ651DBT4YAY80", "ties-llama_2_7b-tulu_all_0.5-science_2500_0.5", "/model", "merged_models"),
    # ("", "ties-llama_2_7b-tulu_all_0.6-science_2500_0.4", "/model", "merged_models"),
    # ("", "ties-llama_2_7b-tulu_all_0.7-science_2500_0.3", "/model", "merged_models"),
    # ("01HRRW8K2D2BW11TMRFF5JRH9X", "ties-llama_2_7b-tulu_all_0.8-science_2500_0.2", "/model", "merged_models"),
    # ("", "ties-llama_2_7b-tulu_all_0.9-science_2500_0.1", "/model", "merged_models"),

    # ("", "ties-llama_2_7b-tulu_all_0.1-science_upsample_0.9", "/model", "merged_models"),
    # ("", "ties-llama_2_7b-tulu_all_0.2-science_upsample_0.8", "/model", "merged_models"),
    # ("", "ties-llama_2_7b-tulu_all_0.3-science_upsample_0.7", "/model", "merged_models"),
    # ("", "ties-llama_2_7b-tulu_all_0.4-science_upsample_0.6", "/model", "merged_models"),
    # ("", "ties-llama_2_7b-tulu_all_0.5-science_upsample_0.5", "/model", "merged_models"),
    # ("", "ties-llama_2_7b-tulu_all_0.6-science_upsample_0.4", "/model", "merged_models"),
    # ("", "ties-llama_2_7b-tulu_all_0.7-science_upsample_0.3", "/model", "merged_models"),
    # ("", "ties-llama_2_7b-tulu_all_0.8-science_upsample_0.2", "/model", "merged_models"),
    # ("", "ties-llama_2_7b-tulu_all_0.9-science_upsample_0.1", "/model", "merged_models"),

    # slerp rerun all:
    ("", "slerp-llama_2_7b-tulu_all_0.1-science_100_0.9", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.2-science_100_0.8", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.3-science_100_0.7", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.4-science_100_0.6", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.5-science_100_0.5", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.6-science_100_0.4", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.7-science_100_0.3", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.8-science_100_0.2", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.9-science_100_0.1", "/model", "merged_models"),

    ("", "slerp-llama_2_7b-tulu_all_0.1-science_200_0.9", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.2-science_200_0.8", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.3-science_200_0.7", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.4-science_200_0.6", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.5-science_200_0.5", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.6-science_200_0.4", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.7-science_200_0.3", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.8-science_200_0.2", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.9-science_200_0.1", "/model", "merged_models"),

    ("", "slerp-llama_2_7b-tulu_all_0.1-science_500_0.9", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.2-science_500_0.8", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.3-science_500_0.7", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.4-science_500_0.6", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.5-science_500_0.5", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.6-science_500_0.4", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.7-science_500_0.3", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.8-science_500_0.2", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.9-science_500_0.1", "/model", "merged_models"),

    ("", "slerp-llama_2_7b-tulu_all_0.2-science_1000_0.8", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.3-science_1000_0.7", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.4-science_1000_0.6", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.5-science_1000_0.5", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.6-science_1000_0.4", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.7-science_1000_0.3", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.8-science_1000_0.2", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.9-science_1000_0.1", "/model", "merged_models"),

    ("", "slerp-llama_2_7b-tulu_all_0.1-science_2500_0.9", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.2-science_2500_0.8", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.3-science_2500_0.7", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.4-science_2500_0.6", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.5-science_2500_0.5", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.6-science_2500_0.4", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.7-science_2500_0.3", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.8-science_2500_0.2", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.9-science_2500_0.1", "/model", "merged_models"),

    ("", "slerp-llama_2_7b-tulu_all_0.1-science_upsample_0.9", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.2-science_upsample_0.8", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.3-science_upsample_0.7", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.4-science_upsample_0.6", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.5-science_upsample_0.5", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.6-science_upsample_0.4", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.7-science_upsample_0.3", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.8-science_upsample_0.2", "/model", "merged_models"),
    ("", "slerp-llama_2_7b-tulu_all_0.9-science_upsample_0.1", "/model", "merged_models"),
    
    # ("01HRR3JBES59RECVY8PS85TMEK", "dare_linear-llama_2_7b-tulu_all_0.1-science_100_0.9", "/model", "merged_models"),
    # ("01HRR4NG6PEEXS1X0W12HC08AF", "dare_linear-llama_2_7b-tulu_all_0.2-science_100_0.8", "/model", "merged_models"),
    # ("01HRR5QBJTMGCENFRQFCR2XK23", "dare_linear-llama_2_7b-tulu_all_0.3-science_100_0.7", "/model", "merged_models"),
    # ("01HRR6TWNMFXE3XYF8R3Q5828D", "dare_linear-llama_2_7b-tulu_all_0.4-science_100_0.6", "/model", "merged_models"),
    # ("01HRR7W9S2P6VT9BT5CGJ91261", "dare_linear-llama_2_7b-tulu_all_0.5-science_100_0.5", "/model", "merged_models"),
    # ("01HRR8Y00J7M1B57HWADP80VRC", "dare_linear-llama_2_7b-tulu_all_0.6-science_100_0.4", "/model", "merged_models"),
    # ("01HRRA06QM1QXABGZ8E5G6874V", "dare_linear-llama_2_7b-tulu_all_0.7-science_100_0.3", "/model", "merged_models"),
    # ("01HRRB25RDCFBXX6BB7T4ZQXZ1", "dare_linear-llama_2_7b-tulu_all_0.8-science_100_0.2", "/model", "merged_models"),
    # ("01HRRC40ZQSENBJN6PAMZPKFFK", "dare_linear-llama_2_7b-tulu_all_0.9-science_100_0.1", "/model", "merged_models"),

    # ("01HRR3XS6H7M4X35FEMYR20NWA", "dare_linear-llama_2_7b-tulu_all_0.1-science_200_0.9", "/model", "merged_models"),
    # ("01HRR51ADRT6HVSWNR3C5HP2JA", "dare_linear-llama_2_7b-tulu_all_0.2-science_200_0.8", "/model", "merged_models"),
    # ("01HRR636RJ1GXK84X6TGHGS8BF", "dare_linear-llama_2_7b-tulu_all_0.3-science_200_0.7", "/model", "merged_models"),
    # ("01HRR75FP8976KP3B384BPQRJ7", "dare_linear-llama_2_7b-tulu_all_0.4-science_200_0.6", "/model", "merged_models"),
    # ("01HRR88286ADDJK03JEJCMS17H", "dare_linear-llama_2_7b-tulu_all_0.5-science_200_0.5", "/model", "merged_models"),
    # ("01HRR99ZD6KTGE6CRRERRE4GA3", "dare_linear-llama_2_7b-tulu_all_0.6-science_200_0.4", "/model", "merged_models"),
    # ("01HRRABQ5KYQEG4QK53E450V42", "dare_linear-llama_2_7b-tulu_all_0.7-science_200_0.3", "/model", "merged_models"),
    # ("01HRRBCRD4RQQZQTC01G2C6KR4", "dare_linear-llama_2_7b-tulu_all_0.8-science_200_0.2", "/model", "merged_models"),
    # ("01HRRCEZBHS0F37235CV4RPW5Q", "dare_linear-llama_2_7b-tulu_all_0.9-science_200_0.1", "/model", "merged_models"),

    # ("01HRGFH0CTYWMCHEWF68JRJ8N5", "dare_linear-llama_2_7b-tulu_all_0.1-science_500_0.9", "/model", "merged_models"),
    # ("01HRGGJ79E0CV0PA4JDNZAKGV8", "dare_linear-llama_2_7b-tulu_all_0.2-science_500_0.8", "/model", "merged_models"),
    # ("01HRGHP5D6EXSMXFKMMCZD89CQ", "dare_linear-llama_2_7b-tulu_all_0.3-science_500_0.7", "/model", "merged_models"),
    # ("01HRGJQX583F0YA501Q7DP5E2X", "dare_linear-llama_2_7b-tulu_all_0.4-science_500_0.6", "/model", "merged_models"),
    # ("01HRGKRHWMQQB3FPRYKFHEQSC8", "dare_linear-llama_2_7b-tulu_all_0.5-science_500_0.5", "/model", "merged_models"),
    # ("01HRGMSFYZS11CVJ1W17NP857H", "dare_linear-llama_2_7b-tulu_all_0.6-science_500_0.4", "/model", "merged_models"),
    # ("01HRGNTP64XZZG14VTHH5DAG1Z", "dare_linear-llama_2_7b-tulu_all_0.7-science_500_0.3", "/model", "merged_models"),
    # ("01HRGPXCN2SZE6HCP3PS14EW4F", "dare_linear-llama_2_7b-tulu_all_0.8-science_500_0.2", "/model", "merged_models"),
    # ("01HRGQZ70ECVCG6P14GV341B31", "dare_linear-llama_2_7b-tulu_all_0.9-science_500_0.1", "/model", "merged_models"),

    # ("01HRGFW40QP3HNCQ1YT3CVHQ9C", "dare_linear-llama_2_7b-tulu_all_0.1-science_1000_0.9", "/model", "merged_models"),
    # ("01HRGGXGNQ732CVSYSEFRXE85S", "dare_linear-llama_2_7b-tulu_all_0.2-science_1000_0.8", "/model", "merged_models"),
    # ("01HRGJ1F70NZKXT9XNV9FA0QFM", "dare_linear-llama_2_7b-tulu_all_0.3-science_1000_0.7", "/model", "merged_models"),
    # ("01HRGK2BNQNG5HTQ0JWXDV7V3A", "dare_linear-llama_2_7b-tulu_all_0.4-science_1000_0.6", "/model", "merged_models"),
    # ("01HRGM3HA1TMTVPJZDT0HF7BTN", "dare_linear-llama_2_7b-tulu_all_0.5-science_1000_0.5", "/model", "merged_models"),
    # ("01HRGN4GFE3SREJ832VRD67YEV", "dare_linear-llama_2_7b-tulu_all_0.6-science_1000_0.4", "/model", "merged_models"),
    # ("01HRGP5ZG92S6YQECB45AQTEJ5", "dare_linear-llama_2_7b-tulu_all_0.7-science_1000_0.3", "/model", "merged_models"),
    # ("01HRGQ8KTKVCWZ9RFFBEVM86W1", "dare_linear-llama_2_7b-tulu_all_0.8-science_1000_0.2", "/model", "merged_models"),
    # ("01HRGRBJ7MKH3WS1Z24HRZK6TJ", "dare_linear-llama_2_7b-tulu_all_0.9-science_1000_0.1", "/model", "merged_models"),

    # ("01HRGG79MRWMG4396QRDT5H09X", "dare_linear-llama_2_7b-tulu_all_0.1-science_2500_0.9", "/model", "merged_models"),
    # ("01HRGH97B43HYYCHW8APNYHZVH", "dare_linear-llama_2_7b-tulu_all_0.2-science_2500_0.8", "/model", "merged_models"),
    # ("01HRGJCEE9PB6KB61WNP4A0BS0", "dare_linear-llama_2_7b-tulu_all_0.3-science_2500_0.7", "/model", "merged_models"),
    # ("01HRGKE3MZSBHAFK2TF2D8E1X9", "dare_linear-llama_2_7b-tulu_all_0.4-science_2500_0.6", "/model", "merged_models"),
    # ("01HRGMEK6TYGWBYAXS2E327KE8", "dare_linear-llama_2_7b-tulu_all_0.5-science_2500_0.5", "/model", "merged_models"),
    # ("01HRGNFZMH8605NMQ5H89MGSVA", "dare_linear-llama_2_7b-tulu_all_0.6-science_2500_0.4", "/model", "merged_models"),
    # ("01HRGPH0HPS2M291P87R50TSM4", "dare_linear-llama_2_7b-tulu_all_0.7-science_2500_0.3", "/model", "merged_models"),
    # ("01HRGQM52YBS1PQ0Z7K2GRAPYE", "dare_linear-llama_2_7b-tulu_all_0.8-science_2500_0.2", "/model", "merged_models"),
    # ("01HRGRPCBWPKW91RYV18P05A32", "dare_linear-llama_2_7b-tulu_all_0.9-science_2500_0.1", "/model", "merged_models"),

    # ("01HRR4A53A56K8418G7ZSP684B", "dare_linear-llama_2_7b-tulu_all_0.1-science_upsample_0.9", "/model", "merged_models"),
    # ("01HRR5C5T7GN19SR9RKPS9KXZD", "dare_linear-llama_2_7b-tulu_all_0.2-science_upsample_0.8", "/model", "merged_models"),
    # ("01HRR6FAPMZSSZ9F19C8KYFYNH", "dare_linear-llama_2_7b-tulu_all_0.3-science_upsample_0.7", "/model", "merged_models"),
    # ("01HRR7GYTFZD5Q5S5A18XJTKW4", "dare_linear-llama_2_7b-tulu_all_0.4-science_upsample_0.6", "/model", "merged_models"),
    # ("01HRR8JYEX5KDR1M0GJ94B55WH", "dare_linear-llama_2_7b-tulu_all_0.5-science_upsample_0.5", "/model", "merged_models"),
    # ("01HRR9N27XMRK0PHF94MG11FMB", "dare_linear-llama_2_7b-tulu_all_0.6-science_upsample_0.4", "/model", "merged_models"),
    # ("01HRRAQ1EJSKXEDP3C5BPSWJ0Z", "dare_linear-llama_2_7b-tulu_all_0.7-science_upsample_0.3", "/model", "merged_models"),
    # ("01HRRBR8V06A7RS5M0W0BYMVBW", "dare_linear-llama_2_7b-tulu_all_0.8-science_upsample_0.2", "/model", "merged_models"),
    # ("01HRRCTJYBCA22JQ3HBFGWZME1", "dare_linear-llama_2_7b-tulu_all_0.9-science_upsample_0.1", "/model", "merged_models"),

    # ("01HR1DXMHWC68VF29PVBKS5XHF", "linear_weighted-llama_2_7b-tulu_all_0.1-science_100_0.9", "/model", "merged_models"),
    # ("01HR1FKFG7THD3W3NT7AN3CG7Z", "linear_weighted-llama_2_7b-tulu_all_0.2-science_100_0.8", "/model", "merged_models"),
    # ("01HR1H98XMHCMWY14PSV9GSEPT", "linear_weighted-llama_2_7b-tulu_all_0.3-science_100_0.7", "/model", "merged_models"),
    # ("01HR1JSMA5C4HT4RQ8N735SZDC", "linear_weighted-llama_2_7b-tulu_all_0.4-science_100_0.6", "/model", "merged_models"),
    # ("01HR1MECQQGG3099GYEMBS0C2V", "linear_weighted-llama_2_7b-tulu_all_0.5-science_100_0.5", "/model", "merged_models"),
    # ("01HR1P76BBM78HP3G90DHZ9BTM", "linear_weighted-llama_2_7b-tulu_all_0.6-science_100_0.4", "/model", "merged_models"),
    # ("01HR1R0ZN268ZKJ92S3DN2HFCD", "linear_weighted-llama_2_7b-tulu_all_0.7-science_100_0.3", "/model", "merged_models"),
    # ("01HR1SM405T0CFD93BXXKJZ801", "linear_weighted-llama_2_7b-tulu_all_0.8-science_100_0.2", "/model", "merged_models"),
    # ("01HR1VA7WPNB5917CGWMCFQVQC", "linear_weighted-llama_2_7b-tulu_all_0.9-science_100_0.1", "/model", "merged_models"),

    # ("01HR1E7WBFW1QFPD6R2D9VEXV1", "linear_weighted-llama_2_7b-tulu_all_0.1-science_200_0.9", "/model", "merged_models"),
    # ("01HR1FY3ZBY2P8FMTVE1M90YSR", "linear_weighted-llama_2_7b-tulu_all_0.2-science_200_0.8", "/model", "merged_models"),
    # ("01HR1HKNBZK07A94QS1ERNR3NC", "linear_weighted-llama_2_7b-tulu_all_0.3-science_200_0.7", "/model", "merged_models"),
    # ("01HR1K48WJPCE1EJ9GDXQRPXRQ", "linear_weighted-llama_2_7b-tulu_all_0.4-science_200_0.6", "/model", "merged_models"),
    # ("01HR1MS7CDTT3J2G35XSAGPK3Z", "linear_weighted-llama_2_7b-tulu_all_0.5-science_200_0.5", "/model", "merged_models"),
    # ("01HR1PKYWBASRSPQQ132VDXQ4S", "linear_weighted-llama_2_7b-tulu_all_0.6-science_200_0.4", "/model", "merged_models"),
    # ("01HR1RBJG2K6K7RPXZPY2ENDSS", "linear_weighted-llama_2_7b-tulu_all_0.7-science_200_0.3", "/model", "merged_models"),
    # ("01HR1SYX83HK4FVX09G8BQ4KMV", "linear_weighted-llama_2_7b-tulu_all_0.8-science_200_0.2", "/model", "merged_models"),
    # ("01HR1VME4FMFVAAWXZ93875DHN", "linear_weighted-llama_2_7b-tulu_all_0.9-science_200_0.1", "/model", "merged_models"),

    # ("01HR1EJRGNFZNJTM597J4PW70C", "linear_weighted-llama_2_7b-tulu_all_0.1-science_500_0.9", "/model", "merged_models"),
    # ("01HR1G8J9PANW5039P9BSWP0HM", "linear_weighted-llama_2_7b-tulu_all_0.2-science_500_0.8", "/model", "merged_models"),
    # ("01HR1HYJK2P8BVZP18F629QDGQ", "linear_weighted-llama_2_7b-tulu_all_0.3-science_500_0.7", "/model", "merged_models"),
    # ("01HR1KEV7ZM1Y3M95D01HW18WZ", "linear_weighted-llama_2_7b-tulu_all_0.4-science_500_0.6", "/model", "merged_models"),
    # ("01HR1N404F41QV9Q2RK7WQ4GGR", "linear_weighted-llama_2_7b-tulu_all_0.5-science_500_0.5", "/model", "merged_models"),
    # ("01HR1Q0WDERCYYNXFJ6TPDD12X", "linear_weighted-llama_2_7b-tulu_all_0.6-science_500_0.4", "/model", "merged_models"),
    # ("01HR1RNDNJM9SQ6W4QX2RJGM2J", "linear_weighted-llama_2_7b-tulu_all_0.7-science_500_0.3", "/model", "merged_models"),
    # ("01HR1T9E5JEEJDW3PJJA7HWKCC", "linear_weighted-llama_2_7b-tulu_all_0.8-science_500_0.2", "/model", "merged_models"),
    # ("01HR1VZ8Z02SM8Q7FAV9Y75T1N", "linear_weighted-llama_2_7b-tulu_all_0.9-science_500_0.1", "/model", "merged_models"),

    # ("01HR1EXQ4QYCFP2PWQMYKNGQ2Q", "linear_weighted-llama_2_7b-tulu_all_0.1-science_1000_0.9", "/model", "merged_models"),
    # ("01HR1GK1DSWQZ1GXMKR774MR94", "linear_weighted-llama_2_7b-tulu_all_0.2-science_1000_0.8", "/model", "merged_models"),
    # ("01HR1J93KY0X6NTSY9XSFWYNT3", "linear_weighted-llama_2_7b-tulu_all_0.3-science_1000_0.7", "/model", "merged_models"),
    # ("01HR1KSJSN04VXSGY81F42QAPF", "linear_weighted-llama_2_7b-tulu_all_0.4-science_1000_0.6", "/model", "merged_models"),
    # ("01HR1NF26VA6R88CBKJH2V9WTN", "linear_weighted-llama_2_7b-tulu_all_0.5-science_1000_0.5", "/model", "merged_models"),
    # ("01HR1QBJZQPDQAKY6NBG7MJPWK", "linear_weighted-llama_2_7b-tulu_all_0.6-science_1000_0.4", "/model", "merged_models"),
    # ("01HR1RZR4MM7EPX12YVZRX0J2B", "linear_weighted-llama_2_7b-tulu_all_0.7-science_1000_0.3", "/model", "merged_models"),
    # ("01HR1TMEXWQYSE2M6T4TFEY4NQ", "linear_weighted-llama_2_7b-tulu_all_0.8-science_1000_0.2", "/model", "merged_models"),
    # ("01HR1W9T40B11V0QYF2QR4WFJP", "linear_weighted-llama_2_7b-tulu_all_0.9-science_1000_0.1", "/model", "merged_models"),

    # ("01HR1F8DY7JBDWHA2YSSK0F3DJ", "linear_weighted-llama_2_7b-tulu_all_0.1-science_2500_0.9", "/model", "merged_models"),
    # ("01HR1GY6Z341Y7JP2GK43MS211", "linear_weighted-llama_2_7b-tulu_all_0.2-science_2500_0.8", "/model", "merged_models"),
    # ("01HR2WXSCK21YWN3TMZTHAFCYA", "linear_weighted-llama_2_7b-tulu_all_0.3-science_2500_0.7", "/model", "merged_models"),
    # ("01HR1M4AGE8SQN4PFWSJWE27A0", "linear_weighted-llama_2_7b-tulu_all_0.4-science_2500_0.6", "/model", "merged_models"),
    # ("01HR1NV0VTR3GT3C4H38M0NKG6", "linear_weighted-llama_2_7b-tulu_all_0.5-science_2500_0.5", "/model", "merged_models"),
    # ("01HR1QPAWMTXAW30EH5YGMSRGH", "linear_weighted-llama_2_7b-tulu_all_0.6-science_2500_0.4", "/model", "merged_models"),
    # ("01HR1S9VTDDA4RE2MHEX091HMR", "linear_weighted-llama_2_7b-tulu_all_0.7-science_2500_0.3", "/model", "merged_models"),
    # ("01HR1TZQ7M8DEW9J0QPZ1QE7NB", "linear_weighted-llama_2_7b-tulu_all_0.8-science_2500_0.2", "/model", "merged_models"),
    # ("01HR1WNVCYM2YKVQMMXK8SEBBC", "linear_weighted-llama_2_7b-tulu_all_0.9-science_2500_0.1", "/model", "merged_models"),

    # ("01HRR7F468FR9P4AKD5E61AF3V", "linear_weighted-llama_2_7b-tulu_all_0.1-science_upsample_0.9", "/model", "merged_models"),
    # ("01HRR7SD3F34MD46J7VZ890D4B", "linear_weighted-llama_2_7b-tulu_all_0.2-science_upsample_0.8", "/model", "merged_models"),
    # ("01HRR84EP1B3BSCNQ1X5FZ8E12", "linear_weighted-llama_2_7b-tulu_all_0.3-science_upsample_0.7", "/model", "merged_models"),
    # ("01HRR8F9ZPCQJ2EEDKCR04Q3ES", "linear_weighted-llama_2_7b-tulu_all_0.4-science_upsample_0.6", "/model", "merged_models"),
    # ("01HRR8SZSZJQJA77K2FD1NHVBB", "linear_weighted-llama_2_7b-tulu_all_0.5-science_upsample_0.5", "/model", "merged_models"),
    # ("01HRR96DV8926P0TPWWMDT6WPZ", "linear_weighted-llama_2_7b-tulu_all_0.6-science_upsample_0.4", "/model", "merged_models"),
    # ("01HRR9GZ1Z3KAQ3S2NGS5DP6VP", "linear_weighted-llama_2_7b-tulu_all_0.7-science_upsample_0.3", "/model", "merged_models"),
    # ("01HRR9WXB4WBEWA052JF3J22N4", "linear_weighted-llama_2_7b-tulu_all_0.8-science_upsample_0.2", "/model", "merged_models"),
    # ("01HRRA77V6F1MNNER2KQ9C3GH3", "linear_weighted-llama_2_7b-tulu_all_0.9-science_upsample_0.1", "/model", "merged_models"),
]

# model to evaluate, each in the followng format: model name, their beaker id, checkpoint subfolder
models = [    
    # llama2 models
    ("llama2-7B", "01HCJYBBWA629B8GJTHPT496TT", None, "vanilla_lm"),
    # ("llama2-13B", "01HCJZQBM2KGQZSZRPF4HKVBZX", None, "vanilla_lm"),
    # ("llama2-70B", "01HCK281AFAXV2Y7T54NMNSC55", None, "vanilla_lm"),
    # ("llama2-chat-7B", "01HCT5D48MSRF0PCNAWNSJDN54", None, "tuned_lm"),
    # ("llama2-chat-13B", "01HCT5Q7A6FE8RZKY8TYN64ZW2", None, "tuned_lm"),
    # ("llama2-chat-70B", "01HCT63DVK7YPT6P9SN35XH417", None, "tuned_lm"),
]

#--------------- experiments about number of supervision tasks -------------------------

if not merge_models:
    # for experiment_group, model_info in itertools.product(experiment_groups, models):
    # for dataset, model_info, experiment_group in itertools.product(pairwise_trained_datasets, models, experiment_groups):
    for (beaker, model_name, model_path, save_subdir), model_info, experiment_group in itertools.product(datasets, models, experiment_groups):
        if beaker == "" or len(beaker) == 0:
            beaker = "jacobm/" + model_name
        # if dataset != "/model":
        #     model_path = f'/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/checkpoints/domain_addition/{dataset}/'
        # else:
        #     model_path = "/model"
        # print(f"Submitting {experiment_group} for model: {dataset}")
        d = copy.deepcopy(d1)

        # model_name = model_info[0] + f"_{model_info[2]}" if model_info[2] is not None else model_info[0]
        # if lora:
            # name = f"open_instruct_eval_{experiment_group}_{model_name}_{dataset}_{today}".replace('/', '-')
        # else:
        name = f"oi_eval_{experiment_group}_{model_name}"
        # shorter_name = name.replace('llama_2_7b-', '')

        # if dataset == "/model":
        d['tasks'][0]['datasets'][1]['source']['beaker'] = beaker
        d['description'] = name
        d['tasks'][0]['name'] = name

        # if "dave" in dataset or "another" in dataset:
        #     save_dir = f"/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/{dataset}/{experiment_group}/"
        # else:
        #     save_dir = f"/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/{dataset.replace('/', '-')}/{experiment_group}/"
        # save_dir = f"/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/results/domain_addition/with_daves_tulu_model/daves_tulu_model/{experiment_group}/"
        # save_dir = "/output/"
        save_dir = f"/net/nfs.cirrascale/allennlp/jacobm/modular_adaptation/new_baselines_results/{save_subdir}/{model_name}/{experiment_group}/"

        if experiment_group == "mmlu_0shot":
            d['tasks'][0]['arguments'][0] = f'''
                python -m eval.mmlu.run_eval \
                --ntrain 0 \
                --data_dir /data/mmlu/ \
                --save_dir {save_dir} \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path} \
                --eval_batch_size 4 \
                --load_in_8bit \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
            '''
        elif experiment_group == "mmlu_5shot":
            d['tasks'][0]['arguments'][0] = f'''
                python -m eval.mmlu.run_eval \
                --ntrain 5 \
                --data_dir /data/mmlu/ \
                --save_dir {save_dir} \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path} \
                --eval_batch_size 4 \
                --load_in_8bit \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
            '''
        elif experiment_group == "bbh_direct":
            d['tasks'][0]['arguments'][0] = f'''
                python -m eval.bbh.run_eval \
                --data_dir /data/bbh \
                --save_dir {save_dir} \
                --use_vllm \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path} \
                --max_num_examples_per_task 40 \
                --no_cot \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
            '''
        elif experiment_group == "bbh_cot":
            d['tasks'][0]['arguments'][0] = f'''
                python -m eval.bbh.run_eval \
                --data_dir /data/bbh \
                --save_dir {save_dir} \
                --use_vllm \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path} \
                --max_num_examples_per_task 40 \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
            '''
        elif experiment_group == "gsm_direct":
            d['tasks'][0]['arguments'][0] = f'''
                python -m eval.gsm.run_eval \
                --data_dir /data/gsm/ \
                --max_num_examples 200 \
                --save_dir {save_dir} \
                --use_vllm \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path} \
                --n_shot 8 \
                --no_cot \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
            '''
        elif experiment_group == "gsm_cot":
            d['tasks'][0]['arguments'][0] = f'''
                python -m eval.gsm.run_eval \
                --data_dir /data/gsm/ \
                --max_num_examples 200 \
                --save_dir {save_dir} \
                --use_vllm \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path} \
                --n_shot 8 \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
            ''' 
        elif experiment_group == "tydiqa_goldp_1shot":
            d["tasks"][0]["arguments"][0] = f'''
                python -m eval.tydiqa.run_eval \
                --data_dir /data/tydiqa/ \
                --n_shot 1 \
                --max_num_examples_per_lang 100 \
                --max_context_length 512 \
                --save_dir {save_dir} \
                --use_vllm \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path} \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
            '''
        elif experiment_group == "tydiqa_no_context_1shot":
            d["tasks"][0]["arguments"][0] = f'''
                python -m eval.tydiqa.run_eval \
                --data_dir /data/tydiqa/ \
                --no_context \
                --n_shot 1 \
                --max_num_examples_per_lang 100 \
                --max_context_length 512 \
                --save_dir {save_dir} \
                --use_vllm \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path} \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
            '''
        elif experiment_group == "codex_eval_temp_0.1":
            d['tasks'][0]['arguments'][0] = f'''
                python -m eval.codex_humaneval.run_eval \
                --data_file /data/codex_humaneval/HumanEval.jsonl.gz \
                --eval_pass_at_ks 1 5 10 20 \
                --unbiased_sampling_size_n 20 \
                --temperature 0.1 \
                --save_dir {save_dir} \
                --use_vllm \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path}
            '''
        elif experiment_group == "codex_eval_temp_0.8":
            d['tasks'][0]['arguments'][0] = f'''
                python -m eval.codex_humaneval.run_eval \
                --data_file /data/codex_humaneval/HumanEval.jsonl.gz \
                --eval_pass_at_ks 1 5 10 20 \
                --unbiased_sampling_size_n 20 \
                --temperature 0.8 \
                --save_dir {save_dir} \
                --use_vllm \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path}
            '''
        elif experiment_group == "truthfulqa":
            d['tasks'][0]['arguments'][0] = f'''
            python -m eval.truthfulqa.run_eval \
                --data_dir /data/truthfulqa \
                --save_dir {save_dir} \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path} \
                --metrics truth info mc \
                --preset qa \
                --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
                --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
                --eval_batch_size 20 \
                --load_in_8bit \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
            '''
        elif experiment_group == "toxigen":
            d['tasks'][0]['arguments'][0] = f'''
            python -m eval.toxigen.run_eval \
                --data_dir /data/toxigen/ \
                --save_dir {save_dir} \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path} \
                --eval_batch_size 32 \
                --use_vllm \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
            '''
        elif experiment_group == "alpaca_farm":
            d['tasks'][0]['arguments'][0] = f'''
            python -m eval.alpaca_farm.run_eval \
                --use_vllm \
                --model_name_or_path {model_path} \
                --tokenizer_name_or_path {model_path} \
                --save_dir {save_dir} \
                --use_chat_format \
                --chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format
            '''
        else:
            raise ValueError("experiment_group not supported")
        
        # TODO: fix if I use lora ever
        if lora:
            d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].strip() + f' --lora_weight_path /net/nfs.cirrascale/allennlp/jacobm/tulu_7B_lora_exp/pairwise_experts/{dataset}/'

        # if a specific checkpoint is specified, load model from that checkpoint
        if model_info[2] is not None:
            assert "--model_name_or_path /model" in d['tasks'][0]['arguments'][0]
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--model_name_or_path /model", "--model_name_or_path /model/"+model_info[2])]
            assert "--tokenizer_name_or_path /model" in d['tasks'][0]['arguments'][0]
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--tokenizer_name_or_path /model", "--tokenizer_name_or_path /model/"+model_info[2])]

        # for vanilla_lm, remove the chat formatting function
        if model_info[3] == "vanilla_lm" and not lora:
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--use_chat_format", "")]

        if "13B" in model_info[0]:
            # find the batch size argument, and reduce by 4x
            if "--eval_batch_size" in d['tasks'][0]['arguments'][0]:
                original_batch_size = re.search("--eval_batch_size (\d+)", d['tasks'][0]['arguments'][0]).group(1)
                new_batch_size = max(1, int(original_batch_size) // 2)
                d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--eval_batch_size {}".format(original_batch_size), "--eval_batch_size {}".format(new_batch_size))]


        if "30B" in model_info[0] or "34B" in model_info[0]:
            # find the batch size argument, and reduce by 4x
            if "--eval_batch_size" in d['tasks'][0]['arguments'][0]:
                original_batch_size = re.search("--eval_batch_size (\d+)", d['tasks'][0]['arguments'][0]).group(1)
                new_batch_size = max(1, int(original_batch_size) // 4)
                d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--eval_batch_size {}".format(original_batch_size), "--eval_batch_size {}".format(new_batch_size))]

            if "codex_eval" in experiment_group:
                # request 2x more GPUs
                d['tasks'][0]['resources']['gpuCount'] = 2 * d['tasks'][0]['resources']['gpuCount']
        
        elif "70B" in model_info[0] or "65B" in model_info[0] or "40B" in model_info[0]:
            # find the batch size argument, and reduce by 4x
            if "--eval_batch_size" in d['tasks'][0]['arguments'][0]:
                original_batch_size = re.search("--eval_batch_size (\d+)", d['tasks'][0]['arguments'][0]).group(1)
                new_batch_size = max(1, int(original_batch_size) // 4)
                d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--eval_batch_size {}".format(original_batch_size), "--eval_batch_size {}".format(new_batch_size))]

            if "codex_eval" in experiment_group:
                # request 4x more GPUs
                d['tasks'][0]['resources']['gpuCount'] = 4 * d['tasks'][0]['resources']['gpuCount']
            else:
                # request 2x more GPUs
                d['tasks'][0]['resources']['gpuCount'] = 2 * d['tasks'][0]['resources']['gpuCount']

        if model_info[0].startswith("hf-"):  # if it's a huggingface model, load it from the model hub
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--use_chat_format", "")]
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--model_name_or_path /model", "--model_name_or_path "+model_info[1])]
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--tokenizer_name_or_path /model", "--model_name_or_path "+model_info[1])]
        # else:  # if it's a beaker model, mount the beaker dataset to `/model`
            # d['tasks'][0]['datasets'][1]['source']['beaker'] = 01HKG46RNVAP3NSHNDH019R5KB # model_info[1]

        if "llama2-chat" in model_info[0]:
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace(
                "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format", 
                "--chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format")
            ]
        elif "code_llama_instruct" in model_info[0]:
            d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace(
                "--chat_formatting_function eval.templates.create_prompt_with_tulu_chat_format", 
                "--chat_formatting_function eval.templates.create_prompt_with_llama2_chat_format")
            ] 

        # TODO: vllm doesn't support lora yet
        if any([x in model_info[0] for x in ["opt", "pythia", "falcon"]]) or lora:
            if "--use_vllm" in d['tasks'][0]['arguments'][0]:
                print(f"Removing --use_vllm for {model_info[0]}")
                d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--use_vllm", "")] 

        # print(d)

        fn = "beaker_configs/auto_created/{}.yaml".format(name)
        file = open(fn, "w")
        yaml.dump(d, file, default_flow_style=True)
        file.close()

        cmd = "beaker experiment create {} --workspace ai2/modular-adaptation-science".format(fn)
        subprocess.Popen(cmd, shell=True)

# merge and eval models
else:
    # for experiment_group, model_info in itertools.product(experiment_groups, models):
    for i in range(1, len(datasets) - 1):
        for j in range(i + 1, len(datasets)):
            target_dataset_1 = datasets[i]
            target_dataset_2 = datasets[j]
            for model_info, experiment_group in itertools.product(models, experiment_groups):
                print(f"Submitting {experiment_group} for model: {model_info[0]}")
                print(f'merging datasets {target_dataset_1} and {target_dataset_2}')
                d = copy.deepcopy(d1)

                model_name = model_info[0] + f"_{model_info[2]}" if model_info[2] is not None else model_info[0]
                name = f"open_instruct_eval_{experiment_group}_merge_{target_dataset_1}_and_{target_dataset_2}_{today}"
                d['description'] = name
                d['tasks'][0]['name'] = name

                d['tasks'][0]['arguments'][0] = f'''
                python -u -m eval.merge_models     \
                    --base_model /model \
                    --target_lora_modules {target_dataset_1}  {target_dataset_2} \
                    --results_dir /output/ \
                    --task {experiment_group}
                '''        
                # if lora:
                    # d['tasks'][0]['arguments'][0] = d['tasks'][0]['arguments'][0].strip() + f' --lora_weight_path /net/nfs.cirrascale/allennlp/jacobm/tulu_7B_lora_exp/{dataset}/'

                if "13B" in model_info[0]:
                    # find the batch size argument, and reduce by 4x
                    if "--eval_batch_size" in d['tasks'][0]['arguments'][0]:
                        original_batch_size = re.search("--eval_batch_size (\d+)", d['tasks'][0]['arguments'][0]).group(1)
                        new_batch_size = max(1, int(original_batch_size) // 2)
                        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--eval_batch_size {}".format(original_batch_size), "--eval_batch_size {}".format(new_batch_size))]


                if "30B" in model_info[0] or "34B" in model_info[0]:
                    # find the batch size argument, and reduce by 4x
                    if "--eval_batch_size" in d['tasks'][0]['arguments'][0]:
                        original_batch_size = re.search("--eval_batch_size (\d+)", d['tasks'][0]['arguments'][0]).group(1)
                        new_batch_size = max(1, int(original_batch_size) // 4)
                        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--eval_batch_size {}".format(original_batch_size), "--eval_batch_size {}".format(new_batch_size))]

                    if "codex_eval" in experiment_group:
                        # request 2x more GPUs
                        d['tasks'][0]['resources']['gpuCount'] = 2 * d['tasks'][0]['resources']['gpuCount']
                
                elif "70B" in model_info[0] or "65B" in model_info[0] or "40B" in model_info[0]:
                    # find the batch size argument, and reduce by 4x
                    if "--eval_batch_size" in d['tasks'][0]['arguments'][0]:
                        original_batch_size = re.search("--eval_batch_size (\d+)", d['tasks'][0]['arguments'][0]).group(1)
                        new_batch_size = max(1, int(original_batch_size) // 4)
                        d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--eval_batch_size {}".format(original_batch_size), "--eval_batch_size {}".format(new_batch_size))]

                    if "codex_eval" in experiment_group:
                        # request 4x more GPUs
                        d['tasks'][0]['resources']['gpuCount'] = 4 * d['tasks'][0]['resources']['gpuCount']
                    else:
                        # request 2x more GPUs
                        d['tasks'][0]['resources']['gpuCount'] = 2 * d['tasks'][0]['resources']['gpuCount']

                if model_info[0].startswith("hf-"):  # if it's a huggingface model, load it from the model hub
                    d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--use_chat_format", "")]
                    d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--model_name_or_path /model", "--model_name_or_path "+model_info[1])]
                    d['tasks'][0]['arguments'] = [d['tasks'][0]['arguments'][0].replace("--tokenizer_name_or_path /model", "--model_name_or_path "+model_info[1])]
                else:  # if it's a beaker model, mount the beaker dataset to `/model`
                    d['tasks'][0]['datasets'][1]['source']['beaker'] = model_info[1]

                fn = "beaker_configs/auto_created/{}.yaml".format(name)
                file = open(fn, "w")
                yaml.dump(d, file, default_flow_style=True)
                file.close()

                cmd = "beaker experiment create {} --workspace ai2/lora-instruct".format(fn)
                subprocess.Popen(cmd, shell=True)
