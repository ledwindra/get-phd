import pandas as pd
from tqdm import tqdm
from transformers import pipeline

models = [
    # ALBERT
    "vumichien/albert-base-v2-squad2",
    # BART
    "valhalla/bart-large-finetuned-squadv1",
    # BERT
    "deepset/bert-base-cased-squad2",
    # BigBird
    "google/bigbird-roberta-base",
    # BigBird-Pegassus
    "google/bigbird-pegasus-large-arxiv",
    # BLOOM
    "bigscience/bloom", # not tested yet because of the size is too large
    # CamemBERT
    "ydshieh/roberta-base-squad2",
    # CANINE
    "Splend1dchan/canine-c-squad",
    # ConvBERT
    "YituTech/conv-bert-base",
    # Data2VecText -- not found on https://huggingface.co/docs/transformers/main/en/model_doc/data2vec-text
    # DeBERTa
    "Palak/microsoft_deberta-large_squad",
    # DeBERTa-v2
    "kamalkraj/deberta-v2-xlarge",
    # DistilBERT
    "distilbert-base-uncased",
    # ELECTRA
    "bhadresh-savani/electra-base-squad2",
    # ERNIE -- not found on https://huggingface.co/docs/transformers/model_doc/ernie#transformers.ErnieForQuestionAnswering
    # ErnieM
    "susnato/ernie-m-base_pytorch",
    # Falcon -- not found on https://huggingface.co/docs/transformers/model_doc/falcon#transformers.FalconForQuestionAnswering
    # FlauBERT
    "flaubert/flaubert_base_cased",
    # FNet
    "google/fnet-base",
    # Funnel Transformer
    "funnel-transformer/small",
    # OpenAI GPT-2
    "gpt2",
    # GPT Neo
    "EleutherAI/gpt-neo-1.3B",
    # GPT NeoX
    "trl-internal-testing/tiny-random-GPTNeoXForCausalLM",
    # GPT-J
    "hf-internal-testing/tiny-random-gptj",
    # I-BERT
    "kssteven/ibert-roberta-base",
    # LayoutLMv2
    "microsoft/layoutlmv2-base-uncased",
    # LayoutLMv3
    "microsoft/layoutlmv3-base",
    # LED
    "allenai/led-base-16384",
    # LiLT
    "SCUT-DLVCLab/lilt-roberta-en-base",
    # Longformer
    "allenai/longformer-large-4096-finetuned-triviaqa",
    # LUKE
    "studio-ousia/luke-base",
    # LXMERT
    "unc-nlp/lxmert-base-uncased",
    # MarkupLM
    "microsoft/markuplm-base-finetuned-websrc",
    # mBART
    "facebook/mbart-large-cc25",
    # MEGA
    "mnaylor/mega-base-wikitext",
    # Megatron-BERT
    "nvidia/megatron-bert-cased-345m",
    # MobileBERT
    "csarron/mobilebert-uncased-squad-v2"
    # MPNet
    "microsoft/mpnet-base",
    # MPT -- not found on https://huggingface.co/docs/transformers/model_doc/mpt#transformers.MptForQuestionAnswering
    # MRA
    "uw-madison/mra-base-512-4",
    # MT5
    "google/mt5-small",
    # MVP
    "RUCAIBox/mvp",
    # Nezha
    "sijunhe/nezha-cn-base",
    # Nyströmformer
    "uw-madison/nystromformer-512",
    # OPT
    "facebook/opt-350m",
    # QDQBert
    "bert-base-uncased",
    # Reformer
    "google/reformer-crime-and-punishment",
    # RemBERT
    "google/rembert",
    # RoBERTa
    "deepset/roberta-base-squad2",
    # RoBERTa-PreLayerNorm
    "andreasmadsen/efficient_mlm_m0.40",
    # RoCBert
    "ArthurZ/dummy-rocbert-qa",
    # RoFormer
    "junnyu/roformer_chinese_base",
    # Splinter
    "tau/splinter-base",
    # SqueezeBERT
    "squeezebert/squeezebert-uncased",
    # T5
    "t5-small",
    # UMT5 -- not found on https://huggingface.co/docs/transformers/model_doc/umt5
    # XLM
    "xlm-mlm-en-2048"
    # XLM-RoBERTa
    "deepset/roberta-base-squad2",
    # XLM-RoBERTa-X
    "xlm-roberta-xlarge",
    # XLNet
    "xlnet-base-cased",
    # X-MOD -- not found on https://huggingface.co/docs/transformers/model_doc/xmod#transformers.XmodForQuestionAnswering
    # YOSO
    "uw-madison/yoso-4096"
]

def qna(number):
    if number == 1:
        question = "Is there any information on where the person attended their Ph.D. program? Answer with the following format: YES/NO"
    else:
        question = "Where did the person receive their Ph.D. program? If there is not any information, return NONE. Otherwise, answer with the following format: UNIVERSITY_NAME"

    return question


def dataset(model_name):
    df = pd.read_csv("../data/faculty-raw.csv", sep="|")
    df["has_phd_info"] = ""
    df["phd_where"] = ""
    question_answerer = pipeline("question-answering", model=model_name)
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Progress"):
        try:
            response1 = question_answerer(question=qna(1), context=row["bio"])
            response2 = question_answerer(question=qna(2), context=row["bio"])
            df.loc[index, "has_phd_info"] = response1["answer"]
            df.loc[index, "phd_where"] = response2["answer"]
        except ValueError:
            df.loc[index, "has_phd_info"] = "VALUE ERROR"
            df.loc[index, "phd_where"] = "VALUE ERROR"
    df.drop("bio", axis=1, inplace=True)
    df = df.apply(lambda x: x.str.upper())
    output_file = model_name.replace("/", "-")
    
    return df.to_csv(f"../data/{output_file}.csv", index=False, sep="|")
