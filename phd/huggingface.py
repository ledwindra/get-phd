import pandas as pd
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
    # CamemBERT
    # CANINE
    # ConvBERT
    # Data2VecText
    # DeBERTa
    # DeBERTa-v2
    # DistilBERT
    # ELECTRA
    # ERNIE
    # ErnieM
    # Falcon
    # FlauBERT
    # FNet
    # Funnel Transformer
    # OpenAI GPT-2
    # GPT Neo
    # GPT NeoX
    # GPT-J
    # I-BERT
    # LayoutLMv2
    # LayoutLMv3
    # LED,
    # LiLT
    # Longformer
    # LUKE
    # LXMERT
    # MarkupLM
    # mBART
    # MEGA
    # Megatron-BERT
    # MobileBERT
    # MPNet
    # MPT
    # MRA
    # MT5
    # MVP
    # Nezha
    # Nyströmformer
    # OPT
    # QDQBert
    # Reformer
    # RemBERT
    # RoBERTa
    # RoBERTa-PreLayerNorm
    # RoCBert
    # RoFormer
    # Splinter
    # SqueezeBERT
    # T5
    # UMT5
    # XLM
    # XLM-RoBERTa
    # XLM-RoBERTa-X
    # XLNet
    # X-MOD
    # YOSO
]

def qna(model_name, question):
    """Model name: vumichien/albert-base-v2-squad2
    Question 1: Is there any information on where the person attended their Ph.D. program? Answer with the following format: YES/NO
    Question 2: If there is not any information on where the person attended their Ph.D. program, return NONE. Otherwise, where did the person receive their Ph.D. program? Answer with the following format: UNIVERSITY_NAME
    """
    df = pd.read_csv("../data/faculty-raw.csv", sep="|")
    df["has_phd_info"] = ""
    # df["phd_where"] = ""
    question_answerer = pipeline("question-answering", model=model_name)
    for index, row in df.iterrows():
        response = question_answerer(question=question, context=row["bio"])
        df.loc[index, "has_phd_info"] = response["answer"]
        # df.loc[index, "phd_where"] = func(response, 1)
    df.drop("bio", axis=1, inplace=True)
    df = df.apply(lambda x: x.str.upper())
    output_file = model_name.replace("/", "-")
    
    return df.to_csv(f"../data/{output_file}.csv", index=False, sep="|")
