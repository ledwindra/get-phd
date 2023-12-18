# About
This repository aims to test question answering models. The tasks are simple, I want to get an information — from a relatively unstructured data — regarding someone's doctoral education background. Specifically, the questions address two things:
1. whether the text has any information on whether the person attended a Ph.D. program or not. The expected output is either `YES` or `NO`;
2. the university name where the person attended the Ph.D. program. The expected output is either `UNIVERSITY_NAME` (if the answer to the second question ys `YES`), or `NONE` (if the answer to the first question is `NO`).

My motivation here is to evaluate which model that's suitable for this kind of seemingly simple problem. I use both open source and closed source models (see details below). Doing this task programmatically without an LLM model may seem difficult to operate.

I hope you enjoy this repository!

# Model(s) used
Note that for every model, I don't do any fine-tuning. In addition, every parameter is set to default.

## Google Gemini API
Refer to its documentation [page](https://ai.google.dev/).

## Huggingface
Refer to its question answering [page](https://huggingface.co/docs/transformers/tasks/question_answering).

# Result
Here's a table that evaluates how compliant (in %) the model to answer in accordance to the format that I require:

|Model|Question 1|Question 2|
|-|-|-|
|ALBERT||||
|BART||||
|BERT||||
|BigBird||||
|BigBird-Pegassus||||
|BLOOM||||
|CamemBERT||||
|CANINE||||
|ConvBERT||||
|Data2VecText||||
|DeBERTa||||
|DeBERTa-v2||||
|DistilBERT||||
|ELECTRA||||
|ERNIE||||
|ErnieM||||
|Falcon||||
|FlauBERT||||
|FNet||||
|Funnel Transformer||||
|OpenAI GPT-2||||
|Gemini Pro|100%|100%||
|GPT Neo||||
|GPT NeoX||||
|GPT-J||||
|I-BERT||||
|LayoutLMv2||||
|LayoutLMv3||||
|LED||||
|LiLT||||
|Longformer||||
|LUKE||||
|LXMERT||||
|MarkupLM||||
|mBART||||
|MEGA||||
|Megatron-BERT||||
|MobileBERT||||
|MPNet||||
|MPT||||
|MRA||||
|MT5||||
|MVP||||
|Nezha||||
|Nyströmformer||||
|OPT||||
|QDQBert||||
|Reformer||||
|RemBERT||||
|RoBERTa||||
|RoBERTa-PreLayerNorm||||
|RoCBert||||
|RoFormer||||
|Splinter||||
|SqueezeBERT||||
|T5||||
|UMT5||||
|XLM||||
|XLM-RoBERTa||||
|XLM-RoBERTa-X||||
|XLNet||||
|X-MOD||||
|YOSO||||

Here's how I check the format compliance:

```python
df = pd.read_csv("choose-dataset.csv", sep="|)

# question 1
# should be only YES or NO
print(df.has_phd_info.unique())

# question 2
# should contain university name only and nothing else
print(df.phd_where.unique()) 
```