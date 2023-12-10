# About
This repository aims to test one of open source models for question answering. The task is simple, I want to get an information — from a relatively unstructured data — about someone whether they have a PhD/doctorate degree or not. And if so, the return value should be the university name. `None` value indicates two things:
1. the person doesn't have a PhD/doctorate degree;
2. the person has a PhD/doctorate degree, but the information where they obtained it is not available.

It may be easier to parse if the document is structured like a curriculum vitae. Otherwise, it may be laborious especially if someone has to do it manually, instead of programmatically.

Enjoy!

# Model(s) used
I have tried several models available on Huggingface, but [tinyroberta-squad2](https://huggingface.co/deepset/tinyroberta-squad2) works well for my use case.

# How to
Below are some expected outputs when you run the models:

```python
>>> from transformers import pipeline
2023-12-10 21:16:05.907725: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
>>> question = "Where did the person receive a doctorate/PhD degree?"
>>> context = """ Claudia Goldin is the Henry Lee Professor of Economics at Harvard University and was the director of the NBER’s Development of the American Economy program from 1989 to 2017. She is a co-director of the NBER's Gender in the Economy group. 
... 
... An economic historian and a labor economist, Goldin's research covers a wide range of topics, including the female labor force, the gender gap in earnings, income inequality, technological change, education, and immigration. Most of her research interprets the present through the lens of the past and explores the origins of current issues of concern. Her most recent book is Career & Family: Women's Century-Long Journey toward Equity (Princeton University Press, 2021).
... 
... She is the author and editor of several books, among them Understanding the Gender Gap: An Economic History of American Women (Oxford 1990), The Regulated Economy: A Historical Approach to Political Economy (with G. Libecap; University of Chicago Press 1994), The Defining Moment: The Great Depression and the American Economy in the Twentieth Century (with M. Bordo and E. White; University of Chicago Press 1998), Corruption and Reform: Lesson’s from America’s Economic History (with E. Glaeser; Chicago 2006), and Women Working Longer: Increased Employment at Older Ages (with L. Katz; Chicago 2018). Her book The Race between Education and Technology (with L. Katz; Belknap Press, 2008, 2010) was the winner of the 2008 R.R. Hawkins Award for the most outstanding scholarly work in all disciplines of the arts and sciences.
... 
... Goldin is best known for her historical work on women in the U.S. economy. Her most influential papers in that area have concerned the history of women’s quest for career and family, coeducation in higher education, the impact of the “Pill” on women’s career and marriage decisions, women’s surnames after marriage as a social indicator, the reasons why women are now the majority of undergraduates, and the new lifecycle of women’s employment. 
... 
... Goldin was the president of the American Economic Association in 2013 and was president of the Economic History Association in 1999/2000. She is a member of the National Academy of Sciences and the American Philosophical Society and a fellow of the American Academy of Political and Social Science, the American Academy of Arts and Sciences, the Society of Labor Economists (SOLE), the Econometric Society, and the Cliometric Society. She received the IZA Prize in Labor Economics in 2016 and in 2009 SOLE awarded Goldin the Mincer Prize for life-time contributions to the field of labor economics. She received the 2019 BBVA Frontiers in Knowledge award and the 2020 Nemmers award, both in economics. From 1984 to 1988 she was editor of the Journal of Economic History.  She is the recipient of several teaching awards. Goldin received her B.A. from Cornell University and her Ph.D. from the University of Chicago."""
>>> QA_input = {"question": question, "context": context}
>>> model_name = "deepset/tinyroberta-squad2"
>>> nlp = pipeline("question-answering", model=model_name, tokenizer=model_name)
Some weights of the PyTorch model were not used when initializing the TF 2.0 model TFRobertaForQuestionAnswering: ['roberta.embeddings.position_ids']
- This IS expected if you are initializing TFRobertaForQuestionAnswering from a PyTorch model trained on another task or with another architecture (e.g. initializing a TFBertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing TFRobertaForQuestionAnswering from a PyTorch model that you expect to be exactly identical (e.g. initializing a TFBertForSequenceClassification model from a BertForSequenceClassification model).
All the weights of TFRobertaForQuestionAnswering were initialized from the PyTorch model.
If your task is similar to the task the model of the checkpoint was trained on, you can already use TFRobertaForQuestionAnswering for predictions without further training.
>>> res = nlp(QA_input)["answer"].upper()
>>> res
'UNIVERSITY OF CHICAGO'
```