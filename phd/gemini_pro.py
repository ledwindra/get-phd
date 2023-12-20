import google.generativeai as genai
import pandas as pd


def api_key():
    api_key = input("Please input your API key: ")
    
    return genai.configure(api_key=api_key)


def generation_config():
    # Set up the model
    gc = {
      "temperature": 0.9,
      "top_p": 1,
      "top_k": 1,
      "max_output_tokens": 2048}
    
    return gc


def safety_settings():
    ss = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE"   
        }
    ]

    return ss


def model(generation_config, safety_settings, context, question):
    model = genai.GenerativeModel(model_name="gemini-pro", generation_config=generation_config,safety_settings=safety_settings)
    prompt_parts = [f"context={context}\n\nquestion={question}"]
    response = model.generate_content(prompt_parts)
    
    return response


def qna(number):
    if number == 1:
        question = "Is there any information on where the person attended their Ph.D. program? Answer with the following format: YES/NO"
    else:
        question = "Where did the person receive their Ph.D. program? If there is not any information, return NONE. Otherwise, answer with the following format: UNIVERSITY_NAME"

    return question


def dataset():
    gc = generation_config()
    ss = safety_settings()
    df = pd.read_csv("../data/faculty-raw.csv", sep="|")
    df["model_name"] = "GEMINI PRO"
    df["has_phd_info"] = ""
    df["phd_where"] = ""
    for index, row in df.iterrows():
        response1 = model(gc, ss, row["bio"], qna(1)).text
        response2 = model(gc, ss, row["bio"], qna(2)).text
        df.loc[index, "has_phd_info"] = response1
        df.loc[index, "phd_where"] = response2
        print(df.loc[index].to_dict())
    df.drop("bio", axis=1, inplace=True)
    df = df.apply(lambda x: x.str.upper())

    return df.to_csv("../data/gemini-pro.csv", index=False, sep="|")
