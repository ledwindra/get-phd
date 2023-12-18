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


def model(generation_config, safety_settings, context):
    model = genai.GenerativeModel(model_name="gemini-pro", generation_config=generation_config,safety_settings=safety_settings)
    question1 = "Is there any information on where the person attended their Ph.D. program? Answer with the following format: question1=YES/NO"
    question2 = "If the answer is NO, return NONE. If the answer is YES, where did the person receive their Ph.D. program? Answer with the following format: question2=UNIVERSITY_NAME"
    prompt_parts = [f"context={context}\n\nquestion1={question1}\n\nquestion2={question2}"]
    response = model.generate_content(prompt_parts)
    
    return response


def dataset():
    gc = generation_config()
    ss = safety_settings()
    func = lambda response, x: response[x].split("=")[1].replace("\"", "")
    df = pd.read_csv("../data/faculty-raw.csv", sep="|")
    df["model_name"] = "GEMINI PRO"
    df["has_phd_info"] = ""
    df["phd_where"] = ""
    for index, row in df.iterrows():
        response = model(gc, ss, row["bio"])
        response = response.text
        response = response.split("\n")
        df.loc[index, "has_phd_info"] = func(response, 0)
        df.loc[index, "phd_where"] = func(response, 1)
        print(df.loc[index].to_dict())
    df.drop("bio", axis=1, inplace=True)
    df = df.apply(lambda x: x.str.upper())

    return df.to_csv("../data/gemini-pro.csv", index=False, sep="|")
