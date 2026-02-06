# Base functions/modules
import os
from pathlib import Path
from langchain_groq import ChatGroq

# Bespoke functions/modules:
from load_keys import *


def load_llm(model_type:str = 'openai/gpt-oss-120b',api_key_path:str = "keys/groq.json"):
    
    p = Path(api_key_path)

    groq_api_key = load_groq_key(p)
    os.environ['GROQ_API_KEY'] = groq_api_key
    
    model = ChatGroq(
            model=model_type,
            temperature=0.2,
            max_retries=2,
        )
    
    return model

def give_message(model,message):

    full_answer = model.invoke(message)
    extracted_answer = full_answer.content

    return extracted_answer