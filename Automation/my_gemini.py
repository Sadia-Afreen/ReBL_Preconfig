# import openai
import datetime
import math
import time 
import json
import tiktoken
from utils import *
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types


# Replace your key here 
load_dotenv()
# openai.organization = os.getenv('OPENAI_ORGANIZE')
# openai.api_key = os.getenv('OPENAI_API_KEY')
client = genai.Client(os.getenv('GOOGLE_GEMINI_API_KEY'))

def generate_text(prompt, history, package_name=None, model="gemini-2.0-flash", max_tokens=128000, attempts = 3):

    gemini_history = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        gemini_history.append(types.Content(
            role=role,
            parts=[types.Part.from_text(text=msg["content"])]
        ))
    
    config = types.GenerateContentConfig(
        temperature=0.3,
        candidate_count=1,
        stop_sequences=None
    )

    for times in range(attempts):
        try:
            response = client.models.generate_content(
                model=model,
                contents=gemini_history + [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
                config=config
            )
            
            history.append({"role": "user", "content": prompt})
            
            return response, history
            
        except Exception as e:
            print(f"Attempt {times + 1} failed with error: {str(e)}")
            if times < 2: 
                if package_name is not None:
                    save_chat_history(history, package_name)
                print(f"Take a 60*{times+1} seconds break before the next attempt...")
                time.sleep(60*(times+1)) 
            else: 
                print(f"All {attempts} attempts failed. Please try again later.")
                if package_name is not None:
                    save_chat_history(history, package_name)
                raise e
    
def save_chat_history(history, package_name):
    curr_time = datetime.datetime.now()
    curr_time_string = curr_time.strftime("%Y-%m-%d %H-%M-%S")
    file_name = f"./chat_history/{package_name}_chat_{curr_time_string}.json"
    with open(file_name, 'w') as file:
        json.dump(history, file)

def get_model_name(response):
     return "gemini-2.0-flash"

def get_message(response):
    return response.text


