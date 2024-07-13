import  google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=gemini_api_key)

LLM = genai.GenerativeModel('models/gemini-pro-vision')

def return_llm_response(img):
    response = LLM.generate_content(
    [
        "just give the distribution of the given image (eg: normal distribution , Uniform Distribution, Skewed Distributions (Right-skewed or Left-skewed),Log-Normal Distribution,Poisson Distribution,Multinomial Distribution,Binomial Distribution,Gamma Distribution,Beta Distribution).Only use one of eg distributions only. only give the type of the distribution nothing else.", 
        img
    ], 
    stream=True
    )
    response.resolve()
    return response.text


