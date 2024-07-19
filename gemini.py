import  google.generativeai as genai
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np
from scipy import stats
import statsmodels.api as sm
import seaborn as sns

load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')

genai.configure(api_key=gemini_api_key)

LLM = genai.GenerativeModel('models/gemini-1.5-flash')

def return_llm_response(img):
    response = LLM.generate_content(
    [
        "just give the distribution of the given image (eg: normal distribution ,Uniform Distribution, Right-skewed , Left-skewed,Multimodal Distribution,Bimodal Distribution).Only use one of eg distributions only. Only use the two words in the output.", 
        img
    ], 
    stream=True
)
    response.resolve()
    return response.text

def create_plot(column):
    plt.hist(column, bins=5, edgecolor='black')
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)  
    plt.close() 
    
    return img_io

def open(image):
    img = Image.open(image)
    
    return img

