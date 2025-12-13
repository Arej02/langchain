from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI 
import os

load_dotenv()


# Use a highly reliable model for text generation
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

# The invoke method works exactly the same
result = model.invoke("What is the capital of Nepal")
print(result.content)