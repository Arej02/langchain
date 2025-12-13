from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

static_prompt="What is the height of Mount Everst?"
result=model.invoke(static_prompt)

print(result.content)