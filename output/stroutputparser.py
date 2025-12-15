from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

template1=PromptTemplate(
    template="Explain in detail about the {topic}",
    input_variables=['topic']
)

template2=PromptTemplate(
    template="Summarize the given topic in 5 sentences./n {text}",
    input_variables=['text']
)

parser=StrOutputParser()

chain=template1 | model | parser | template2 | model | parser

result=chain.invoke({'topic':'black hole'})

print(result)