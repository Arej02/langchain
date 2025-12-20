from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model=ChatGroq(model="llama-3.1-8b-instant")

text_loader=TextLoader('cover_letter.txt',encoding='utf-8')
docs=text_loader.load()
print(type(docs))

print(docs[0].page_content)

prompt=PromptTemplate(
    template="Summarize the given cover letter. {content}",
    input_variables=['content']
)

parser=StrOutputParser()

chain= prompt | model | parser

result=chain.invoke({'content':docs[0].page_content})
print(result)






