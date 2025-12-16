from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel,Field
from typing import Annotated
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

model=ChatGroq(model="llama-3.1-8b-instant",)

class Person(BaseModel):
    name:Annotated[str,Field(...,title="Name",description="Enter the name of the person",example="Arya")]
    age:Annotated[int,Field(...,title="Age",description="Enter the age of the person",example=22,ge=0,le=120)]
    city:Annotated[str,Field(...,title="City",description="Enter the city of the person",example="Kathmandu")]


parser=PydanticOutputParser(pydantic_object=Person)

prompt=PromptTemplate(
    # While forming prompt we send an extra information about the type of out of the LLM
    template="Generate a fictional name, age , city and {place}\n{format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain=prompt | model | parser

result=chain.invoke({"place":"Nepal"})

print(result)
chain.get_graph().print_ascii()

