from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Create an object for the model:
model=ChatGoogleGenerativeAI(model="gemini-2.5-flash") # This is the Langchain chat model wrapper that connects to the Gemini LLM and handles prompts sending and receiving

# Create one template:
template=PromptTemplate( # This is a parameterized prompt where we can fit it teh palceholders in the topic dynamically.
    template="Summarize about the {topic}",
    input_variables=['topic']
)

# Create a object for parser:
parser=StrOutputParser() # This is a parser that converts output in strings and enforces consistent output type

# Create a simple chain:
chain= template | model | parser # Prompt->Model->Parser

result=chain.invoke({'topic':'phylogenetic trees'}) # Chaining lets us invoke the entire pipeline with a single call instead of manually invoking each step.

print(result)
chain.get_graph().print_ascii()