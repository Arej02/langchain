from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# prompt_template=PromptTemplate(
#     template="""
#     Describe {topic} in {length} sentences.
#     If you do not have enough information just repsond by saying not enough inforamtion available rather than guessing randomly. 
#     If the given length is long and the concept requires mathematical explanation please do so.
#     """,
#     input_variables=["topic","length"]
# )

prompt_template=load_prompt("template.json")

user_topic=input("\nEnter the topic of interest:")
user_length=input("\nEnter the length of response:")

dynamic_prompt=prompt_template.format(topic=user_topic,length=user_length)

response=model.invoke(dynamic_prompt)
print(response.content)



