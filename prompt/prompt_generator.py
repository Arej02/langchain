from langchain_core.prompts import PromptTemplate

prompt_template=PromptTemplate(
    template="""
    Describe {topic} in {length} sentences.
    If you do not have enough information just repsond by saying not enough inforamtion available rather than guessing randomly. 
    If the given length is long and the concept requires mathematical explanation please do so.
    """,
    input_variables=["topic","length"],
    validation_template=True # Make sures the we have kept correct placeholder
)

prompt_template.save("template.json")