from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


load_dotenv()

model = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)

notes_prompt=PromptTemplate(
    template="Generate ten line notes on {text}",
    input_variables=['text']
)

quiz_prompt=PromptTemplate(
    template="Generate five quizes on the {topic}",
    input_variables=["topic"]
    )

merge_prompt=PromptTemplate(
    template="Merge the notes and quizes provided of {notes} and {quiz}",
    input_variables=["notes","quiz"]
)

parser=StrOutputParser()

parallel_chain = RunnableParallel(
    notes=notes_prompt | model | parser,
    quiz=quiz_prompt | model | parser
)

merge_chain = merge_prompt | model | parser

chain=parallel_chain | merge_chain

text="""Faster R-CNN was originally published in NIPS 2015. After publication, it went through a couple of revisions which we'll later discuss. As we mentioned in our previous blog post, Faster R-CNN is the third iteration of the R-CNN papers — which had Ross Girshick as author & co-author.

Everything started with “Rich feature hierarchies for accurate object detection and semantic segmentation” (R-CNN) in 2014, which used an algorithm called Selective Search to propose possible regions of interest and a standard Convolutional Neural Network (CNN) to classify and adjust them. It quickly evolved into Fast R-CNN, published in early 2015, where a technique called Region of Interest Pooling allowed for sharing expensive computations and made the model much faster. Finally came Faster R-CNN, where the first fully differentiable model was proposed."""

result=chain.invoke({
    "text":text,
    "topic":"Faster R-CNN"
    })

print(result)

chain.get_graph().print_ascii()