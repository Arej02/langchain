from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from typing import Annotated,List,Literal


load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class Review(BaseModel):

    key_themes: List[str] = Field(
        title="Key Themes",
        description=(
            "List 2–4 short noun phrases capturing the main topics of the review "
            "(e.g., 'water resistance failure', 'device durability', 'long-term reliability')."
        ),
        min_items=2,
        max_items=4
    )
    summary: Annotated[
        str,
        Field(
            title="Summary",
            description="Concise, neutral summary of the user's experience in 1–2 sentences.",
            max_length=300
        )
    ]
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment expressed in the review"
    )
    sentiment_confidence: float = Field(
    ge=0.0,
    le=1.0,
    description="Model confidence in the sentiment classification"
    )

structured_model = model.with_structured_output(
    Review
)
review="Had this phone since launch. never careless with it. Last year i went swimming and tested the water resistance by taking video and picture under water. no problems until 12 months later. phone is still in pristine condition but somehow when i went swimming with it this time it died within 3 sec. now its dead forever. looks like the water resistance only last for couple years, idk. any opinions"

result=structured_model.invoke(review)

print(result)