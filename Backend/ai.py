from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Access the OpenAI API key
openai_api_key = os.getenv("MY_OPENAI_KEY")

# Read from the text file
with open("ai_response.txt", "r") as file:
    ai_response = file.read()

prompt = """
You're a creative brand strategist, given the following customer information, come up with \
next marketing strategy based on AI generated response, \
explained step by step.

Customer Information:
{cust_info}

Customer Information:
{ai_response}

Your Response:
* Predicted Segment:
* Next Marketing Strategy:
"""

prompt_template = PromptTemplate(
    input_variables=["clusters","ai_response"],
    template=prompt,
)

llm = ChatOpenAI(openai_api_key=openai_api_key,model_name='gpt-3.5-turbo', temperature=0)

chain = LLMChain(
    llm=llm,
    prompt = prompt_template,
    verbose=True)