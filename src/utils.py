import os
from dotenv import load_dotenv
import sys

load_dotenv()
WORKDIR=os.getenv("WORKDIR")
os.chdir(WORKDIR)
sys.path.append(WORKDIR)

from typing import Annotated, List, Literal, Union
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import  HumanMessage, AnyMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_aws.chat_models import ChatBedrock


class State(TypedDict):
    user_query: HumanMessage
    messages: Annotated[list, add_messages]

class GraphConfig(TypedDict):
    thread_id: int
    user_id: Union[str, int]
    llm: Literal['groq','openai','google','amazon']
    temperature: float

class GraphInput(TypedDict):
    user_query: HumanMessage

class GraphOutput(TypedDict):
    messages: List[AnyMessage]

def get_model(model, temperature):    
    if model == "openai":
        return ChatOpenAI(temperature=temperature, model="gpt-4o-mini")
    elif model == "google":
        return ChatGoogleGenerativeAI(temperature=temperature, model="gemini-1.5-pro-exp-0827")
    elif model == 'groq':
        return ChatGroq(temperature=temperature, model="llama-3.1-70b-versatile")
    elif model == 'amazon':
        return ChatBedrock(model_id = 'anthropic.claude-3-5-sonnet-20240620-v1:0', model_kwargs = {'temperature':temperature})
    else:
        raise ValueError