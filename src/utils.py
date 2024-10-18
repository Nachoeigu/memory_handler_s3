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
from constants import AVAILABLE_MODELS
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
    llm: Literal[*AVAILABLE_MODELS]
    temperature: float = 0
    number_retrieved_msgs: int = 10

class GraphInput(TypedDict):
    user_query: HumanMessage

class GraphOutput(TypedDict):
    messages: List[AnyMessage]

def get_model(model, temperature):  
    provider, model_name = model.split('|')
    if provider == "openai":
        return ChatOpenAI(temperature=temperature, model= model_name)
    elif provider == "google":
        return ChatGoogleGenerativeAI(temperature=temperature, model=model_name)
    elif provider == 'groq':
        return ChatGroq(temperature=temperature, model=model_name)
    elif provider == 'amazon':
        return ChatBedrock(model_id = model_name, model_kwargs = {'temperature':temperature})
    else:
        raise ValueError