import os
from dotenv import load_dotenv
import sys

load_dotenv()
WORKDIR=os.getenv("WORKDIR")
os.chdir(WORKDIR)
sys.path.append(WORKDIR)

from src.utils import GraphConfig, State, get_model
from datetime import datetime
from src.memory_handler import MemoryHandlerS3, ChatHistory
from constants import SYSTEM_MSG
from langchain_core.messages import SystemMessage

def retrieve_chat_history(state: State, config: GraphConfig):
    user_id = str(config['configurable']["user_id"])
    number_retrieved_msgs = config['configurable']["number_retrieved_msgs"]
    s3_memory_instance = MemoryHandlerS3()
    chat_history = s3_memory_instance.retrieve_chat_history(user_id = user_id, number_retrieved_msgs = number_retrieved_msgs)
    if chat_history is not None:
        chat_history = s3_memory_instance.convert_memory_in_langchain_obj(chat_history=chat_history)
    else:
        chat_history = []
    
    return {"messages": chat_history + state['user_query']}

def chatbot(state: State, config: GraphConfig):
    llm = get_model(model = config['configurable']["llm"],temperature = config['configurable']["temperature"])
    output = llm.invoke([SystemMessage(content = SYSTEM_MSG)] + state['messages'])
    output.etl_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    return {"messages": [output]}

def saving_chat_history(state: State, config: GraphConfig):
    user_id = str(config['configurable']["user_id"])
    s3_memory_instance = MemoryHandlerS3()
    new_data = s3_memory_instance.convert_langchain_obj_in_memory(chat_history=state['messages'])
    s3_memory_instance.upload_chat_history(user_id=user_id, new_data=ChatHistory(**new_data))

