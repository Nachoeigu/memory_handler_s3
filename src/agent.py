import os
from dotenv import load_dotenv
import sys

load_dotenv()
WORKDIR=os.getenv("WORKDIR")
os.chdir(WORKDIR)
sys.path.append(WORKDIR)

import yaml
from langgraph.graph import StateGraph, START, END
from datetime import datetime
from langchain_core.messages import  HumanMessage
import uuid
from src.utils import *
from src.nodes import *
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')



graph_builder = StateGraph(State, 
                      input = GraphInput,
                      output = GraphOutput,
                      config_schema = GraphConfig)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("saving_chat_history", saving_chat_history)
graph_builder.add_node("retrieve_chat_history", retrieve_chat_history)

graph_builder.add_edge(START, "retrieve_chat_history")
graph_builder.add_edge('retrieve_chat_history', "chatbot")
graph_builder.add_edge('chatbot', "saving_chat_history")
graph_builder.add_edge('saving_chat_history', END)


graph = graph_builder.compile()


if __name__ == '__main__':
    with open(f"{WORKDIR}/agent_config.yaml", "r") as file:
        yaml_config = yaml.safe_load(file)

    configuration = {
        "configurable": {
            "thread_id": yaml_config['thread_id'],
            "user_id":yaml_config['user_id'],
            'temperature': yaml_config['temperature'],
            'llm': yaml_config['llm']
        }
    }
    while True:
        user_input = input("User: ")
        if user_input in ['q','quit','exit']:
            break
        output = graph.invoke(
            input = {"user_query": [HumanMessage(content = user_input, id = uuid.uuid4(), etl_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z')]},
            config = configuration          
            )
        
        logging.info("Reply:" + output['messages'][-1].content)
