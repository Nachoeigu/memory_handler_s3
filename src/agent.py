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
from src.routers import *
import logging
from langgraph.checkpoint.memory import MemorySaver


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
graph_builder.add_conditional_edges('chatbot', 
                                    chatbot_router)
graph_builder.add_edge('saving_chat_history', END)

graph = graph_builder.compile()

if __name__ == '__main__':
    memory = MemorySaver()

    graph = graph_builder.compile(interrupt_after=['chatbot'],
                                  checkpointer=memory)

    with open(f"{WORKDIR}/agent_config.yaml", "r") as file:
        yaml_config = yaml.safe_load(file)

    configuration = {
        "configurable": {
            "thread_id": yaml_config['thread_id'],
            "user_id":yaml_config['user_id'],
            'temperature': yaml_config['temperature'],
            'llm': yaml_config['llm'],
            'number_retrieved_msgs': yaml_config['number_retrieved_msgs']
        }
    }
    user_input = input("User: ")
    for event in graph.stream(
        input = {"user_query": [
            HumanMessage(
                content = user_input, id = uuid.uuid4(), 
                etl_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            )
        ]},
        config = configuration     
        ):

        if event.get("chatbot","") != "":
            logging.info(event['chatbot']['messages'][-1].content)

    while True:
        user_input = input("User: ")
        user_input = HumanMessage(
                content = user_input, id = uuid.uuid4(), 
                etl_time = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            )
        if user_input.content in ['q','quit','exit']:
            graph.get_state(config = configuration).next
        else:
            graph.update_state(configuration, {'messages': [user_input]}, as_node = 'chatbot')

        for event in graph.stream(
                input = None,
                config = configuration):

            if event.get("chatbot","") != "":
                logging.info(event['chatbot']['messages'][-1].content)
        
        if user_input.content in ['q','quit','exit']:
            break

