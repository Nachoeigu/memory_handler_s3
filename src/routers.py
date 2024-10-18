import os
from dotenv import load_dotenv
import sys

load_dotenv()
WORKDIR=os.getenv("WORKDIR")
os.chdir(WORKDIR)
sys.path.append(WORKDIR)

from src.utils import State
from typing import Literal

def chatbot_router(state: State) -> Literal['chatbot','saving_chat_history']:
    if state['messages'][-1].type == 'human':
        return 'chatbot'
    else:
        return 'saving_chat_history'
