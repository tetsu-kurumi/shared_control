import sys
import os
import time
from langchain_community.llms import Ollama
from langchain_community.cache import InMemoryCache

import ollama


sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from shared_control_proj.shared_control import config

class LLM:
    def __init__(self):
        self.llm = Ollama(model="llama3:8b", top_k=25, top_p = 0.5, cache=False)
        self.user_message = config.USER_MESSAGE

    # @gcache.decorate
    def infer_goal(self, utterance):
        if config.HOLDING_LETTER:
            prompt = config.SYSTEM_MESSAGE_PLACE + self.user_message + utterance
        else:
            prompt = config.SYSTEM_MESSAGE_PICK + self.user_message + utterance
        result = self.llm.invoke(prompt)
        return result

