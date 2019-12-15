from enum import Enum
import re

class ActionType(Enum):
    TRAVERSE  = 1
    TERMINATE = 2


class Action:
    """Data structure for describing an max_player's action"""
    def __init__(self,
                 agent,
                 # optional arguments
                 action_type: ActionType=None,
                 description='',
                 end_time=0,
                 callback=None):
        self.agent = agent
        self.action_type = action_type
        self.description = description
        self.end_time = end_time
        self.callback = callback
        self.summary = re.search('(\w+->\w+)', self.description).group() if '->' in description else 'T'

    def execute(self):
        if self.callback is not None:
            self.callback()
            print('[DONE]' + self.description)

    def describe(self):
        print(self.description)
