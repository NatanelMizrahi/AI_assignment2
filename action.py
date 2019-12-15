from enum import IntEnum
import re


class ActionType(IntEnum):
    """ forces order on actions being executed on the same cycle: first uncompleted TRAVERSE moves (ARRIVE),
        then TERMINATE, and lastly beginning TRAVERSE operations"""
    NOOP      = 0
    ARRIVE    = 1
    TERMINATE = 2
    TRAVERSE  = 3


class Action:
    """Data structure for describing an max_player's action"""
    def __init__(self,
                 agent,
                 # optional arguments
                 action_type: ActionType=ActionType.NOOP,
                 description='',
                 end_time=0,
                 callback=None):
        self.agent = agent
        self.action_type = action_type
        self.description = description
        self.end_time = end_time
        self.callback = callback
        self.summary = re.search('(\w+->\w+)', self.description).group() if '->' in description else 'T'

    def __lt__(self, other):
        return self.action_type <= other.action_type

    def __repr__(self):
        return self.description

    def execute(self):
        if self.callback is not None:
            self.callback()
            print('[DONE]' + self.description)

    def describe(self):
        print(self.description)
