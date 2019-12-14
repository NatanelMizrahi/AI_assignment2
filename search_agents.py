from environment import Environment, EvacuateNode, Option
from utils.data_structures import Stack
from agents.base_agents import Human
from minimax import MiniMaxTree
from configurator import Configurator, debug
from action import Action


class GameAgent(Human):
    """Base class for search agents"""

    def __init__(self, name, start_loc: EvacuateNode, max_expand=float('inf')):
        super().__init__(name, start_loc)
        self.max_expand = max_expand

    def act(self, env: Environment):
        """pop the next action in strategy and execute it"""
        self.time += self.max_expand * Configurator.T
        if (env.mode == "Adversarial"):
            MMtree = MiniMaxTree(env, self)
            choice, value = MMtree.minimax(MMtree.root, env.depth, float('-inf'), float('inf'), True)
            print ("Decided on Action")
            choice.action.execute()
#
#
# class GreedySearch(SearchAgent):
#     """A search agent that expands one node at a time in a search tree when devising a strategy"""
#     def __init__(self, name, start_loc: EvacuateNode):
#         super().__init__(name, start_loc, max_expand=1)
#
#
# class RTAStar(SearchAgent):
#     """A search agent that expands a limited number of nodes at a time in a search tree when devising a strategy"""
#     def __init__(self, name, start_loc: EvacuateNode):
#         super().__init__(name, start_loc, max_expand=Configurator.limit)
#
#
# class AStar(SearchAgent):
#     """A search agent with an unlimited amount of expansions in a search tree when devising a strategy"""
#     def __init__(self, name, start_loc: EvacuateNode):
#         super().__init__(name, start_loc, max_expand=100000)
#
