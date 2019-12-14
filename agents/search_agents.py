from environment import Environment, EvacuateNode, Option
from agents.base_agents import Human
from minimax import MiniMaxTree
from configurator import Configurator


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
            print("Decided on Action")
            choice.action.execute()
