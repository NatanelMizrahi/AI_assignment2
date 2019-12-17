from environment import Environment, EvacuateNode
from agents.agent import Agent
from minimax import MiniMaxTree


class GameAgent(Agent):
    def __init__(self, name, start_loc: EvacuateNode):
        super().__init__(name, start_loc)

    def act(self, env: Environment):
        """find the best action and execute it"""
        if not self.is_available(env):
            return
        best_move = MiniMaxTree(env, self, env.get_other_agent(self), mode=env.mode).get_best_move()
        best_move.execute()

        env.print_queued_actions("POST2 %s" % self.name)

