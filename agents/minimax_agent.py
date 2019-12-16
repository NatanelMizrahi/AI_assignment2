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
        MMtree = MiniMaxTree(env, self, env.get_other_agent(self), mode=env.mode)
        best_choice, _ = MMtree.minimax(state_node=MMtree.root,
                                        depth=env.depth,
                                        a=float('-inf'),
                                        b=float('inf'),
                                        is_max=True)
        MMtree.display()      # view the minimax decision tree
        MMtree.restore_env()  # restore the environment to the state it was in before this function was called
        best_choice.action.execute()

