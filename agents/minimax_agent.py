from environment import Environment, EvacuateNode
from agents.agent import Agent
from minimax import MiniMaxTree


class GameAgent(Agent):
    def __init__(self, name, start_loc: EvacuateNode, max_expand=float('inf')):
        super().__init__(name, start_loc)
        self.max_expand = max_expand

    def act(self, env: Environment):
        """pop the next action in strategy and execute it"""
        if not self.is_available(env):
            return
        MMtree = MiniMaxTree(env, self, env.get_other_agent(self), mode=env.mode)
        best_choice, _ = MMtree.minimax(state_node=MMtree.root,
                                        depth=env.depth,
                                        a=float('-inf'),
                                        b=float('inf'),
                                        is_max=True)
        MMtree.display()
        MMtree.restore_env()
        # self.register_action(env, best_choice.action)
        best_choice.action.execute()
        env.print_queued_actions("POST EXEC")
        # env.G.display()

