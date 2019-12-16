from typing import Union, List
from utils.tree import display_tree
from environment import Environment, State, EvacuateNode, Option
from configurator import Configurator, debug
from action import Action, ActionType


class MiniMaxTree:
    def __init__(self, env: Environment, max_player, min_player, mode='Adversarial'):
        self.max_player = max_player
        self.min_player = min_player
        self.env = env
        self.mode = mode
        self.root: Option = self.get_root_node()
        self.nodes: List[Option] = []  # for debug
        self.sync_env() # start by

    def sync_env(self):
        """ some of the actions in the environment may not have completed (e.g. agents in transit).
            This function brings the environment to the point that all the agents have finished their prior moves
            before creating the minimax tree. NOT TO BE CONFUSED WITH restore_env()"""
        self.env.execute_all_env_actions()

    def get_initial_state(self):
        return self.env.get_state(self.max_player)

    def get_root_node(self):
        """creates a root node for the search tree representing the initial state"""
        return Option(Action(self.max_player, description='ROOT'), self.get_initial_state())

    def restore_env(self):
        """restore environment to actual state after finding a strategy"""
        self.env.apply_state(self.root.state)

    def get_final_state(self, terminal_node):
        """ calculates the terminal state by backtracking the terminal node to get all the actions leading to the
            terminal state and executes them.
            This partially solves the concurrency issue caused by the fact that the environment is inherently NOT
            turn based: one agent can do a single long traverse move while the other can do several shorter moves"""
        path_to_root = self.backtrack(terminal_node)
        all_agents_actions = [option.action for option in path_to_root]

        self.restore_env()
        self.env.agent_actions = {}
        self.env.add_agent_actions(all_agents_actions)
        self.env.execute_all_env_actions()
        return self.env.get_state(self.max_player)

    def minimax(self, state_node: Option, depth, a, b, is_max=True):
        self.nodes.append(state_node)
        state_node.is_max = is_max
        state = state_node.state
        if depth == 0 or state.is_goal():
            final_state = self.get_final_state(state_node)
            utility = self.heuristic(final_state)
            state_node.state = final_state
            state_node.value = utility
            return state_node, utility

        if is_max:  # MAX player
            value = float('-inf')
            options = self.expand_node(state, self.max_player)
            choice = None
            for opt in options:
                opt.parent = state_node
                min_option, min_option_value = self.minimax(opt, depth-1, a, b, is_max=False)
                opt.value = min_option_value
                temp = max(value, min_option_value)
                if temp > value:
                    choice = opt
                    value = temp
                a = max(a, value)
                if a >= b:
                    print("PRUNED: " + state.ID)
                    break
        else:  # MIN player
            value = float('inf')
            options = self.expand_node(state, self.min_player)
            choice = None
            for opt in options:
                opt.parent = state_node
                max_option, max_option_value = self.minimax(opt, depth-1, a, b, is_max=True)
                opt.value = max_option_value
                temp = min(value, max_option_value)
                if temp < value:
                    choice = opt
                    value = temp
                b = min(b, value)
                if a >= b:
                    print("PRUNED: " + state.ID)
                    break
        state_node.value = value
        return choice, value

    def heuristic(self, state: State):
        h1 = self.heuristic_helper(self.max_player, state)
        h2 = self.heuristic_helper(self.min_player, state)
        return (h1 - h2) if self.mode is 'Adversarial' else (h1 + h2)

    def heuristic_helper(self, agent, state: State):
        """given a state for an max_player, returns how many people can (!) be saved by the max_player"""
        self.env.apply_state(state)
        if agent.terminated:
            return agent.n_saved #TODO: fix

        def num_rescuable_carrying():
            if any([self.env.can_reach_before_deadline(v) for v in self.env.get_shelters()]):
                return agent.n_carrying
            return 0

        def get_evac_candidates():
            """find nodes that can be reached before hurricane hits them.
                :returns: a list of (node, pickup_time, pickup_path) tuples"""
            src = agent.loc
            self.env.G.dijkstra(src)
            evacuation_candidates = []
            require_evac_nodes = self.env.get_require_evac_nodes()
            for v in require_evac_nodes:
                if self.env.can_reach_before_deadline(v) and self.env.can_reach_before_other_agent(agent, v):
                    # nodes we can reach in time
                    time_after_pickup = self.env.time + v.d
                    pickup_shortest_path = list(self.env.G.get_shortest_path(src, v))
                    evacuation_candidates.append((v, time_after_pickup, pickup_shortest_path))
            return evacuation_candidates

        def can_reach_shelter(evacuation_candidates):
            can_save = []
            V = self.env.G.get_vertices()
            for u, time_after_pickup, pickup_shortest_path in evacuation_candidates:
                self.env.G.dijkstra(u)  # calculate minimum distance from node after pickup
                shelter_candidates = [(v, time_after_pickup + v.d, list(self.env.G.get_shortest_path(u, v))) for v in V
                                      if v.is_shelter() and time_after_pickup + v.d <= v.deadline]
                if len(shelter_candidates) != 0:
                    can_save.append(u)
            return can_save

        evac_candidates = get_evac_candidates()
        can_save_nodes = can_reach_shelter(evac_candidates)
        n_already_saved = agent.n_saved
        n_can_save = num_rescuable_carrying()
        n_can_save += sum([v.n_people for v in can_save_nodes])
        debug('[#{}]:{} can save {} (nodes = {}, carrying ={})'
              .format(state.ID, agent.name, n_can_save, can_save_nodes, agent.n_carrying))
        return n_can_save + n_already_saved # TODO: fix

    def expand_node(self, state: State, agent):
        """returns Options (action, state) for all possible moves."""
        self.env.apply_state(state)
        neighbours = agent.get_possible_steps(self.env, verbose=False) # options to proceed
        options = []
        for dest in neighbours + [ActionType.TERMINATE]:
            new_option = self.successor(state, agent, dest)
            options.append(new_option)
        return options

    def successor(self, state: State, agent, dest: Union[EvacuateNode, ActionType]):
        """
        :param agent: the agent to be executing the action
        :param state: a state of the environment in the search tree node
        :param dest: a destination node (TRAVERSE action) or ActionType.TERMINATE (for terminate action)
        :return: Option (action,state) action resulting in the successor state
        """
        self.env.apply_state(state)
        if dest == ActionType.TERMINATE:
            def terminate_agent():
                agent.terminate(self.env)
            action = Action(
                agent=agent,
                action_type=ActionType.TERMINATE,
                description='*[T={:>3}] "TERMINATE" action for {}'.format(agent.time, agent.name),
                end_time=agent.time,
                callback=terminate_agent)
            agent.local_terminate()
        else:
            def move_agent():
                agent.traverse(self.env, dest)
                print('TRV', agent.loc, dest, agent.loc.agents, dest.agents)
            action = Action(
                agent=agent,
                action_type=ActionType.TRAVERSE,
                description='*[T={:>3}] "TRAVERSE {}->{}" action for {}'.format(agent.time, agent.loc, dest, agent.name),
                end_time=agent.time,
                callback=move_agent)
            agent.goto(self.env, dest)
        return Option(action, self.env.get_state(agent))

    def backtrack(self, option):
        path_to_root = [option]
        v = option
        while v is not self.root:
            v = v.parent
            path_to_root.append(v)
        return reversed(path_to_root)

    def display(self):
        """plots the search tree"""
        if not Configurator.view_strategy:
            return
        nodes = {node: node.summary() for node in self.nodes}
        max_nodes = [nodes[v] for v in nodes.keys() if v.is_max]
        min_nodes = [nodes[v] for v in nodes.keys() if not v.is_max]
        E = []
        edge_labels = {}
        for node in nodes.keys():
            if node.parent is not None:
                e = (nodes[node], nodes[node.parent])
                E.append(e)
                edge_labels[e[0], e[1]] = node.action.summary
                edge_labels[e[1], e[0]] = node.action.summary
        display_tree(nodes[self.root], min_nodes, max_nodes, E, edge_labels)
