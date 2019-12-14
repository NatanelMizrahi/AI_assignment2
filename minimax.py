from utils.data_structures import Heap, Stack
from typing import Union
from utils.tree import display_tree
from environment import Environment, State, EvacuateNode, Option
from configurator import Configurator, debug
from action import Action, ActionType


class MiniMaxTree:
    def __init__(self, env: Environment, agent):
        self.agent = agent
        self.env = env
        self.root = self.get_root_node()
        self.hist = [] # used for debug

    def get_initial_state(self):
        return self.env.get_state(self.agent)

    def get_root_node(self):
        """creates a root node for the search tree representing the initial state"""
        return self.get_initial_state()

    def restore_env(self):
        """restore environment to actual state after finding a strategy"""
        self.env.apply_state(self.root.state)

    def minimax(self, state: State, depth, a, b, IsMax=True):
        if depth == 0 or state.is_goal():
            return state, self.heuristic(state)
        if IsMax:
            value = float('-inf')
            options = self.expand_node(state, self.agent)
            choice = None
            for opt in options:
                temp = max(value, (self.minimax(opt.state, depth-1, a, b, False))[1])
                if temp > value:
                    choice = opt
                    value = temp
                a = max(a, value)
                if a >= b:
                    break
            return choice, value
        else: #MIN
            value = float('inf')
            options = self.expand_node(state, self.env.get_other_agent(self.agent))
            choice = None
            for opt in options:
                temp = min(value, (self.minimax(opt.state, depth-1, a, b, True))[1])
                if temp < value:
                    choice = opt
                    value = temp
                b = min(b, value)
                if a >= b:
                    break
            return choice, value

    def heuristic(self, state: State=None):
        h1 = self.heuristic_helper(self.agent, state)
        h2 = self.heuristic_helper(self.env.get_other_agent(self.agent), state)
        return h1 - h2

    def heuristic_helper(self, agent, state: State=None):
        """given a state for an agent, returns how many people can (!) be saved by the agent"""
        self.env.apply_state(state)
        src = agent.loc
        self.env.G.dijkstra(src)
        V = self.env.G.get_vertices()
        require_evac_nodes = list(self.env.require_evac_nodes)
        # find nodes that can be reached before hurricane hits them. create (node, required_pickup_time) pairs
        evac_candidates, can_save = [], []
        for v in require_evac_nodes:
            if self.env.time + v.d >= v.deadline: #nodes we can reach in time
                evac_candidates.append((v, self.env.time + v.d, list(self.env.G.get_shortest_path(src, v))))
        for u, time_after_pickup, pickup_shortest_path in evac_candidates:
            self.env.G.dijkstra(u) # calculate minimum distance from node after pickup
            shelter_candidates = [(v, time_after_pickup + v.d, list(self.env.G.get_shortest_path(u, v))) for v in V
                                  if v.is_shelter() and time_after_pickup + v.d <= v.deadline]
            if shelter_candidates:
                can_save.append(u)
            # debug('\npossible routes for evacuating {}:'.format(u))
            # for shelter, total_time, dropoff_shortest_path in shelter_candidates:
                # debug('pickup:(T{}){}(T{}) | drop-off:{}(T{}): Shelter(D{})'.format(self.env.time,
                #                                                                     pickup_shortest_path,
                #                                                                     time_after_pickup,
                #                                                                     dropoff_shortest_path,
                #                                                                     total_time,
                #                                                                     shelter.deadline))
        n_can_save = sum([v.n_people for v in can_save])
        # debug('h(x) = {} = # of doomed people (doomed_nodes = {})'.format(n_doomed_people, doomed_nodes))
        return n_can_save

    def total_cost(self, state):
        # assumes environment's state was updated before calling this function
        h = 0 if state.is_goal() else self.heuristic(state)
        g = state.agent.penalty
        debug('cost = g + h = {} + {} = {}'.format(g, h, g+h))
        return g + h

    def expand_node(self, state: State, agent):
        """returns Options (action, state) for all possible moves."""
        self.env.apply_state(state)
        # debug("Expanding node ID={0.ID} (cost = {0.cost}):".format(plan))
        # state.describe()
        neighbours = agent.get_possible_steps(self.env, verbose=True) # options to proceed
        options = []
        for dest in neighbours + [ActionType.TERMINATE]:
            # debug("\ncreated state:")
            # result_state.describe()
            new_option = self.successor(state, agent, dest)
            options.append(new_option)
        return options

    def successor(self, state: State, agent, dest: Union[EvacuateNode, ActionType]):
        """
        :param state: a state of the environment in the search tree node
        :param dest: a destination node (GOTO action) or ActionType.TERMINATE (for terminate action)
        :return: Option (action,state) action resulting in the successor state
        """
        self.env.apply_state(state)
        if dest == ActionType.TERMINATE:
            def terminate_agent():
                agent.terminate(self.env)
            action = Action(
                agent=agent,
                description='*[T={:>3}] "TERMINATE" action for {}'.format(agent.time, agent.name),
                callback=terminate_agent)
            agent.local_terminate()
        else:
            def move_agent():
                agent.goto2(self.env, dest)
            action = Action(
                agent=agent,
                description='*[T={:>3}] "GOTO {}->{}" action for {}'.format(agent.time, agent.loc, dest, agent.name),
                callback=move_agent)
            agent.local_goto(self.env, dest)
        action.describe()
        return Option(action, self.env.get_state(agent))
    #
    # def display(self):
    #     """plots the search tree"""
    #     if not Configurator.view_strategy:
    #         return
    #     state_nodes = self.hist + self.fringe.heap
    #     for node in state_nodes:
    #         node.tmp = node.summary() + ' {}'.format(node.ID)
    #     V = [node.tmp for node in state_nodes]
    #     E = [(node.tmp, node.parent.tmp) for node in state_nodes if node.parent is not None]
    #     display_tree(V[0], V, E)
