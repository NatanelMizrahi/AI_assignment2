from typing import Union, List
from utils.tree import display_tree
from environment import Environment, State, EvacuateNode
from configurator import Configurator, debug
from action import Action, ActionType


class Option:
    def __init__(self,
                 action: Action,
                 state: State):
        self.action = action
        self.state = state
        self.value = None
        self.is_max = False
        self.parent: Option = None

    def summary(self):
        return '{} v={}'.format(self.state.summary(), self.value)


class MiniMaxTree:
    def __init__(self, env: Environment, max_player, min_player, mode='adversarial'):
        self.max_player = max_player
        self.min_player = min_player
        self.env = env
        self.mode = mode
        self.nodes: List[Option] = []  # for debug
        self.env_initial_state: State = self.get_initial_state()
        self.root: Option = self.get_root_node() #TODO: was before sync env!!

    def sync_env(self):
        """ some of the actions in the environment may not have completed (e.g. agents in transit).
            This function brings the environment to the point that all the agents have finished their prior moves
            before creating the minimax tree. NOT TO BE CONFUSED WITH restore_env()"""
        self.env.print_queued_actions("PRE:: T=%d" % self.env.time)
        self.env.get_state(self.max_player).describe()
        self.env.execute_all_env_actions()
        self.env.print_queued_actions("POST:: T=%d" % self.env.time)
        self.env.get_state(self.max_player).describe()

    def get_initial_state(self):
        return self.env.get_state(self.max_player)

    def get_root_node(self):
        """Completes queued actions in env + creates a root node for the search tree representing the initial state"""
        self.sync_env()
        return Option(Action(self.max_player, description='Dummy action (Tree Root)'), self.get_initial_state())

    def restore_root_state(self):
        """restore environment to the state in the root node - right after completing the actions initially in queue"""
        self.env.apply_state(self.root.state)

    def restore_env(self):
        """restore environment to actual state after finding a strategy"""
        self.env.apply_state(self.env_initial_state)

    def get_final_state(self, terminal_node):
        """ calculates the terminal state by backtracking the terminal node to get all the actions leading to the
            terminal state and executes them.
            This partially solves the concurrency issue caused by the fact that the environment is inherently NOT
            turn based: one agent can do a single long traverse move while the other can do several shorter moves"""
        path_to_root = self.backtrack(terminal_node)
        all_agents_actions = [option.action for option in path_to_root]

        self.restore_root_state()
        self.env.agent_actions = {}
        self.env.add_agent_actions(all_agents_actions)
        self.env.execute_all_env_actions()
        return self.env.get_state(self.max_player)

    def get_other_player(self, player):
        return player is self.min_player and self.max_player or self.min_player

    #TODO: review this tie-breaker
    def tie_breaker(self, agent, option, current_best_option):
        print('Tie: %s VS. %s' % (current_best_option.state.ID, option.state.ID))
        if self.mode is 'semi-cooperative':
            current_best_state = self.get_final_state(current_best_option)
            other_state = self.get_final_state(option)
            current_h = self.heuristic(self, current_best_state, True, 'cooperative')
            other_h = self.heuristic(self, other_state, True, 'cooperative')
            return other_h > current_h

        agent_state = agent is option.state.agent and option.state.agent_state or option.state.agent2_state
        return (Configurator.tie_breaker is 'goal' and option.state.is_goal()) or \
               (Configurator.tie_breaker is 'shelter' and agent_state.loc.is_shelter())

    def get_best_move(self):
        best_move, _ = self.minimax(state_node=self.root,
                                    depth=self.env.depth,
                                    a=float('-inf'),
                                    b=float('inf'),
                                    current_player=self.max_player,
                                    is_max=True)
        self.display()      # view the minimax decision tree
        self.restore_env()  # restore the environment to the state it was in before this function was called
        self.env.print_queued_actions("POST %s" % self.max_player.name)

        return best_move.action

    def minimax(self, state_node: Option, depth, a, b, current_player, is_max=True):
        self.nodes.append(state_node)
        state_node.is_max = is_max
        state = state_node.state
        other_player = self.get_other_player(current_player)
        if depth == 0 or state.is_goal():
            final_state = self.get_final_state(state_node)
            utility = self.heuristic(final_state, is_max)
            state_node.state = final_state
            state_node.value = utility
            return state_node, utility

        if is_max:  # MAX player
            value = float('-inf')
            options = self.expand_node(state, current_player)
            choice = None
            for opt in options:
                opt.parent = state_node
                is_coop = self.mode is 'cooperative'
                min_option, min_option_value = self.minimax(opt, depth-1, a, b, other_player, is_max=is_coop)
                temp = max(value, min_option_value)
                if temp > value or (temp == opt.value and self.tie_breaker(current_player, opt, choice)):
                    choice = opt
                    value = temp
                a = max(a, value)
                if a >= b:
                    print("PRUNED: " + state.ID)
                    break
        else:  # MIN player
            value = float('inf')
            options = self.expand_node(state, current_player)
            choice = None
            for opt in options:
                opt.parent = state_node
                max_option, max_option_value = self.minimax(opt, depth-1, a, b, other_player, is_max=True)
                temp = min(value, max_option_value)
                if temp < value or (temp == opt.value and self.tie_breaker(current_player, opt, choice)):
                    choice = opt
                    value = temp
                b = min(b, value)
                if a >= b:
                    print("PRUNED: " + state.ID)
                    break
        state_node.value = value
        return choice, value

    def heuristic(self, state: State, is_max, alternative_mode=None):
        mode = alternative_mode or self.mode
        sign = is_max and 1 or -1
        if mode is 'semi-cooperative':
            agent = is_max and self.max_player or self.min_player
            return sign * self.heuristic_helper(agent, state)
        h1 = self.heuristic_helper(self.max_player, state)
        h2 = self.heuristic_helper(self.min_player, state)
        if mode is 'adversarial':
            return h1 - h2
        if mode is 'cooperative':
            return h1 + h2

    def heuristic_helper(self, agent, state: State):
        """given a state for an max_player, returns how many people can (!) be saved by the max_player"""
        self.env.apply_state(state, active_agent=agent)
        if agent.terminated:
            return agent.n_saved

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
        n_can_save_carrying = num_rescuable_carrying()
        n_can_save_new = sum([v.n_people for v in can_save_nodes])
        total_can_save = n_can_save_carrying + n_can_save_new + n_already_saved
        debug('[#{0}]: {1.name} can save {2} (nodes:{3} (={4}) + rescuable carrying ={5} + saved before={1.n_saved})'
              .format(state.ID, agent, total_can_save, can_save_nodes, n_can_save_new, n_can_save_carrying))
        return total_can_save

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

        if dest == ActionType.TERMINATE: #or (not agent.is_reachable(self.env, dest)):
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
