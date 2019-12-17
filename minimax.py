from typing import Union, List
from utils.tree import display_tree
from environment import Environment, State, EvacuateNode
from configurator import Configurator, debug
from action import Action, ActionType

inf = float('inf')


class Option:
    """Represents an (action, result-state) node in a Minimax tree"""
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
        self.root: Option = self.get_root_node()

    def sync_env(self):
        """ some of the actions in the environment may not have completed (e.g. agents in transit).
            This function brings the environment to the point that all the agents have finished their prior moves
            before creating the minimax tree. NOT TO BE CONFUSED WITH restore_env()"""
        self.env.execute_all_env_actions()

    def get_initial_state(self):
        return self.env.get_state(self.max_player)

    def get_root_node(self):
        """ First, creates the proper initial state by completing queued actions in the environment.
            Then, creates a root node for the search tree representing the initial state"""
        self.sync_env()
        return Option(Action(self.max_player, description='Dummy action (Tree Root)'), self.get_initial_state())

    def restore_root_state(self):
        """restore environment to the state in the root node - right after completing the actions initially in queue"""
        self.env.apply_state(self.root.state)
        self.env.agent_actions = {}

    def restore_env(self):
        """restore environment to actual state after finding a strategy"""
        self.env.apply_state(self.env_initial_state)

    def get_terminal_state(self, terminal_node):
        """ calculates the terminal state by backtracking the terminal node to get all the actions leading to the
            terminal state and executes them.
            This partially solves the concurrency issue caused by the fact that the environment is inherently NOT
            turn based: one agent can do a single long traverse move while the other can do several shorter moves"""
        path_to_root = self.backtrack(terminal_node)
        all_agents_actions = [option.action for option in path_to_root]

        self.restore_root_state()
        self.env.add_agent_actions(all_agents_actions)
        self.env.execute_all_env_actions()
        return self.env.get_state(self.max_player)

    def get_other_player(self, player):
        return player is self.min_player and self.max_player or self.min_player

    @staticmethod
    def tie_breaker(agent, option, current_best_option):
        debug('Tie: %s VS. %s' % (current_best_option.state.ID, option.state.ID))
        agent_state = agent is option.state.agent and option.state.agent_state or option.state.agent2_state
        return (Configurator.tie_breaker is 'goal' and option.state.is_goal()) or \
               (Configurator.tie_breaker is 'shelter' and agent_state.loc.is_shelter())

    def semi_coop_tie_breaker(self):
        """In case a tie is detected in semi-coop mode rerun Minimax in cooperative mode and return the result"""
        print("Semi-cooperative agent tie-breaker called: returning best cooperative move")
        self.display()  # view the minimax decision tree before making a new cooperative mode
        self.restore_env()
        best_coop_move = MiniMaxTree(self.env, self.max_player, self.min_player, mode='cooperative').get_best_move()
        return best_coop_move

    def get_best_move(self):
        depth = 1 if self.mode == 'semi_cooperative' else self.env.depth
        best_move, _ = self.minimax(state_node=self.root,
                                    depth=depth,
                                    a=-inf, b=inf,
                                    current_player=self.max_player,
                                    is_max=True)

        self.display()          # view the minimax decision tree
        self.restore_env()      # restore the environment to the state it was in before creating the tree
        return best_move

    def minimax(self, state_node: Option, depth, a, b, current_player, is_max=True):
        other_player = self.get_other_player(current_player)
        self.nodes.append(state_node)
        state_node.is_max = is_max
        state = state_node.state

        if depth == 0 or state.is_goal():
            terminal_state = self.get_terminal_state(state_node)
            utility = self.heuristic(terminal_state)
            state_node.state = terminal_state
            state_node.value = utility
            return state_node, utility

        if is_max:  # MAX player
            value = -inf
            max_tie_value = -inf  # keeps track of the maximum value in a tie for semi-coop
            options = self.expand_node(state, current_player)
            choice = None
            for opt in options:
                opt.parent = state_node
                is_coop = self.mode == 'cooperative'
                min_option, min_option_value = self.minimax(opt, depth-1, a, b, other_player, is_max=is_coop)
                temp = max(value, min_option_value)
                tie = (value == opt.value) and (choice is not None)
                if tie:
                    max_tie_value = max(temp, max_tie_value)
                if temp > value or (tie and self.tie_breaker(current_player, opt, choice)):
                    choice = opt
                    value = temp
                a = max(a, value)
                if a >= b:
                    print("PRUNED: " + state.ID)
                    break

            # handle special case for semi_cooperative agents. See doc @semi_coop_tie_breaker()
            if self.mode == 'semi_cooperative' and max_tie_value == value:
                return self.semi_coop_tie_breaker(), None

        else:  # MIN player
            value = inf
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

    def heuristic(self, state: State):
        h1 = self.heuristic_helper(self.max_player, state)
        if self.mode == 'semi_cooperative':
            return h1

        h2 = self.heuristic_helper(self.min_player, state)
        if self.mode == 'adversarial':
            return h1 - h2
        if self.mode == 'cooperative':
            return h1 + h2

    def heuristic_helper(self, agent, state: State):
        """given a state for an max_player, returns how many people can (!) be saved by the max_player"""
        self.env.apply_state(state, active_agent=agent)
        if agent.terminated:
            return agent.n_saved

        def num_rescuable_carrying():
            can_reach_a_shelter = any([self.env.can_reach_before_deadline(v) for v in self.env.get_shelters()])
            return agent.n_carrying if can_reach_a_shelter else 0

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
        neighbours = agent.get_possible_steps(self.env, verbose=False)  # options to proceed
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
                description='*[T={:>3}] {}: TERMINATE'.format(agent.time, agent.name),
                end_time=agent.time,
                callback=terminate_agent)
            agent.local_terminate()
        else:
            def move_agent():
                agent.traverse(self.env, dest)
            action = Action(
                agent=agent,
                action_type=ActionType.TRAVERSE,
                description='*[T={:>3}] {}: TRAVERSE {}->{}'.format(agent.time, agent.name, agent.loc, dest),
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
        if Configurator.skip_strategy:
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
