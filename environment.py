from utils.data_structures import Node, Edge, Graph
from typing import List, Set, TypeVar, Union
from copy import copy as shallow_copy
from action import Action

AgentType = TypeVar('Agent')
GameMode = Union['Adversarial', 'Cooperative', 'Semi-cooperative']


class EvacuateNode(Node):
    """Represents a node with people that are waiting for evacuation"""
    def __init__(self, label, deadline: int, n_people=0):
        super().__init__(label)
        self.deadline = deadline
        self.n_people = n_people
        self.n_people_initial = self.n_people
        self.evacuated = (n_people == 0)
        self.agents: Set[AgentType] = set([])

    def is_shelter(self):
        return False

    def summary(self):
        return '{}\n(D{}|P{}/{})'.format(self.label, self.deadline, self.n_people, self.n_people_initial)\
               + Node.summary(self)

    def describe(self):
        return self.summary() + '\n' + '\n'.join([agent.summary() for agent in self.agents])


class ShelterNode(EvacuateNode):
    """Represents a node with a shelter"""
    def is_shelter(self):
        return True

    def summary(self):
        return '{}\n(D{})'.format(self.label, self.deadline) + Node.summary(self)

    def describe(self):
        return 'Shelter\n' + super().describe()


class SmartGraph(Graph):
    """A variation of a graph that accounts for edge and node deadlines when running dijkstra"""

    def __init__(self, V: List[Node]=[], E: List[Edge]=[], env=None):
        """:param env: the enclosing environment in which the graph "lives". Used to access the environment's time."""
        super().__init__(V, E)
        self.env = env


class State:
    def __init__(self,
                 agent: AgentType,
                 agent_state: AgentType,
                 agent2: AgentType,
                 agent2_state: AgentType,
                 require_evac_nodes: Set[EvacuateNode]):
        """creates a new state. Inherits env and max_player data, unless overwritten"""
        self.agent = agent
        self.agent_state = agent_state
        self.agent2 = agent2
        self.agent2_state = agent2_state
        self.require_evac_nodes = require_evac_nodes

    def is_goal(self):
        return self.agent_state.terminated or self.agent2_state.terminated

    def describe(self):
        print("State: [{:<30}Evac:{}]"
              .format(self.agent.summary(), self.require_evac_nodes))

    def summary(self):
        return '\n'.join([self.agent_state.summary(),
                          self.agent2_state.summary(),
                          repr(self.require_evac_nodes or [])])


class Environment:
    def __init__(self, G, mode: GameMode='Adverserial', depth=3):
        self.time = 0
        self.G: SmartGraph = G
        self.mode = mode
        self.depth = depth
        self.agents: List[AgentType] = []
        self.require_evac_nodes: Set[EvacuateNode] = self.init_required_evac_nodes()
        self.agent_actions = {}

    def tick(self):
        self.time += 1
        self.execute_agent_actions()

    def all_terminated(self):
        return all([agent.terminated for agent in self.agents])

    def total_unsaved(self):
        return sum([v.n_people for v in self.require_evac_nodes])

    def init_required_evac_nodes(self):
        return set([v for v in self.G.get_vertices() if (not v.is_shelter() and v.n_people > 0)])

    #TODO: convert to pointers
    def get_other_agent(self, agent):
        if agent.name == self.agents[0].name:
            return self.agents[1]
        else:
            return self.agents[0]

    def get_require_evac_nodes(self):
        return shallow_copy(self.require_evac_nodes)

    # def get_agent_actions(self):
    #     return {max_player: max_player.actions_seq for max_player in self.agents}

    def add_agent_actions(self, agent_actions_to_add):
        for action in agent_actions_to_add:
            if action.end_time not in self.agent_actions:
                self.agent_actions[action.end_time] = []
            self.agent_actions[action.end_time].append(action)

    def execute_agent_actions(self):
        queued_actions = self.agent_actions.get(self.time)
        if queued_actions:
            for action in queued_actions:
                print('[EXECUTING]' + action.description)
                action.execute()
            del self.agent_actions[self.time]

    def get_state(self, agent: AgentType):
        return State(
            agent,
            agent.get_agent_state(),
            self.get_other_agent(agent),
            self.get_other_agent(agent).get_agent_state(),
            self.get_require_evac_nodes()
        )

    def apply_state(self, state: State):
        """applies a state to the environment, in terms of the max_player's state variables & node evacuation status"""
        agent, to_copy = state.agent, state.agent_state
        agent2, to_copy2 = state.agent2, state.agent2_state
        agent.update(to_copy)
        agent2.update(to_copy2)
        self.time = agent.time #TODO not correct
        self.require_evac_nodes = shallow_copy(state.require_evac_nodes)
        for v in self.G.get_vertices():
            v_requires_evac = v in self.require_evac_nodes
            v.evacuated = not v_requires_evac
            v.n_people = v.n_people_initial if v_requires_evac else v.n_people


class Option:
    def __init__(self,
                 action: Action,
                 state: State):
        self.action = action
        self.state = state
        self.value = None
        self.parent: Option = None

    def summary(self):
        return '\n'.join([self.state.summary(), 'h={}'.format(self.value)])
