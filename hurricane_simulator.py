import re
from configurator import Configurator
from random import choice as rand_choice
from utils.data_structures import Edge
from environment import Environment, ShelterNode, EvacuateNode, Graph
from agents.minimax_agent import GameAgent


class Simulator:
    """Hurricane evacuation simulator"""

    def __init__(self, mode, depth):
        self.G: Graph = self.get_graph()
        self.env: Environment = Environment(self.G, mode, depth)

    def get_graph(self):
        if Configurator.graph_path is 'random':
            return Configurator.randomize_config()
        return self.parse_graph(Configurator.graph_path)

    def parse_graph(self, path):
        """Parse and create graph from tests file, syntax same as in assignment instructions"""
        num_v_pattern          = re.compile("#N\s+(\d+)")
        shelter_vertex_pattern = re.compile("#(V\d+)\s+D(\d+)\s+S")
        person_vertex_pattern  = re.compile("#(V\d+)\s+D(\d+)\s+P(\d+)")
        edge_pattern           = re.compile("#(E\d+)\s+(\d+)\s+(\d+)\s+W(\d+)")

        shelter_nodes = []
        person_nodes = []
        name_2_node = {}
        n_vertices = 0
        E = []

        with open(path, 'r') as f:
            for line in f.readlines():
                if not line.startswith('#'):
                    continue

                match = num_v_pattern.match(line)
                if match:
                    n_vertices = int(match.group(1))

                match = shelter_vertex_pattern.match(line)
                if match:
                    name, deadline = match.groups()
                    new_node = ShelterNode(name, int(deadline))
                    shelter_nodes.append(new_node)
                    name_2_node[new_node.label] = new_node

                match = person_vertex_pattern.match(line)
                if match:
                    name, deadline, n_people = match.groups()
                    new_node = EvacuateNode(name, int(deadline), int(n_people))
                    person_nodes.append(new_node)
                    name_2_node[new_node.label] = new_node

                match = edge_pattern.match(line)
                if match:
                    name, v1_name, v2_name, weight = match.groups()
                    v1 = name_2_node['V'+v1_name]
                    v2 = name_2_node['V'+v2_name]
                    E.append(Edge(v1, v2, int(weight), name))

        V = person_nodes + shelter_nodes
        if n_vertices != len(V):
            raise Exception("Error: |V| != N")
        return Graph(V, E)

    def init_agents(self):
        shelters = [v for v in self.G.get_vertices() if v.is_shelter()]
        for i in range(0, 2):
            start_vertex = rand_choice(shelters)
            new_agent = GameAgent('A' + str(i), start_vertex)
            self.env.agents.append(new_agent)
            start_vertex.agents.add(new_agent)

    def run_agents(self):
        if self.env.any_agent_available():
            for agent in self.env.agents:
                title = 'T=%d: %s ' % (self.env.time, agent.name)
                self.env.G.display(title + 'PRE')
                agent.act(self.env)
                self.env.G.display(title+'POST')

    def run_simulation(self):
        self.init_agents()
        print('** STARTING SIMULATION **')
        while not self.env.all_terminated():
            print('\nT=%d' % self.env.time)
            self.env.execute_agent_actions()
            self.run_agents()
            self.env.time += 1
        self.env.G.display('Final State: T=' + str(self.env.time))
