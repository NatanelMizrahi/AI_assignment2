import argparse
from random import sample
from datetime import datetime
from utils.data_structures import Edge
from environment import ShelterNode, EvacuateNode, Graph


class Configurator:
    """static configurator class"""
    @staticmethod
    def get_user_config():
        parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                         description='''
        Environment simulator for the Hurricane Evacuation Problem
        examples: 
        python test.py -g tests/all.config -L V2 V1 --mode cooperative -K 4
        python test.py -g random --max_neighbors 4 --mode adversarial''')

        parser.add_argument('-g', '--graph_path',
                            # default='random',
                            default='tests/all.config',
                            help='path to graph initial configuration file')

        parser.add_argument('-nn', '--max_neighbors',
                            default=3, type=int,
                            help='for random configurations, limits the number of neighbors for each node')

        parser.add_argument('-K', '--base_penalty',
                            default='2',   type=int,
                            help='base penalty for losing an evacuation vehicle')

        parser.add_argument('-m', '--mode',
                            default='adversarial',
                            choices=['adversarial', 'cooperative', 'semi_cooperative'],
                            help='game mode')

        parser.add_argument('-t', '--tie_breaker',
                            default='goal',  choices=['goal', 'shelter'],
                            help='tie breaker for same value nodes in the minimax tree')

        parser.add_argument('-L', '--agent_locs',
                            default=None, nargs=2,
                            help='agent locations by order. E.g "--agent_locs V1 V3" => initially A1 in V1, A2 in V3')

        # debug command line arguments
        parser.add_argument('-q', '--quiet',
                            default=False,  action='store_true',
                            help='disable debug prints (enabled by default)')

        parser.add_argument('-s', '--skip_strategy',
                            default=False, action='store_true',
                            help='disable plotting search agents strategy trees (enabled by default)')

        args = vars(parser.parse_args())
        for k, v in args.items():
            setattr(Configurator, k, v)
        print("Environment Configured.")

    @staticmethod
    def randomize_config():
        """Randomize an initial configuration for the hurricane evacuation graph until it meets the constraints below"""

        def is_legal_config(G):
            inf = float('inf')
            V = G.get_vertices()
            # Graph cannot be empty
            if not V:
                return False

            # Graph must have at least one shelter node
            shelters = [v for v in G.get_vertices() if v.is_shelter()]
            has_shelter = len(shelters) > 0
            if not has_shelter:
                return False

            # number of neighbours for each vertex must be no more than limit
            num_neighbors = [len(G.neighbours(v)) for v in V]
            if any([n > Configurator.max_neighbors for n in num_neighbors]):
                return False

            # All nodes must be connected
            G.dijkstra(shelters[0])
            all_reachable = all([v.d < inf for v in V])
            if not all_reachable:
                return False

            # all nodes must have some evacuation path (pickup from any shelter + drop-off to another)
            need_evac = [v for v in V if not v.is_shelter() and v.n_people > 0]
            for s1 in shelters:
                G.dijkstra(s1)
                for s2 in shelters:
                    pickup_time = [v.d for v in need_evac]
                    G.dijkstra(s2)
                    dropoff_time = [v.d for v in need_evac]
                    total_time = zip(need_evac, pickup_time, dropoff_time)
                    for v, pickup, dropoff in total_time:
                        if (pickup < v.deadline) and (dropoff < s2.deadline):  # found an evacuation route for v

                            need_evac.remove(v)

            if len(need_evac) > 0:
                # some nodes can't be saved to begin with
                return False

            return True

        def rand_bool(prob=0.5):
            return sample(range(60), 1)[0] < 60 * prob

        def rand_weight(u, v):
            return sample(range(1, min(u.deadline, v.deadline)+1), 1)[0]

        G = Graph()
        while not is_legal_config(G):
            V = []
            E = []
            N = sample(range(4, 8), 1)[0]
            L = ['V{}'.format(i) for i in range(N)]
            D = sample(range(1, 2 * N), N)
            P = sample(range(0, 20), N)

            for node_args in zip(L, D, P):
                node_type = ShelterNode if rand_bool(prob=0.2) else EvacuateNode
                u = node_type(*node_args)
                V.append(u)
                for v in V[:-1]:
                    if rand_bool(prob=0.33):
                        E.append(Edge(u, v, rand_weight(u, v), 'E0'))
            G = Graph(V, E)

        _, Configurator.base_penalty = sample(range(5), 2)
        print('base penalty: {}'.format(Configurator.base_penalty))
        filename = 'tests/{:%d-%m__%H-%M-%S}.config'.format(datetime.now())
        lines = []
        # save the new configuration in file for review/ reuse
        with open(filename, 'w') as config_file:
            lines.append('#N {}'.format(N))
            for v in V:
                if v.is_shelter():
                    lines.append('#{} D{} S'.format(v.label, v.deadline))
                else:
                    lines.append('#{} D{} P{}'.format(v.label, v.deadline, v.n_people))
            for e in E:
                lines.append('#{} {} {} W{}'.format(e.name, e.v1.label[1:], e.v2.label[1:], e.w))
            config_file.write('\n'.join(lines))

        return G


def debug(s):
    if not Configurator.quiet:
        print(s)

