from environment import Environment, EvacuateNode
from configurator import Configurator, debug
from action import Action, ActionType
from copy import copy as shallow_copy
from typing import List


class Agent:
    def __init__(self, name, start_loc: EvacuateNode):
        self.loc: EvacuateNode = start_loc
        self.actions_seq: List[Action] = []
        self.name = name
        self.n_saved = 0
        self.penalty = 0
        self.n_carrying = 0
        self.terminated = False
        self.time = 0
        self.dest = None
        self.eta = 0
        self.goto_str = ''  # used for debug
        print("{}({}) created in {}".format(self.name, self.__class__.__name__, start_loc))

    def is_available(self, env: Environment):
        return (not self.terminated) and (self.time <= env.time)

    def act(self, env: Environment):
        pass

    def get_score(self):
        return self.n_saved - self.penalty

    def get_possible_steps(self, env: Environment, verbose=False):
        possible_steps = env.G.neighbours(self.loc)
        if verbose:
            for i, v in enumerate(possible_steps):
                print('{}. {} -> {}'.format(i, self.loc.label, v.summary().replace('\n', ' ')))
            print('{}. TERMINATE\n'.format(len(possible_steps)))
        return possible_steps

    def is_reachable(self, env: Environment, v: EvacuateNode, verbose=False):
        """returns True iff transit to node v can be finished within v's deadline"""
        e = env.G.get_edge(self.loc, v)
        if self.time + e.w > v.deadline:
            if verbose:
                print('cannot reach {} from {} before deadline. time={} e.w={} deadline={}'
                      .format(v.label, self.loc, self.time, e.w, v.deadline))
            return False
        return True

    def traverse_end_time(self, env, v):
        return self.time + env.G.get_edge(self.loc, v).w

    def traverse(self, env: Environment, v: EvacuateNode):
        """Move max_player, taking transit time into account"""
        self.register_goto_callback(env, v)

    def arrive(self, env: Environment, v: EvacuateNode):
        self.loc.agents.remove(self)
        self.loc = v
        v.agents.add(self)
        self.goto_str = ''
        self.eta = 0
        self.dest = None
        self.try_evacuate(env, v)

    def goto(self, env: Environment, v: EvacuateNode):
        """simulates a traverse operation locally for an agent without updating the environment's entire state"""
        if not self.is_reachable(env, v, verbose=True):
            self.local_terminate()
            return
        self.time = self.traverse_end_time(env, v)
        self.loc = v

    def local_terminate(self):
        """simulates a terminate operation locally for an agent without updating the environment's entire state"""
        self.penalty = self.n_carrying + Configurator.base_penalty  # TODO: fix formula
        self.terminated = True

    def register_goto_callback(self, env: Environment, v):
        if not self.is_reachable(env, v, verbose=True):
            self.terminate(env)
            return

        def goto_node():
            self.arrive(env, v)
        end_time = self.traverse_end_time(env, v)
        goto_action = Action(
            agent=self,
            action_type=ActionType.ARRIVE,
            description='{}: Go from {} to {} (end_time: {})'.format(self.name, self.loc, v.label, end_time),
            callback=goto_node,
            end_time=end_time
        )
        self.dest = v
        self.eta = end_time
        self.goto_str = '->{}'.format(v)
        self.register_action(env, goto_action)

    def try_evacuate(self, env: Environment, v: EvacuateNode):
        if self.terminated:
            return
        if v.is_shelter():
            if self.n_carrying > 0:
                debug('{0.name} Dropped off {0.n_carrying} people'.format(self))
                self.n_saved += self.n_carrying
                self.n_carrying = 0
        elif not v.evacuated:
            debug('{} Picked up {} people'.format(self.name, v.n_people))
            self.n_carrying += v.n_people
            v.evacuated = True
            v.n_people = 0
            env.require_evac_nodes.remove(v)

    def terminate(self, env: Environment):
        if self.n_carrying > 0:
            self.penalty += Configurator.base_penalty + self.n_carrying
        self.terminated = True

        terminate_action = Action(
            agent=self,
            action_type=ActionType.TERMINATE,
            description='Terminating {}. Score = {}'.format(self.name, self.get_score())
        )
        self.register_action(env, terminate_action)

    def register_action(self, env: Environment, action: Action, verbose=True):
        if action.action_type is not ActionType.TERMINATE:  # immediate action - no delay
            env.add_agent_actions([action])
        self.actions_seq.append(action)
        self.time = max(self.time, action.end_time)
        if verbose:
            print('\n[STARTED]' + action.description)

    def summary(self):
        terminate_string = '[${}]'.format(self.get_score()) if self.terminated else ''
        return '{0.name}:{0.loc}{0.goto_str}|S{0.n_saved}|C{0.n_carrying}|T{0.time}'.format(self) + terminate_string

    def get_agent_state(self):
        return shallow_copy(self)

    def update(self, other):
        for k, v in other.__dict__.items():
            setattr(self, k, v)

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        return self.name


