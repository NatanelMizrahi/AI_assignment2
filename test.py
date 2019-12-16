from hurricane_simulator import Simulator
from configurator import Configurator

if __name__ == '__main__':
    Configurator.get_user_config()
    mode = Configurator.mode
    depth = 3
    sim = Simulator(mode, depth)
    sim.run_simulation()
