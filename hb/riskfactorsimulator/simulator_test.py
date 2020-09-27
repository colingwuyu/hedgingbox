import unittest
import json
from hb.riskfactorsimulator.simulator import Simulator


class SimulatorTest(unittest.TestCase):
    def test_simulator(self):
        test_json ="""
        {
            "ir": 0.015,
            "equity": [
                {
                    "name": "AMZN",
                    "riskfactors": ["Spot", 
                                    "Vol 3Mx100",
                                    "Vol 2Mx100",
                                    "Vol 4Wx100"],
                    "process_param": {
                        "process_type": "Heston",
                        "param": {
                            "spot": 100,
                            "spot_var": 0.096024,
                            "drift": 0.25,
                            "dividend": 0.0,
                            "kappa": 6.288453,
                            "theta": 0.397888,
                            "epsilon": 0.753137,
                            "rho": -0.696611
                        } 
                    }
                },
                {
                    "name": "SPX",
                    "riskfactors": ["Spot", "Vol 3Mx100"],
                    "process_param": {
                        "process_type": "GBM",
                        "param": {
                            "spot": 100,
                            "drift": 0.10,
                            "dividend": 0.01933,
                            "vol": 0.25
                        } 
                    }
                }
            ],
            "correlation": [
                {
                    "equity1": "AMZN",
                    "equity2": "SPX",
                    "corr": 0.8
                }
            ]
        }
        """
        simulator = Simulator.load_json(test_json)
        simulator.set_time_step(1/360)
        simulator.set_num_steps(90)
        with open("sim.json", 'w') as sim_file:
            sim_file.write(str(simulator))
        test_data = json.loads("""
        [
            {
                "name": "AMZN",
                "data": {
                            "time_step_day": 1,
                            "Spot": [3400.0,3414.063507540342,3360.1097430892696,3514.713081433771,3399.4403346846934,3388.775188349936,3296.0554086124134,3330.74487143777],
                            "Vol 3Mx100": [0.3321,0.3321,0.3321,0.3321,0.3321,0.3321,0.3321,0.3321]
                        }
            }
        ]
        """)
        simulator.load_json_data(test_data)
        print(simulator)

        simulator.generate_paths(100)
        print(simulator.get_spot("AMZN",path_i=0,step_i=1))
        imp_vol_surf = simulator.get_implied_vol_surface("AMZN",path_i=0,step_i=30)
        print(imp_vol_surf.get_black_vol(t=60/360,k=100.))
        import matplotlib.pyplot as plt
        for i in range(100):
            plt.plot(simulator.get_spot("AMZN", path_i=i))
        plt.show()
        for path_i in range(100):
            for step_i in range(90):
                print(simulator.get_implied_vol_surface("AMZN", path_i=path_i, step_i=step_i).get_black_vol(t=90/360-step_i/360,k=100.))
        plt.show()
        print(simulator.get_spot("SPX",path_i=0,step_i=1))
        imp_vol_surf = simulator.get_implied_vol_surface("SPX",path_i=0,step_i=30)
        print(imp_vol_surf.get_black_vol(t=60/360,k=100.))

