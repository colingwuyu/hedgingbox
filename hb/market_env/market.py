from hb.instrument.instrument import Instrument
from typing import List, Union
import dm_env



class MarketEnv(dm_env.Environment):
    """Market Environment
    """
    def __init__(
            equity_model: str,
            risk_free_rate: float
        ):
        self._equity_model = equity_model
        self._risk_free_rate = risk_free_rate
    
    def add_instruments(self, instruments: Union[Instrument, List[Instrument]]):
        ...

    def calibrate(self):
        ...

    def get_instruments(str_instruments: Union[str, List[str]]) -> Union[Instrument, List[Instrument]]:
        ...

    def use_instruments(str_instruments: Union[str, List[str]]):
        ...
        
    def reset(self):
        ...

    def step(self, action):
        ...

    def observation_spec(self):
        ...

    def action_spec(self):
        ...

    
    

    
