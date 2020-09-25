from hb.utils.date import get_period_from_str


class RiskFactor(object):
    __slots__ = ["_name", "_data"]

    def set_name(self, name: str):
        self._name = name

    def get_name(self) -> str:
        return self._name

    def name(self, name: str):
        self._name = name
        return self

    def set_data(self, data):
        self._data = data
    
    def get_data(self):
        return self._data
    
    def data(self, data):
        self._data = data
        return self

    def __repr__(self):
        return self._name

class ImpliedVolRiskFactor(RiskFactor):
    __slots__ = ["_strike", "_maturity"]

    def set_name(self, name):
        assert name[:3] == "Vol"
        self._name = name
        str_maturity=name[4:].split("x")[0]
        str_strike=name[4:].split("x")[1]
        self._maturity = get_period_from_str(str_maturity)
        self._strike = float(str_strike)

    def name(self, name):
        self.set_name(name)
        return self

    def get_strike(self):
        return self._strike

    def get_maturity(self):
        return self._maturity

class RiskFactorFactory:
    @staticmethod
    def create_riskfactor(riskfactor_name: str):
        if "Vol" in riskfactor_name:
            return ImpliedVolRiskFactor().name(riskfactor_name)
        else:
            return RiskFactor().name(riskfactor_name)