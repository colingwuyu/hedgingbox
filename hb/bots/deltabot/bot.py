from acme import specs
from acme.agents import agent
from hb.bots.deltabot import actor as delta_hedge_actor
from hb.bots import fake_learner


class DeltaHedgeBot(agent.Agent):
    """Delta Hedging Bot.


    """

    def __init__(self,
                 environment_spec: specs.EnvironmentSpec,
                 ):
        """Initialize the delta hedging bot

        Args:
            environment_spec (specs.EnvironmentSpec): description of the actions, observations, etc.
        """
        # Create the actor
        actor = delta_hedge_actor.DeltaHedgeActor(environment_spec.actions)
        learner = fake_learner.FakeLeaner()

        super().__init__(
            actor=actor,
            learner=learner,
            min_observations=100,
            observations_per_step=1e9)
