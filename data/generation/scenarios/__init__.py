"""Injected business scenarios A-J.

Brief section 19 requires the dataset to contain *identifiable* situations, not
just plausible noise. Each scenario is applied at named products, stores and
date windows, and registered in ``ground_truth/scenario_config.json`` with its
expected direction.

That registry does double duty. It is how Step 2's validation confirms each
scenario is actually visible in the data, and it is the seed for the Step 21
golden evaluation set - "why did Product X decline in November?" has a known
correct answer precisely because it was injected here.

One design note that matters more than it looks: **Scenario H is a distribution
loss, not a demand loss.** Stores in the affected region stop stocking the
product, so observed sales fall while underlying demand per stocking store holds.
A root-cause agent that concludes "demand collapsed in North" has got it wrong,
and the data has to be able to prove that.
"""

from data.generation.scenarios.injector import ScenarioInjector, ScenarioRecord

__all__ = ["ScenarioInjector", "ScenarioRecord"]
