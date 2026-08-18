from __future__ import annotations

from cais_stability.core import DynamicGovernance
from cais_stability.domains import default_scenarios


def main() -> None:
    governance = DynamicGovernance()
    print("CAIS-Stability smoke verification")
    for scenario in default_scenarios():
        decision = governance.decide(scenario.state(), proposed_action=0.8)
        print(f"{scenario.name}: mode={decision.mode.value}, action={decision.action:.3f}")
    print("CAIS-Stability smoke verification complete")


if __name__ == "__main__":
    main()
