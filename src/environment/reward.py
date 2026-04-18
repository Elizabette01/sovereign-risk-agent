"""
reward.py — Multi-objective reward function for the Sovereign Risk environment.

Reward design philosophy:
The agent must balance five competing objectives that real-world fiscal
policymakers face: debt sustainability, affordable borrowing costs, smooth
adjustment paths, output stability, and climate resilience. Each is captured
by a reward component. This multi-objective structure means the agent cannot
just "pick one objective and maximise it" — it must trade off between them,
just as real governments do.

By returning both the scalar total reward and a component breakdown, we enable
interpretability analysis: we can understand *why* an agent chose a particular
policy by examining which components dominate its decisions.

References:
- Ghosh et al. (2013) "Fiscal Fatigue, Fiscal Space, and Debt Sustainability"
  — justification for quadratic debt penalty structure.
- Alesina et al. (2019) "Effects of Austerity: Expenditure vs Tax Adjustments"
  — adjustment costs and political constraints.
- Blanchard and Leigh (2013) "Growth Forecast Errors and Fiscal Multipliers"
  — stimulus costs during recessions.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

from .config import ProfileConfig

logger = logging.getLogger(__name__)


def compute_reward(
    state: dict,
    prev_state: dict,
    action: int,
    config: ProfileConfig,
) -> Tuple[float, Dict[str, float]]:
    """Compute the multi-objective reward for one fiscal year.

    The reward penalises high debt, interest burden, large policy swings,
    output volatility, and climate damage. It rewards debt reduction and
    positive economic growth.

    Parameters
    ----------
    state:
        Current period's raw state dict. Expected keys:
        'growth', 'debt', 'primary_balance', 'interest_rate',
        'climate_damage', 'adaptation_capital', 'risk_premium'.
    prev_state:
        Previous period's raw state dict (same keys).
    action:
        Integer action chosen by the agent (0–5).
    config:
        Profile-specific calibration parameters.

    Returns
    -------
    Tuple[float, Dict[str, float]]:
        (total_reward, component_dict)
        - total_reward: scalar reward signal for the RL agent.
        - component_dict: named breakdown of each reward component.
          This is logged per step and used for interpretability analysis.
    """
    debt = state["debt"]
    growth = state["growth"]
    interest_rate = state["interest_rate"]
    primary_balance = state["primary_balance"]
    climate_damage = state["climate_damage"]

    prev_debt = prev_state["debt"]
    prev_pb = prev_state["primary_balance"]

    # ------------------------------------------------------------------
    # COMPONENT 1: Debt penalty (quadratic above the Maastricht threshold)
    #
    # Rationale: a linear penalty treats each additional percentage point of
    # debt equally. But fiscal stress is convex — going from 80% to 100%
    # is much more dangerous than going from 40% to 60%, because market
    # confidence can collapse suddenly at high debt levels (Ghosh et al.,
    # 2013 "fiscal fatigue"). The quadratic form captures this non-linearity.
    # Threshold: 60% (EU Maastricht criterion; IMF benchmark for advanced economies).
    # ------------------------------------------------------------------
    debt_threshold = 60.0
    if debt > debt_threshold:
        debt_penalty = -0.01 * (debt - debt_threshold) ** 2
    else:
        debt_penalty = 0.0

    # ------------------------------------------------------------------
    # COMPONENT 2: Interest burden penalty
    #
    # Rationale: when interest payments crowd out spending on public goods,
    # the government loses fiscal space. We penalise the share of revenue
    # consumed by interest payments above a 15% threshold. This is consistent
    # with IMF (2016) recommendations that interest/revenue above 15% signals
    # fiscal stress. The proxy for revenue is the primary deficit + 30
    # (a rough approximation of government revenue as a share of GDP).
    # ------------------------------------------------------------------
    interest_burden = max(0.0, interest_rate * debt / 100.0)  # ~ interest payments % GDP
    revenue_proxy = abs(config.initial_primary_balance) + interest_burden + 30.0
    interest_share = interest_burden / max(revenue_proxy, 1.0)
    interest_penalty = -5.0 * max(0.0, interest_share - 0.15)

    # ------------------------------------------------------------------
    # COMPONENT 3: Fiscal adjustment cost penalty
    #
    # Rationale: large year-on-year fiscal swings are costly. Large austerity
    # packages depress aggregate demand and carry political costs (Alesina et
    # al., 2019). Large stimulus packages can create future fiscal pressures.
    # The penalty is proportional to the size of the change, not its direction —
    # both sharp austerity and sharp stimulus are penalised.
    # A 5pp swing (severe austerity) costs −2.5 in reward.
    # ------------------------------------------------------------------
    pb_change = abs(primary_balance - prev_pb)
    adjustment_penalty = -0.5 * pb_change

    # ------------------------------------------------------------------
    # COMPONENT 4: Output instability penalty
    #
    # Rationale: growth volatility creates welfare losses through uncertainty
    # (Ramey & Ramey, 1995). We penalise deviation from trend growth (growth_base),
    # not the level of growth itself. A deep recession is bad; equally, a boom
    # that is unsustainably above trend creates vulnerabilities.
    # ------------------------------------------------------------------
    growth_deviation_sq = (growth - config.growth_base) ** 2
    output_penalty = -0.1 * growth_deviation_sq

    # ------------------------------------------------------------------
    # COMPONENT 5: Climate damage penalty
    #
    # Rationale: direct welfare loss from unmitigated climate damage.
    # Scaled so that a 5% of GDP climate event costs −10 in reward, roughly
    # comparable to the cost of 20pp of excess debt. This calibration means
    # the agent has a meaningful incentive to invest in adaptation (action 5),
    # but adaptation is not overwhelmingly dominant relative to fiscal concerns.
    # ------------------------------------------------------------------
    climate_penalty = -2.0 * climate_damage

    # ------------------------------------------------------------------
    # COMPONENT 6: Debt reduction reward
    #
    # Rationale: moving toward fiscal sustainability should be rewarded, not
    # just penalising bad outcomes. This encourages gradual consolidation over
    # the episode rather than only reacting to crises. Only positive progress
    # (actual debt reduction) is rewarded; debt increases receive no "anti-reward"
    # beyond the debt level penalty above.
    # ------------------------------------------------------------------
    debt_improvement = prev_debt - debt
    reduction_reward = 0.5 * debt_improvement if debt_improvement > 0.0 else 0.0

    # ------------------------------------------------------------------
    # COMPONENT 7: Growth reward
    #
    # Rationale: economic growth improves welfare and makes debt sustainable
    # (through the r−g mechanism). Positive growth is rewarded; negative growth
    # adds no penalty here (already captured in output_penalty above).
    # ------------------------------------------------------------------
    growth_reward = 0.3 * max(growth, 0.0)

    # ------------------------------------------------------------------
    # TOTAL
    # ------------------------------------------------------------------
    total = (
        debt_penalty
        + interest_penalty
        + adjustment_penalty
        + output_penalty
        + climate_penalty
        + reduction_reward
        + growth_reward
    )

    # Apply optional profile-level scaling (default 1.0)
    total *= config.reward_scale

    components = {
        "debt_penalty": debt_penalty * config.reward_scale,
        "interest_penalty": interest_penalty * config.reward_scale,
        "adjustment_penalty": adjustment_penalty * config.reward_scale,
        "output_penalty": output_penalty * config.reward_scale,
        "climate_penalty": climate_penalty * config.reward_scale,
        "reduction_reward": reduction_reward * config.reward_scale,
        "growth_reward": growth_reward * config.reward_scale,
    }

    return float(total), components
