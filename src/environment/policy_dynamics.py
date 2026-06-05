"""
dynamics.py — Transition functions for the Sovereign Risk Gymnasium environment.

This module contains the economic mechanics of the simulation. Each function corresponds to one part of the government's fiscal environment:

1. generate_growth — AR(1) GDP growth process
2. generate_interest_rate — random-walk interest rate
3. generate_climate_shock — two-stage climate damage model
4. compute_next_debt — government budget constraint (accounting identity)
5. update_adaptation_capital — ND-GAIN readiness capital accumulation

Design principle: every function is pure (no side effects, no global state) and accepts the Gymnasium-managed RNG (np.random.Generator) rather than calling np.random directly. This ensures all randomness is seed-controlled from the environment's reset() call.
"""

from __future__ import annotations

import logging
import math
from typing import Tuple

import numpy as np

from .config import ProfileConfig

logger = logging.getLogger(__name__)


# -----------------------------------------------------------
# 1. GDP Growth Process
# -----------------------------------------------------------

def generate_growth(
    prev_growth: float,
    config: ProfileConfig,
    rng: np.random.Generator,
) -> float:
    """Generate next-period real GDP growth using a calibrated AR(1) process.

    Economic rationale:
    Growth is persistent — a recession year tends to be followed by another weak year — but mean-reverts toward the long-run average (growth_base).
    This is captured by an AR(1) model:

        g(t) = μ(1 − ρ) + ρ·g(t−1) + ε(t)

    where μ = growth_base, ρ = growth_ar1_coef, and ε ~ F(0, σ).

    The innovation ε can be drawn from a normal or Student's t distribution, depending on which fit better during calibration. The t distribution has heavier tails, better capturing the large growth collapses seen during financial crises (−10% in 2009) and COVID (−15% in 2020).

    For Student's t: we draw from standard_t(df) and scale by shock_std.
    The standard_t distribution has unit variance (for df > 2), so this scaling gives the correct calibrated volatility.

    Parameters
    ----------
    prev_growth:
        Last period's real GDP growth rate (%).
    config:
        Profile-specific calibration parameters.
    rng:
        Gymnasium-managed random number generator.

    Returns
    -------
    float: next-period growth rate (%), clipped to [−30, 30] to match the
    winsorisation bounds applied during dataset construction.
    """
    mu = config.growth_base
    rho = config.growth_ar1_coef
    sigma = config.growth_shock_std

    if config.growth_shock_distribution == "t" and config.growth_shock_t_df is not None:
        # Student's t innovation: heavier tails than normal
        shock = rng.standard_t(config.growth_shock_t_df) * sigma
    else:
        # Normal innovation: appropriate for Advanced economies with stable histories
        shock = rng.normal(0.0, sigma)

    # AR(1) update with mean reversion
    new_growth = mu * (1.0 - rho) + rho * prev_growth + shock

    # Clip to match dataset winsorisation (prevents physically impossible values)
    new_growth = float(np.clip(new_growth, -30.0, 30.0))

    if math.isnan(new_growth):
        raise RuntimeError(
            f"NaN in generate_growth: prev_growth={prev_growth}, shock={shock}"
        )

    return new_growth


# ---------------------------------------------------------------------------
# 2. Real Interest Rate Process
# ---------------------------------------------------------------------------

def generate_interest_rate(
    prev_rate: float,
    config: ProfileConfig,
    rng: np.random.Generator,
) -> float:
    """Generate next-period real interest rate via a random walk with drift.

    Economic rationale:
    Real interest rates are highly persistent and do not strongly mean-revert over short horizons. A random walk with drift is the standard representation for nominal/real rates in the fiscal sustainability literature (Blanchard, 2019; IMF WP/19/155).

    The drift (rate_shock_mean) is typically small and positive, reflecting the slow normalisation of rates from the post-GFC low-rate environment. The volatility (rate_shock_std) captures uncertainty around future rate paths.

        r(t) = r(t−1) + ε(t),  ε ~ N(rate_shock_mean, rate_shock_std)

    Parameters
    ----------
    prev_rate:
        Last period's real interest rate (%).
    config:
        Profile-specific calibration parameters.
    rng:
        Gymnasium-managed random number generator.

    Returns
    -------
    float: next-period real interest rate (%), clipped to [−80, 85] to match
    the winsorisation bounds applied during dataset construction.
    """
    shock = rng.normal(config.rate_shock_mean, config.rate_shock_std)
    new_rate = prev_rate + shock
    new_rate = float(np.clip(new_rate, -80.0, 85.0))

    if math.isnan(new_rate):
        raise RuntimeError(
            f"NaN in generate_interest_rate: prev_rate={prev_rate}, shock={shock}"
        )

    return new_rate


# ---------------------------------------------------------------------------
# 3. Climate Shock Process
# ---------------------------------------------------------------------------

def generate_climate_shock(
    config: ProfileConfig,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Generate climate damage and its impact on the primary fiscal balance.

    Economic rationale:
    Climate disasters are rare but potentially large. The generating process has two stages:

    Stage 1 — Event occurrence:
        Each year, a disaster occurs with probability p (Bernoulli draw). This probability is calibrated from the EM-DAT disaster database and ND-GAIN vulnerability scores.

    Stage 2 — Damage magnitude (conditional on event):
        If an event occurs, damage is drawn from a truncated normal distribution with mean = climate_conditional_mean and std = climate_conditional_std.
        The draw is floored at 0 (damage cannot be negative) and capped at climate_max_damage (typically 10% of GDP) to prevent unrealistic extremes.

    Stage 3 — Fiscal impact:
        Only disasters exceeding 1% of GDP trigger a meaningful fiscal response. Below this threshold the government can absorb costs through existing contingency funds without affecting the primary balance. The fiscal cost is scaled proportionally to the damage drawn, anchored to the calibrated climate_fiscal_cost for an average major event.

    Parameters
    ----------
    config:
        Profile-specific calibration parameters.
    rng:
        Gymnasium-managed random number generator.

    Returns
    -------
    Tuple[float, float]:
        (damage_pct_gdp, fiscal_cost)
        - damage_pct_gdp: climate damage as % of GDP (0.0 if no event)
        - fiscal_cost: impact on primary balance in pp of GDP (negative = worsening; 0 if no event or damage < 1%)
    """
    # Stage 1: does an event occur this year?
    event_occurs = rng.random() < config.climate_event_probability

    if not event_occurs:
        return 0.0, 0.0

    # Stage 2: how large is the damage?
    damage = rng.normal(config.climate_conditional_mean, config.climate_conditional_std)
    damage = float(np.clip(damage, 0.0, config.climate_max_damage))

    # Stage 3: what is the fiscal cost?
    # Only triggered when damage exceeds 1% of GDP (below this, normal
    # contingency reserves cover costs without affecting the primary balance).
    # Scale proportionally: larger-than-average events cost proportionally more.
    cond_mean = config.climate_conditional_mean
    if damage > 1.0 and cond_mean > 0:
        fiscal_cost = config.climate_fiscal_cost * (damage / cond_mean)
    else:
        fiscal_cost = 0.0

    return damage, float(fiscal_cost)


# ---------------------------------------------------------------------------
# 4. Debt Accumulation Identity
# ---------------------------------------------------------------------------

def compute_next_debt(
    prev_debt: float,
    growth: float,
    interest_rate: float,
    primary_balance: float,
    stock_flow_shock: float,
) -> float:
    """Compute next-period debt-to-GDP using the government budget constraint.

    This is an accounting identity derived from the government's intertemporal budget constraint (Blanchard, 1990):

        d(t) = [(1 + r/100) / (1 + g/100)] × d(t−1) − pb(t) + sf(t)

    where:
        d   = debt-to-GDP ratio (e.g. 60.0 = 60% of GDP)
        r   = real interest rate (%)
        g   = real GDP growth rate (%)
        pb  = primary balance as % of GDP (positive = surplus)
        sf  = stock-flow adjustment (% GDP)

    The signs are guaranteed by construction:
        Higher growth  (g↑) → lower debt  (denominator increases)
        Higher rate    (r↑) → higher debt (numerator increases)
        Primary surplus (pb > 0) → lower debt (directly reduces stock)
        Stock-flow (sf) → captures privatisations, recapitalisations, etc.

    Note: this identity is in levels (not changes). The ratio r/(1+g) is the "snowball effect". If r > g, debt grows even with a balanced budget.
    This is the mechanism behind the "r−g" literature (Blanchard, 2019).

    Parameters
    ----------
    prev_debt:
        Previous period debt-to-GDP ratio.
    growth:
        Real GDP growth rate (%). A value of 3.0 means 3%.
    interest_rate:
        Real interest rate (%). A value of 5.0 means 5%.
    primary_balance:
        Primary balance as % of GDP. Positive = surplus.
    stock_flow_shock:
        Stock-flow adjustment drawn from N(sf_mean, sf_std).

    Returns
    -------
    float: next-period debt-to-GDP ratio, floored at 0 (debt cannot be negative).

    Raises
    ------
    RuntimeError: if the result is NaN (indicates upstream numerical issue).
    """
    # Guard against growth = -100% (economy would disappear, causing div/zero)
    g_factor = max(1.0 + growth / 100.0, 0.01)
    r_factor = 1.0 + interest_rate / 100.0

    next_debt = (r_factor / g_factor) * prev_debt - primary_balance + stock_flow_shock

    # Debt-to-GDP cannot be negative (a government cannot have "negative debt"
    # in the sense of a creditor to the economy as a whole at the macro level)
    next_debt = max(next_debt, 0.0)

    if math.isnan(next_debt):
        raise RuntimeError(
            f"NaN in compute_next_debt: prev_debt={prev_debt}, g={growth}, "
            f"r={interest_rate}, pb={primary_balance}, sf={stock_flow_shock}"
        )

    return float(next_debt)


# ---------------------------------------------------------------------------
# 5. Adaptation Capital Evolution
# ---------------------------------------------------------------------------

def update_adaptation_capital(
    prev_capital: float,
    investment_action: bool,
    damage: float,
) -> float:
    """Update adaptation capital based on policy choice and climate damage received.

    Adaptation capital is proxied by the ND-GAIN Readiness Index, which measures a country's ability to leverage investments for adaptation actions. It covers economic readiness, governance readiness, and social readiness.

    The dynamic is:
    - Without investment, the index decays slowly (−0.002/year) as infrastructure ages and institutional capacity erodes without maintenance.
    - Active adaptation investment (action 5) adds +0.005/year — a realistic rate consistent with ND-GAIN readiness improvements of ~0.005–0.01/year in countries with sustained national adaptation plans (UNEP, 2023).
    - Climate damage erodes capital (−0.001 × damage_pct_gdp) — major disasters destroy adaptive infrastructure: roads, hospitals, communication networks.

    The index is bounded to [0, 1] (the ND-GAIN readiness scale).

    Parameters
    ----------
    prev_capital:
        Previous period's adaptation capital (ND-GAIN readiness, in [0, 1]).
    investment_action:
        True if the agent chose action 5 (climate adaptation investment).
    damage:
        Climate damage this period as % of GDP.

    Returns
    -------
    float: updated adaptation capital in [0, 1].
    """
    if investment_action:
        delta = +0.005  # investment raises capacity
    else:
        delta = -0.002  # natural decay without active maintenance

    # Damage erodes the infrastructure base
    damage_erosion = -0.001 * damage

    new_capital = prev_capital + delta + damage_erosion

    # Bounded to [0, 1] — ND-GAIN readiness is defined on this scale
    return float(np.clip(new_capital, 0.0, 1.0))
