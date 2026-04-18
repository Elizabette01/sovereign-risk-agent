# Notebook 04 — Feature Engineering: Full Explanation

**MSc Dissertation: *Agentic AI for Sovereign Risk Assessment under Climate-Related Fiscal Stress***

This document explains every decision, computation, and output in `notebooks/04_feature_engineering.ipynb`. It is written as a teaching companion so you understand not just *what* was done but *why* — including the economic theory behind each feature and how it connects to your RL environment.

---

## What the notebook does, in one paragraph

The notebook takes the 51-column master panel (9,059 rows, 196 countries, 1980–2029) and produces a ready-to-train dataset for your Reinforcement Learning agent. It restricts the data to a reliable historical window, fixes data quality issues, engineers 29 new features, normalises the 7 MDP state variables using robust scaling, classifies countries into a 3×3 experimental matrix, and saves four output files.

**Outputs:**

| File | Shape | Purpose |
|---|---|---|
| `master_panel_engineered.csv` | 7,559 × 80 | Full dataset for analysis and validation |
| `rl_training_data.csv` | 7,559 × 19 | Direct input to Gymnasium environment |
| `scaling_parameters.json` | — | Median/IQR per economy type per state variable |
| `calibration_profiles.csv` | 9 × 25 | Starting state distributions for the 9 agent profiles |

---

## Setup

**What it does:** Installs `statsmodels` if not present, imports all libraries, sets pandas display options, ensures the `data/processed/` directory exists, and fixes the working directory so the kernel can find data files regardless of how it was launched.

**Why `statsmodels`:** The HP (Hodrick-Prescott) filter used in Step 5c lives in `statsmodels.tsa.filters.hp_filter`. This is the industry-standard econometric package for time-series filtering.

**Why sort by `iso3, year` immediately:** All subsequent `groupby('iso3').diff()`, `.shift()`, and `.rolling()` operations assume chronological order *within* each country. Sorting once at the start prevents subtle bugs where operations accidentally cross country boundaries.

---

## Step 1: Data Boundary Decisions

These three decisions are made before any feature engineering because they affect every subsequent calculation. They cannot be retrofitted later without re-running the whole pipeline.

### 1a — Year range restriction (1990–2024)

**Decision:** Drop rows before 1990; flag 2025–2029 as projections.

**Why 1990?** The 1980s had very poor data coverage across most variables, especially for developing countries. Including them would introduce a lot of NaNs and selection bias (only the largest/richest countries have 1980s data). The modern sovereign debt crisis literature also generally uses post-1990 data, so restricting to this window aligns the dissertation with standard practice.

**Why flag projections rather than drop them?** The WEO projections (2025–2029) are not observations — they are IMF model forecasts. Using them to compute historical statistics (means, quantiles, rolling averages) would contaminate the calibration. However, they may still be useful for scenario analysis, so they are retained in the dataset with the `is_projection = True` flag. The flag is used throughout the notebook to exclude these rows from any statistical computation.

**Result:** 1,500 rows dropped (pre-1990), 980 projection rows flagged.

### 1b — Fix 4 unclassified countries

**The problem:** Four countries had `economy_type = NaN` in the master panel: ETH (Ethiopia), VEN (Venezuela), WBG (West Bank and Gaza), UVK (Kosovo). Since the entire experimental design depends on economy type, these NaN rows would be excluded from every economy-type-specific computation.

**The assignments and justification:**
- **ETH (Ethiopia)** → `Developing`: Ethiopia is a low-income country, classified as such by the IMF WEO and the World Bank.
- **VEN (Venezuela)** → `Emerging Market`: Despite its economic crisis, Venezuela was historically classified by the IMF as an emerging market economy. Its oil-based economy and middle-income history place it in this category rather than developing.
- **WBG (West Bank and Gaza)** → `Developing`: A fragile, aid-dependent economy with no monetary sovereignty. The IMF and World Bank classify it with developing economies.
- **UVK (Kosovo)** → `Emerging Market`: Kosovo adopted the Euro unilaterally after independence. The IMF WEO classifies it as an emerging and developing European economy, but its European context and institutional setup are closer to emerging market.

### 1c — Fill climate NaNs with zero (2000–2024)

**The conceptual point:** In the EM-DAT database, a country-year only appears if a disaster was recorded. If a country-year does not appear, it means no disaster occurred — not that the data is missing. Therefore, NaN in `climate_disaster_count` within the 2000–2024 coverage window means zero disasters, not unknown. Treating it as missing would bias the rolling average upward (it would average only disaster years, ignoring quiet years).

**Why only within 2000–2024?** Before 2000, the EM-DAT coverage is less comprehensive and NaN may genuinely mean "no data collected." After 2024, EM-DAT data was not available at the time of extraction. Outside this window, NaN retains its meaning of "unknown."

---

## Step 2: Debt Dynamics Features

### 2a — Year-on-year change in debt-to-GDP (`debt_to_gdp_change`)

**Formula:** `debt_to_gdp.diff()` within each country.

**Economic meaning:** The debt-to-GDP ratio is a stock (level), but the RL agent needs to know the *trajectory* — is debt rising or falling? A country with 80% debt-to-GDP but improving by 5pp/year is in a very different position from one with 60% debt but deteriorating by 5pp/year. This feature is the "velocity" of the debt stock.

**MDP connection:** This feature is used in the calibration profiles and as a diagnostic. It is not a direct state variable (the level `debt_to_gdp` is), but the RL environment's transition function will implicitly capture this through the debt accumulation equation.

**Extreme outliers (|change| > 50pp):** These are flagged but retained. Real examples:
- Zimbabwe hyperinflation years (GDP collapse denominator effect)
- Greece 2010-2012 debt crisis (debt reclassification)
- Ireland 2010 (bank bailout)
These are real events, not data errors, and are important for the RL agent to see during training.

### 2b — Interest payments proxy and debt service ratio

**Formula:**
```
interest_payments = -(fiscal_balance - primary_balance)
debt_service_ratio = interest_payments / govt_revenue
```

**Why this identity works:** By definition, the fiscal balance equals the primary balance minus interest payments. Rearranging:
```
interest_payments = primary_balance - fiscal_balance
```
If the primary balance is +2% of GDP and the fiscal balance is -1% of GDP, then interest payments consumed 3% of GDP.

**Debt service ratio:** This is the share of government revenue absorbed by interest payments. It is a critical fiscal stress indicator — a country paying 25% of all revenue just on interest has very little room for discretionary spending or climate adaptation investment.

**MDP connection:** Maps directly to the interest burden component of the reward function.

---

## Step 3: Fiscal Policy Features

### 3a — Fiscal impulse (`fiscal_impulse`)

**Formula:**
- Where `weo_structural_balance` available: `fiscal_impulse = structural_balance.diff()`
- Otherwise: `fiscal_impulse = primary_balance.diff()`

**Why the structural (cyclically-adjusted) balance is preferred:** During a recession, tax revenues automatically fall and spending automatically rises — even if the government does nothing. These "automatic stabilisers" make the primary balance look looser than actual policy. The cyclically-adjusted balance removes this effect, showing only deliberate policy changes.

**Why only 33.5% coverage on structural balance?** The IMF only estimates cyclically-adjusted balances for countries where it has reliable potential output estimates — mostly advanced economies. For developing countries, the primary balance diff is the standard fallback.

**Economic interpretation:** A positive fiscal impulse means fiscal *tightening* (the government is cutting spending or raising taxes, reducing the deficit). A negative impulse means fiscal *loosening* (stimulus). The RL agent's action space spans this spectrum, so understanding historical magnitudes is essential for calibration.

### 3b & 3c — Tax revenue and government expenditure ratios

These are existing columns (`wdi_tax_revenue_gdp`, `wdi_gov_expenditure_gdp`) that require no transformation. They are verified for coverage and used in the calibration profiles to measure fiscal capacity — the government's ability to raise revenue and its spending commitments.

---

## Step 4: Climate Risk Features

### 4a — Climate damage as % of GDP (`climate_damage_gdp_pct`)

**Formula:**
```
climate_damage_gdp_pct = (climate_damage_adj_000usd × 1000) / GDP_in_USD × 100
```

**Unit handling (important):** The raw damage column is in thousands of USD. The WEO GDP column (`weo_gdp_current`) is in *billions of USD* (confirmed by checking USA 2020 = $21,354 billion ≈ $21.4 trillion). The World Bank column (`wdi_gdp_current_usd`) is in actual USD. We use `wdi_gdp_current_usd` as the primary denominator (more reliable currency), with `weo_gdp_current × 1e9` as a fallback.

**Why normalise by GDP?** A $1 billion flood is catastrophic for a small island state (perhaps 50% of GDP) but negligible for the US (0.004% of GDP). Normalising makes the variable scale-free and comparable across the entire panel.

**The top 20 list** should show small island developing states (SIDS) like Haiti, Tonga, and Samoa, plus disaster-prone countries like Honduras and Myanmar. If it shows large countries, there is likely a unit error.

### 4b — 5-year rolling mean of climate damage (`climate_damage_gdp_pct_5yr`)

**Formula:** `rolling(5, min_periods=1).mean()` within each country.

**Why smooth it?** Individual years are dominated by single events (one hurricane can make a country appear 100× more climate-stressed than average). A 5-year rolling mean captures the *underlying trend* in climate exposure — a country's structural climate risk, not just one bad year.

**This is the state variable** `state_climate_shock` in the MDP. The RL agent observes this smoothed version rather than the raw annual damage.

**`min_periods=1`:** Countries with fewer than 5 years of data still get a value (the mean of however many years are available). Without this, countries with data only from 2020 onwards would have NaN for 2020–2023.

### 4c — 5-year rolling mean of disaster frequency (`climate_disaster_freq_5yr`)

The same smoothing applied to the count of disasters per year. While `climate_damage_gdp_pct_5yr` captures financial severity, this captures event *frequency*. Some countries (e.g., Bangladesh) have very frequent small-to-medium disasters; others have rare but catastrophic events.

### 4d — Climate vulnerability × debt interaction (`climate_debt_interaction`)

**Formula:** `ndgain_vulnerability × debt_to_gdp`

**Why this interaction matters (dissertation core):** This is the key theoretical contribution. Consider two countries both with 80% debt-to-GDP:
- Country A: ND-GAIN vulnerability = 0.20 (low risk) → interaction = 16
- Country B: ND-GAIN vulnerability = 0.60 (high risk) → interaction = 48

Country B faces triple the compounding fiscal-climate stress. A climate disaster for Country B simultaneously:
1. Destroys GDP (shrinks the denominator of debt-to-GDP)
2. Forces emergency spending (increases the numerator)
3. Damages infrastructure (reduces future tax capacity)
4. While the country is already highly indebted (limited fiscal space)

This non-linear compounding is what the interaction term captures.

---

## Step 5: Macroeconomic Context Features

### 5a — Implicit interest rate and real interest rate

**Implicit interest rate:**
```
implicit_interest_rate = (interest_payments / debt_to_gdp) × 100
```
This gives the *effective average* rate the government pays on its outstanding debt stock. It differs from the marginal new-issuance rate because the debt stock includes old bonds issued at historical rates. It is capped at [0%, 100%] to remove impossible values caused by very small debt stocks or data inconsistencies.

**Real interest rate:**
```
real_interest_rate = implicit_interest_rate - inflation
```
Fisher equation: the real rate is what matters for debt sustainability, because inflation erodes the real value of the debt stock. A country with 15% nominal rates but 12% inflation has a real rate of only 3%.

### 5b — Interest rate–growth differential (r − g)

**Formula:** `r_minus_g = real_interest_rate - gdp_growth`

**Why this is the most important variable in debt sustainability:**

The debt accumulation equation (simplified) is:
```
Δ(debt/GDP) ≈ (r − g) × debt/GDP − primary_surplus/GDP
```

When r > g: debt-to-GDP rises automatically, even with a zero primary deficit. The government must run a **primary surplus** just to keep debt stable.

When r < g: the economy grows faster than the interest burden. The government can run a **primary deficit** and still have stable debt. This is the "free lunch" scenario Blanchard (2019) famously argued existed for many advanced economies.

This is the single number that determines whether a country's debt is self-stabilising or explosive. It is the `state_risk_premium` variable in your MDP.

**Distribution by economy type:** Advanced economies typically have lower r-g (especially post-2010 with near-zero interest rates and moderate growth). Emerging markets and developing countries have more variable and often higher r-g, reflecting higher borrowing costs and growth volatility.

### 5c — Output gap

**What it measures:** How far is actual GDP from potential (full employment) GDP, expressed as a percentage of potential. A negative gap = recession (output below potential); positive = boom (output above potential).

**Why it matters for fiscal policy:** A government running a deficit during a recession may be doing so entirely due to automatic stabilisers (falling tax receipts, rising welfare spending) rather than active policy loosening. The output gap allows us to decompose observed fiscal balances into cyclical and structural components.

**HP filter (Ravn-Uhlig, 2002):**
- The Hodrick-Prescott filter decomposes a time series into trend (potential) and cycle (gap)
- The smoothing parameter λ = 6.25 is the standard value for annual data (Ravn and Uhlig showed that the standard λ = 1,600 is too high for annual data)
- Applied only to countries with ≥ 15 years of GDP data (shorter series produce unreliable trend estimates)
- Formula: `output_gap = (log(actual) − log(trend)) × 100`

**Coverage:** WEO output gap data is mostly available for advanced economies (13.1% overall). The HP filter extends coverage to ~80%+ of the panel.

---

## Step 6: Lagged Features

All five lag features are computed as `groupby('iso3').shift(n)` — crucially, within each country. Without the groupby, the shift would cross country boundaries (the first row of country B would get the last row of country A), which is a catastrophic data contamination bug.

| Feature | Lag | Purpose |
|---|---|---|
| `debt_to_gdp_lag1` | 1 year | Was debt rising or falling into the current year? |
| `debt_to_gdp_lag2` | 2 years | Multi-year debt trend context |
| `gdp_growth_lag1` | 1 year | Is this year's growth a recovery or continuation? |
| `primary_balance_lag1` | 1 year | Has fiscal policy been consistently tight or loose? |
| `climate_damage_gdp_pct_lag1` | 1 year | Did a climate shock just hit? (recovery period) |

**For the RL environment:** These features are used in calibration rather than in the MDP state vector directly, but understanding them helps validate the environment's transition function. If `debt_to_gdp_lag1` is very close to `debt_to_gdp`, the country has stable debt; large differences indicate volatile fiscal dynamics.

**Countries flagged for short time series (<10 years):** These may be very small territories, newly created states, or countries that only recently joined IMF/World Bank reporting. They are retained but flagged — you will need to decide in the methodology chapter whether to include them in RL training.

---

## Step 7: Climate Risk Tier Classification

**Goal:** Assign each country to Low / Medium / High climate risk tier to enable the 3×3 experimental matrix.

**Variable used:** `ndgain_vulnerability` (ND-GAIN composite of exposure, sensitivity, and adaptive capacity). This is the best available cross-country climate vulnerability index with panel coverage back to 1995.

**Method:** For each country, take the mean vulnerability score over the most recent 5 years of available ND-GAIN data (typically 2019–2023). Then split countries into terciles.

**Why the most recent 5 years?** Climate risk is not static — vulnerability has been rising in most countries. Using the most recent period gives the most policy-relevant classification for the 2025+ simulation period.

**Why terciles over fixed thresholds?** Terciles guarantee exactly 1/3 of countries in each tier. A fixed threshold (e.g., vulnerability > 0.5 = High) might produce 10% High and 70% Low, leaving cells in the matrix severely underpopulated. Balanced cells are important for RL training stability.

**The 3×3 matrix (example from actual run):**
```
                   Low    Medium    High   Total
Advanced            43        14       2      59
Emerging Market     14        34       5      53
Developing           5        13      55      73
Total               62        61      62     185
```

**Key observation:** Advanced economies cluster in Low vulnerability (43 of 59 are Low), while developing economies cluster in High vulnerability (55 of 73 are High). This is the empirical pattern the dissertation is built around — the most climate-vulnerable countries are also the least fiscally resilient.

**Flagged cells:** Cells with <5 countries (e.g., Advanced × High, Developing × Low) are flagged in the notebook output. These represent structurally atypical combinations and may produce unreliable RL training results. This is an important methodological limitation to discuss.

---

## Step 8: Normalisation for RL State Space

**Why normalise at all?** Neural network-based RL agents (DQN, PPO) are sensitive to input scales. If debt-to-GDP ranges from 0–300% but GDP growth ranges from −15% to +15%, the network will implicitly weight debt-to-GDP much more heavily. Normalisation puts all inputs on a comparable scale.

**Why robust scaling over z-score (standard normalisation)?**

Z-score standardisation: `(x − mean) / std`
Robust scaling: `(x − median) / IQR`

In macroeconomic data:
- Greece 2012: debt-to-GDP = 174%. This outlier would pull the mean far from the centre and inflate the standard deviation, compressing the signal for all other countries.
- Robust scaling uses the median (unaffected by outliers) and IQR (the 25th–75th percentile range, also unaffected by extremes).
- The result is that "normal" countries get values near 0, while truly extreme countries get large positive/negative values — which is the right behaviour for an RL agent.

**Why scale within economy type, not globally?**

A debt-to-GDP of 100% means:
- For Japan (Advanced): high but manageable — Japan has financed 200%+ debt domestically for decades
- For Zambia (Developing): near-default territory — market access disappears, IMF intervention likely

If you used global scaling, the same value would map to the same scaled number despite having completely different implications. Economy-type scaling ensures the agent learns within the context appropriate to each economy.

**The 7 state variables and their sources:**

| State Variable | Source Column | What it represents |
|---|---|---|
| `state_output_growth` | `gdp_growth` | Economic momentum |
| `state_debt_to_gdp` | `debt_to_gdp` | Fiscal stock position |
| `state_primary_balance` | `weo_primary_balance` | Current fiscal policy stance |
| `state_interest_rate` | `real_interest_rate` | Cost of debt service |
| `state_climate_shock` | `climate_damage_gdp_pct_5yr` | Climate stress signal |
| `state_adaptation_capital` | `ndgain_readiness` | Capacity to adapt |
| `state_risk_premium` | `r_minus_g` | Debt sustainability margin |

**Scaling parameters file (`scaling_parameters.json`):** Stores the median and IQR for each state variable × economy type combination. This file *must* be used when building the Gymnasium environment to ensure the training and environment observations are on the same scale. If you re-scale in the environment using different parameters, the agent's trained policy will produce incorrect actions.

---

## Step 9: Country Profile Calibration Table

**What it is:** A 9-row table (one per cell in the 3×3 matrix) containing the median and IQR of each state variable for historical observations (2000–2024, excluding projections).

**How it will be used:** When the Gymnasium environment initialises an episode for a given profile (e.g., "Emerging Market / High Climate Risk"), it will draw the starting state from the historical distribution of that profile. This ensures the RL agent trains on realistic starting conditions rather than arbitrary initial values.

**Reading the calibration table:**
- `n_countries`: How many countries contribute data to this cell
- `n_observations`: Country × year observations (= n_countries × ~25 years, roughly)
- `debt_to_gdp_median`: Typical debt level for this profile
- `climate_shock_median`: Typical 5-year climate damage exposure
- `risk_premium_median`: Typical r−g differential

**The stress scenario profile** is identified as the profile with the highest median debt-to-GDP. This is the "worst case" calibration for the dissertation's analysis of climate-fiscal interaction under stress.

---

## Step 10: Final Datasets

### `master_panel_engineered.csv` (7,559 × 80)

The full dataset. The original 51 columns plus all 29 engineered features:
- `is_projection` — boolean
- `debt_to_gdp_change`, `interest_payments`, `debt_service_ratio` — debt dynamics
- `fiscal_impulse_structural`, `fiscal_impulse_simple`, `fiscal_impulse` — fiscal features
- `climate_damage_gdp_pct`, `climate_damage_gdp_pct_5yr`, `climate_disaster_freq_5yr`, `climate_debt_interaction` — climate features
- `implicit_interest_rate`, `real_interest_rate`, `r_minus_g`, `hp_output_gap`, `output_gap` — macro features
- `debt_to_gdp_lag1`, `debt_to_gdp_lag2`, `gdp_growth_lag1`, `primary_balance_lag1`, `climate_damage_gdp_pct_lag1` — lags
- `climate_risk_tier` — classification
- `state_output_growth`, `state_debt_to_gdp`, `state_primary_balance`, `state_interest_rate`, `state_climate_shock`, `state_adaptation_capital`, `state_risk_premium` — normalised state variables

Use this file for: exploratory data analysis, feature validation, dissertation charts, regression analysis.

### `rl_training_data.csv` (7,559 × 19)

The streamlined file for the RL environment. Contains only what the agent needs:
- Identifiers: `iso3`, `year`, `economy_type`, `climate_risk_tier`
- Normalised state variables (7 columns prefixed `state_`)
- Un-normalised source columns (for debugging and economic interpretation)
- `is_projection` flag

Use this file for: building the `gymnasium.Env` class, training DQN/PPO agents.

---

## Key Numbers to Know

| Metric | Value |
|---|---|
| Final dataset rows | 7,559 (1990–2029, 196 countries) |
| Engineered columns added | 29 |
| Projection rows flagged | 980 |
| Climate zeros filled (2000–2024) | varies by column |
| HP filter countries | ~160+ |
| Countries per climate tier | ~62 each |
| Advanced × High vulnerability | 2 countries (methodological flag) |
| Developing × Low vulnerability | 5 countries (borderline) |

---

## What Comes Next (Notebook 05)

The `rl_training_data.csv` and `calibration_profiles.csv` are the direct inputs to:

1. **Gymnasium Environment (`sovereign_risk_env.py`)**: Build a custom `gym.Env` that initialises from calibration profiles, applies the scaling parameters, and implements the MDP transition function and reward function.

2. **DQN Training**: Train a DQN agent on each of the 9 profiles separately, logging the reward curve and policy behaviour.

3. **PPO Training**: Same as above with the PPO algorithm.

4. **Cross-profile analysis**: Compare how the two algorithms perform across the 3×3 matrix, with emphasis on the High Climate Risk × Developing profile as the stress scenario.
