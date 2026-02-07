# Agentic AI for Sovereign Risk Assessment  
### An Agent-Based Modelling and Reinforcement Learning Approach to Climate-Related Fiscal Stress

## Overview

This project forms part of my MSc dissertation in Computing and investigates how **agentic artificial intelligence** can be used to model and assess **sovereign risk under climate-related fiscal stress**.

Traditional sovereign risk assessment relies heavily on econometric and statistical models that assume rational behaviour, linear relationships, and static policy responses. However, real-world government decision-making under climate stress is dynamic, adaptive, and influenced by uncertainty, political constraints, and feedback effects.

This project proposes an **Agent-Based Modelling (ABM)** framework combined with **Reinforcement Learning (RL)** to simulate how governments respond to climate-induced fiscal shocks over time. The aim is to explore whether agentic AI systems can provide **richer behavioural insights** and **emergent risk patterns** that complement or challenge conventional econometric approaches.

---

## Research Objectives

The key objectives of this dissertation are:

- To model a sovereign state as an **intelligent decision-making agent** operating under climate-related fiscal stress.
- To simulate **dynamic policy responses** (e.g. taxation, borrowing, spending adjustments) using reinforcement learning.
- To capture **non-linear interactions and emergent behaviour** using an agent-based environment.
- To compare agentic AI outcomes against **baseline econometric or rule-based benchmarks**.
- To evaluate the strengths, limitations, and interpretability of agentic AI in sovereign risk analysis.

---

## Problem Statement

Climate change introduces increasing fiscal uncertainty through:
- Climate disasters and recovery costs
- Revenue volatility
- Debt sustainability pressures
- Policy trade-offs between short-term stability and long-term resilience

Existing sovereign risk models struggle to:
- Represent adaptive policy behaviour
- Capture feedback loops between fiscal decisions and economic conditions
- Simulate long-horizon decision-making under uncertainty

This project addresses these gaps by modelling governments as **learning agents** interacting with a stochastic climate–economic environment.

---

## Methodology Overview

The system is built around three core components:

### 1. Agent-Based Environment
- Represents the macro-fiscal system and climate conditions
- Encodes state variables such as debt levels, GDP growth, climate shocks, and fiscal stress indicators
- Evolves over discrete time steps

### 2. Government Agent
- Acts as a rational but bounded decision-maker
- Observes the environment state
- Selects policy actions (e.g. spending cuts, borrowing, investment)
- Learns optimal strategies via reinforcement learning

### 3. Reinforcement Learning Framework
- Uses reward signals aligned with fiscal sustainability and economic stability
- Trains the agent through repeated simulation episodes
- Enables adaptive behaviour rather than hard-coded rules

---

## Project Structure

The repository is organised to clearly separate **data, environment logic, agents, training, and evaluation**, making the system modular and extensible.

```text
agentic-sovereign-risk/
│
├── data/
│   ├── raw/
│   │   └── climate_fiscal_data.csv
│   ├── processed/
│   │   └── state_variables.csv
│   └── README.md
│
├── environment/
│   ├── __init__.py
│   ├── climate_model.py
│   ├── fiscal_model.py
│   └── sovereign_env.py
│
├── agents/
│   ├── __init__.py
│   ├── government_agent.py
│   └── policy_space.py
│
├── rl/
│   ├── __init__.py
│   ├── reward_function.py
│   ├── q_learning.py
│   └── ppo_agent.py
│
├── training/
│   ├── train_agent.py
│   ├── hyperparameters.yaml
│   └── training_logs/
│
├── evaluation/
│   ├── benchmark_models.py
│   ├── metrics.py
│   └── results_analysis.py
│
├── experiments/
│   ├── scenario_analysis.py
│   └── stress_tests.py
│
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   └── results_visualisation.ipynb
│
├── config/
│   └── config.yaml
│
├── requirements.txt
├── README.md
└── LICENSE
