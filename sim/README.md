# Synthetic Sovereign Hedge — Simulation Stack

## Overview
Implements the Technical Architecture Brief: a stochastic simulation engine that validates
the KES short/FX arbitrage thesis from the White Paper and produces a static interactive
dashboard showing all modelled outcomes.

## Quick Start
```bash
# From this directory:
docker compose up --build
```
Then open **http://localhost:8080** in your browser.

The engine container runs first (≈30–60 seconds for 10,000 Monte Carlo paths), writes
all datasets to a shared Docker volume, then exits. The dashboard container starts after
the engine succeeds and serves the static dashboard.

## Re-running the Engine
If you want to regenerate datasets (e.g., after changing parameters):
```bash
docker compose run --rm engine
docker compose up dashboard
```

## What is Simulated

### Core Assumption Tests (7 tests)
| ID  | Test |
|-----|------|
| T1  | FIC positive cash flow at inception |
| T2  | FIC DSCR ≥ 1.25 (lender covenant) |
| T3  | Class B IRR ≥ 16% Kenyan bond yield |
| T4  | FIC retains positive FX profit after sweeteners |
| T5  | Class C IRR > Class B (risk ordering preserved) |
| T6  | KES debt GBP cost shrinks after devaluation |
| T7  | Capital flight survival: SPV exit > hold in Kenya |

### Deterministic Scenarios (Phase 1)
Four pre-modelled scenarios with smooth linear devaluation paths:
- Slow 20% drop (month 36)
- Medium 35% drop (month 24)
- Severe 50% drop (month 18)
- Catastrophic 60% drop (month 12)

### Monte Carlo (Phase 2 — 10,000 paths)
- Merton Jump-Diffusion FX model (KES/GBP)
- Trigger probability for each tranche (Class A/B/C)
- Investor IRR distribution in GBP
- Pareto frontier: IRR vs duration risk
- Counterfactual: SPV exit vs holding KES in Kenyan bonds

## Dashboard Tabs
| Tab | Content |
|-----|---------|
| Overview | KPI cards + Pareto chart |
| Assumption Tests | Pass/fail table with evidence |
| Deterministic Scenarios | DSCR, cash flow, FX charts |
| Monte Carlo | Trigger probabilities, distributions |
| Pareto Frontier | IRR vs domestic bond scatter |
| IRR Distribution | Histograms for Class B & C |
| Capital Flight | SPV exit vs hold-in-Kenya comparison |
| FX Paths | 200-path Jump-Diffusion fan chart |

## Key Parameters (edit `engine/simulate.py` → `PARAMS`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `kes_gbp_initial_spot` | 166.6 | Initial KES/GBP rate |
| `devaluation_jump_prob_annual` | 30% | Annual probability of a crash jump |
| `jump_severity_mean` | -35% | Mean jump magnitude |
| `property_value_gbp` | £360,000 | Centenary Quay property value |
| `monthly_rent_gbp` | £1,850 | Gross monthly rent |
| `spv_total_raise_kes` | 70,000,000 | Total SPV capital raise |
| `n_paths` | 10,000 | Monte Carlo paths |

## File Structure
```
sim/
├── docker-compose.yml
├── engine/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── simulate.py          ← Main simulation engine
└── dashboard/
    ├── Dockerfile
    ├── nginx.conf
    └── index.html           ← Static dashboard (Plotly.js)
```
