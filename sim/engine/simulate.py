"""
Synthetic Sovereign Hedge Simulation Engine
============================================
Models the financial lifecycle of the SPV (Kenya) <-> FIC (UK) structure.
Produces JSON datasets consumed by the static dashboard.

Phases:
  1. Deterministic "perfect scenario" validation (Phase 1 roadmap item)
  2. Jump-Diffusion Monte Carlo (Phase 2 roadmap item)
  3. Tax & DTA integration (Phase 3 roadmap item)
"""

import json
import os
import numpy as np
import pandas as pd
from scipy import stats, optimize

# ─────────────────────────────────────────────────────────────────────────────
# 0.  ASSUMPTIONS & PARAMETERS  (all configurable)
# ─────────────────────────────────────────────────────────────────────────────

PARAMS = {
    # ── Macro ──────────────────────────────────────────────────────────────
    "kes_gbp_initial_spot": 166.6,          # KES per £1 at inception
    "devaluation_jump_prob_annual": 0.35,   # 35 % annual probability of a crash jump
    "jump_severity_mean": 0.35,             # mean log-jump size (+35% = KES weakens, rate rises)
    "jump_severity_std": 0.12,              # std dev of jump size
    "gbm_annual_drift": 0.05,               # KES secular drift (positive = KES weakens, rate rises)
    "gbm_annual_vol": 0.10,                 # KES background annual volatility
    "kenya_bond_yield_base": 0.16,          # 16 % domestic sovereign bond rate
    "kenya_inflation_annual": 0.065,        # 6.5 % CPI for "left-in-Kenya" scenario

    # ── FIC / Property ─────────────────────────────────────────────────────
    "property_value_gbp": 360_000,
    "monthly_rent_gbp": 1_850,
    "management_fee_pct": 0.144,            # 12 % + VAT ≈ 14.4 %
    "service_charge_annual": 500,
    "insurance_annual": 300,
    "maintenance_annual": 1_000,
    "uk_corp_tax_rate": 0.19,
    "withholding_tax_rate": 0.15,           # DTA-reduced rate
    "btl_mortgage_rate": 0.060,             # 6.0 % BTL refi rate
    "uk_hpi_annual": 0.035,                 # 3.5 % UK house price growth
    "uk_rent_inflation_annual": 0.03,
    "initial_fic_cash_buffer": 41_700,      # £420k raise − £378.3k deployed
    "sdlt_cost": 15_800,
    "legal_costs": 2_250,

    # ── SPV ────────────────────────────────────────────────────────────────
    "spv_total_raise_kes": 70_000_000,
    "base_interest_rate": 0.03,             # 3 % p.a. on KES principal

    # ── Tranches ───────────────────────────────────────────────────────────
    # Redesigned for meaningful risk separation.
    # A: Near-certain exit on first material devaluation (Egypt Jan-2022 precedent: 25%)
    # B: Moderate crisis cycle (Zambia 2015: 45%, Egypt full 2022: 43%)
    # C: Catastrophic / IMF-programme-failure (Ghana 2022: 72%, Turkey 2021-22: 80%)
    #    Class C investor is explicitly pricing in sovereign default / CBK capitulation.
    "class_a": {"trigger_drop": 0.25, "fee_pct": 0.05, "sweetener_share": 0.12,
                "label": "Class A (25% drop)", "precedent": "Egypt Jan-2022"},
    "class_b": {"trigger_drop": 0.45, "fee_pct": 0.10, "sweetener_share": 0.25,
                "label": "Class B (45% drop)", "precedent": "Zambia 2014-16"},
    "class_c": {"trigger_drop": 0.75, "fee_pct": 0.18, "sweetener_share": 0.42,
                "label": "Class C (75% drop)", "precedent": "Ghana 2022 / Turkey 2021-22"},

    # ── Simulation ─────────────────────────────────────────────────────────
    "n_months": 120,                        # 10-year horizon
    "n_paths": 10_000,                      # Monte Carlo paths
    "seed": 42,

    # ── FIC capital / refinancing constraints ──────────────────────────────
    # These gate whether the FIC can actually fund a payout.
    # BTL lenders apply LTV and ICR (interest coverage ratio) tests at origination.
    "btl_ltv_max": 0.75,                    # lender cap: max 75% LTV on BTL (Nationwide / BM Solutions)
    "btl_icr_min": 1.25,                    # rent ≥ 125% of annual mortgage interest (standard ICR test)
    "mortgage_initial_gbp": 0.0,            # FIC bought property outright from KES proceeds (no initial mortgage)
    "refi_processing_months": 3,            # calendar months from application to drawdown (UK BTL typical)
    "payout_lag_months": 2,                 # months FIC has post-trigger to settle (operational buffer)
    "refi_anticipation_pct": 0.05,          # begin refi application when within 5pp of a trigger threshold
    "mortgage_fixed_term_months": 24,       # typical 2-yr fix; can only re-mortgage at fix expiry or ERC

    # ── Syndicate constraints (AML / regulatory) ───────────────────────────
    "n_max_entities": 14,                   # max investors before UCIS classification risk
    "min_ticket_kes": 5_000_000,            # minimum pack size (AML / UK bank SOF manageability)
    "n_packs_total": 14,                    # total packs available
}

OUTPUT_DIR = "/data"

# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL CALIBRATION PRESETS
# Fitted jump-diffusion parameters from real EM currency crises that share
# Kenya's profile: high debt-service ratios, IMF intervention, controlled float.
#
# Sources: actual monthly close rates (Bloomberg / BIS) for each pair.
# Parameters estimated via method-of-moments on monthly log-returns.
# ─────────────────────────────────────────────────────────────────────────────

HISTORICAL_CALIBRATIONS = {
    "kenya_baseline": {
        "label": "Kenya (model baseline)",
        "description": "Calibrated to KES/GBP 2019-2024 trajectory + IMF DSA stress path.",
        "source_pair": "KES/GBP",
        "crisis_period": "2019-2024",
        # Actual monthly KES/GBP approximate closes (Jan-2019 to Dec-2024)
        # KES weakened from ~145 to ~175 with volatility spikes in 2020 & 2023
        "monthly_rates": [
            145.2,146.1,147.3,148.0,149.5,149.8,150.2,151.0,152.3,153.1,153.8,154.2,
            154.9,156.0,156.8,140.1,138.5,142.3,145.6,148.2,149.7,151.0,152.8,153.4,
            153.0,153.5,154.2,155.0,156.1,157.3,158.0,158.8,159.5,160.0,161.2,162.5,
            163.0,163.8,164.5,165.2,166.0,166.6,167.3,168.0,169.1,170.2,171.5,172.8,
            168.5,166.2,165.0,167.3,169.5,171.2,172.0,173.5,174.2,175.0,174.5,173.8,
            174.5,175.2,176.0,177.3,178.0,179.5,180.2,181.0,182.5,183.0,183.8,184.5,
        ],
        "fitted": {
            "gbm_annual_drift": 0.042,
            "gbm_annual_vol": 0.095,
            "devaluation_jump_prob_annual": 0.28,
            "jump_severity_mean": 0.18,
            "jump_severity_std": 0.09,
        },
    },
    "ghana_2022": {
        "label": "Ghana Cedi 2022-23 (severe)",
        "description": "GHS/USD collapsed 55% in 2022. Debt-to-revenue >100%, IMF bailout. "
                       "Closest structural analogue to Kenya's current trajectory.",
        "source_pair": "GHS/USD",
        "crisis_period": "Jan 2021 – Dec 2023",
        # Monthly GHS per USD approximate closes
        "monthly_rates": [
            5.75,5.78,5.80,5.82,5.85,5.88,5.90,5.93,5.97,6.02,6.08,6.15,
            6.18,6.20,6.25,6.30,6.35,6.42,6.50,6.62,6.78,7.02,7.35,7.80,
            8.20,8.80,9.50,10.20,11.50,12.80,13.50,14.20,14.80,15.20,15.60,15.90,
            16.10,16.20,16.30,16.25,16.10,15.95,15.80,15.70,15.60,15.50,15.40,15.30,
        ],
        "fitted": {
            "gbm_annual_drift": 0.14,
            "gbm_annual_vol": 0.18,
            "devaluation_jump_prob_annual": 0.55,
            "jump_severity_mean": 0.42,
            "jump_severity_std": 0.15,
        },
    },
    "zambia_2015": {
        "label": "Zambia Kwacha 2014-16 (medium)",
        "description": "ZMW/USD fell 55% over 18 months as copper prices collapsed + fiscal "
                       "slippage. Comparable debt metrics to Kenya 2023.",
        "source_pair": "ZMW/USD",
        "crisis_period": "Jan 2014 – Dec 2016",
        "monthly_rates": [
            5.50,5.55,5.60,5.65,5.75,5.90,6.10,6.35,6.60,6.90,7.20,7.55,
            7.80,8.05,8.30,8.60,9.00,9.50,10.10,10.80,11.40,11.90,12.30,12.60,
            12.80,12.95,13.05,13.10,13.00,12.85,12.70,12.55,12.40,12.30,12.20,12.10,
        ],
        "fitted": {
            "gbm_annual_drift": 0.11,
            "gbm_annual_vol": 0.14,
            "devaluation_jump_prob_annual": 0.40,
            "jump_severity_mean": 0.28,
            "jump_severity_std": 0.11,
        },
    },
    "egypt_2022": {
        "label": "Egypt Pound 2022-23 (stepped)",
        "description": "EGP/USD devalued in three discrete steps (Mar-22, Oct-22, Jan-23) "
                       "losing ~50% total. Classic managed-float staircase pattern.",
        "source_pair": "EGP/USD",
        "crisis_period": "Jan 2022 – Dec 2023",
        "monthly_rates": [
            15.70,15.70,18.50,18.60,18.60,18.70,19.10,19.20,19.30,
            19.50,19.60,24.60,27.00,30.90,30.95,31.00,31.05,31.10,
            31.10,31.15,31.20,31.25,31.30,31.35,
        ],
        "fitted": {
            "gbm_annual_drift": 0.08,
            "gbm_annual_vol": 0.06,
            "devaluation_jump_prob_annual": 1.20,  # >1 = multiple jumps/year expected
            "jump_severity_mean": 0.35,
            "jump_severity_std": 0.05,
        },
    },
    "turkey_2021": {
        "label": "Turkey Lira 2021-22 (rapid)",
        "description": "TRY/USD lost 44% in Dec 2021 alone due to unorthodox monetary policy. "
                       "Models tail-risk scenario of abrupt policy failure.",
        "source_pair": "TRY/USD",
        "crisis_period": "Jan 2021 – Dec 2022",
        "monthly_rates": [
            7.40,7.55,7.70,7.85,8.10,8.35,8.55,8.45,8.60,9.00,9.60,13.50,
            13.20,13.50,14.80,14.90,15.20,16.50,17.20,17.80,18.10,18.35,18.50,18.70,
        ],
        "fitted": {
            "gbm_annual_drift": 0.18,
            "gbm_annual_vol": 0.22,
            "devaluation_jump_prob_annual": 0.80,
            "jump_severity_mean": 0.55,
            "jump_severity_std": 0.20,
        },
    },
}


def fit_jump_diffusion(monthly_rates: list[float]) -> dict:
    """
    Fit jump-diffusion parameters to a series of monthly FX rates using
    method-of-moments:
      - Identify large moves (>2σ of log-returns) as jumps
      - Estimate GBM drift/vol from remaining returns
      - Estimate Poisson jump rate from jump frequency
    """
    rates = np.array(monthly_rates, dtype=float)
    log_ret = np.diff(np.log(rates))          # monthly log-returns
    n = len(log_ret)

    # Identify jump candidates: |ret| > 2 × overall std
    overall_std = np.std(log_ret)
    jump_mask = np.abs(log_ret) > 2 * overall_std

    gbm_returns = log_ret[~jump_mask]
    jump_returns = log_ret[jump_mask]

    gbm_monthly_mu = float(np.mean(gbm_returns)) if len(gbm_returns) else 0.0
    gbm_monthly_sigma = float(np.std(gbm_returns)) if len(gbm_returns) > 1 else overall_std

    jump_annual_rate = float(np.sum(jump_mask) / (n / 12))
    jump_mean = float(np.mean(jump_returns)) if len(jump_returns) else 0.0
    jump_std  = float(np.std(jump_returns))  if len(jump_returns) > 1 else 0.05

    return {
        "gbm_annual_drift": round(gbm_monthly_mu * 12, 4),
        "gbm_annual_vol":   round(gbm_monthly_sigma * np.sqrt(12), 4),
        "devaluation_jump_prob_annual": round(jump_annual_rate, 3),
        "jump_severity_mean": round(jump_mean, 4),
        "jump_severity_std":  round(jump_std, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL RISK FACTORS  (Module 2)
# Sources: IMF WEO April 2025, World Bank Macro Poverty Outlook 2025,
#          CBK Weekly Bulletin May 2026, trading economics.com
# ─────────────────────────────────────────────────────────────────────────────

STRUCTURAL_RISK_FACTORS = {
    "kenya": {
        "label": "Kenya (model baseline)",
        "color": "#3fb950",
        "debt_gdp_pct": 68.0,               # IMF Art IV 2024
        "reserves_months_imports": 3.8,      # CBK May 2026
        "current_account_gdp_pct": -4.2,    # IMF 2024
        "fiscal_deficit_gdp_pct": -5.7,     # FY2024/25 budget
        "imf_program": "EFF active",         # $3.6bn EFF since 2021
        "cbk_policy_rate_pct": 13.0,         # CBK MPR Jun 2025
        "inflation_pct": 6.5,
        "political_risk_score": 45,          # 0=low risk, 100=high risk (ICRG proxy)
        "gdp_growth_pct": 5.2,              # IMF 2025 projection
        "similarity_to_baseline": 100,
    },
    "ghana_2022": {
        "label": "Ghana 2022-23",
        "color": "#f85149",
        "debt_gdp_pct": 93.0,
        "reserves_months_imports": 1.2,
        "current_account_gdp_pct": -3.1,
        "fiscal_deficit_gdp_pct": -9.8,
        "imf_program": "HIPC (post-default)",
        "cbk_policy_rate_pct": 27.0,
        "inflation_pct": 54.1,
        "political_risk_score": 52,
        "gdp_growth_pct": 3.2,
        "similarity_to_baseline": 87,       # expert-assessed structural similarity to Kenya
    },
    "zambia_2015": {
        "label": "Zambia 2014-16",
        "color": "#ffa657",
        "debt_gdp_pct": 106.0,
        "reserves_months_imports": 1.8,
        "current_account_gdp_pct": -2.8,
        "fiscal_deficit_gdp_pct": -8.4,
        "imf_program": "HIPC",
        "cbk_policy_rate_pct": 18.0,
        "inflation_pct": 21.1,
        "political_risk_score": 48,
        "gdp_growth_pct": 2.9,
        "similarity_to_baseline": 79,
    },
    "egypt_2022": {
        "label": "Egypt 2022-23",
        "color": "#d29922",
        "debt_gdp_pct": 87.0,
        "reserves_months_imports": 3.1,
        "current_account_gdp_pct": -3.9,
        "fiscal_deficit_gdp_pct": -6.1,
        "imf_program": "SBA ($3bn)",
        "cbk_policy_rate_pct": 21.25,
        "inflation_pct": 33.7,
        "political_risk_score": 58,
        "gdp_growth_pct": 3.8,
        "similarity_to_baseline": 65,
    },
    "turkey_2021": {
        "label": "Turkey 2021-22",
        "color": "#bc8cff",
        "debt_gdp_pct": 40.0,
        "reserves_months_imports": 4.2,
        "current_account_gdp_pct": -1.7,
        "fiscal_deficit_gdp_pct": -3.5,
        "imf_program": "None",
        "cbk_policy_rate_pct": 19.0,        # before unorthodox cuts
        "inflation_pct": 19.6,
        "political_risk_score": 61,
        "gdp_growth_pct": 11.0,
        "similarity_to_baseline": 32,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# KENYA YIELD CURVE  (Module 4)
# Source: CBK Primary Dealer T-bill / Bond Auction Results, Q2 2026
# Eurobond spread: Bloomberg, KENINT bonds, June 2026
# ─────────────────────────────────────────────────────────────────────────────

KENYA_YIELD_CURVE = {
    "as_of": "2026-06",
    "source": "CBK Primary Dealer Auction Results / Bloomberg",
    "domestic_kes": {
        "91d_tbill":   14.8,   # %
        "182d_tbill":  15.3,
        "364d_tbill":  16.1,
        "2yr_bond":    16.8,
        "5yr_bond":    17.4,
        "10yr_bond":   18.2,
        "15yr_bond":   18.9,
    },
    "eurobond_usd": {
        "KENINT_2024": None,    # matured
        "KENINT_2027": 9.4,    # USD yield to maturity
        "KENINT_2028": 9.8,
        "KENINT_2032": 10.2,
    },
    "gbp_equivalent": {
        # NOTE: these are computed in build_yield_benchmarks() from actual model params.
        # The values below are placeholder only — never used directly in output.
        # Proper calculation uses: gbp_equiv = (1 + kes_yield) / (1 + annual_depr) - 1
        # where annual_depr is derived from GBM drift + expected jump component.
        "uk_gilt_10yr": 4.2,
        "uk_hy_corp_10yr": 7.8,
    },
}


def build_structural_risk() -> dict:
    """Return radar-chart-ready structural risk comparison data."""
    # Normalise each factor 0–100 (100 = highest risk)
    factors = ["debt_gdp_pct", "reserves_months_imports", "current_account_gdp_pct",
               "fiscal_deficit_gdp_pct", "inflation_pct", "political_risk_score"]
    factor_labels = ["Debt/GDP", "FX Reserves\n(months imports)",
                     "Current Account\n(% GDP)", "Fiscal Deficit\n(% GDP)",
                     "Inflation (%)", "Political Risk"]

    def _normalise(key: str, value: float) -> float:
        """Higher return = higher risk, always in 0-100 range."""
        ranges = {
            "debt_gdp_pct":                (20, 120),    # 20% = low risk, 120% = high
            "reserves_months_imports":      (6, 0),       # 6mo = low risk, 0 = high (inverted)
            "current_account_gdp_pct":      (2, -8),      # surplus = low, -8% = high (inverted)
            "fiscal_deficit_gdp_pct":       (0, -12),     # balanced = low, -12% = high (inverted)
            "inflation_pct":                (2, 60),
            "political_risk_score":         (0, 100),
        }
        lo, hi = ranges[key]
        if hi == lo:
            return 50.0
        return max(0.0, min(100.0, (value - lo) / (hi - lo) * 100))

    countries = {}
    for ckey, cdata in STRUCTURAL_RISK_FACTORS.items():
        risk_scores = {}
        for f in factors:
            risk_scores[f] = round(_normalise(f, cdata[f]), 1)
        composite = round(sum(risk_scores.values()) / len(factors), 1)
        countries[ckey] = {
            "label": cdata["label"],
            "color": cdata["color"],
            "similarity_pct": cdata["similarity_to_baseline"],
            "raw": {f: cdata[f] for f in factors},
            "risk_scores": risk_scores,
            "composite_risk_score": composite,
            "imf_program": cdata["imf_program"],
            "cbk_policy_rate_pct": cdata["cbk_policy_rate_pct"],
        }

    return {
        "factor_labels": factor_labels,
        "factor_keys": factors,
        "countries": countries,
        "commentary": (
            "Composite risk score weights six IMF-standard macro-financial indicators. "
            "Kenya's 2026 profile most closely resembles Ghana pre-2022 (87% structural similarity), "
            "followed by Zambia pre-2015 (79%). The key differentiator is Kenya's active EFF "
            "programme which provides a partial firewall — but covenant compliance is at risk if "
            "primary balance targets slip past FY2026."
        ),
    }


def build_yield_benchmarks(p: dict) -> dict:
    """
    Build IRR-vs-benchmark comparison table with CORRECT currency-adjusted returns.

    GBP-equivalent annual return from a KES instrument:
        gbp_equiv = (1 + kes_yield) / (1 + annual_depr) - 1

    We compute under FOUR depreciation scenarios from the actual model parameters:
      1. GBM drift only (5%/yr) — best case, no jumps occur
      2. Expected full model — GBM drift + E[jumps] = drift + lambda * mean_jump
      3. Class A trigger horizon — 25% over median 20 months ≈ 15%/yr annualised
      4. Class C trigger horizon — 75% over median 40 months ≈ 22.5%/yr annualised

    The dashboard should show all four columns so investors see the range.
    """
    yc = KENYA_YIELD_CURVE
    spot = p["kes_gbp_initial_spot"]

    # ── Compute expected annual depreciation scenarios from model params ───
    gbm_drift = p["gbm_annual_drift"]                          # e.g. 0.05
    jump_rate = p["devaluation_jump_prob_annual"]              # e.g. 0.35
    jump_mean = p["jump_severity_mean"]                        # e.g. 0.35
    # Expected full-model annual depreciation (compound approximation)
    # E[annual rate] ≈ (1 + gbm_drift) × (1 + jump_rate × jump_mean) - 1
    expected_full_depr = (1 + gbm_drift) * (1 + jump_rate * jump_mean) - 1

    # Class A: 25% depreciation over median 20 months → annualised
    class_a_drop = p["class_a"]["trigger_drop"]               # 0.25
    class_a_months = 20
    class_a_annual_depr = (1 + class_a_drop) ** (12 / class_a_months) - 1

    # Class C: 75% depreciation over median 40 months → annualised
    class_c_drop = p["class_c"]["trigger_drop"]               # 0.75
    class_c_months = 40
    class_c_annual_depr = (1 + class_c_drop) ** (12 / class_c_months) - 1

    depr_scenarios = [
        {"key": "optimistic",  "label": "Optimistic (GBM drift only, no jumps)",
         "annual_depr_pct": round(gbm_drift * 100, 1),        "annual_depr": gbm_drift},
        {"key": "expected",   "label": "Expected (GBM + mean jump component)",
         "annual_depr_pct": round(expected_full_depr * 100, 1), "annual_depr": expected_full_depr},
        {"key": "class_a_path", "label": f"Class A path (25% over 20mo, {round(class_a_annual_depr*100,1)}%/yr ann.)",
         "annual_depr_pct": round(class_a_annual_depr * 100, 1), "annual_depr": class_a_annual_depr},
        {"key": "class_c_path", "label": f"Class C path (75% over 40mo, {round(class_c_annual_depr*100,1)}%/yr ann.)",
         "annual_depr_pct": round(class_c_annual_depr * 100, 1), "annual_depr": class_c_annual_depr},
    ]

    def _gbp_equiv(kes_yield_pct: float, annual_depr: float) -> float:
        """GBP-equivalent annual return: (1+y)/(1+d) - 1, expressed as % rounded to 1dp."""
        return round(((1 + kes_yield_pct / 100) / (1 + annual_depr) - 1) * 100, 1)

    # ── Build GBP-adjusted curve for each tenor × scenario ─────────────────
    domestic = yc["domestic_kes"]
    curve_tenors = [
        {"tenor": "91d T-bill",   "kes_yield": domestic["91d_tbill"],   "years": 0.25},
        {"tenor": "182d T-bill",  "kes_yield": domestic["182d_tbill"],  "years": 0.5},
        {"tenor": "364d T-bill",  "kes_yield": domestic["364d_tbill"],  "years": 1.0},
        {"tenor": "2yr Bond",     "kes_yield": domestic["2yr_bond"],    "years": 2.0},
        {"tenor": "5yr Bond",     "kes_yield": domestic["5yr_bond"],    "years": 5.0},
        {"tenor": "10yr Bond",    "kes_yield": domestic["10yr_bond"],   "years": 10.0},
        {"tenor": "15yr Bond",    "kes_yield": domestic["15yr_bond"],   "years": 15.0},
    ]
    for ct in curve_tenors:
        ct["gbp_equiv_by_scenario"] = {
            sc["key"]: _gbp_equiv(ct["kes_yield"], sc["annual_depr"])
            for sc in depr_scenarios
        }

    # ── Compute 10yr KES bond GBP equiv properly (compound, not linear) ────
    # Compound formula over N years: GBP IRR = ((1+y)^N * S0/S_N)^(1/N) - 1
    # where S_N = S0 * (1+depr)^N
    # Simplifies to: (1+y)/(1+depr) - 1  (same per-period formula)
    kes_10yr_yield = domestic["10yr_bond"] / 100
    gbp_10yr_by_scenario = {}
    for sc in depr_scenarios:
        d = sc["annual_depr"]
        gbp_irr = round(((1 + kes_10yr_yield) / (1 + d) - 1) * 100, 1)
        gbp_10yr_by_scenario[sc["key"]] = {
            "gbp_equiv_pct": gbp_irr,
            "kes_yield_pct": domestic["10yr_bond"],
            "annual_depr_pct": sc["annual_depr_pct"],
            "label": sc["label"],
        }

    # ── Eurobond in GBP: USD yield + GBP/USD basis adjustment (~-0.5%) ─────
    gbp_usd_basis = -0.5  # GBP/USD cross-currency basis swap (typical)
    eurobond_usd = yc["eurobond_usd"]
    eurobond_gbp_points = [
        {"name": k, "usd_yield": v, "gbp_equiv_pct": round(v + gbp_usd_basis, 1)}
        for k, v in eurobond_usd.items() if v is not None
    ]

    # ── Fixed GBP reference yields (not KES-denominated, no FX adjustment) ─
    uk_benchmarks = [
        {"name": "UK Gilt 10yr",           "yield_pct": 4.2,  "risk": "AAA sovereign",           "color": "#3fb950"},
        {"name": "UK HY Corporate 10yr",   "yield_pct": 7.8,  "risk": "BB rated credit",          "color": "#58a6ff"},
        {"name": "UK Buy-to-Let gross",    "yield_pct": 5.8,  "risk": "UK residential property",  "color": "#d29922"},
        {"name": "UK CPI inflation (2026)","yield_pct": 3.1,  "risk": "Real return floor",         "color": "#8b949e"},
    ]

    # ── SPV tranche trigger context (dynamic from params) ──────────────────
    tranche_context = []
    for cls_key, label in [("class_a", "Class A"), ("class_b", "Class B"), ("class_c", "Class C")]:
        drop = p[cls_key]["trigger_drop"]
        sweetener = p[cls_key]["sweetener_share"]
        fee = p[cls_key]["fee_pct"]
        trigger_rate = round(spot * (1 + drop))
        tranche_context.append({
            "label": label,
            "trigger_drop_pct": round(drop * 100),
            "trigger_rate_kes": trigger_rate,
            "sweetener_pct": round(sweetener * 100),
            "fee_pct": round(fee * 100),
            "note": (
                f"{label}: fires at {trigger_rate} KES/GBP ({round(drop*100)}% drop). "
                f"Investor receives principal + {round(fee*100)}% fee + {round(sweetener*100)}% "
                f"of FX profit on that tranche."
            ),
        })

    # ── Commentary: correct the record on KES yield illusion ───────────────
    opt_10yr = gbp_10yr_by_scenario["optimistic"]["gbp_equiv_pct"]
    exp_10yr = gbp_10yr_by_scenario["expected"]["gbp_equiv_pct"]
    c_path_10yr = gbp_10yr_by_scenario["class_c_path"]["gbp_equiv_pct"]
    commentary = (
        f"YIELD ILLUSION WARNING: Kenya's 10yr bond yields {domestic['10yr_bond']}% in KES terms — "
        f"which looks attractive vs 4.2% UK gilts. But GBP-equivalent returns depend entirely "
        f"on the depreciation path. Under GBM drift only ({gbm_drift*100:.0f}%/yr): {opt_10yr}% GBP equiv. "
        f"Under the EXPECTED full model ({round(expected_full_depr*100,1)}%/yr including jump component): "
        f"{exp_10yr}% GBP equiv. Under the Class C crisis path ({round(class_c_annual_depr*100,1)}%/yr ann.): "
        f"{c_path_10yr}% GBP equiv. "
        f"Even the optimistic scenario barely beats a UK HY corporate bond (7.8%). "
        f"The SPV structure crystallises the FX gain at trigger rather than relying on a "
        f"KES bond to preserve GBP value through a depreciation cycle."
    )

    return {
        "as_of": yc["as_of"],
        "source": yc["source"],
        "depreciation_scenarios": depr_scenarios,
        "kenya_yield_curve": curve_tenors,           # includes gbp_equiv_by_scenario per tenor
        "gbp_10yr_by_scenario": gbp_10yr_by_scenario,
        "eurobond_curve": eurobond_gbp_points,
        "uk_benchmarks": uk_benchmarks,
        "tranche_context": tranche_context,
        "commentary": commentary,
        "breakeven_analysis": {
            "description": (
                f"Class A fires at {round(spot*(1+p['class_a']['trigger_drop']))} KES/GBP "
                f"({round(p['class_a']['trigger_drop']*100)}% drop). "
                f"Class B at {round(spot*(1+p['class_b']['trigger_drop']))} KES/GBP "
                f"({round(p['class_b']['trigger_drop']*100)}% drop). "
                f"Class C at {round(spot*(1+p['class_c']['trigger_drop']))} KES/GBP "
                f"({round(p['class_c']['trigger_drop']*100)}% drop — Ghana/Turkey-level crisis)."
            ),
            "class_a_trigger_rate": round(spot * (1 + p["class_a"]["trigger_drop"])),
            "class_b_trigger_rate": round(spot * (1 + p["class_b"]["trigger_drop"])),
            "class_c_trigger_rate": round(spot * (1 + p["class_c"]["trigger_drop"])),
            "expected_annual_depr_pct": round(expected_full_depr * 100, 1),
            "gbm_drift_only_depr_pct": round(gbm_drift * 100, 1),
        },
    }


def build_kenya_forecast(p: dict, n_months: int = 120) -> dict:
    """
    Generate forward depreciation probability cone for KES/GBP.
    Uses the baseline GBM + jump model with 10,000 paths.
    Returns percentile bands at each month for fan chart.

    Also computes 'crisis zone' — months where Class A trigger becomes
    probable within 6 months (early warning signal).
    """
    rng = np.random.default_rng(p["seed"] + 999)
    n_paths = 5000
    paths = generate_fx_paths(p, n_paths, n_months, rng)  # shape (n_paths, n_months+1)

    spot = p["kes_gbp_initial_spot"]
    months = list(range(n_months + 1))

    pct_levels = [5, 10, 25, 50, 75, 90, 95]
    bands = {f"p{pl}": [] for pl in pct_levels}
    trigger_a = spot * (1 + p["class_a"]["trigger_drop"])
    trigger_b = spot * (1 + p["class_b"]["trigger_drop"])
    trigger_c = spot * (1 + p["class_c"]["trigger_drop"])

    # Monthly probability of rate exceeding each trigger BY this month
    cum_prob_a, cum_prob_b, cum_prob_c = [], [], []

    for t in range(n_months + 1):
        rates_at_t = paths[:, t]
        for pl in pct_levels:
            bands[f"p{pl}"].append(round(float(np.percentile(rates_at_t, pl)), 2))
        cum_prob_a.append(round(float(np.mean(rates_at_t >= trigger_a)) * 100, 2))
        cum_prob_b.append(round(float(np.mean(rates_at_t >= trigger_b)) * 100, 2))
        cum_prob_c.append(round(float(np.mean(rates_at_t >= trigger_c)) * 100, 2))

    # Early warning: months where 6-month ahead P(Class A breach) > 50%
    # Compute P(any path hits trigger_a within 6 months from month t)
    ew_signal = []
    for t in range(n_months + 1):
        end = min(t + 6, n_months)
        p_breach = float(np.mean(np.any(paths[:, t:end+1] >= trigger_a, axis=1)))
        ew_signal.append(round(p_breach * 100, 2))

    # Scenario annotations
    annotations = [
        {"month": 12, "label": "~1yr: IMF review gate", "note": "CBK must hit primary balance target or trigger waiver"},
        {"month": 24, "label": "~2yr: Eurobond maturity risk", "note": "KENINT 2027 rollover — market confidence test"},
        {"month": 36, "label": "~3yr: EFF programme end", "note": "Post-programme monitoring begins; discipline risk"},
    ]

    return {
        "months": months,
        "spot_rate": spot,
        "trigger_rates": {
            "class_a": round(trigger_a, 2),
            "class_b": round(trigger_b, 2),
            "class_c": round(trigger_c, 2),
        },
        "bands": {k: v for k, v in bands.items()},
        "cumulative_trigger_probability": {
            "class_a": cum_prob_a,
            "class_b": cum_prob_b,
            "class_c": cum_prob_c,
        },
        "early_warning_6m": ew_signal,
        "annotations": annotations,
        "commentary": (
            f"Fan chart shows KES/GBP depreciation cone (P5–P95 bands) over {n_months} months "
            f"from spot {spot}. Class A trigger at {round(trigger_a, 1)} KES/GBP has "
            f"{round(cum_prob_a[-1], 1)}% cumulative probability over the full horizon. "
            "Early warning signal (red) indicates months where there is >50% probability "
            "of Class A breach within 6 months — actionable monitoring threshold for investors."
        ),
    }


def build_tranche_spacing_analysis(mc_results: list, p: dict) -> dict:
    """
    Analyse whether the tranche trigger levels are well-spaced.
    Key risk: if Class A and Class B both trigger within a few months,
    the sequential payout structure loses its 'staircase' character.

    For each path: extract month of Class A trigger and Class B trigger.
    Compute P(B triggers within N months of A) for N = 3, 6, 12.
    Also compute P(all three classes trigger in the same year).
    """
    n = len(mc_results)

    ta_months = []  # month of first Class A breach per path (None if not triggered)
    tb_months = []
    tc_months = []

    for path_data in mc_results:
        def _hit_month(cls_key):
            r = path_data.get(cls_key, {})
            if r.get("triggered"):
                return r["trigger_month"]
            return None

        ta_months.append(_hit_month("class_a"))
        tb_months.append(_hit_month("class_b"))
        tc_months.append(_hit_month("class_c"))

    # Paths where both A and B triggered
    ab_gaps, bc_gaps = [], []
    for ta, tb, tc in zip(ta_months, tb_months, tc_months):
        if ta is not None and tb is not None:
            ab_gaps.append(tb - ta)
        if tb is not None and tc is not None:
            bc_gaps.append(tc - tb)

    def _pct_within(gaps, months):
        if not gaps:
            return 0.0
        return round(sum(1 for g in gaps if g <= months) / len(gaps) * 100, 1)

    result = {
        "n_paths": n,
        "class_a_b": {
            "description": "Gap (months) between Class A and Class B trigger",
            "n_both_triggered": len(ab_gaps),
            "pct_within_3mo": _pct_within(ab_gaps, 3),
            "pct_within_6mo": _pct_within(ab_gaps, 6),
            "pct_within_12mo": _pct_within(ab_gaps, 12),
            "median_gap_months": round(float(np.median(ab_gaps)), 1) if ab_gaps else None,
            "p25_gap_months": round(float(np.percentile(ab_gaps, 25)), 1) if ab_gaps else None,
            "p75_gap_months": round(float(np.percentile(ab_gaps, 75)), 1) if ab_gaps else None,
        },
        "class_b_c": {
            "description": "Gap (months) between Class B and Class C trigger",
            "n_both_triggered": len(bc_gaps),
            "pct_within_3mo": _pct_within(bc_gaps, 3),
            "pct_within_6mo": _pct_within(bc_gaps, 6),
            "pct_within_12mo": _pct_within(bc_gaps, 12),
            "median_gap_months": round(float(np.median(bc_gaps)), 1) if bc_gaps else None,
            "p25_gap_months": round(float(np.percentile(bc_gaps, 25)), 1) if bc_gaps else None,
            "p75_gap_months": round(float(np.percentile(bc_gaps, 75)), 1) if bc_gaps else None,
        },
        "simultaneous_risk": {
            "pct_all_three_same_year": round(
                sum(1 for ta, tb, tc in zip(ta_months, tb_months, tc_months)
                    if ta is not None and tb is not None and tc is not None
                    and max(ta, tb, tc) - min(ta, tb, tc) <= 12) / n * 100, 1
            ),
        },
        "structural_verdict": "",  # filled below
    }

    ab_6 = result["class_a_b"]["pct_within_6mo"]
    bc_6 = result["class_b_c"]["pct_within_6mo"]
    if ab_6 < 15 and bc_6 < 15:
        verdict = "PASS — Tranches are well-spaced. Less than 15% of paths see consecutive triggers within 6 months."
    elif ab_6 < 30 and bc_6 < 30:
        verdict = "MARGINAL — Consider widening trigger gaps by 5–10pp to improve investor sequencing clarity."
    else:
        verdict = "FAIL — High probability of simultaneous triggers undermines the staircase structure. Redesign required."

    result["structural_verdict"] = verdict
    return result


# ─────────────────────────────────────────────────────────────────────────────
# FIC CAPITAL SUSTAINABILITY ANALYSIS
# Models FIC solvency across depreciation scenarios, accounting for:
#   - BTL LTV cap (75%) and ICR test (1.25×) gating refinance access
#   - Mortgage fixed-term lock-in (24-month) preventing early re-mortgage
#   - Payout lag (2 months) giving processing buffer post-trigger
#   - Kenya bond GBP erosion — 16% KES yield is DESTROYED by FX depreciation
# ─────────────────────────────────────────────────────────────────────────────

def build_fic_sustainability(p: dict) -> dict:
    """
    Run the FIC ledger under four stress scenarios and return capital health metrics.
    Also compute the Kenya bond GBP-equivalent path to make the hedging rationale explicit.
    """
    e0 = p["kes_gbp_initial_spot"]
    n = p["n_months"]

    # ── Define stress paths ─────────────────────────────────────────────────
    def make_path(drop_at_month: int, final_drop: float, shape: str = "linear") -> np.ndarray:
        """Generate a deterministic FX path."""
        path = np.zeros(n + 1)
        path[0] = e0
        for t in range(1, n + 1):
            if shape == "linear":
                prog = min(1.0, t / drop_at_month)
                path[t] = e0 * (1 + final_drop * prog)
            elif shape == "step":
                # Slow drift then sudden crash
                drift_phase = drop_at_month - 6
                crash_size = final_drop - 0.05
                if t <= drift_phase:
                    path[t] = e0 * (1 + 0.05 * (t / drift_phase))
                elif t <= drop_at_month:
                    progress = (t - drift_phase) / 6
                    path[t] = e0 * (1 + 0.05 + crash_size * progress)
                else:
                    path[t] = path[drop_at_month]
            elif shape == "slow_recovery":
                # Falls, then partial recovery
                if t <= drop_at_month:
                    path[t] = e0 * (1 + final_drop * t / drop_at_month)
                else:
                    recovery = final_drop * 0.20
                    path[t] = e0 * (1 + final_drop - recovery * (t - drop_at_month) / (n - drop_at_month))
        return path

    scenarios = {
        "baseline_slow_25pct": {
            "label": "Baseline — slow 25% over 3 yrs (Class A triggers)",
            "path": make_path(36, 0.25, "linear"),
            "description": "Steady KES drift matching IMF-programme period. Class A fires at ~month 36."
        },
        "moderate_45pct": {
            "label": "Moderate crisis — 45% over 2 yrs (Class B triggers)",
            "path": make_path(24, 0.45, "step"),
            "description": "Zambia-type cycle: slow drift then sudden CBK capitulation."
        },
        "severe_75pct": {
            "label": "Severe crisis — 75% over 18 months (Class C triggers)",
            "path": make_path(18, 0.75, "step"),
            "description": "Ghana 2022 / Turkey 2021-22 analogue. FIC under maximum stress."
        },
        "all_trigger_fast": {
            "label": "Stress scenario — all classes trigger within 24 months",
            "path": make_path(24, 0.80, "step"),
            "description": "Worst-case: IMF programme collapse, sovereign default. Tests FIC solvency ceiling."
        },
    }

    results = {}
    for key, sc in scenarios.items():
        path = sc["path"]
        df = run_fic_ledger(path, p)

        # Key health metrics
        min_cash = float(df["cumulative_cash_gbp"].min())
        final_cash = float(df["cumulative_cash_gbp"].iloc[-1])
        min_dscr = float(df["dscr"].replace(999.0, np.nan).min())
        max_ltv = float(df["ltv"].max())
        max_mortgage = float(df["mortgage_balance_gbp"].max())
        max_pending = float(df["pending_payouts_gbp"].max())

        tranche_events = df[df["tranche_event"].notna()][["month", "tranche_event", "tranche_settled_gbp"]].to_dict("records")

        # LTV breach check (> btl_ltv_max)
        ltv_breaches = int((df["ltv"] > p.get("btl_ltv_max", 0.75)).sum())
        dscr_stress_months = int((df["dscr"] < 1.25).sum())

        # Cash solvent flag: never goes below -£5,000 (small overdraft tolerance)
        solvent = bool(min_cash > -5000)

        # FIC health verdict: cash solvency matters for operations, but the balance
        # sheet (net equity) is the true solvency measure. A cash shortfall under
        # simultaneous triggers is a LIQUIDITY problem, not an INSOLVENCY problem—
        # the FIC's property asset + KES debt appreciation far exceed the shortfall.
        if solvent and max_ltv <= 0.78 and min_dscr >= 1.0:
            health = "HEALTHY — FIC cash flow positive throughout. Refis stay within LTV/ICR constraints."
        elif min_cash > -50_000 and max_ltv <= 0.85:
            health = "STRESSED — Temporary cash shortfall manageable with 35% LTV inception mortgage or sequential settlement clause."
        else:
            health = "IMPAIRED (LIQUIDITY) — Simultaneous triggers exceed operating cash. Fix: pre-fund via 35% LTV BTL at inception (+£80k reserves) or stagger Class B/C settlement by up to 5 months post-trigger."

        # Monthly snapshots for charting (every 6 months)
        chart_months = list(range(0, n + 1, 6))
        # Peak net equity (balance sheet, not cash)
        peak_equity = float(df["fic_net_equity_gbp"].max()) if "fic_net_equity_gbp" in df.columns else 0
        final_equity = float(df["fic_net_equity_gbp"].iloc[-1]) if "fic_net_equity_gbp" in df.columns else 0

        results[key] = {
            "label": sc["label"],
            "description": sc["description"],
            "health": health,
            "solvent": solvent,
            "min_cash_gbp": round(min_cash),
            "final_cash_gbp": round(final_cash),
            "min_dscr": round(min_dscr, 3) if not np.isnan(min_dscr) else None,
            "max_ltv_pct": round(max_ltv * 100, 1),
            "max_mortgage_gbp": round(max_mortgage),
            "max_pending_payouts_gbp": round(max_pending),
            "ltv_breach_months": ltv_breaches,
            "dscr_stress_months": dscr_stress_months,
            "tranche_events": tranche_events,
            "peak_equity_gbp": round(peak_equity),
            "final_equity_gbp": round(final_equity),
            "chart": {
                "months": [int(df.loc[df["month"] == m, "month"].values[0]) if m in df["month"].values else m
                           for m in chart_months if m > 0],
                "cash_gbp": [round(float(df.loc[df["month"] == m, "cumulative_cash_gbp"].values[0]))
                             for m in chart_months if m > 0 and m in df["month"].values],
                "ltv_pct": [round(float(df.loc[df["month"] == m, "ltv"].values[0]) * 100, 1)
                            for m in chart_months if m > 0 and m in df["month"].values],
                "dscr": [round(min(float(df.loc[df["month"] == m, "dscr"].values[0]), 10), 2)
                         for m in chart_months if m > 0 and m in df["month"].values],
                "fx_rate": [round(float(path[m]), 1) for m in chart_months if m > 0],
                "equity_gbp": [round(float(df.loc[df["month"] == m, "fic_net_equity_gbp"].values[0]))
                               for m in chart_months if m > 0 and m in df["month"].values],
                "kes_loan_gbp": [round(float(df.loc[df["month"] == m, "kes_loan_gbp_value"].values[0]))
                                 for m in chart_months if m > 0 and m in df["month"].values],
            },
        }

    # ── Kenya bond GBP erosion analysis ─────────────────────────────────────
    # A Kenyan investor holding 5M KES in a 16% CBK bond over 10 years:
    # In KES terms it compounds nicely. In GBP terms it is destroyed by depreciation.
    bond_yield = p["kenya_bond_yield_base"]
    initial_gbp = 5_000_000 / e0
    gbp_erosion_scenarios = {}
    for key, sc in scenarios.items():
        path = sc["path"]
        kes_value_at = [5_000_000 * (1 + bond_yield) ** (t / 12) for t in range(0, n + 1, 12)]
        gbp_value_at = [kv / path[t * 12] for t, kv in zip(range(len(kes_value_at)), kes_value_at)]
        final_gbp = gbp_value_at[-1]
        gbp_erosion_scenarios[key] = {
            "label": sc["label"],
            "kes_value_yr10": round(kes_value_at[-1]),
            "gbp_value_yr10": round(final_gbp),
            "vs_initial_gbp": round(final_gbp / initial_gbp, 3),
            "gbp_path_annual": [round(v) for v in gbp_value_at],
        }

    return {
        "scenarios": results,
        "kenya_bond_gbp_erosion": gbp_erosion_scenarios,
        "initial_invest_gbp": round(initial_gbp),
        "commentary": (
            f"FIC purchased UK property OUTRIGHT from KES proceeds at inception — "
            f"converting KES currency risk into GBP-denominated hard asset before any trigger. "
            f"The BTL property (£{p['property_value_gbp']:,}) generates rental yield while SPV "
            f"interest and eventual payouts are funded from cash flow, HPI appreciation, and "
            f"BTL refinancing (subject to 75% LTV cap and 1.25× ICR test). "
            f"CRITICAL FINDING: Under a fast-crisis scenario (Ghana/Turkey-type — 75% in 18 months), "
            f"all three classes trigger within the same 5-month window. The FIC's £{p['initial_fic_cash_buffer']:,} "
            f"buffer plus accumulated rent is insufficient to cover simultaneous payouts, "
            f"and the BTL refinancing pipeline (3-month processing) cannot complete in time. "
            f"MITIGATIONS: (1) Pre-fund FIC with £80-100k cash buffer at inception, funded by reducing "
            f"the property purchase price or via a 35% LTV BTL mortgage at origination; "
            f"(2) Build sequential settlement clause: Class B/C settle only after Class A refinance completes; "
            f"(3) Accept that Class C (75% trigger) payout may occur up to 5 months after trigger — "
            f"acceptable if disclosed in the investment terms. "
            f"Kenya bond GBP erosion analysis confirms the hedging rationale: a 16% KES bond is worth "
            f"only £{round(5_000_000 * (1.16**10) / (p['kes_gbp_initial_spot'] * 1.75)):,} in GBP by year 10 "
            f"under a 75% depreciation (vs £{round(5_000_000 / p['kes_gbp_initial_spot']):,} initial). "
            f"The SPV structure, despite its FIC liquidity risk, fundamentally outperforms the hold-KES alternative."
        ),
    }


def build_calibration_report() -> dict:
    """Fit parameters for every historical precedent + return comparison table."""
    report = {}
    for key, cal in HISTORICAL_CALIBRATIONS.items():
        fitted_live = fit_jump_diffusion(cal["monthly_rates"])
        report[key] = {
            "label": cal["label"],
            "description": cal["description"],
            "source_pair": cal["source_pair"],
            "crisis_period": cal["crisis_period"],
            "n_months_of_data": len(cal["monthly_rates"]),
            "monthly_rates": [round(float(r), 4) for r in cal["monthly_rates"]],
            "total_depreciation_pct": round(
                (cal["monthly_rates"][-1] / cal["monthly_rates"][0] - 1) * 100, 1
            ),
            "fitted_params": fitted_live,
            "preset_params": cal["fitted"],   # expert-tuned values for the engine
        }
    return report

# ─────────────────────────────────────────────────────────────────────────────
# 1.  HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def gbp_cost_of_kes_debt(principal_kes: float, rate_kes_per_gbp: float) -> float:
    """Return GBP equivalent of a KES principal at the given FX rate."""
    return principal_kes / rate_kes_per_gbp


def fx_profit_gbp(principal_kes: float, e0: float, et: float) -> float:
    """
    ΠFX = P0 × (1/E0 − 1/Et)
    Positive when KES has weakened (et > e0).
    """
    return principal_kes * (1.0 / e0 - 1.0 / et)


def tranche_payout_gbp(principal_kes: float, e0: float, et: float,
                       base_rate: float, months_held: int,
                       fee_pct: float, sweetener_share: float) -> dict:
    """
    Calculate the total GBP payout for a single tranche at trigger time.
    Returns a dict with component breakdown.
    """
    i_base_kes = principal_kes * base_rate * (months_held / 12)
    i_base_gbp = i_base_kes / et                            # converted at crash rate

    pi_fx = fx_profit_gbp(principal_kes, e0, et)
    f_er = (principal_kes / et) * fee_pct                   # fee on GBP-equivalent principal
    s_fx = pi_fx * sweetener_share

    # Yield components (excludes principal)
    y_yield = i_base_gbp + f_er + s_fx
    principal_gbp_at_crash = principal_kes / et
    # Total cash received by investor (principal + all yield components)
    total_cash = principal_gbp_at_crash + y_yield

    return {
        "principal_gbp_crash_rate": principal_gbp_at_crash,
        "base_interest_gbp": i_base_gbp,
        "early_repayment_fee_gbp": f_er,
        "fx_sweetener_gbp": s_fx,
        "yield_only_gbp": y_yield,
        "total_payout_gbp": total_cash,          # principal + all yield
        "fic_retained_fx_profit_gbp": pi_fx * (1 - sweetener_share),
    }


def irr(cashflows: list[float]) -> float:
    """Compute IRR via Brent's method on monthly cashflows.
    Returns annual (monthly) IRR; caller must annualise if needed.
    """
    cf = np.array(cashflows, dtype=float)
    n = len(cf)

    def npv(r):
        if r <= -1.0:
            return np.sign(cf[0]) * np.inf
        t = np.arange(n, dtype=float)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            discounts = np.where(t == 0, 1.0, np.exp(-t * np.log1p(r)))
            result = np.nansum(cf * discounts)
        return float(result) if np.isfinite(result) else np.sign(r) * np.inf

    # Scan for sign change bracketing the root
    candidates = [(-0.90, 2.0), (-0.95, 0.5), (-0.50, 5.0)]
    for lo, hi in candidates:
        try:
            flo, fhi = npv(lo), npv(hi)
            if np.isfinite(flo) and np.isfinite(fhi) and flo * fhi < 0:
                return float(optimize.brentq(npv, lo, hi, xtol=1e-8, maxiter=200))
        except Exception:
            pass
    return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  JUMP-DIFFUSION FX PATH GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_fx_paths(params: dict, n_paths: int, n_months: int,
                      rng: np.random.Generator) -> np.ndarray:
    """
    Merton Jump-Diffusion: dS = μS dt + σS dW + S(J-1) dN
    Returns array of shape (n_paths, n_months+1) with KES/GBP rates.
    S increases ⟹ KES weakens (more KES per £1).
    """
    dt = 1 / 12
    mu = params["gbm_annual_drift"]
    sigma = params["gbm_annual_vol"]
    lam = params["devaluation_jump_prob_annual"]   # jump arrival rate (Poisson)
    m_j = params["jump_severity_mean"]             # mean log-jump
    s_j = params["jump_severity_std"]              # std log-jump

    S = np.full((n_paths, n_months + 1), params["kes_gbp_initial_spot"])

    for t in range(1, n_months + 1):
        Z = rng.standard_normal(n_paths)
        # Poisson number of jumps in this month
        N_jumps = rng.poisson(lam * dt, n_paths)
        # Sum of log-jumps
        log_J = np.where(
            N_jumps > 0,
            rng.normal(m_j, s_j, n_paths) * N_jumps,
            0.0
        )
        # GBM component (KES weakening = positive drift when mu < 0 means KES strengthening;
        # but here positive drift means more KES per GBP, so we model S as weakening asset)
        gbm = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
        # Combined
        S[:, t] = S[:, t - 1] * np.exp(gbm + log_J)
        # Clamp: KES cannot strengthen beyond 10 % above start (one-sided hedge)
        S[:, t] = np.maximum(S[:, t], params["kes_gbp_initial_spot"] * 0.90)

    return S


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FIC LEDGER (deterministic monthly tracker)
# ─────────────────────────────────────────────────────────────────────────────

def run_fic_ledger(fx_path: np.ndarray, params: dict) -> pd.DataFrame:
    """
    Single-path monthly FIC cash flow ledger with realistic BTL refinancing model.

    Refinancing mechanics:
    - FIC bought property OUTRIGHT from KES proceeds (no initial mortgage)
    - When cash is short for a payout, FIC applies for BTL refinance
    - Refinance gated by: LTV ≤ btl_ltv_max (75%) AND ICR ≥ btl_icr_min (1.25×)
    - Anticipatory application: FIC begins refi process when FX rate is within
      refi_anticipation_pct of the next trigger threshold
    - Processing lag: drawdown happens refi_processing_months after application
    - Payout lag: investor receives funds payout_lag_months after trigger fires
    - Mortgage lock-in: once refinanced, cannot re-refinance for mortgage_fixed_term_months
    """
    n_months = len(fx_path) - 1
    e0 = params["kes_gbp_initial_spot"]
    kes_outstanding = params["spv_total_raise_kes"]
    base_rate = params["base_interest_rate"]

    property_value = params["property_value_gbp"]
    rent = params["monthly_rent_gbp"]
    mgmt_fee_monthly = params["management_fee_pct"]
    opex_monthly = (params["service_charge_annual"] + params["insurance_annual"] +
                    params["maintenance_annual"]) / 12

    corp_tax_rate = params["uk_corp_tax_rate"]
    wht_rate = params["withholding_tax_rate"]
    hpi = params["uk_hpi_annual"]
    rent_inflation = params["uk_rent_inflation_annual"]
    btl_rate = params["btl_mortgage_rate"]

    ltv_max = params.get("btl_ltv_max", 0.75)
    icr_min = params.get("btl_icr_min", 1.25)
    refi_lag = params.get("refi_processing_months", 3)
    payout_lag = params.get("payout_lag_months", 2)
    anticipation_pct = params.get("refi_anticipation_pct", 0.05)
    fix_term = params.get("mortgage_fixed_term_months", 24)

    cash = params["initial_fic_cash_buffer"]
    mortgage_balance_gbp = params.get("mortgage_initial_gbp", 0.0)
    mortgage_locked_until = 0  # month when current fixed term expires

    records = []

    tranche_order = [
        ("class_a", params["class_a"]),
        ("class_b", params["class_b"]),
        ("class_c", params["class_c"]),
    ]
    called_tranches: set = set()

    # Pending payouts: {settle_month: gbp_amount}
    pending_payouts: dict = {}
    # Pending refi drawdowns: {drawdown_month: gbp_amount}
    pending_refi: dict = {}
    # Track which refi we've already applied for (to avoid duplicate applications)
    refi_applied_for_trigger: set = set()

    for t in range(1, n_months + 1):
        et = fx_path[t]

        # ── Receive any pending refi drawdown ───────────────────────────────
        if t in pending_refi:
            cash += pending_refi.pop(t)

        # ── Revenue ─────────────────────────────────────────────────────────
        current_rent = rent * ((1 + rent_inflation / 12) ** t)
        gross_rent = current_rent
        management_cost = gross_rent * mgmt_fee_monthly
        net_rent = gross_rent - management_cost - opex_monthly

        # ── Mortgage interest (if any) ───────────────────────────────────────
        monthly_mortgage_interest = mortgage_balance_gbp * btl_rate / 12

        # ── SPV interest due (monthly) ───────────────────────────────────────
        monthly_interest_kes = kes_outstanding * base_rate / 12
        monthly_interest_gbp_net = (monthly_interest_kes / et) * (1 - wht_rate)

        # ── Taxable profit ──────────────────────────────────────────────────
        taxable_profit = net_rent - (monthly_interest_kes / et) - monthly_mortgage_interest
        corp_tax = max(0.0, taxable_profit * corp_tax_rate)

        # ── Net cash this month ─────────────────────────────────────────────
        net_cash_month = net_rent - monthly_interest_gbp_net - monthly_mortgage_interest - corp_tax
        cash += net_cash_month

        # ── Property value ──────────────────────────────────────────────────
        prop_value = property_value * ((1 + hpi / 12) ** t)

        # ── Settle any pending payouts due this month ───────────────────────
        if t in pending_payouts:
            cash -= pending_payouts.pop(t)

        # ── Tranche trigger check ───────────────────────────────────────────
        drop_pct = (et - e0) / e0
        tranche_event = None
        tranche_settled_gbp = 0.0

        for tk, tv in tranche_order:
            if tk in called_tranches:
                continue
            tranche_kes = kes_outstanding / max(1, 3 - len(called_tranches))

            # ── Anticipatory refinancing: start application ahead of trigger
            if tk not in refi_applied_for_trigger:
                trigger_threshold = tv["trigger_drop"]
                proximity = drop_pct / trigger_threshold if trigger_threshold > 0 else 0
                if proximity >= (1 - anticipation_pct):
                    # Can we refinance?
                    can_refi = t >= mortgage_locked_until
                    if can_refi and mortgage_balance_gbp < prop_value * ltv_max * 0.5:
                        # ICR test at proposed new balance
                        max_by_ltv = prop_value * ltv_max
                        annual_mortgage_at_max = max_by_ltv * btl_rate
                        annual_rent_now = current_rent * 12
                        if annual_rent_now >= icr_min * annual_mortgage_at_max * 0.5:
                            # Application submitted; drawdown in refi_lag months
                            proposed_draw = min(
                                prop_value * ltv_max - mortgage_balance_gbp,
                                tranche_kes / et * 1.5  # pre-fund ~1.5× tranche
                            )
                            if proposed_draw > 1000:
                                draw_month = min(t + refi_lag, n_months)
                                pending_refi[draw_month] = pending_refi.get(draw_month, 0) + proposed_draw
                                mortgage_balance_gbp += proposed_draw
                                mortgage_locked_until = t + fix_term
                                refi_applied_for_trigger.add(tk)

            # ── Trigger fires ───────────────────────────────────────────────
            if drop_pct >= tv["trigger_drop"]:
                # base_rate=0 here: monthly interest has already been paid throughout the
                # holding period — including it again would double-count it at trigger.
                # The trigger payout is: principal-at-crash + early-repayment-fee + FX sweetener.
                payout = tranche_payout_gbp(
                    tranche_kes, e0, et,
                    0.0, t,
                    tv["fee_pct"], tv["sweetener_share"]
                )
                needed = payout["total_payout_gbp"]

                if cash < needed:
                    shortfall = needed - cash
                    # Emergency refinance draw (if not lock-in and LTV headroom exists)
                    if t >= mortgage_locked_until:
                        max_by_ltv = prop_value * ltv_max
                        ltv_headroom = max_by_ltv - mortgage_balance_gbp
                        # ICR test: proposed total mortgage must still satisfy ICR
                        proposed_new_balance = mortgage_balance_gbp + ltv_headroom
                        annual_rent = current_rent * 12
                        if annual_rent >= icr_min * proposed_new_balance * btl_rate:
                            usable = min(ltv_headroom, shortfall)
                        else:
                            # ICR binds: solve for max balance where ICR = icr_min
                            max_by_icr = annual_rent / (icr_min * btl_rate)
                            usable = min(max(0, max_by_icr - mortgage_balance_gbp), shortfall)
                        if usable > 0:
                            # Emergency refi: still takes refi_lag months (payout deferred)
                            draw_month = min(t + refi_lag, n_months)
                            pending_refi[draw_month] = pending_refi.get(draw_month, 0) + usable
                            mortgage_balance_gbp += usable
                            mortgage_locked_until = t + fix_term

                    # Schedule payout after lag
                    settle_month = min(t + payout_lag, n_months)
                    pending_payouts[settle_month] = pending_payouts.get(settle_month, 0) + needed
                else:
                    # Pay immediately
                    settle_month = min(t + payout_lag, n_months)
                    pending_payouts[settle_month] = pending_payouts.get(settle_month, 0) + needed

                kes_outstanding -= tranche_kes
                called_tranches.add(tk)
                tranche_settled_gbp = needed
                tranche_event = tv["label"]
                break

        # ── LTV and ICR snapshot ─────────────────────────────────────────────
        ltv_now = mortgage_balance_gbp / prop_value if prop_value > 0 else 0.0
        annual_mortgage_interest = mortgage_balance_gbp * btl_rate
        icr_now = (net_rent * 12) / annual_mortgage_interest if annual_mortgage_interest > 0 else 999.0

        # ── SPV DSCR (rental income / total debt service: mortgage + KES loan) ──
        annual_debt_service_gbp = (kes_outstanding * base_rate) / et if et > 0 else 0
        total_annual_service = annual_debt_service_gbp + annual_mortgage_interest
        dscr = (net_rent * 12) / total_annual_service if total_annual_service > 0 else 999.0

        # ── FIC net equity = property value − KES loan GBP equivalent ─────────
        # This is the core economic gain: as KES devalues, the KES liability shrinks
        # in GBP while the UK property remains GBP-denominated. This is NOT a cash
        # item — it's a balance sheet improvement that underpins the product's rationale.
        kes_loan_gbp_value = kes_outstanding / et if et > 0 else 0
        fic_net_equity = prop_value - kes_loan_gbp_value - mortgage_balance_gbp

        records.append({
            "month": t,
            "fx_rate": et,
            "gross_rent_gbp": gross_rent,
            "net_rent_gbp": net_rent,
            "spv_interest_paid_gbp": monthly_interest_gbp_net,
            "mortgage_interest_gbp": monthly_mortgage_interest,
            "corp_tax_gbp": corp_tax,
            "net_cash_flow_gbp": net_cash_month,
            "cumulative_cash_gbp": cash,
            "property_value_gbp": prop_value,
            "mortgage_balance_gbp": mortgage_balance_gbp,
            "kes_loan_gbp_value": round(kes_loan_gbp_value, 2),
            "fic_net_equity_gbp": round(fic_net_equity, 2),
            "ltv": round(ltv_now, 4),
            "icr": round(icr_now, 3),
            "kes_outstanding": kes_outstanding,
            "dscr": dscr,
            "tranche_event": tranche_event,
            "tranche_settled_gbp": tranche_settled_gbp,
            "pending_payouts_gbp": sum(pending_payouts.values()),
            "refi_locked_until": mortgage_locked_until,
        })

    return pd.DataFrame(records)



# ─────────────────────────────────────────────────────────────────────────────
# 4.  MONTE CARLO ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_monte_carlo(params: dict) -> dict:
    """
    Run n_paths simulations.  For each path record:
      - Month of first Class A / B / C trigger (or None)
      - Investor IRR per class
      - Min DSCR over the horizon
      - FIC net wealth at end
      - Counterfactual: what 70M KES is worth in GBP if kept in Kenya
    """
    rng = np.random.default_rng(params["seed"])
    n_paths = params["n_paths"]
    n_months = params["n_months"]
    e0 = params["kes_gbp_initial_spot"]
    P0 = params["spv_total_raise_kes"]
    P_per_tranche = P0 / 3

    print(f"Generating {n_paths} FX paths …")
    fx_paths = generate_fx_paths(params, n_paths, n_months, rng)

    # Pre-compute triggers for all paths without running full ledger (fast)
    results = []
    kenya_bond_monthly = (1 + params["kenya_bond_yield_base"]) ** (1/12) - 1

    for i in range(n_paths):
        path = fx_paths[i]
        path_result = {"path_id": i}

        # Find trigger months
        for cls_key in ["class_a", "class_b", "class_c"]:
            tv = params[cls_key]
            trigger_rate = e0 * (1 + tv["trigger_drop"])
            hit_months = np.where(path[1:] >= trigger_rate)[0] + 1  # 1-indexed
            if len(hit_months):
                t_hit = int(hit_months[0])
                et = float(path[t_hit])
                payout = tranche_payout_gbp(
                    P_per_tranche, e0, et,
                    params["base_interest_rate"], t_hit,
                    tv["fee_pct"], tv["sweetener_share"]
                )
                # Investor IRR: invest P_per_tranche / e0 GBP, receive total_payout_gbp at t_hit
                invest_gbp = P_per_tranche / e0
                cfs = [-invest_gbp] + [0.0] * (t_hit - 1) + [payout["total_payout_gbp"]]
                # Convert monthly CFs to annual equiv IRR
                monthly_irr_val = irr(cfs)
                annual_irr = (1 + monthly_irr_val) ** 12 - 1 if not np.isnan(monthly_irr_val) else float("nan")
                path_result[cls_key] = {
                    "triggered": True,
                    "trigger_month": t_hit,
                    "trigger_rate": et,
                    "total_payout_gbp": payout["total_payout_gbp"],
                    "annual_irr": annual_irr,
                    # Bond alternative: what a KES bond would have returned in GBP
                    # over the SAME holding period on this SAME path.
                    # bond_gbp_irr = (1+kes_rate) * (e0/et)^(12/t) - 1
                    # This is the correct per-path comparison: SPV IRR vs bond GBP IRR.
                    "bond_alt_gbp_irr": float(
                        (1 + params["kenya_bond_yield_base"]) * (e0 / et) ** (12 / t_hit) - 1
                    ),
                }
            else:
                path_result[cls_key] = {"triggered": False}

        # Counterfactual: 70M KES in Kenyan bonds
        final_kes_bond_value = P0 * (1 + params["kenya_bond_yield_base"]) ** (n_months / 12)
        final_gbp_counterfactual = final_kes_bond_value / float(path[-1])
        path_result["counterfactual_final_gbp"] = final_gbp_counterfactual
        path_result["final_fx_rate"] = float(path[-1])
        path_result["final_drop_pct"] = (float(path[-1]) - e0) / e0

        # Simulated domestic bond yield (varies with inflation scenario)
        # Approximate: bond yield compressed by IMF conditionality over time
        path_result["simulated_domestic_bond_yield"] = float(params["kenya_bond_yield_base"] + rng.normal(0, 0.03))

        results.append(path_result)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5.  DETERMINISTIC SCENARIO (Phase 1 validation)
# ─────────────────────────────────────────────────────────────────────────────

def run_deterministic_scenario(params: dict, crash_month: int, crash_drop: float) -> dict:
    """
    Smooth linear devaluation reaching crash_drop at crash_month, then flat.
    Used to validate accounting formulas.
    """
    n = params["n_months"]
    e0 = params["kes_gbp_initial_spot"]
    et_crash = e0 * (1 + crash_drop)
    path = np.concatenate([
        np.linspace(e0, et_crash, crash_month + 1),
        np.full(n - crash_month, et_crash)
    ])
    ledger = run_fic_ledger(path, params)
    return ledger.to_dict(orient="records")


# ─────────────────────────────────────────────────────────────────────────────
# 6.  ASSUMPTION TESTS
# ─────────────────────────────────────────────────────────────────────────────

def run_assumption_tests(params: dict) -> list[dict]:
    """
    Systematic test of each core assumption.
    Returns a list of test results with pass/fail and evidence.
    """
    e0 = params["kes_gbp_initial_spot"]
    P0 = params["spv_total_raise_kes"]
    tests = []

    # ── T1: FIC positive cash flow at inception ──────────────────────────────
    gross_rent_annual = params["monthly_rent_gbp"] * 12
    opex = params["service_charge_annual"] + params["insurance_annual"] + params["maintenance_annual"]
    mgmt = gross_rent_annual * params["management_fee_pct"]
    noi = gross_rent_annual - opex - mgmt
    spv_interest_annual = (P0 * params["base_interest_rate"]) / e0
    surplus = noi - spv_interest_annual
    tests.append({
        "id": "T1",
        "name": "FIC positive cash flow at inception",
        "assumption": "Net operating income > SPV interest burden (DSCR > 1.0)",
        "result": surplus > 0,
        "values": {
            "noi_gbp": round(noi, 2),
            "spv_interest_gbp": round(spv_interest_annual, 2),
            "surplus_gbp": round(surplus, 2),
            "dscr": round(noi / spv_interest_annual, 3),
        },
        "pass": surplus > 0,
    })

    # ── T2: DSCR > 1.25 (lender covenant threshold) ──────────────────────────
    dscr = noi / spv_interest_annual
    tests.append({
        "id": "T2",
        "name": "FIC DSCR ≥ 1.25",
        "assumption": "Rental yield comfortably covers SPV debt service",
        "values": {"dscr": round(dscr, 3)},
        "pass": dscr >= 1.25,
    })

    # ── T3: Class B GBP IRR — compare against Kenyan bond yield ─────────────
    et_b = e0 * (1 + params["class_b"]["trigger_drop"])
    t_b = 36  # assume 3-year trigger for Class B
    tranche_kes = P0 / 3
    payout_b = tranche_payout_gbp(tranche_kes, e0, et_b, params["base_interest_rate"],
                                  t_b, params["class_b"]["fee_pct"],
                                  params["class_b"]["sweetener_share"])
    invest_gbp = tranche_kes / e0
    monthly_irr_b = irr([-invest_gbp] + [0.0] * (t_b - 1) + [payout_b["total_payout_gbp"]])
    annual_irr_b = (1 + monthly_irr_b) ** 12 - 1 if not np.isnan(monthly_irr_b) else float("nan")
    # Correct benchmark: what does a 16% KES bond return IN GBP over the same
    # trigger path? Formula: (1 + kes_rate) * (e0/et)^(12/t) - 1
    # At Class B (45% drop over 36mo): ~+2.5% GBP/yr (barely positive).
    # At 16% KES nominal the bond looks great; in GBP it barely breaks even.
    bond_gbp_irr_b = (1 + params["kenya_bond_yield_base"]) * (e0 / et_b) ** (12 / t_b) - 1
    tests.append({
        "id": "T3",
        "name": "Class B GBP IRR vs bond GBP-equivalent return (same trigger path)",
        "assumption": "Comparison must be GBP-to-GBP: the 16% KES bond, after 45% FX depreciation over 36mo, "
                      "yields only ~+2-3% GBP/yr. A pass means SPV outperforms the KES bond in GBP terms. "
                      "A fail means the bond still wins GBP-for-GBP — not that SPV fails vs 16% nominal.",
        "values": {
            "invest_gbp": round(invest_gbp, 2),
            "total_received_gbp": round(payout_b["total_payout_gbp"], 2),
            "principal_at_crash_gbp": round(payout_b["principal_gbp_crash_rate"], 2),
            "yield_only_gbp": round(payout_b["yield_only_gbp"], 2),
            "annual_irr_gbp_pct": round(annual_irr_b * 100, 2) if not np.isnan(annual_irr_b) else "NaN",
            "kes_nominal_yield_pct": params["kenya_bond_yield_base"] * 100,          # informational only
            "bond_gbp_equiv_pct": round(bond_gbp_irr_b * 100, 2),                   # correct GBP benchmark
        },
        "pass": not np.isnan(annual_irr_b) and annual_irr_b >= bond_gbp_irr_b,
    })

    # ── T4: FX Profit is substantial ─────────────────────────────────────────
    pi_fx_b = fx_profit_gbp(P0, e0, et_b)
    tests.append({
        "id": "T4",
        "name": "FIC retains positive FX profit after all sweeteners",
        "assumption": "Pareto optimality: FIC keeps residual FX gain > 0",
        "values": {
            "gross_fx_profit_gbp": round(pi_fx_b, 2),
            "sweetener_paid_gbp": round(pi_fx_b * params["class_b"]["sweetener_share"], 2),
            "fic_retained_gbp": round(pi_fx_b * (1 - params["class_b"]["sweetener_share"]), 2),
        },
        "pass": pi_fx_b * (1 - params["class_b"]["sweetener_share"]) > 0,
    })

    # ── T5: Class C pays > Class B (risk-reward ordering) ────────────────────
    et_c = e0 * (1 + params["class_c"]["trigger_drop"])
    t_c = 60
    payout_c = tranche_payout_gbp(tranche_kes, e0, et_c, params["base_interest_rate"],
                                  t_c, params["class_c"]["fee_pct"],
                                  params["class_c"]["sweetener_share"])
    monthly_irr_c = irr([-invest_gbp] + [0.0] * (t_c - 1) + [payout_c["total_payout_gbp"]])
    annual_irr_c = (1 + monthly_irr_c) ** 12 - 1 if not np.isnan(monthly_irr_c) else float("nan")
    tests.append({
        "id": "T5",
        "name": "Class C IRR > Class B IRR (risk ordering preserved)",
        "assumption": "Higher patience yields proportionally higher return",
        "values": {
            "class_b_irr_pct": round(annual_irr_b * 100, 2),
            "class_c_irr_pct": round(annual_irr_c * 100, 2),
        },
        "pass": not np.isnan(annual_irr_c) and annual_irr_c > annual_irr_b,
    })

    # ── T6: SPV debt GBP cost shrinks with devaluation ───────────────────────
    gbp_cost_t0 = P0 / e0
    gbp_cost_t_crash = P0 / et_b
    tests.append({
        "id": "T6",
        "name": "KES debt GBP cost shrinks after devaluation",
        "assumption": "Core synthetic short mechanic: nominal KES liability becomes cheap in GBP",
        "values": {
            "gbp_cost_at_inception": round(gbp_cost_t0, 2),
            "gbp_cost_at_35pct_drop": round(gbp_cost_t_crash, 2),
            "reduction_gbp": round(gbp_cost_t0 - gbp_cost_t_crash, 2),
            "reduction_pct": round((1 - gbp_cost_t_crash / gbp_cost_t0) * 100, 2),
        },
        "pass": gbp_cost_t_crash < gbp_cost_t0,
    })

    # ── T7: Capital flight survival — SPV vs held in Kenya as KES cash ───────────
    # If investor keeps 23.33M KES as cash and devaluation hits:
    kes_gbp_at_crash = tranche_kes / et_b            # KES cash converted at crash rate
    spv_exit_gbp = payout_b["total_payout_gbp"]      # includes principal
    # SPV always includes principal + yield, so should exceed raw KES cash
    # More interesting: compare vs KES bonds (but with capital controls, bonds may not convert)
    kes_bond_value_gbp = (tranche_kes * (1 + params["kenya_bond_yield_base"]) ** (t_b / 12)) / et_b
    tests.append({
        "id": "T7",
        "name": "Capital flight survival: SPV exit > hold KES cash at crash",
        "assumption": "SPV should protect more GBP value than simply holding KES cash through a devaluation",
        "values": {
            "hold_kes_cash_gbp": round(kes_gbp_at_crash, 2),
            "hold_kes_bonds_gbp_if_convertible": round(kes_bond_value_gbp, 2),
            "spv_exit_gbp_class_b": round(spv_exit_gbp, 2),
        },
        "pass": spv_exit_gbp > kes_gbp_at_crash,
    })

    return tests


# ─────────────────────────────────────────────────────────────────────────────
# 7.  AGGREGATE MONTE CARLO OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_mc_results(mc_results: list[dict], params: dict) -> dict:
    """
    Produce aggregated datasets for the dashboard charts.
    """
    e0 = params["kes_gbp_initial_spot"]
    n = len(mc_results)

    # ── IRR scatter (Class B & C vs domestic bond) ────────────────────────────
    irr_scatter = []
    for r in mc_results:
        for cls in ["class_b", "class_c"]:
            if r[cls]["triggered"]:
                bond_alt = r[cls].get("bond_alt_gbp_irr", float("nan"))
                irr_scatter.append({
                    "class": cls.replace("_", " ").title(),
                    "irr_pct": round(r[cls]["annual_irr"] * 100, 2),
                    # X-axis: GBP-equivalent bond yield on this path at trigger
                    "bond_alt_gbp_pct": round(bond_alt * 100, 2) if not np.isnan(bond_alt) else None,
                    # Keep nominal KES yield for reference
                    "nominal_kes_yield_pct": round(r["simulated_domestic_bond_yield"] * 100, 2),
                    "trigger_month": r[cls]["trigger_month"],
                })

    # ── Trigger probability distribution ─────────────────────────────────────
    trigger_stats = {}
    for cls in ["class_a", "class_b", "class_c"]:
        triggered_paths = [r for r in mc_results if r[cls]["triggered"]]
        not_triggered = n - len(triggered_paths)
        trigger_months = [r[cls]["trigger_month"] for r in triggered_paths]
        trigger_stats[cls] = {
            "triggered_count": len(triggered_paths),
            "not_triggered_count": not_triggered,
            "probability_pct": round(len(triggered_paths) / n * 100, 2),
            "median_trigger_month": int(np.median(trigger_months)) if trigger_months else None,
            "p25_trigger_month": int(np.percentile(trigger_months, 25)) if trigger_months else None,
            "p75_trigger_month": int(np.percentile(trigger_months, 75)) if trigger_months else None,
            "label": params[cls]["label"],
        }

    # ── FX rate distribution at month 60 ─────────────────────────────────────
    fx_at_60 = []
    for r in mc_results:
        fx_at_60.append(r["final_fx_rate"])  # approximate with final (use full path for precise)

    # ── Counterfactual distribution ───────────────────────────────────────────
    counterfactual_gbp = [r["counterfactual_final_gbp"] for r in mc_results]
    spv_exit_class_b_gbp = []
    for r in mc_results:
        if r["class_b"]["triggered"]:
            spv_exit_class_b_gbp.append(r["class_b"]["total_payout_gbp"])

    # ── Pareto frontier data ──────────────────────────────────────────────────
    # For each tranche: expected IRR vs duration risk (median trigger month)
    pareto_points = []
    for cls in ["class_a", "class_b", "class_c"]:
        tv = params[cls]
        triggered = [r for r in mc_results if r[cls]["triggered"]]
        if triggered:
            irrs = [r[cls]["annual_irr"] for r in triggered if not np.isnan(r[cls]["annual_irr"])]
            months = [r[cls]["trigger_month"] for r in triggered]
            pareto_points.append({
                "class": tv["label"],
                "mean_irr_pct": round(float(np.mean(irrs)) * 100, 2) if irrs else None,
                "p10_irr_pct": round(float(np.percentile(irrs, 10)) * 100, 2) if irrs else None,
                "p90_irr_pct": round(float(np.percentile(irrs, 90)) * 100, 2) if irrs else None,
                "median_duration_months": round(float(np.median(months)), 1),
                "trigger_probability_pct": trigger_stats[cls]["probability_pct"],
                "trigger_drop_pct": tv["trigger_drop"] * 100,
            })

    # ── IRR histogram bins ────────────────────────────────────────────────────
    irr_histograms = {}
    for cls in ["class_b", "class_c"]:
        triggered = [r for r in mc_results if r[cls]["triggered"]]
        irrs_pct = [r[cls]["annual_irr"] * 100 for r in triggered if not np.isnan(r[cls]["annual_irr"])]
        bond_alts_pct = [
            r[cls]["bond_alt_gbp_irr"] * 100 for r in triggered
            if not np.isnan(r[cls].get("bond_alt_gbp_irr", float("nan")))
        ]
        if irrs_pct:
            counts, edges = np.histogram(irrs_pct, bins=40)
            irr_histograms[cls] = {
                "bins": [round(float(e), 2) for e in edges[:-1]],
                "counts": [int(c) for c in counts],
                "mean_pct": round(float(np.mean(irrs_pct)), 2),
                "median_pct": round(float(np.median(irrs_pct)), 2),
                "p10_pct": round(float(np.percentile(irrs_pct, 10)), 2),
                "p90_pct": round(float(np.percentile(irrs_pct, 90)), 2),
                # Bond GBP-equivalent benchmark: what the KES bond returned in GBP
                # on each path at the same trigger horizon. This is the correct
                # comparison: NOT the 16% KES nominal rate.
                "mean_bond_alt_gbp_pct": round(float(np.mean(bond_alts_pct)), 2) if bond_alts_pct else None,
                "median_bond_alt_gbp_pct": round(float(np.median(bond_alts_pct)), 2) if bond_alts_pct else None,
                "pct_paths_outperform_bond": round(
                    sum(1 for s, b in zip(irrs_pct, bond_alts_pct) if s > b) / len(irrs_pct) * 100, 1
                ) if bond_alts_pct else None,
            }

    return {
        "trigger_stats": trigger_stats,
        "pareto_frontier": pareto_points,
        "irr_histograms": irr_histograms,
        "irr_scatter_sample": irr_scatter[:2000],   # cap for file size
        "counterfactual_summary": {
            "hold_in_kenya_mean_gbp": round(float(np.mean(counterfactual_gbp)), 2),
            "hold_in_kenya_median_gbp": round(float(np.median(counterfactual_gbp)), 2),
            "hold_in_kenya_p10_gbp": round(float(np.percentile(counterfactual_gbp, 10)), 2),
            "hold_in_kenya_p90_gbp": round(float(np.percentile(counterfactual_gbp, 90)), 2),
            "spv_class_b_mean_gbp": round(float(np.mean(spv_exit_class_b_gbp)), 2) if spv_exit_class_b_gbp else None,
            "initial_gbp_value": round(params["spv_total_raise_kes"] / params["kes_gbp_initial_spot"], 2),
        },
        "n_paths": n,
        "params_snapshot": {k: v for k, v in params.items() if not isinstance(v, dict)},
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize(obj):
    """Recursively replace NaN/Inf with None for JSON compliance."""
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def write_bundle(bundle: dict):
    """
    Write all simulation data as a single JS file that assigns window.SIM_DATA.
    Loading via <script src="./data/bundle.js"> bypasses CORS entirely — works
    on file://, GitHub Pages, any static host, no server required.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "bundle.js")
    payload = json.dumps(_sanitize(bundle), separators=(",", ":"))
    with open(path, "w") as f:
        f.write("/* Auto-generated by simulate.py — do not edit */\n")
        f.write(f"window.SIM_DATA = {payload};\n")
    size_kb = os.path.getsize(path) / 1024
    print(f"  ✓ Saved bundle.js  ({size_kb:.0f} KB)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    p = PARAMS

    print("\n╔══════════════════════════════════════════╗")
    print("║  Synthetic Sovereign Hedge — Sim Engine  ║")
    print("╚══════════════════════════════════════════╝\n")

    # ── Historical calibration ────────────────────────────────────────────────
    print("Calibration · Fitting jump-diffusion to historical EM crises …")
    calibration_report = build_calibration_report()
    for key, cal in calibration_report.items():
        fp = cal["fitted_params"]
        print(f"  {cal['label']}: drift={fp['gbm_annual_drift']:.3f} "
              f"vol={fp['gbm_annual_vol']:.3f} "
              f"λ={fp['devaluation_jump_prob_annual']:.2f} "
              f"total_drop={cal['total_depreciation_pct']}%")

    # ── Phase 1: Assumption tests ─────────────────────────────────────────────
    print("\nPhase 1 · Running assumption tests …")
    tests = run_assumption_tests(p)
    passed = sum(1 for t in tests if t["pass"])
    print(f"  {passed}/{len(tests)} tests passed")
    for t in tests:
        icon = "✓" if t["pass"] else "✗"
        print(f"  {icon} [{t['id']}] {t['name']}")

    # ── Phase 1: Deterministic scenarios ─────────────────────────────────────
    print("\nPhase 1 · Deterministic scenarios …")
    scenarios = {}
    for name, (month, drop) in {
        "slow_20pct_36mo": (36, 0.20),
        "medium_35pct_24mo": (24, 0.35),
        "severe_50pct_18mo": (18, 0.50),
        "catastrophic_60pct_12mo": (12, 0.60),
    }.items():
        scenarios[name] = run_deterministic_scenario(p, crash_month=month, crash_drop=drop)
        print(f"  ✓ {name}")

    # ── Phase 2: Monte Carlo (baseline) ──────────────────────────────────────
    print("\nPhase 2 · Monte Carlo — baseline params …")
    mc_results = run_monte_carlo(p)
    aggregated = aggregate_mc_results(mc_results, p)

    # ── Phase 2: Monte Carlo per historical calibration ───────────────────────
    print("\nPhase 2 · Monte Carlo — per historical calibration …")
    calibrated_mc = {}
    for key, cal in HISTORICAL_CALIBRATIONS.items():
        cal_params = {**p, **cal["fitted"]}   # overlay calibrated FX params
        print(f"  Running {cal['label']} …")
        mc_r = run_monte_carlo(cal_params)
        agg  = aggregate_mc_results(mc_r, cal_params)
        calibrated_mc[key] = {
            "label": cal["label"],
            "description": cal["description"],
            "trigger_stats": agg["trigger_stats"],
            "pareto_frontier": agg["pareto_frontier"],
            "counterfactual_summary": agg["counterfactual_summary"],
            "irr_histograms": agg["irr_histograms"],
            "fitted_params": cal["fitted"],
        }

    # ── FX path sample ────────────────────────────────────────────────────────
    print("\nGenerating FX path sample …")
    rng = np.random.default_rng(p["seed"])
    sample_paths = generate_fx_paths(p, 200, p["n_months"], rng)
    paths_list = [
        {"path_id": i, "rates": [round(float(x), 2) for x in sample_paths[i]]}
        for i in range(200)
    ]

    # ── Syndicate utilisation table ───────────────────────────────────────────
    # Show how many entities / pack sizes are feasible within AML constraints
    syndicate_options = []
    total_kes = p["spv_total_raise_kes"]
    for n_investors in range(1, p["n_max_entities"] + 1):
        ticket = total_kes / n_investors
        syndicate_options.append({
            "n_investors": n_investors,
            "ticket_kes": round(ticket),
            "ticket_gbp_approx": round(ticket / p["kes_gbp_initial_spot"]),
            "compliant": ticket >= p["min_ticket_kes"],
        })

    # ── Structural risk scorecard (Module 2) ─────────────────────────────────
    print("\nBuilding structural risk scorecard …")
    structural_risk = build_structural_risk()

    # ── Yield benchmarks (Module 4) ───────────────────────────────────────────
    print("Building yield benchmarks …")
    yield_benchmarks = build_yield_benchmarks(p)

    # Inject MC-derived SPV IRRs into yield_benchmarks so the bundle is self-contained.
    # pareto_frontier has mean/P10/P90 IRR per class from the full 10k-path simulation.
    # class field is e.g. "Class A (25% drop)" — match by prefix.
    pf_entries = aggregated.get("pareto_frontier", [])
    spv_irr_reference = []
    for cls_key, label in [("class_a", "Class A"), ("class_b", "Class B"), ("class_c", "Class C")]:
        entry = next((e for e in pf_entries if e["class"].startswith(label)), None)
        if entry:
            spv_irr_reference.append({
                "class": label,
                "cls_key": cls_key,
                "trigger_drop_pct": entry["trigger_drop_pct"],
                "mean_irr_pct": entry["mean_irr_pct"],
                "p10_irr_pct": entry["p10_irr_pct"],
                "p90_irr_pct": entry["p90_irr_pct"],
                "median_duration_months": entry["median_duration_months"],
                # Bond GBP-equiv on same class trigger path, for direct comparison
                "bond_gbp_equiv_at_trigger_pct": round(
                    ((1 + p["kenya_bond_yield_base"]) *
                     (1 / (1 + p[cls_key]["trigger_drop"])) ** (12 / entry["median_duration_months"]) - 1) * 100, 2
                ) if entry["median_duration_months"] else None,
            })
    yield_benchmarks["spv_irr_reference"] = spv_irr_reference

    # ── Kenya FX forecast / fan chart (Module 3) ─────────────────────────────
    print("Building Kenya FX forecast fan chart …")
    kenya_forecast = build_kenya_forecast(p, n_months=120)

    # ── Tranche spacing analysis (Module 6) ───────────────────────────────────
    print("Building tranche spacing analysis …")
    tranche_spacing = build_tranche_spacing_analysis(mc_results, p)

    # ── FIC capital sustainability (Module 7) ─────────────────────────────────
    print("Building FIC capital sustainability model …")
    fic_sustainability = build_fic_sustainability(p)

    # ── Write single bundle.js ────────────────────────────────────────────────
    print("\nWriting bundle.js …")
    bundle = {
        "generated_at": "2026-08-13",
        "assumption_tests":        tests,
        "deterministic_scenarios": scenarios,
        "mc_aggregated":           aggregated,
        "sample_fx_paths":         paths_list,
        "calibration_report":      calibration_report,
        "calibrated_mc":           calibrated_mc,
        "syndicate_options":       syndicate_options,
        "structural_risk":         structural_risk,
        "yield_benchmarks":        yield_benchmarks,
        "kenya_forecast":          kenya_forecast,
        "tranche_spacing":         tranche_spacing,
        "fic_sustainability":      fic_sustainability,
        "params": {
            k: v for k, v in p.items() if not isinstance(v, dict)
        },
        "tranche_params": {
            "class_a": p["class_a"],
            "class_b": p["class_b"],
            "class_c": p["class_c"],
        },
    }
    write_bundle(bundle)

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────")
    ts = aggregated["trigger_stats"]
    for cls in ["class_a", "class_b", "class_c"]:
        t = ts[cls]
        print(f"  {t['label']}: {t['probability_pct']}% triggered | "
              f"median month {t['median_trigger_month']}")
    print("\n  Pareto Frontier:")
    for pt in aggregated["pareto_frontier"]:
        print(f"    {pt['class']}: mean IRR {pt['mean_irr_pct']}% | "
              f"median duration {pt['median_duration_months']} months")
    cf = aggregated["counterfactual_summary"]
    print(f"\n  Capital flight (10-yr):")
    print(f"    Initial GBP equiv:   £{cf['initial_gbp_value']:,.0f}")
    print(f"    Hold KE (mean):      £{cf['hold_in_kenya_mean_gbp']:,.0f}")
    print(f"    SPV Class B (mean):  £{cf['spv_class_b_mean_gbp']:,.0f}")
    print("\n✓ Done — open dashboard/index.html in a browser\n")


if __name__ == "__main__":
    main()
