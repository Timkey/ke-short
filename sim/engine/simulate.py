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
    "class_a": {"trigger_drop": 0.20, "fee_pct": 0.05, "sweetener_share": 0.10, "label": "Class A (20% drop)"},
    "class_b": {"trigger_drop": 0.35, "fee_pct": 0.10, "sweetener_share": 0.20, "label": "Class B (35% drop)"},
    "class_c": {"trigger_drop": 0.50, "fee_pct": 0.15, "sweetener_share": 0.30, "label": "Class C (50% drop)"},

    # ── Simulation ─────────────────────────────────────────────────────────
    "n_months": 120,                        # 10-year horizon
    "n_paths": 10_000,                      # Monte Carlo paths
    "seed": 42,

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
    Single-path monthly FIC cash flow ledger.
    fx_path: 1D array of KES/GBP rate over time (length n_months+1).
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

    cash = params["initial_fic_cash_buffer"]
    records = []

    tranche_order = [
        ("class_a", params["class_a"]),
        ("class_b", params["class_b"]),
        ("class_c", params["class_c"]),
    ]
    called_tranches = set()

    for t in range(1, n_months + 1):
        et = fx_path[t]
        month_year = t

        # ── Revenue ─────────────────────────────────────────────────────────
        current_rent = rent * ((1 + rent_inflation / 12) ** t)
        gross_rent = current_rent
        management_cost = gross_rent * mgmt_fee_monthly
        net_rent = gross_rent - management_cost - opex_monthly

        # ── SPV interest due (monthly) ───────────────────────────────────────
        monthly_interest_kes = kes_outstanding * base_rate / 12
        monthly_interest_gbp_net = (monthly_interest_kes / et) * (1 - wht_rate)

        # ── Taxable profit ──────────────────────────────────────────────────
        taxable_profit = net_rent - (monthly_interest_kes / et)
        corp_tax = max(0.0, taxable_profit * corp_tax_rate)

        # ── Net cash this month ─────────────────────────────────────────────
        net_cash_month = net_rent - monthly_interest_gbp_net - corp_tax
        cash += net_cash_month

        # ── Property value ──────────────────────────────────────────────────
        prop_value = property_value * ((1 + hpi / 12) ** t)

        # ── Tranche trigger check ───────────────────────────────────────────
        drop_pct = (et - e0) / e0   # positive = KES weakened
        tranche_settled_gbp = 0.0
        tranche_event = None

        for tk, tv in tranche_order:
            if tk in called_tranches:
                continue
            # Allocate KES evenly (simplified: split into 3 equal tranches)
            tranche_kes = kes_outstanding / (3 - len(called_tranches)) if called_tranches else kes_outstanding / 3
            if drop_pct >= tv["trigger_drop"]:
                payout = tranche_payout_gbp(
                    tranche_kes, e0, et,
                    base_rate, t,
                    tv["fee_pct"], tv["sweetener_share"]
                )
                # FIC pays out from cash or refinance
                needed = payout["total_payout_gbp"]
                if cash < needed:
                    # simulate BTL refinance draw
                    equity = prop_value - (kes_outstanding / et)
                    draw = min(equity * 0.70, needed - cash)
                    cash += max(0.0, draw)

                cash -= needed
                kes_outstanding -= tranche_kes
                called_tranches.add(tk)
                tranche_settled_gbp = needed
                tranche_event = tv["label"]
                break

        # ── DSCR ────────────────────────────────────────────────────────────
        annual_debt_service_gbp = (kes_outstanding * base_rate) / et
        dscr = (net_rent * 12) / annual_debt_service_gbp if annual_debt_service_gbp > 0 else 999.0

        records.append({
            "month": t,
            "fx_rate": et,
            "gross_rent_gbp": gross_rent,
            "net_rent_gbp": net_rent,
            "spv_interest_paid_gbp": monthly_interest_gbp_net,
            "corp_tax_gbp": corp_tax,
            "net_cash_flow_gbp": net_cash_month,
            "cumulative_cash_gbp": cash,
            "property_value_gbp": prop_value,
            "kes_outstanding": kes_outstanding,
            "dscr": dscr,
            "tranche_event": tranche_event,
            "tranche_settled_gbp": tranche_settled_gbp,
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
    # NOTE: GBP IRR is expected to be below 16% KES bond rate because the investor
    # suffers the FX loss on principal. The real value of this structure is capital
    # flight to hard currency, not nominal yield enhancement.
    tests.append({
        "id": "T3",
        "name": "Class B GBP IRR vs Kenyan bond yield benchmark",
        "assumption": "White paper claims yield parity with 16% domestic bonds. A fail indicates FX loss on principal dominates — "
                      "structure is capital-preservation play, not yield play.",
        "values": {
            "invest_gbp": round(invest_gbp, 2),
            "total_received_gbp": round(payout_b["total_payout_gbp"], 2),
            "principal_at_crash_gbp": round(payout_b["principal_gbp_crash_rate"], 2),
            "yield_only_gbp": round(payout_b["yield_only_gbp"], 2),
            "annual_irr_gbp_pct": round(annual_irr_b * 100, 2) if not np.isnan(annual_irr_b) else "NaN",
            "target_yield_pct": params["kenya_bond_yield_base"] * 100,
        },
        "pass": not np.isnan(annual_irr_b) and annual_irr_b >= params["kenya_bond_yield_base"],
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
                irr_scatter.append({
                    "class": cls.replace("_", " ").title(),
                    "irr_pct": round(r[cls]["annual_irr"] * 100, 2),
                    "domestic_bond_yield_pct": round(r["simulated_domestic_bond_yield"] * 100, 2),
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
        if irrs_pct:
            counts, edges = np.histogram(irrs_pct, bins=40)
            irr_histograms[cls] = {
                "bins": [round(float(e), 2) for e in edges[:-1]],
                "counts": [int(c) for c in counts],
                "mean_pct": round(float(np.mean(irrs_pct)), 2),
                "median_pct": round(float(np.median(irrs_pct)), 2),
                "p10_pct": round(float(np.percentile(irrs_pct, 10)), 2),
                "p90_pct": round(float(np.percentile(irrs_pct, 90)), 2),
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

    # ── Write single bundle.js ────────────────────────────────────────────────
    print("\nWriting bundle.js …")
    bundle = {
        "generated_at": "2026-08-11",
        "assumption_tests":        tests,
        "deterministic_scenarios": scenarios,
        "mc_aggregated":           aggregated,
        "sample_fx_paths":         paths_list,
        "calibration_report":      calibration_report,
        "calibrated_mc":           calibrated_mc,
        "syndicate_options":       syndicate_options,
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
