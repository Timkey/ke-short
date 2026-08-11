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
}

OUTPUT_DIR = "/data"

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


def save_json(data, filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(_sanitize(data), f, indent=2)
    print(f"  ✓ Saved {filename}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    p = PARAMS

    print("\n╔══════════════════════════════════════════╗")
    print("║  Synthetic Sovereign Hedge — Sim Engine  ║")
    print("╚══════════════════════════════════════════╝\n")

    # ── Phase 1: Assumption tests ─────────────────────────────────────────────
    print("Phase 1 · Running assumption tests …")
    tests = run_assumption_tests(p)
    passed = sum(1 for t in tests if t["pass"])
    print(f"  {passed}/{len(tests)} tests passed")
    for t in tests:
        icon = "✓" if t["pass"] else "✗"
        print(f"  {icon} [{t['id']}] {t['name']}")
    save_json(tests, "assumption_tests.json")

    # ── Phase 1: Deterministic scenario (35 % crash at month 24) ─────────────
    print("\nPhase 1 · Deterministic scenario: 35 % crash at month 24 …")
    det_ledger = run_deterministic_scenario(p, crash_month=24, crash_drop=0.35)
    save_json(det_ledger, "deterministic_ledger.json")

    # Additional deterministic scenarios
    scenarios = {}
    for name, (month, drop) in {
        "slow_20pct_36mo": (36, 0.20),
        "medium_35pct_24mo": (24, 0.35),
        "severe_50pct_18mo": (18, 0.50),
        "catastrophic_60pct_12mo": (12, 0.60),
    }.items():
        scenarios[name] = run_deterministic_scenario(p, crash_month=month, crash_drop=drop)
    save_json(scenarios, "deterministic_scenarios.json")

    # ── Phase 2: Monte Carlo ──────────────────────────────────────────────────
    print("\nPhase 2 · Monte Carlo simulation …")
    mc_results = run_monte_carlo(p)

    print("Phase 2 · Aggregating results …")
    aggregated = aggregate_mc_results(mc_results, p)
    save_json(aggregated, "mc_aggregated.json")

    # Save a sample of raw paths for the FX distribution chart
    rng = np.random.default_rng(p["seed"])
    sample_paths = generate_fx_paths(p, 200, p["n_months"], rng)
    paths_list = []
    for i in range(200):
        paths_list.append({
            "path_id": i,
            "rates": [round(float(x), 2) for x in sample_paths[i]],
        })
    save_json(paths_list, "sample_fx_paths.json")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Summary ──────────────────────────────────────────")
    ts = aggregated["trigger_stats"]
    for cls in ["class_a", "class_b", "class_c"]:
        t = ts[cls]
        print(f"  {t['label']}: {t['probability_pct']}% triggered | "
              f"median month {t['median_trigger_month']}")
    print(f"\n  Pareto Frontier:")
    for pt in aggregated["pareto_frontier"]:
        print(f"    {pt['class']}: mean IRR {pt['mean_irr_pct']}% | "
              f"median duration {pt['median_duration_months']} months")

    cf = aggregated["counterfactual_summary"]
    print(f"\n  Capital flight comparison (10-year horizon):")
    print(f"    Initial GBP value of 70M KES:     £{cf['initial_gbp_value']:,.0f}")
    print(f"    Hold in Kenya (mean):              £{cf['hold_in_kenya_mean_gbp']:,.0f}")
    print(f"    SPV Class B exit (mean):           £{cf['spv_class_b_mean_gbp']:,.0f}")
    print("\n✓ All datasets written to /data/\n")


if __name__ == "__main__":
    main()
