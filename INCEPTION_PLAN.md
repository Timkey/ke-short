# KES Synthetic Sovereign Hedge — Data-Driven Product Inception Plan
## Branch: `feature/data-driven-inception`

---

## 1. Strategic Objective

Transform the proof-of-concept simulation into a **client-ready, fact-grounded investment presentation** 
that can withstand due diligence. Every assumption must be traceable to a published source.
Every chart must tell a decision-relevant story.

---

## 2. Core Problems to Solve

### 2.1 Tranche Trigger Calibration (Currently Arbitrary)
Current triggers (20% / 35% / 50%) were set by hand. We need:
- **Historical trigger exceedance rates** — what % of real EM crises exceeded each threshold within 10 years?
- **Kenya-specific probability cone** — using IMF DSA scenarios, CBK reserve trajectory, and debt-service ratios
- **Optimal spacing analysis** — triggers must not cluster (client hits Class A and B almost simultaneously = poor structure)
- **Proposed approach**: fit a Markov regime-switching model to KES/GBP history; use regime transition probabilities to set triggers at the 33rd, 67th, 90th percentile of first-exceedance distribution

### 2.2 Benchmark Bond Yield (Currently Hardcoded at 16%)
Kenya's yield environment is dynamic:
- **Kenya 91-day T-bill**: ~15–18% (CBK auction data, 2023–2025)
- **Kenya 10-year bond**: ~17–19%
- **Kenya Eurobond (USD)**: 9–11% + USD/KES risk premium ≈ 22–28% GBP equivalent
- **UK Gilt 10yr**: ~4.2% (2026)
- **Proposed**: dynamic benchmark selector in dashboard — user picks comparison rate, all IRR charts update

### 2.3 Rate Stabilisation Mechanisms (Missing Entirely)
Kenya has active intervention tools:
- **CBK FX reserves** (currently ~$7.1bn = ~3.8 months import cover)
- **IMF RCF/EFF facility** ($941m drawn, $2.3bn available)
- **CBK intervention history**: defended 165 KES/USD in 2023, failed at 160
- **Proposed**: overlay "intervention band" on FX path charts; model probability that CBK holds rate in each quarter using logistic regression on reserves/imports ratio

### 2.4 Cross-Country Structural Risk Scorecard (Currently Cosmetic)
Current calibration tab shows parameters but not WHY they differ. We need:

| Factor | Kenya | Ghana 2022 | Zambia 2015 | Egypt 2022 | Turkey 2021 |
|--------|-------|-----------|-------------|-----------|-------------|
| Debt/GDP | 68% | 93% | 106% | 87% | 40% |
| Reserves (months imports) | 3.8 | 1.2 | 1.8 | 3.1 | 4.2 |
| Current Account (% GDP) | -4.2% | -3.1% | -2.8% | -3.9% | -1.7% |
| IMF Program | EFF active | HIPC | HIPC | SBA | None |
| Political Risk Score | 45/100 | 52/100 | 48/100 | 58/100 | 61/100 |
| Similarity to Kenya | baseline | HIGH | HIGH | MEDIUM | LOW |

- **Proposed**: radar chart comparing structural factors; weighting matrix to compute "crisis similarity score"

### 2.5 Master Variable Panel (Missing)
A single control panel that governs ALL analysis:
- Initial KES/GBP spot rate
- CBK reserves level (feeds intervention probability)
- IMF programme status (modifies jump probability)
- UK base rate (feeds mortgage cost)
- Kenya bond yield (feeds benchmark)
- Property yield (feeds FIC cash flows)
- Monte Carlo paths (speed vs accuracy)

When any master variable changes → all charts, all tables, all KPIs recalculate live.

---

## 3. Build Modules (Implementation Plan)

### MODULE 1: Engine — Historical Rate Series in Bundle *(Fix)*
**File**: `sim/engine/simulate.py`  
**Change**: Write `monthly_rates` from `HISTORICAL_CALIBRATIONS` into bundle JSON  
**Effort**: 30 min  
**Impact**: Calibration tab shows REAL historical FX trajectories  

### MODULE 2: Engine — Structural Risk Scorecard Data
**File**: `sim/engine/simulate.py`  
**New section**: `STRUCTURAL_RISK_FACTORS` dict with IMF/World Bank data for each country  
**Output**: `bundle.structural_risk` with radar chart data  
**Sources**: IMF WEO April 2025, World Bank Macro Poverty Outlook  
**Effort**: 1 hour  

### MODULE 3: Engine — Kenya Depreciation Path with Regime Model
**New section**: `_kenya_regime_path()`  
- Fit 2-state Markov (stable / crisis) to KES/GBP 2015–2025  
- Compute forward depreciation bands at 50%, 80%, 95% confidence  
- Identify "trigger warning" zones where Class A breach becomes likely within 6 months  
**Output**: `bundle.kenya_forecast` with probability cones  
**Effort**: 3 hours  

### MODULE 4: Engine — Bond Yield Curve & Benchmark Matrix
**New section**: `KENYA_YIELD_CURVE`  
- 91-day, 182-day, 1yr, 2yr, 5yr, 10yr Kenya government securities  
- Eurobond spread vs UST  
- GBP-equivalent yield (adjusting for FX expectation)  
- SPV tranche IRR vs each benchmark  
**Output**: `bundle.yield_benchmarks`  
**Effort**: 1 hour  

### MODULE 5: Engine — CBK Intervention Model
**New section**: `_intervention_probability(reserves_months, imf_active)`  
- Logistic model: P(intervention) = f(reserves cover, IMF active, months since last shock)  
- Output intervention bands by quarter over 10-year horizon  
**Output**: `bundle.intervention_bands`  
**Effort**: 2 hours  

### MODULE 6: Engine — Tranche Spacing Optimiser
**New section**: `_optimise_triggers(paths, min_spacing_months=6)`  
- For each trigger pair (A,B) and (B,C): compute P(B triggered within 6 months of A)  
- Find trigger levels where spacing probability < 10%  
- Report whether current 20/35/50 structure passes  
**Output**: `bundle.tranche_spacing_analysis`  
**Effort**: 2 hours  

### MODULE 7: Dashboard — Master Variable Control Panel
**New tab**: "Master Controls"  
- 8 key sliders that drive all other tabs  
- "Recalculate All" button that re-runs MC in JS (simplified 1000-path version)  
- Live KPI propagation  
**Effort**: 3 hours  

### MODULE 8: Dashboard — Regime & Depreciation Forecast Tab
**New tab**: "KES Forecast"  
- Probability cone chart (fan chart, like Bank of England style)  
- Regime state probability over time (heat map)  
- CBK intervention band overlay  
- Early warning traffic light (Red/Amber/Green based on reserves)  
**Effort**: 3 hours  

### MODULE 9: Dashboard — Structural Risk Comparison Tab
**New tab**: "EM Risk Comparison"  
- Radar/spider chart: Kenya vs analogues on 6 structural factors  
- Weighted crisis similarity score  
- Counterfactual: "If Kenya follows Ghana path, Class A triggers in month X"  
**Effort**: 2 hours  

### MODULE 10: Dashboard — Yield Benchmark Tab  
**New tab**: "Yield Analysis"  
- Kenya yield curve (term structure)  
- SPV tranche IRR vs benchmark matrix (table + bar chart)  
- GBP-equivalent yield calculator  
- Breakeven analysis: at what FX rate does SPV outperform holding KES bonds?  
**Effort**: 2 hours  

### MODULE 11: Dashboard — Client Demo Narrative Mode
**New feature**: "Presentation Mode" button  
- Hides technical controls, shows clean investor-facing summary  
- Guided tab sequence with explanatory callout boxes  
- Print/PDF export layout  
**Effort**: 2 hours  

---

## 4. Data Sources & Traceability

| Data Point | Source | Last Updated |
|------------|--------|-------------|
| KES/GBP historical rates | CBK Statistical Bulletin | Monthly |
| Kenya debt/GDP | IMF Article IV 2024 | Dec 2024 |
| CBK FX reserves | CBK Weekly Bulletin | Weekly |
| Kenya T-bill yields | CBK Primary Dealer auctions | Weekly |
| Kenya Eurobond (KENINT) | Bloomberg / CBK | Daily |
| IMF programme status | IMF.org press releases | As published |
| Ghana/Zambia/Egypt/Turkey rates | BIS FX database | Monthly |
| UK HPI | ONS UK House Price Index | Monthly |
| UK Gilt yields | DMO / BoE | Daily |
| UK corp tax rate | HMRC | As legislated (25% from Apr 2023) |

---

## 5. Priority Order (what to build first for demo)

1. **Module 1** (historical rates in bundle) — 30 min, immediate visual fix
2. **Module 4** (yield benchmarks) — 1 hour, core investment thesis validation
3. **Module 2** (structural risk scorecard) — 1 hour, differentiation from generic EM pitches
4. **Module 7** (master controls) — 3 hours, "wow factor" for live demo
5. **Module 8** (KES forecast tab) — 3 hours, the scientific backbone
6. **Module 6** (tranche spacing) — 2 hours, product integrity proof
7. **Module 3** (regime model) — 3 hours, most sophisticated analysis
8. **Modules 9–11** — polish and presentation

---

## 6. Key Questions That Must Be Answered

1. **What is the correct KES/GBP benchmark?** (GBP investors care about GBP return, not USD)
2. **Does the structure survive a Ghana-style crisis?** (worst case: 55% drop in 18 months)
3. **Are the tranches too close together?** (risk of simultaneous triggers destroying sequencing logic)
4. **What is the real IRR after UK tax?** (current model has 19% corp tax; UK also has SDLT, CGT)
5. **At what reserves level should an investor exit?** (early warning threshold)
6. **How does the SPV perform if Kenya secures a new IMF programme mid-life?** (positive shock)
7. **What is the AML/KYC cost per investor?** (affects economics at low investor count)

---

## 7. Technical Architecture for v2

```
sim/
├── engine/
│   ├── simulate.py          (extend — add 6 new data modules)
│   ├── calibrate.py         (NEW — Markov regime fitting, standalone)
│   ├── data/
│   │   ├── kenya_rates.csv  (NEW — monthly KES/GBP 2015-2026)
│   │   ├── em_risk.json     (NEW — structural factors per country)
│   │   └── yield_curve.json (NEW — Kenya term structure snapshots)
│   └── requirements.txt     (add: statsmodels, hmmlearn)
├── dashboard/
│   ├── index.html           (extend — add 4 new tabs + master controls)
│   └── data/
│       └── bundle.js        (regenerated, larger ~800KB)
└── docker-compose.yml       (unchanged)
```

---

## 8. What Makes This Client-Ready

A client will ask three questions:

**"Why will KES depreciate?"**  
→ Answer: Module 2 structural scorecard + Module 8 regime model showing Kenya on same trajectory as Ghana 2022 with 18-month lag

**"Will your structure pay out when it matters?"**  
→ Answer: Module 6 tranche spacing + Module 3 regime model showing triggers fire in the right sequence

**"What do I actually earn, net of UK tax and FX costs?"**  
→ Answer: Module 4 yield benchmark table showing GBP net IRR vs UK gilts, UK property, and KES domestic bonds

---

*Plan generated: 2026-08-13. Branch: feature/data-driven-inception.*
