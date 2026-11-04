"""
ESG Portfolio Optimizer: Bottom-Up Telemetry Orchestrator & Stochastic SFDR Art. 9 Capital Allocator.
Author: Juan Diego Ussa Aponte (Valencia, Spain)
Neutral GitHub Library Name: esg-portfolio-optimizer
Architecture: Zero Placeholders, Analytical Jacobian SLSQP + Vectorized Monte Carlo CVaR_95.

Integrates the 3 operational layers of Zubi Group:
1. Bottom-Up Regulatory Telemetry (CoCircular): Ley 7/2022 LER mass-balance & Scope 3 audit.
2. Bottom-Up Physical/SciML Telemetry (Matteco): AEM/NiFe electrolyzer Euler-Maruyama SDE & LCOH.
3. Top-Down Capital Allocation (Zubi Labs / Zubi Capital): Multi-objective non-linear Pareto
   optimization (SLSQP) with Monte Carlo Conditional Value-at-Risk (CVaR 95%) impact verification
   aligned with EU SFDR Article 9 and GHG Protocol Scopes 1, 2, and 3.
"""

from __future__ import annotations

import time
from typing import Sequence, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from cocircular_waste_auditor import (
    audit_waste_transfer_notes,
    build_valencia_construction_waste_batch,
)
from domain_models import (
    ElectrolyzerSimulationResult,
    PortfolioAllocationResult,
    VentureImpactProfile,
    WasteAuditSummary,
)
from matteco_h2_kinetics import (
    build_matteco_catalyst_comparison_cells,
    simulate_electrolyzer_degradation_sde,
)


def build_zubi_ecosystem_portfolio(
    cocircular_audit: WasteAuditSummary | None = None,
    matteco_sim: ElectrolyzerSimulationResult | None = None,
) -> Tuple[VentureImpactProfile, ...]:
    """
    Constructs the Zubi Group portfolio of 5 Valencia impact ventures.
    Dynamically calibrates Matteco and CoCircular abatement efficiencies from bottom-up
    physical and regulatory telemetry when provided.
    """
    # Calibrate Matteco alpha from physical H2 stack simulation and LCOH parity if available
    if matteco_sim is not None:
        # Scale physical module abatement (~187 tCO2e/module) & LCOH advantage vs 3.20 EUR/kg grey H2 parity
        lcoh_parity_factor = 3.20 / max(1.0, matteco_sim.lcoh_eur_per_kg_h2)
        matteco_alpha = round(
            max(6.0, (matteco_sim.annual_co2_abated_tons / 22.31) * lcoh_parity_factor), 2
        )
    else:
        matteco_alpha = 9.65

    # Calibrate CoCircular alpha from audited LER batch Scope 3 savings if available
    if cocircular_audit is not None:
        cocircular_alpha = round(
            max(5.0, (cocircular_audit.net_scope3_avoided_tco2e / 31.0)), 2
        )
    else:
        cocircular_alpha = 8.40

    return (
        VentureImpactProfile(
            venture_id="VNT-MTC-01",
            name="Matteco (Green H2 Catalysts - Paterna)",
            vertical="Deep-Tech / Climate Hardware",
            scope1_tco2=420.0,
            scope2_tco2=680.0,
            scope3_tco2=1850.0,
            min_investment_eur=25000.0,
            max_investment_eur=200000.0,
            expected_financial_roi=0.185,
            abatement_alpha=matteco_alpha,
            abatement_Uncertainty_sigma=0.07,
            sfdr_pai_compliant=True,
        ),
        VentureImpactProfile(
            venture_id="VNT-COC-02",
            name="CoCircular (360 LER Waste SaaS - Valencia)",
            vertical="Circular Economy SaaS",
            scope1_tco2=95.0,
            scope2_tco2=110.0,
            scope3_tco2=1480.0,
            min_investment_eur=20000.0,
            max_investment_eur=130000.0,
            expected_financial_roi=0.155,
            abatement_alpha=cocircular_alpha,
            abatement_Uncertainty_sigma=0.05,
            sfdr_pai_compliant=True,
        ),
        VentureImpactProfile(
            venture_id="VNT-WOO-03",
            name="Woodea (Industrialized Mass Timber CLT)",
            vertical="Sustainable PropTech / ConTech",
            scope1_tco2=290.0,
            scope2_tco2=230.0,
            scope3_tco2=2250.0,
            min_investment_eur=25000.0,
            max_investment_eur=180000.0,
            expected_financial_roi=0.165,
            abatement_alpha=11.20,
            abatement_Uncertainty_sigma=0.06,
            sfdr_pai_compliant=True,
        ),
        VentureImpactProfile(
            venture_id="VNT-OLE-04",
            name="Olesym Chemicals (Palm-Free Green Oleochemicals - ITQ)",
            vertical="Deep-Tech Green Chemistry / Sustainable Oleochemicals",
            scope1_tco2=140.0,
            scope2_tco2=190.0,
            scope3_tco2=890.0,
            min_investment_eur=15000.0,
            max_investment_eur=110000.0,
            expected_financial_roi=0.135,
            abatement_alpha=7.40,
            abatement_Uncertainty_sigma=0.08,
            sfdr_pai_compliant=True,
        ),
        VentureImpactProfile(
            venture_id="VNT-PIN-05",
            name="Zubi Cities / La Pinada (Regenerative Urbanism)",
            vertical="Zero-Carbon Real Estate",
            scope1_tco2=180.0,
            scope2_tco2=210.0,
            scope3_tco2=1120.0,
            min_investment_eur=15000.0,
            max_investment_eur=100000.0,
            expected_financial_roi=0.115,
            abatement_alpha=6.10,
            abatement_Uncertainty_sigma=0.05,
            sfdr_pai_compliant=True,
        ),
    )


def calculate_abatement_tons(
    investment_allocations: np.ndarray, abatement_alphas: np.ndarray
) -> np.ndarray:
    """
    Computes expected annual tons of CO2e abated under concave diminishing marginal returns (MACC):
        A_i(x_i) = 12.0 * alpha_i * sqrt(x_i / 1000.0)
    Strictly non-negative and continuously differentiable for x_i > 0.
    """
    inv_thousands = np.maximum(0.0, np.asarray(investment_allocations, dtype=np.float64)) / 1000.0
    alphas = np.asarray(abatement_alphas, dtype=np.float64)
    return 12.0 * alphas * np.sqrt(inv_thousands)


def calculate_abatement_jacobian(
    investment_allocations: np.ndarray, abatement_alphas: np.ndarray
) -> np.ndarray:
    """
    Analytical first derivative d(A_i)/d(x_i) for rapid, exact SLSQP KKT convergence:
        dA_i / dx_i = 12.0 * alpha_i / (2000.0 * sqrt(x_i / 1000.0))
    """
    inv_thousands = np.maximum(1e-9, np.asarray(investment_allocations, dtype=np.float64) / 1000.0)
    alphas = np.asarray(abatement_alphas, dtype=np.float64)
    return (12.0 * alphas) / (2000.0 * np.sqrt(inv_thousands))


def compute_max_achievable_abatement_tons(
    bounds_eur: Sequence[Tuple[float, float]],
    alphas: np.ndarray,
    total_budget_eur: float,
) -> float:
    """
    Computes the exact physical/mathematical maximum carbon abatement (tCO2e) achievable
    subject to min_i <= x_i <= max_i and sum(x_i) <= total_budget_eur.
    Since A_i(x_i) = 12 * alpha_i * sqrt(x_i / 1000) is strictly concave and separable,
    the exact maximum is found via water-filling on the KKT condition dA_i/dx_i = lambda,
    which allocates x_i proportional to alpha_i^2 within [min_i, max_i].
    """
    lows = np.array([b[0] for b in bounds_eur], dtype=np.float64)
    highs = np.array([b[1] for b in bounds_eur], dtype=np.float64)
    if total_budget_eur < float(np.sum(lows)) - 1e-9:
        raise ValueError(
            f"Infeasible budget ({total_budget_eur}) below sum of minimum bounds ({float(np.sum(lows))})."
        )
    if np.sum(highs) <= total_budget_eur + 1e-9:
        return float(np.sum(calculate_abatement_tons(highs, alphas)))

    # Binary search on the marginal abatement Lagrange multiplier (in kEUR units)
    lows_k = lows / 1000.0
    highs_k = highs / 1000.0
    budget_k = total_budget_eur / 1000.0

    # dA_i / dx_k = 6 * alpha_i / sqrt(x_k) = lam  =>  x_k(lam) = (6 * alpha_i / lam)^2
    lam_low = 1e-6
    lam_high = float(np.max(6.0 * alphas / np.sqrt(np.maximum(1e-9, lows_k)))) + 1e4
    for _ in range(80):
        lam_mid = 0.5 * (lam_low + lam_high)
        unconstrained_k = np.where(alphas > 0.0, (6.0 * alphas / lam_mid) ** 2, 0.0)
        x_k = np.clip(unconstrained_k, lows_k, highs_k)
        if np.sum(x_k) > budget_k:
            lam_low = lam_mid
        else:
            lam_high = lam_mid

    opt_x_eur = np.clip((6.0 * alphas / (0.5 * (lam_low + lam_high))) ** 2, lows_k, highs_k) * 1000.0
    return float(np.sum(calculate_abatement_tons(opt_x_eur, alphas)))


def simulate_monte_carlo_abatement_risk(
    allocations: np.ndarray,
    alphas: np.ndarray,
    sigmas: np.ndarray,
    n_scenarios: int = 2000,
    seed: int = 42,
    correlation_matrix: np.ndarray | None = None,
) -> Tuple[float, float]:
    """
    Evaluates portfolio carbon abatement tail risk under log-normal operational noise
    across n_scenarios Monte Carlo realizations, supporting cross-venture correlation
    via Cholesky decomposition.

    Returns:
        (var_95_abatement_tco2, cvar_95_abatement_tco2):
        - VaR 95%: 5th percentile worst-case carbon abatement (tons CO2e).
        - CVaR 95% (Expected Shortfall): Mean carbon abatement in the worst 5% of scenarios.
    """
    if n_scenarios < 100:
        raise ValueError("n_scenarios must be >= 100 for reliable tail risk estimation.")
    n_ventures = len(allocations)
    det_abatement = calculate_abatement_tons(allocations, alphas)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(size=(n_scenarios, n_ventures))

    if correlation_matrix is not None:
        corr = np.asarray(correlation_matrix, dtype=np.float64)
        if corr.shape != (n_ventures, n_ventures):
            raise ValueError(
                f"correlation_matrix shape {corr.shape} does not match ({n_ventures}, {n_ventures})."
            )
        if not np.allclose(corr, corr.T, atol=1e-8) or not np.allclose(np.diag(corr), 1.0, atol=1e-8):
            raise ValueError("correlation_matrix must be symmetric with unit diagonal.")
        try:
            chol = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError as exc:
            raise ValueError("correlation_matrix must be symmetric positive-definite.") from exc
        z = z @ chol.T

    # Log-normal multiplicative shock with E[shock] = 1.0
    shocks = np.exp(sigmas * z - 0.5 * (sigmas**2))
    scenario_totals = np.sum(det_abatement * shocks, axis=1)

    var_95 = float(np.percentile(scenario_totals, 5))
    tail_mask = scenario_totals <= var_95
    cvar_95 = float(np.mean(scenario_totals[tail_mask])) if np.any(tail_mask) else var_95
    return var_95, cvar_95


def optimize_stochastic_pareto_portfolio(
    ventures: Sequence[VentureImpactProfile],
    total_budget_eur: float = 450000.0,
    min_co2_abatement_target_tons: float = 4500.0,
    n_mc_scenarios: int = 2000,
    seed: int = 42,
    correlation_matrix: np.ndarray | None = None,
) -> PortfolioAllocationResult:
    """
    Solves the constrained multi-objective capital allocation problem for Zubi Capital:
        Maximize_{x}   Sum_i (x_i * ROI_i)
        Subject to:
            1. Sum_i x_i <= total_budget_eur
            2. Sum_i A_i(x_i) >= min_co2_abatement_target_tons
            3. min_investment_i <= x_i <= max_investment_i  (for SFDR PAI-compliant ventures)
    Followed by vectorized Monte Carlo VaR/CVaR 95% verification.
    """
    if not ventures:
        raise ValueError("ventures sequence cannot be empty.")
    if total_budget_eur <= 0.0:
        raise ValueError("total_budget_eur must be > 0.")
    if min_co2_abatement_target_tons < 0.0:
        raise ValueError("min_co2_abatement_target_tons cannot be negative.")

    t0 = time.perf_counter()

    # Filter out any non-PAI-compliant venture capacity under SFDR Art. 9 strict screening
    bounds = [
        (float(v.min_investment_eur), float(v.max_investment_eur))
        if v.sfdr_pai_compliant
        else (0.0, 0.0)
        for v in ventures
    ]
    min_required_sum = sum(b[0] for b in bounds)
    if total_budget_eur < min_required_sum:
        raise ValueError(
            f"Infeasible budget: total_budget_eur ({total_budget_eur}) is less than "
            f"sum of minimum venture commitments ({min_required_sum})."
        )

    rois = np.array([v.expected_financial_roi for v in ventures], dtype=np.float64)
    alphas = np.array(
        [v.abatement_alpha if v.sfdr_pai_compliant else 0.0 for v in ventures], dtype=np.float64
    )
    sigmas = np.array([v.abatement_Uncertainty_sigma for v in ventures], dtype=np.float64)

    # Zero Silent Errors: verify physical/mathematical reachability of the carbon target
    max_achievable_abated = compute_max_achievable_abatement_tons(bounds, alphas, total_budget_eur)
    if min_co2_abatement_target_tons > max_achievable_abated + 1e-6:
        raise ValueError(
            f"Infeasible carbon target: min_co2_abatement_target_tons ({min_co2_abatement_target_tons:.2f} tCO2e) "
            f"exceeds maximum achievable abatement ({max_achievable_abated:.2f} tCO2e) "
            f"under budget {total_budget_eur:,.2f} EUR and venture capacity bounds."
        )

    # Solve internally in kEUR (thousands of EUR) for O(1) Jacobian/Hessian conditioning
    budget_keur = total_budget_eur / 1000.0
    bounds_keur = [(b[0] / 1000.0, b[1] / 1000.0) for b in bounds]
    max_alloc_keur = np.array([b[1] for b in bounds_keur], dtype=np.float64)

    if np.sum(max_alloc_keur) > budget_keur:
        x0_keur = np.array([b[0] for b in bounds_keur], dtype=np.float64)
        rem_keur = budget_keur - np.sum(x0_keur)
        headroom_keur = max_alloc_keur - x0_keur
        if np.sum(headroom_keur) > 0:
            x0_keur = x0_keur + headroom_keur * min(1.0, rem_keur / np.sum(headroom_keur))
    else:
        x0_keur = max_alloc_keur.copy()

    def objective(x_keur: np.ndarray) -> float:
        return float(-np.dot(x_keur, rois))

    def objective_jac(x_keur: np.ndarray) -> np.ndarray:
        return -rois

    def constraint_budget(x_keur: np.ndarray) -> float:
        return float(budget_keur - np.sum(x_keur))

    def constraint_budget_jac(x_keur: np.ndarray) -> np.ndarray:
        return -np.ones_like(x_keur)

    def constraint_carbon(x_keur: np.ndarray) -> float:
        abated = 12.0 * alphas * np.sqrt(np.maximum(0.0, x_keur))
        return float(np.sum(abated) - min_co2_abatement_target_tons)

    def constraint_carbon_jac(x_keur: np.ndarray) -> np.ndarray:
        return (6.0 * alphas) / np.sqrt(np.maximum(1e-9, x_keur))

    constraints = [
        {"type": "ineq", "fun": constraint_budget, "jac": constraint_budget_jac},
        {"type": "ineq", "fun": constraint_carbon, "jac": constraint_carbon_jac},
    ]

    opt_res = minimize(
        objective,
        x0_keur,
        jac=objective_jac,
        method="SLSQP",
        bounds=bounds_keur,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 250},
    )

    alloc = np.clip(opt_res.x * 1000.0, [b[0] for b in bounds], [b[1] for b in bounds])
    abated_per_venture = calculate_abatement_tons(alloc, alphas)
    total_abated = float(np.sum(abated_per_venture))
    total_invested = float(np.sum(alloc))
    total_return = float(np.dot(alloc, rois))
    portfolio_roi_pct = (total_return / total_invested) * 100.0 if total_invested > 0 else 0.0
    cost_per_ton = total_invested / total_abated if total_abated > 0 else 0.0

    # Verify constraint satisfaction within numerical tolerance
    feasible_budget = total_invested <= total_budget_eur + 1e-3
    feasible_carbon = total_abated >= min_co2_abatement_target_tons - 1e-3
    converged = bool(opt_res.success and feasible_budget and feasible_carbon)

    var_95, cvar_95 = simulate_monte_carlo_abatement_risk(
        alloc,
        alphas,
        sigmas,
        n_scenarios=n_mc_scenarios,
        seed=seed,
        correlation_matrix=correlation_matrix,
    )

    exec_time_ms = (time.perf_counter() - t0) * 1000.0

    sfdr_compliant = bool(
        converged
        and all(
            v.sfdr_pai_compliant or alloc[idx] == 0.0 for idx, v in enumerate(ventures)
        )
    )

    return PortfolioAllocationResult(
        success=converged,
        execution_time_ms=round(exec_time_ms, 3),
        venture_names=tuple(v.name for v in ventures),
        allocations_eur=tuple(round(float(x), 2) for x in alloc),
        expected_abatement_per_venture_tco2=tuple(
            round(float(a), 2) for a in abated_per_venture
        ),
        total_invested_eur=round(total_invested, 2),
        expected_financial_return_eur=round(total_return, 2),
        portfolio_roi_pct=round(portfolio_roi_pct, 3),
        expected_total_abatement_tco2=round(total_abated, 2),
        var_95_abatement_tco2=round(var_95, 2),
        cvar_95_abatement_tco2=round(cvar_95, 2),
        marginal_cost_per_ton_eur=round(cost_per_ton, 2),
        sfdr_art9_compliant=sfdr_compliant,
    )


def create_venture_portfolio_data() -> pd.DataFrame:
    """
    Returns a DataFrame view of the Zubi Group portfolio profiles for reporting and tabular export.
    """
    profiles = build_zubi_ecosystem_portfolio()
    return pd.DataFrame(
        [
            {
                "venture_id": p.venture_id,
                "venture": p.name,
                "vertical": p.vertical,
                "scope1_tco2": p.scope1_tco2,
                "scope2_tco2": p.scope2_tco2,
                "scope3_tco2": p.scope3_tco2,
                "min_investment_eur": p.min_investment_eur,
                "max_investment_eur": p.max_investment_eur,
                "financial_roi": p.expected_financial_roi,
                "abatement_alpha": p.abatement_alpha,
                "abatement_sigma": p.abatement_Uncertainty_sigma,
            }
            for p in profiles
        ]
    )


def run_zubi_triple_impact_pipeline() -> None:
    """
    Executes the end-to-end 3-tier Zubi Group pipeline and prints an executive technical report.
    """
    t_total_start = time.perf_counter()
    print("=" * 90)
    print("ESG PORTFOLIO OPTIMIZER: TRIPLE-IMPACT TELEMETRY & STOCHASTIC CAPITAL ALLOCATOR")
    print("Author: Juan Diego Ussa Aponte | Architecture: SciML SDE + LER Auditor + SLSQP/CVaR")
    print("Sector Benchmark: CoCircular LER Auditor · Matteco Green H2 SDE · Zubi Capital SLSQP")
    print("=" * 90)

    # -------------------------------------------------------------------------
    # TIER 1A: CoCircular Bottom-Up Waste & Regulatory Traceability (Ley 7/2022)
    # -------------------------------------------------------------------------
    waste_notes = build_valencia_construction_waste_batch()
    cc_audit = audit_waste_transfer_notes(waste_notes)

    print("\n[1] TIER 1A - COCIRCULAR: LEY 7/2022 LER WASTE TRACEABILITY & SCOPE 3 AUDITOR")
    print(f"    - Albaranes / DIs Procesados (Valencia):  {cc_audit.total_notes_processed} lotes (Cap. 17 LER)")
    print(f"    - Masa Despachada en Obra vs Recibida:    {cc_audit.total_dispatched_tons:,.1f} t vs {cc_audit.total_received_tons:,.1f} t (Merma: {cc_audit.mass_leakage_pct:.2f}%)")
    print(f"    - Cuota de Valorizacion Material Real:    {cc_audit.valorization_rate_pct:.2f}% (Umbral Ley 7/2022 >= 70.0% -> {'CUMPLE' if cc_audit.ley_7_2022_compliant else 'INCUMPLE'})")
    print(f"    - Emisiones Scope 3 Evitadas Auditadas:   {cc_audit.net_scope3_avoided_tco2e:,.2f} tCO2e")
    print(f"    - Alertas Automaticas de Bascula (>3%):   {list(cc_audit.flagged_note_ids)}")

    # -------------------------------------------------------------------------
    # TIER 1B: Matteco Bottom-Up Green H2 Catalyst Kinetics (Paterna Plant)
    # -------------------------------------------------------------------------
    matteco_cell, baseline_cell = build_matteco_catalyst_comparison_cells()
    mt_sim = simulate_electrolyzer_degradation_sde(matteco_cell)
    base_sim = simulate_electrolyzer_degradation_sde(baseline_cell)

    print("\n[2] TIER 1B - MATTECO: AEM/NiFe ELECTROLYZER SDE DYNAMICS & LCOH (PATERNA)")
    print(f"    - Lote Catalizador Matteco:               {mt_sim.catalyst_batch} (1.0 A/cm2, 70 C)")
    print(f"    - Voltaje Celda (Inicial -> 8,000h P95):  {mt_sim.initial_cell_voltage_v:.4f} V -> {mt_sim.p95_terminal_voltage_v:.4f} V (vs {base_sim.p95_terminal_voltage_v:.4f} V convencional)")
    print(f"    - Consumo Especifico & Eficiencia LHV:    {mt_sim.specific_energy_kwh_per_kg_h2:.2f} kWh/kg H2 ({mt_sim.voltage_efficiency_lhv_pct:.1f}% LHV vs {base_sim.voltage_efficiency_lhv_pct:.1f}% std)")
    print(f"    - Coste Nivelado H2 Verde (LCOH):         {mt_sim.lcoh_eur_per_kg_h2:.2f} EUR/kg H2 (vs {base_sim.lcoh_eur_per_kg_h2:.2f} EUR/kg std | Paridad H2 Gris < 3.20 EUR/kg: {'SI' if mt_sim.beats_grey_h2_parity else 'NO'})")
    print(f"    - Abatimiento Anual por Modulo Stack:     {mt_sim.annual_co2_abated_tons:,.2f} tCO2e/year ({mt_sim.annual_h2_production_kg:,.0f} kg H2 verde)")

    # -------------------------------------------------------------------------
    # TIER 2: Zubi Labs / Zubi Capital Stochastic Portfolio Allocator (SFDR Art. 9)
    # -------------------------------------------------------------------------
    ventures = build_zubi_ecosystem_portfolio(cocircular_audit=cc_audit, matteco_sim=mt_sim)
    total_budget = 450000.0
    carbon_mandate = 4900.0

    res_esg = optimize_stochastic_pareto_portfolio(
        ventures, total_budget_eur=total_budget, min_co2_abatement_target_tons=carbon_mandate
    )
    res_greedy = optimize_stochastic_pareto_portfolio(
        ventures, total_budget_eur=total_budget, min_co2_abatement_target_tons=0.0
    )

    print("\n[3] TIER 2 - ZUBI CAPITAL: STOCHASTIC PARETO ALLOCATOR (SLSQP + MONTE CARLO CVaR 95%)")
    df_alloc = pd.DataFrame(
        {
            "Participada Zubi Group": res_esg.venture_names,
            "Capital Asignado (EUR)": [f"{x:,.2f}" for x in res_esg.allocations_eur],
            "Abatimiento Esperado (tCO2e)": [
                f"{a:,.1f}" for a in res_esg.expected_abatement_per_venture_tco2
            ],
            "Retorno Esp. (EUR)": [
                f"{x * v.expected_financial_roi:,.2f}"
                for x, v in zip(res_esg.allocations_eur, ventures)
            ],
        }
    )
    print(df_alloc.to_string(index=False))

    total_ms = (time.perf_counter() - t_total_start) * 1000.0
    print("\n[4] METRICAS DE RIESGO ESTOCASTICO Y FRONTERA DE PARETO (SFDR ARTICULO 9):")
    print(f"    - Presupuesto Desplegado:                 {res_esg.total_invested_eur:,.2f} EUR (Convergencia KKT: {res_esg.success} en {res_esg.execution_time_ms:.2f} ms)")
    print(f"    - ROI Financiero Ponderado Cartera:       {res_esg.portfolio_roi_pct:.2f}% ({res_esg.expected_financial_return_eur:,.2f} EUR/year vs {res_greedy.portfolio_roi_pct:.2f}% Greedy)")
    print(f"    - Abatimiento Esperado vs Greedy ROI:     {res_esg.expected_total_abatement_tco2:,.1f} tCO2e vs {res_greedy.expected_total_abatement_tco2:,.1f} tCO2e (+{res_esg.expected_total_abatement_tco2 - res_greedy.expected_total_abatement_tco2:,.1f} tCO2e)")
    print(f"    - Riesgo de Cola Monte Carlo (VaR / CVaR): VaR 95% = {res_esg.var_95_abatement_tco2:,.1f} tCO2e | CVaR 95% = {res_esg.cvar_95_abatement_tco2:,.1f} tCO2e")
    print(f"    - Coste Marginal Medio de Abatimiento:    {res_esg.marginal_cost_per_ton_eur:.2f} EUR / tCO2e")
    print(f"    - Latencia End-to-End (3 Motores en CPU): {total_ms:.2f} ms")
    print("=" * 90)


# Neutral sector alias for open-source library consumers
run_esg_portfolio_pipeline = run_zubi_triple_impact_pipeline


if __name__ == "__main__":
    run_esg_portfolio_pipeline()
