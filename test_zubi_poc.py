"""
Comprehensive Pytest Verification Suite for Zubi Group Triple-Impact Engine.
Author: Juan Diego Ussa Aponte (Valencia, Spain)

Verifies:
1. Strict domain contracts & Zero Silent Errors (conservation of mass, LER syntax, thermodynamics).
2. CoCircular Ley 7/2022 & RD 553/2020 compliance, scale discrepancy detection, and Scope 3 math.
3. Matteco Butler-Volmer/Tafel polarization physics, Euler-Maruyama SDE bounds, and LCOH parity.
4. Zubi Capital SLSQP KKT convergence, SFDR Art. 9 PAI screening, and Monte Carlo CVaR_95 tail bounds.
"""

from __future__ import annotations

import numpy as np
import pytest

from cocircular_waste_auditor import (
    audit_waste_transfer_notes,
    build_valencia_construction_waste_batch,
)
from domain_models import (
    ElectrolyzerCellTelemetry,
    VentureImpactProfile,
    WasteTransferNoteLER,
)
from esg_carbon_telemetry_simulator import (
    build_zubi_ecosystem_portfolio,
    calculate_abatement_jacobian,
    calculate_abatement_tons,
    optimize_stochastic_pareto_portfolio,
)
from matteco_h2_kinetics import (
    GREY_H2_COST_PARITY_EUR_PER_KG,
    build_matteco_catalyst_comparison_cells,
    compute_initial_cell_voltage,
    simulate_electrolyzer_degradation_sde,
)


# =============================================================================
# 1. DOMAIN MODELS & ZERO SILENT ERRORS TESTS
# =============================================================================


def test_waste_transfer_note_rejects_invalid_ler_and_mass_violation() -> None:
    # Invalid LER code format
    with pytest.raises(ValueError, match="Invalid European Waste List"):
        WasteTransferNoteLER(
            note_id="DI-ERR-01",
            project_site="Obra Test",
            ler_code="INVALID-LER",
            material_name="Hormigón",
            dispatched_mass_tons=10.0,
            received_mass_tons=10.0,
            valorized_mass_tons=9.0,
            treatment_operation="R5",
            virgin_emission_factor_tco2_per_ton=0.15,
            recycling_process_emission_tco2_per_ton=0.02,
        )

    # Conservation of mass violation: valorized > received
    with pytest.raises(ValueError, match="Conservation of mass violation"):
        WasteTransferNoteLER(
            note_id="DI-ERR-02",
            project_site="Obra Test",
            ler_code="17 01 01",
            material_name="Hormigón",
            dispatched_mass_tons=10.0,
            received_mass_tons=9.5,
            valorized_mass_tons=9.8,
            treatment_operation="R5",
            virgin_emission_factor_tco2_per_ton=0.15,
            recycling_process_emission_tco2_per_ton=0.02,
        )

    # Disposal operation D1 claiming valorized mass
    with pytest.raises(ValueError, match="Disposal operation"):
        WasteTransferNoteLER(
            note_id="DI-ERR-03",
            project_site="Obra Test",
            ler_code="17 08 02",
            material_name="Yeso",
            dispatched_mass_tons=10.0,
            received_mass_tons=10.0,
            valorized_mass_tons=5.0,
            treatment_operation="D1",
            virgin_emission_factor_tco2_per_ton=0.15,
            recycling_process_emission_tco2_per_ton=0.02,
        )


def test_electrolyzer_cell_rejects_unphysical_parameters() -> None:
    with pytest.raises(ValueError, match="Thermodynamic violation"):
        ElectrolyzerCellTelemetry(
            cell_id="BAD-CELL",
            catalyst_batch="BATCH-0",
            active_area_cm2=100.0,
            operating_temp_k=343.15,
            current_density_a_cm2=1.0,
            reversible_potential_v=1.55,
            thermoneutral_voltage_v=1.48,  # Invalid: thermoneutral <= reversible
        )

    with pytest.raises(ValueError, match="outside valid liquid/AEM range"):
        ElectrolyzerCellTelemetry(
            cell_id="FROZEN-CELL",
            catalyst_batch="BATCH-0",
            active_area_cm2=100.0,
            operating_temp_k=250.0,  # Below freezing
            current_density_a_cm2=1.0,
        )


# =============================================================================
# 2. COCIRCULAR WASTE AUDITOR TESTS (LEY 7/2022 & RD 553/2020)
# =============================================================================


def test_cocircular_valencia_batch_audit_and_anomaly_detection() -> None:
    notes = build_valencia_construction_waste_batch()
    summary = audit_waste_transfer_notes(notes)

    assert summary.total_notes_processed == 6
    assert summary.total_dispatched_tons == pytest.approx(1060.0)
    assert summary.total_received_tons == pytest.approx(1036.3)
    assert summary.total_valorized_tons == pytest.approx(869.5)
    # 869.5 / 1060.0 = 82.03% >= 70% statutory threshold under Ley 7/2022
    assert summary.valorization_rate_pct == pytest.approx(82.03, abs=0.02)
    assert summary.ley_7_2022_compliant is True
    # Note DI-VLC-2026-004 has 10% scale loss (160t -> 144t) and must be flagged
    assert summary.flagged_note_ids == ("DI-VLC-2026-004",)
    assert summary.net_scope3_avoided_tco2e > 300.0


def test_cocircular_detects_non_compliance_and_hazardous_ler() -> None:
    low_valorization_notes = (
        WasteTransferNoteLER(
            note_id="DI-HAZ-01",
            project_site="Demolición Nave Antigua",
            ler_code="17 06 05*",  # Hazardous asbestos-containing material
            material_name="Materiales de construcción con amianto",
            dispatched_mass_tons=50.0,
            received_mass_tons=50.0,
            valorized_mass_tons=0.0,
            treatment_operation="D5",
            virgin_emission_factor_tco2_per_ton=0.10,
            recycling_process_emission_tco2_per_ton=0.02,
        ),
        WasteTransferNoteLER(
            note_id="DI-LOW-02",
            project_site="Demolición Nave Antigua",
            ler_code="17 09 04",
            material_name="RCD Mezclado",
            dispatched_mass_tons=100.0,
            received_mass_tons=100.0,
            valorized_mass_tons=40.0,
            treatment_operation="R5",
            virgin_emission_factor_tco2_per_ton=0.20,
            recycling_process_emission_tco2_per_ton=0.05,
        ),
    )
    summary = audit_waste_transfer_notes(low_valorization_notes)
    # Per Ley 7/2022 Art. 30.2, valorization rate is evaluated strictly on non-hazardous CDW:
    # 40 / 100 = 40% < 70% -> Non-compliant with Ley 7/2022 (hazardous 50t excluded from denominator)
    assert summary.total_dispatched_tons == pytest.approx(150.0)
    assert summary.valorization_rate_pct == pytest.approx(40.0)
    assert summary.ley_7_2022_compliant is False
    # Hazardous LER note must be automatically flagged for special traceability
    assert "DI-HAZ-01" in summary.flagged_note_ids


def test_cocircular_rejects_empty_batch_and_invalid_annex_codes() -> None:
    with pytest.raises(ValueError, match="Cannot audit an empty sequence"):
        audit_waste_transfer_notes(())

    # Invalid Annex II/III code R99 or D99 must be rejected
    with pytest.raises(ValueError, match="Invalid Annex II/III treatment_operation"):
        WasteTransferNoteLER(
            note_id="DI-ERR-OP",
            project_site="Obra Test",
            ler_code="17 01 01",
            material_name="Hormigón",
            dispatched_mass_tons=10.0,
            received_mass_tons=10.0,
            valorized_mass_tons=9.0,
            treatment_operation="R99",
            virgin_emission_factor_tco2_per_ton=0.15,
            recycling_process_emission_tco2_per_ton=0.02,
        )

    # Recycling process emission exceeding virgin emission factor must be rejected
    with pytest.raises(ValueError, match="Thermodynamic/LCA violation"):
        WasteTransferNoteLER(
            note_id="DI-ERR-LCA",
            project_site="Obra Test",
            ler_code="17 01 01",
            material_name="Hormigón",
            dispatched_mass_tons=10.0,
            received_mass_tons=10.0,
            valorized_mass_tons=9.0,
            treatment_operation="R5",
            virgin_emission_factor_tco2_per_ton=0.10,
            recycling_process_emission_tco2_per_ton=0.25,
        )


# =============================================================================
# 3. MATTECO GREEN H2 SCI-ML & ELECTROCHEMICAL DYNAMICS TESTS
# =============================================================================


def test_matteco_catalyst_outperforms_conventional_baseline() -> None:
    matteco_cell, baseline_cell = build_matteco_catalyst_comparison_cells()

    v0_mt = compute_initial_cell_voltage(matteco_cell)
    v0_base = compute_initial_cell_voltage(baseline_cell)

    # Thermodynamic sanity: operating voltage at 1.0 A/cm2 exceeds thermoneutral voltage (1.481 V)
    assert v0_mt > matteco_cell.thermoneutral_voltage_v
    assert v0_mt < v0_base

    sim_mt = simulate_electrolyzer_degradation_sde(matteco_cell, seed=123)
    sim_base = simulate_electrolyzer_degradation_sde(baseline_cell, seed=123)

    # Matteco NiFe has lower terminal voltage, lower specific energy, and lower LCOH
    assert sim_mt.p95_terminal_voltage_v < sim_base.p95_terminal_voltage_v
    assert sim_mt.specific_energy_kwh_per_kg_h2 < sim_base.specific_energy_kwh_per_kg_h2
    assert sim_mt.lcoh_eur_per_kg_h2 < sim_base.lcoh_eur_per_kg_h2
    assert sim_mt.lcoh_eur_per_kg_h2 < GREY_H2_COST_PARITY_EUR_PER_KG
    assert sim_mt.beats_grey_h2_parity is True


# =============================================================================
# 4. ZUBI CAPITAL STOCHASTIC PARETO ALLOCATOR & SFDR ART. 9 TESTS
# =============================================================================


def test_analytical_jacobian_matches_finite_differences() -> None:
    x = np.array([45000.0, 80000.0, 120000.0, 60000.0, 30000.0])
    alphas = np.array([9.6, 8.4, 11.2, 7.4, 6.1])
    eps = 1e-4
    analytic_grad = calculate_abatement_jacobian(x, alphas)
    numeric_grad = (
        calculate_abatement_tons(x + eps, alphas) - calculate_abatement_tons(x - eps, alphas)
    ) / (2.0 * eps)
    np.testing.assert_allclose(analytic_grad, numeric_grad, rtol=1e-6)


def test_zubi_portfolio_optimizer_convergence_and_cvar_ordering() -> None:
    ventures = build_zubi_ecosystem_portfolio()
    olesym_profile = next(v for v in ventures if v.venture_id == "VNT-OLE-04")
    assert "Olesym Chemicals" in olesym_profile.name
    assert "Palm-Free" in olesym_profile.name
    assert "ITQ" in olesym_profile.name
    assert "Green Chemistry" in olesym_profile.vertical
    assert "Oleochemicals" in olesym_profile.vertical

    # Verify bottom-up Matteco SDE telemetry dynamically calibrates higher alpha than conventional baseline
    mt_cell, base_cell = build_matteco_catalyst_comparison_cells()
    mt_sim = simulate_electrolyzer_degradation_sde(mt_cell, seed=42)
    base_sim = simulate_electrolyzer_degradation_sde(base_cell, seed=42)
    v_mt = build_zubi_ecosystem_portfolio(matteco_sim=mt_sim)
    v_base = build_zubi_ecosystem_portfolio(matteco_sim=base_sim)
    assert v_mt[0].abatement_alpha > v_base[0].abatement_alpha

    res = optimize_stochastic_pareto_portfolio(
        ventures, total_budget_eur=450000.0, min_co2_abatement_target_tons=4900.0, seed=42
    )
    res_greedy = optimize_stochastic_pareto_portfolio(
        ventures, total_budget_eur=450000.0, min_co2_abatement_target_tons=0.0, seed=42
    )

    assert res.success is True
    assert res.sfdr_art9_compliant is True
    assert res.total_invested_eur == pytest.approx(450000.0, abs=1.0)
    assert res.expected_total_abatement_tco2 >= 4900.0 - 1e-2
    assert res.expected_total_abatement_tco2 > res_greedy.expected_total_abatement_tco2 + 180.0
    # Mathematical risk ordering: CVaR_95 <= VaR_95 <= Expected Abatement
    assert res.cvar_95_abatement_tco2 <= res.var_95_abatement_tco2
    assert res.var_95_abatement_tco2 <= res.expected_total_abatement_tco2


def test_sfdr_pai_non_compliant_venture_receives_zero_capital() -> None:
    base_ventures = list(build_zubi_ecosystem_portfolio())
    # Replace last venture with a non-PAI compliant profile with artificially high ROI
    base_ventures[-1] = VentureImpactProfile(
        venture_id="VNT-NON-PAI",
        name="Non-Compliant High ROI Venture",
        vertical="Legacy",
        scope1_tco2=500.0,
        scope2_tco2=500.0,
        scope3_tco2=500.0,
        min_investment_eur=10000.0,
        max_investment_eur=200000.0,
        expected_financial_roi=0.45,
        abatement_alpha=5.0,
        abatement_Uncertainty_sigma=0.05,
        sfdr_pai_compliant=False,
    )
    res = optimize_stochastic_pareto_portfolio(
        base_ventures, total_budget_eur=400000.0, min_co2_abatement_target_tons=3500.0
    )
    assert res.success is True
    assert res.allocations_eur[-1] == 0.0
    assert res.sfdr_art9_compliant is True


def test_infeasible_carbon_target_and_correlated_monte_carlo_shocks() -> None:
    ventures = build_zubi_ecosystem_portfolio()
    # Target strictly exceeds maximum achievable carbon abatement under budget/capacity bounds
    with pytest.raises(ValueError, match="Infeasible carbon target"):
        optimize_stochastic_pareto_portfolio(
            ventures, total_budget_eur=450000.0, min_co2_abatement_target_tons=7500.0
        )

    # Positive cross-sector correlation (e.g., CoCircular, Woodea, and Zubi Cities in construction)
    # increases portfolio variance and deepens worst-case tail risk (lower CVaR_95)
    n = len(ventures)
    corr_pos = np.full((n, n), 0.65, dtype=np.float64)
    np.fill_diagonal(corr_pos, 1.0)

    res_indep = optimize_stochastic_pareto_portfolio(
        ventures, total_budget_eur=450000.0, min_co2_abatement_target_tons=4900.0, seed=42
    )
    res_corr = optimize_stochastic_pareto_portfolio(
        ventures,
        total_budget_eur=450000.0,
        min_co2_abatement_target_tons=4900.0,
        seed=42,
        correlation_matrix=corr_pos,
    )
    assert res_corr.cvar_95_abatement_tco2 < res_indep.cvar_95_abatement_tco2


def test_optimizer_and_sde_reject_invalid_edge_case_inputs() -> None:
    ventures = build_zubi_ecosystem_portfolio()
    # Empty ventures sequence
    with pytest.raises(ValueError, match="ventures sequence cannot be empty"):
        optimize_stochastic_pareto_portfolio(())

    # Budget below sum of minimum commitments (100,000 EUR)
    with pytest.raises(ValueError, match="Infeasible budget"):
        optimize_stochastic_pareto_portfolio(ventures, total_budget_eur=50000.0)

    # Non-positive operating hours in Matteco SDE
    matteco_cell, _ = build_matteco_catalyst_comparison_cells()
    with pytest.raises(ValueError, match="operating_hours must be > 0"):
        simulate_electrolyzer_degradation_sde(matteco_cell, operating_hours=0.0)

    # Negative annualized capex/opex in Matteco SDE
    with pytest.raises(ValueError, match="annualized_capex_opex_eur cannot be negative"):
        simulate_electrolyzer_degradation_sde(matteco_cell, annualized_capex_opex_eur=-100.0)


