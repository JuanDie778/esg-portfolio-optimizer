"""
Matteco Module: SciML Electrochemical Polarization & Euler-Maruyama Catalyst Degradation Engine.
Author: Juan Diego Ussa Aponte (Valencia, Spain)

Physical & Industrial Context (Matteco - Paterna 10,000 m2 Plant / Parc Científic UV):
- Models PGM-free nanolayered NiFe catalysts for Alkaline and AEM (Anion Exchange Membrane)
  water electrolysis.
- Couples deterministic Butler-Volmer/Tafel + Ohmic polarization physics with an Itô Stochastic
  Differential Equation (Euler-Maruyama scheme) for voltage degradation under intermittent
  renewable power cycling.
- Computes Specific Energy Consumption (kWh/kg H2), Levelized Cost of Hydrogen (LCOH in EUR/kg H2),
  and net CO2e abated against Steam Methane Reforming (SMR) grey hydrogen.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

from domain_models import ElectrolyzerCellTelemetry, ElectrolyzerSimulationResult

FARADAY_CONSTANT_C_MOL: float = 96485.33212
H2_MOLAR_MASS_KG_MOL: float = 2.01588e-3
ELECTRONS_PER_H2_MOLECULE: int = 2
# Exact electrochemical charge required per kg of H2 produced: ~26.593 kAh / kg H2
SPECIFIC_CHARGE_KAH_PER_KG_H2: float = (
    ELECTRONS_PER_H2_MOLECULE * FARADAY_CONSTANT_C_MOL / (H2_MOLAR_MASS_KG_MOL * 3.6e6)
)
LHV_H2_KWH_PER_KG: float = 33.33
GREY_H2_SMR_EMISSION_FACTOR_KG_CO2_PER_KG_H2: float = 10.5
GREY_H2_COST_PARITY_EUR_PER_KG: float = 3.20


def build_matteco_catalyst_comparison_cells() -> Tuple[ElectrolyzerCellTelemetry, ElectrolyzerCellTelemetry]:
    """
    Returns two benchmark electrolyzer cell configurations tested at 1.0 A/cm2 and 343.15 K (70 C):
    1. Matteco Nanolayered NiFe Catalyst (Paterna Industrial Batch):
       Ultra-low combined Tafel slope (68 mV/dec), low ASR (0.10 Ohm*cm2), minimal degradation (1.6 uV/h).
    2. Conventional Baseline Alkaline Electrode:
       Higher combined Tafel slope (112 mV/dec), higher ASR (0.22 Ohm*cm2), faster degradation (14.0 uV/h).
    """
    matteco_cell = ElectrolyzerCellTelemetry(
        cell_id="CELL-MTC-AEM-01",
        catalyst_batch="MTC-PATERNA-NIFE-2026A",
        active_area_cm2=500.0,
        operating_temp_k=343.15,
        current_density_a_cm2=1.0,
        reversible_potential_v=1.229,
        thermoneutral_voltage_v=1.481,
        tafel_slope_v_dec=0.068,
        exchange_current_density_a_cm2=1.5e-6,
        area_specific_resistance_ohm_cm2=0.10,
        degradation_drift_uv_per_h=1.6,
        degradation_volatility_uv_sqrt_h=4.0,
        faradaic_efficiency=0.988,
    )

    conventional_cell = ElectrolyzerCellTelemetry(
        cell_id="CELL-STD-ALK-02",
        catalyst_batch="STD-NICKEL-FOAM-BASELINE",
        active_area_cm2=500.0,
        operating_temp_k=343.15,
        current_density_a_cm2=1.0,
        reversible_potential_v=1.229,
        thermoneutral_voltage_v=1.481,
        tafel_slope_v_dec=0.112,
        exchange_current_density_a_cm2=1.2e-7,
        area_specific_resistance_ohm_cm2=0.22,
        degradation_drift_uv_per_h=14.0,
        degradation_volatility_uv_sqrt_h=11.5,
        faradaic_efficiency=0.955,
    )
    return matteco_cell, conventional_cell


def compute_initial_cell_voltage(cell: ElectrolyzerCellTelemetry) -> float:
    """
    Computes initial operating cell voltage V_cell(j, T) from first principles:
        V_cell = E_rev(T) + eta_activation(j) + eta_ohmic(j)
    Uses the exact inverse hyperbolic sine Butler-Volmer activation formulation:
        eta_act = b_tafel * log10( (j / (2*j0)) + sqrt(1 + (j / (2*j0))^2) )
    """
    # Temperature adjustment of reversible Nernst potential (~ -0.85 mV/K above 298.15 K)
    temp_delta = cell.operating_temp_k - 298.15
    e_rev_t = cell.reversible_potential_v - 0.85e-3 * temp_delta

    ratio = cell.current_density_a_cm2 / (2.0 * cell.exchange_current_density_a_cm2)
    eta_act = cell.tafel_slope_v_dec * float(np.log10(ratio + np.sqrt(1.0 + ratio * ratio)))
    eta_ohm = cell.current_density_a_cm2 * cell.area_specific_resistance_ohm_cm2

    v_cell = e_rev_t + eta_act + eta_ohm
    # Under adiabatic/isothermal industrial operation, enforce physical lower bound E_rev(T)
    return float(max(e_rev_t, v_cell))


def simulate_electrolyzer_degradation_sde(
    cell: ElectrolyzerCellTelemetry,
    operating_hours: float = 8000.0,
    n_steps: int = 200,
    n_monte_carlo_paths: int = 500,
    ppa_electricity_price_eur_mwh: float = 34.0,
    num_cells_in_stack_module: int = 120,
    annualized_capex_opex_eur: float = 22000.0,
    seed: int = 42,
) -> ElectrolyzerSimulationResult:
    """
    Simulates catalyst voltage degradation trajectories via a vectorized Euler-Maruyama
    Stochastic Differential Equation (SDE) scheme:
        dV(t) = mu_deg * dt + sigma_deg * sqrt(dt) * Z,   Z ~ N(0, 1)
    with a physical reflecting barrier at V_0 (preventing unphysical spontaneous voltage below initial).

    Computes stack-level hydrogen output, specific energy (kWh/kg H2), LCOH (EUR/kg H2),
    and annual Scope 1-3 CO2e abated by displacing grey SMR hydrogen.
    """
    if operating_hours <= 0.0:
        raise ValueError("operating_hours must be > 0.")
    if n_steps < 10 or n_monte_carlo_paths < 10:
        raise ValueError("Require n_steps >= 10 and n_monte_carlo_paths >= 10.")
    if ppa_electricity_price_eur_mwh < 0.0:
        raise ValueError("ppa_electricity_price_eur_mwh cannot be negative.")
    if num_cells_in_stack_module < 1:
        raise ValueError("num_cells_in_stack_module must be >= 1.")
    if annualized_capex_opex_eur < 0.0:
        raise ValueError("annualized_capex_opex_eur cannot be negative.")

    v0 = compute_initial_cell_voltage(cell)
    dt = operating_hours / float(n_steps)
    sqrt_dt = float(np.sqrt(dt))

    mu_v_per_h = cell.degradation_drift_uv_per_h * 1e-6
    sigma_v_sqrt_h = cell.degradation_volatility_uv_sqrt_h * 1e-6

    rng = np.random.default_rng(seed)
    z_noise = rng.standard_normal(size=(n_monte_carlo_paths, n_steps))

    increments = mu_v_per_h * dt + sigma_v_sqrt_h * sqrt_dt * z_noise
    cumulative_deg = np.cumsum(increments, axis=1)
    # Physical lower bound: voltage remains >= thermoneutral / initial baseline
    v_trajectories = np.maximum(v0, v0 + cumulative_deg)

    terminal_voltages = v_trajectories[:, -1]
    mean_terminal_v = float(np.mean(terminal_voltages))
    p95_terminal_v = float(np.percentile(terminal_voltages, 95))

    # Time-averaged voltage across all Monte Carlo paths and steps
    mean_operating_v = float(np.mean(v_trajectories))

    # Specific energy consumption per kg H2 via Faraday's law
    specific_energy_kwh_kg = (mean_operating_v * SPECIFIC_CHARGE_KAH_PER_KG_H2) / cell.faradaic_efficiency
    voltage_efficiency_lhv_pct = (LHV_H2_KWH_PER_KG / specific_energy_kwh_kg) * 100.0

    # Annual H2 production for the industrial stack module (num_cells_in_stack_module cells)
    total_current_ka = (
        cell.current_density_a_cm2 * cell.active_area_cm2 * num_cells_in_stack_module
    ) / 1000.0
    total_charge_kah = total_current_ka * operating_hours * cell.faradaic_efficiency
    annual_h2_kg = total_charge_kah / SPECIFIC_CHARGE_KAH_PER_KG_H2

    # Levelized Cost of Hydrogen (LCOH in EUR / kg H2)
    electricity_price_eur_kwh = ppa_electricity_price_eur_mwh / 1000.0
    energy_cost_eur_per_kg = specific_energy_kwh_kg * electricity_price_eur_kwh
    capex_opex_per_kg = annualized_capex_opex_eur / annual_h2_kg
    lcoh_eur_per_kg = energy_cost_eur_per_kg + capex_opex_per_kg

    # Net CO2e abated vs grey SMR hydrogen (tons CO2e / year)
    annual_co2_abated_tons = (
        annual_h2_kg * GREY_H2_SMR_EMISSION_FACTOR_KG_CO2_PER_KG_H2
    ) / 1000.0

    return ElectrolyzerSimulationResult(
        cell_id=cell.cell_id,
        catalyst_batch=cell.catalyst_batch,
        initial_cell_voltage_v=round(v0, 4),
        mean_terminal_voltage_v=round(mean_terminal_v, 4),
        p95_terminal_voltage_v=round(p95_terminal_v, 4),
        voltage_efficiency_lhv_pct=round(voltage_efficiency_lhv_pct, 2),
        specific_energy_kwh_per_kg_h2=round(specific_energy_kwh_kg, 2),
        lcoh_eur_per_kg_h2=round(lcoh_eur_per_kg, 3),
        annual_h2_production_kg=round(annual_h2_kg, 1),
        annual_co2_abated_tons=round(annual_co2_abated_tons, 2),
        beats_grey_h2_parity=bool(lcoh_eur_per_kg < GREY_H2_COST_PARITY_EUR_PER_KG),
    )
