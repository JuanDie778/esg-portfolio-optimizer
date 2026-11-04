"""
Domain Models & Strict Validation Contracts for Zubi Group Triple-Impact Engine.
Author: Juan Diego Ussa Aponte (Valencia, Spain)
Design Principle: Zero Silent Errors, Immutable Dataclasses, Regulatory & Physical Bounds.

Covers the three operational tiers of the Zubi Group ecosystem:
1. CoCircular (Ley 7/2022 & RD 553/2020 LER Waste Traceability & Scope 3 Circularity)
2. Matteco (AEM / Alkaline Green Hydrogen Electrolyzer Electrochemistry & LCOH Dynamics)
3. Zubi Labs / Zubi Capital (SFDR Art. 9 Constrained Stochastic Portfolio Allocation)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple

LER_CODE_REGEX = re.compile(r"^\d{2}\s\d{2}\s\d{2}\*?$")
TREATMENT_OPERATION_REGEX = re.compile(r"^(?:R(?:[1-9]|1[0-3])|D(?:[1-9]|1[0-5]))$")


@dataclass(frozen=True)
class WasteTransferNoteLER:
    """
    Represents an official Waste Identification Document (DI / Albarán de Báscula)
    under Spanish Law 7/2022 and Royal Decree 553/2020 for construction/industrial waste.
    """

    note_id: str
    project_site: str
    ler_code: str
    material_name: str
    dispatched_mass_tons: float
    received_mass_tons: float
    valorized_mass_tons: float
    treatment_operation: str  # e.g., 'R5' (inorganic recycling), 'R3', 'R4', 'D1' (landfill)
    virgin_emission_factor_tco2_per_ton: float
    recycling_process_emission_tco2_per_ton: float

    def __post_init__(self) -> None:
        if not self.note_id or not self.note_id.strip():
            raise ValueError("note_id cannot be empty.")
        if not self.project_site or not self.project_site.strip():
            raise ValueError("project_site cannot be empty.")
        if not LER_CODE_REGEX.match(self.ler_code.strip()):
            raise ValueError(
                f"Invalid European Waste List (LER) code '{self.ler_code}'. "
                "Expected format 'XX XX XX' or 'XX XX XX*' (e.g., '17 01 01')."
            )
        if self.dispatched_mass_tons <= 0.0:
            raise ValueError(
                f"dispatched_mass_tons must be > 0, got {self.dispatched_mass_tons}."
            )
        if self.received_mass_tons < 0.0:
            raise ValueError(
                f"received_mass_tons cannot be negative, got {self.received_mass_tons}."
            )
        if self.valorized_mass_tons < 0.0:
            raise ValueError(
                f"valorized_mass_tons cannot be negative, got {self.valorized_mass_tons}."
            )
        if self.valorized_mass_tons > self.received_mass_tons + 1e-9:
            raise ValueError(
                f"Conservation of mass violation in note {self.note_id}: "
                f"valorized_mass_tons ({self.valorized_mass_tons}) cannot exceed "
                f"received_mass_tons ({self.received_mass_tons})."
            )
        op_clean = self.treatment_operation.strip().upper()
        if not TREATMENT_OPERATION_REGEX.match(op_clean):
            raise ValueError(
                f"Invalid Annex II/III treatment_operation '{self.treatment_operation}'. "
                "Must be a valid Recovery code ('R1'..'R13') or Disposal code ('D1'..'D15')."
            )
        if op_clean.startswith("D") and self.valorized_mass_tons > 1e-9:
            raise ValueError(
                f"Regulatory inconsistency in note {self.note_id}: Disposal operation "
                f"'{self.treatment_operation}' cannot claim positive valorized mass."
            )
        if self.virgin_emission_factor_tco2_per_ton <= 0.0:
            raise ValueError("virgin_emission_factor_tco2_per_ton must be strictly positive.")
        if self.recycling_process_emission_tco2_per_ton < 0.0:
            raise ValueError("recycling_process_emission_tco2_per_ton cannot be negative.")
        if self.recycling_process_emission_tco2_per_ton > self.virgin_emission_factor_tco2_per_ton:
            raise ValueError(
                f"Thermodynamic/LCA violation in note {self.note_id}: "
                f"recycling_process_emission_tco2_per_ton ({self.recycling_process_emission_tco2_per_ton}) "
                f"exceeds virgin_emission_factor_tco2_per_ton ({self.virgin_emission_factor_tco2_per_ton})."
            )

    @property
    def is_hazardous(self) -> bool:
        """Hazardous waste codes under the European List of Waste end with an asterisk (*)."""
        return self.ler_code.strip().endswith("*")


@dataclass(frozen=True)
class WasteAuditSummary:
    """
    Aggregated telemetry and regulatory compliance output from the CoCircular auditor.
    """

    total_notes_processed: int
    total_dispatched_tons: float
    total_received_tons: float
    total_valorized_tons: float
    mass_discrepancy_tons: float
    mass_leakage_pct: float
    valorization_rate_pct: float
    ley_7_2022_compliant: bool
    net_scope3_avoided_tco2e: float
    flagged_note_ids: Tuple[str, ...]


@dataclass(frozen=True)
class ElectrolyzerCellTelemetry:
    """
    Electrochemical and operating parameters for a Matteco PGM-free (NiFe) AEM/Alkaline cell.
    Enforces fundamental thermodynamic and kinetic bounds.
    """

    cell_id: str
    catalyst_batch: str
    active_area_cm2: float
    operating_temp_k: float
    current_density_a_cm2: float
    reversible_potential_v: float = 1.229
    thermoneutral_voltage_v: float = 1.481
    tafel_slope_v_dec: float = 0.068
    exchange_current_density_a_cm2: float = 1.5e-6
    area_specific_resistance_ohm_cm2: float = 0.10
    degradation_drift_uv_per_h: float = 1.8
    degradation_volatility_uv_sqrt_h: float = 4.5
    faradaic_efficiency: float = 0.985

    def __post_init__(self) -> None:
        if not self.cell_id or not self.cell_id.strip():
            raise ValueError("cell_id cannot be empty.")
        if self.active_area_cm2 <= 0.0:
            raise ValueError("active_area_cm2 must be > 0.")
        if not (273.15 <= self.operating_temp_k <= 423.15):
            raise ValueError(
                f"operating_temp_k ({self.operating_temp_k} K) outside valid liquid/AEM range [273.15, 423.15] K."
            )
        if self.current_density_a_cm2 <= 0.0:
            raise ValueError("current_density_a_cm2 must be > 0.")
        if self.reversible_potential_v <= 1.0 or self.thermoneutral_voltage_v <= self.reversible_potential_v:
            raise ValueError(
                "Thermodynamic violation: require 1.0 < reversible_potential_v < thermoneutral_voltage_v."
            )
        if self.tafel_slope_v_dec <= 0.0:
            raise ValueError("tafel_slope_v_dec must be > 0.")
        if self.exchange_current_density_a_cm2 <= 0.0:
            raise ValueError("exchange_current_density_a_cm2 must be > 0.")
        if self.area_specific_resistance_ohm_cm2 <= 0.0:
            raise ValueError("area_specific_resistance_ohm_cm2 must be > 0.")
        if self.degradation_drift_uv_per_h < 0.0:
            raise ValueError("degradation_drift_uv_per_h cannot be negative.")
        if self.degradation_volatility_uv_sqrt_h < 0.0:
            raise ValueError("degradation_volatility_uv_sqrt_h cannot be negative.")
        if not (0.0 < self.faradaic_efficiency <= 1.0):
            raise ValueError("faradaic_efficiency must be in (0.0, 1.0].")


@dataclass(frozen=True)
class ElectrolyzerSimulationResult:
    """
    Output of Matteco's Euler-Maruyama stochastic degradation & techno-economic LCOH simulation.
    """

    cell_id: str
    catalyst_batch: str
    initial_cell_voltage_v: float
    mean_terminal_voltage_v: float
    p95_terminal_voltage_v: float
    voltage_efficiency_lhv_pct: float
    specific_energy_kwh_per_kg_h2: float
    lcoh_eur_per_kg_h2: float
    annual_h2_production_kg: float
    annual_co2_abated_tons: float
    beats_grey_h2_parity: bool


@dataclass(frozen=True)
class VentureImpactProfile:
    """
    Portfolio company profile for Zubi Labs / Zubi Capital multi-objective allocation.
    """

    venture_id: str
    name: str
    vertical: str
    scope1_tco2: float
    scope2_tco2: float
    scope3_tco2: float
    min_investment_eur: float
    max_investment_eur: float
    expected_financial_roi: float
    abatement_alpha: float
    abatement_Uncertainty_sigma: float
    sfdr_pai_compliant: bool = True

    def __post_init__(self) -> None:
        if not self.venture_id or not self.name:
            raise ValueError("venture_id and name cannot be empty.")
        if self.scope1_tco2 < 0 or self.scope2_tco2 < 0 or self.scope3_tco2 < 0:
            raise ValueError("GHG Scope 1, 2, and 3 emissions cannot be negative.")
        if self.min_investment_eur < 0 or self.max_investment_eur < self.min_investment_eur:
            raise ValueError("Require 0 <= min_investment_eur <= max_investment_eur.")
        if self.abatement_alpha <= 0.0:
            raise ValueError("abatement_alpha must be > 0.")
        if not (0.0 <= self.abatement_Uncertainty_sigma < 1.0):
            raise ValueError("abatement_Uncertainty_sigma must be in [0.0, 1.0).")

    @property
    def total_baseline_tco2(self) -> float:
        return self.scope1_tco2 + self.scope2_tco2 + self.scope3_tco2


@dataclass(frozen=True)
class PortfolioAllocationResult:
    """
    Output of Zubi Capital's SLSQP + Monte Carlo CVaR portfolio optimizer.
    """

    success: bool
    execution_time_ms: float
    venture_names: Tuple[str, ...]
    allocations_eur: Tuple[float, ...]
    expected_abatement_per_venture_tco2: Tuple[float, ...]
    total_invested_eur: float
    expected_financial_return_eur: float
    portfolio_roi_pct: float
    expected_total_abatement_tco2: float
    var_95_abatement_tco2: float
    cvar_95_abatement_tco2: float
    marginal_cost_per_ton_eur: float
    sfdr_art9_compliant: bool
