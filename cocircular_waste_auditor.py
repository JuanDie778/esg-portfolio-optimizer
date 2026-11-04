"""
CoCircular Module: Construction & Demolition Waste (CDW) LER Traceability & Scope 3 Auditor.
Author: Juan Diego Ussa Aponte (Valencia, Spain)

Regulatory Context (Spain & EU):
- Ley 7/2022, de 8 de abril, de residuos y suelos contaminados para una economía circular:
  Mandates >= 70% by weight valorization of non-hazardous construction and demolition waste (CDW).
- Real Decreto 553/2020: Regulates electronic waste transfer traceability via Documentos de
  Identificación (DI) and European List of Waste (Códigos LER, Chapter 17).
- GHG Protocol Scope 3 (Categories 1 & 5): Quantifies net avoided CO2e emissions by substituting
  virgin raw materials with secondary valorized aggregates, metals, and timber.
"""

from __future__ import annotations

from typing import Sequence, Tuple
import numpy as np

from domain_models import WasteAuditSummary, WasteTransferNoteLER

LEY_7_2022_MIN_VALORIZATION_PCT: float = 70.0
DEFAULT_MASS_DISCREPANCY_TOLERANCE_PCT: float = 3.0


def build_valencia_construction_waste_batch() -> Tuple[WasteTransferNoteLER, ...]:
    """
    Generates a realistic batch of Construction & Demolition Waste (RCD) transfer notes
    across flagship sustainable projects in the Valencia ecosystem (La Pinada, Woodea, Marina).
    Includes intentional scale discrepancies to demonstrate automated anomaly detection.
    """
    return (
        WasteTransferNoteLER(
            note_id="DI-VLC-2026-001",
            project_site="Eco-Barrio La Pinada (Paterna) - Fase Urbanización",
            ler_code="17 01 01",
            material_name="Hormigón estructural triturado (Árido reciclado)",
            dispatched_mass_tons=420.0,
            received_mass_tons=416.5,  # 0.83% normal moisture/scale variance
            valorized_mass_tons=395.0,
            treatment_operation="R5",
            virgin_emission_factor_tco2_per_ton=0.145,
            recycling_process_emission_tco2_per_ton=0.022,
        ),
        WasteTransferNoteLER(
            note_id="DI-VLC-2026-002",
            project_site="Woodea Valencia Timber Hub - Estructura CLT",
            ler_code="17 02 01",
            material_name="Madera estructural limpia y recortes CLT",
            dispatched_mass_tons=95.0,
            received_mass_tons=94.2,
            valorized_mass_tons=91.0,
            treatment_operation="R3",
            virgin_emission_factor_tco2_per_ton=0.720,
            recycling_process_emission_tco2_per_ton=0.065,
        ),
        WasteTransferNoteLER(
            note_id="DI-VLC-2026-003",
            project_site="Eco-Barrio La Pinada (Paterna) - Cimentación",
            ler_code="17 04 05",
            material_name="Acero corrugado y perfiles metálicos recuperados",
            dispatched_mass_tons=130.0,
            received_mass_tons=129.1,
            valorized_mass_tons=126.5,
            treatment_operation="R4",
            virgin_emission_factor_tco2_per_ton=1.850,
            recycling_process_emission_tco2_per_ton=0.410,
        ),
        WasteTransferNoteLER(
            note_id="DI-VLC-2026-004",
            project_site="Rehabilitación Sostenible Ciutat Vella",
            ler_code="17 09 04",
            material_name="Residuos mezclados de construcción y demolición",
            dispatched_mass_tons=160.0,
            received_mass_tons=144.0,  # 10.0% mass loss -> ANOMALY (>3% threshold)
            valorized_mass_tons=82.0,
            treatment_operation="R5",
            virgin_emission_factor_tco2_per_ton=0.210,
            recycling_process_emission_tco2_per_ton=0.055,
        ),
        WasteTransferNoteLER(
            note_id="DI-VLC-2026-005",
            project_site="Adecuación Nave Industrial Paterna",
            ler_code="17 05 04",
            material_name="Tierras y piedras de excavación no contaminadas",
            dispatched_mass_tons=210.0,
            received_mass_tons=208.0,
            valorized_mass_tons=175.0,
            treatment_operation="R10",
            virgin_emission_factor_tco2_per_ton=0.085,
            recycling_process_emission_tco2_per_ton=0.015,
        ),
        WasteTransferNoteLER(
            note_id="DI-VLC-2026-006",
            project_site="Rehabilitación Sostenible Ciutat Vella",
            ler_code="17 08 02",
            material_name="Materiales de construcción a base de yeso (Rechazo)",
            dispatched_mass_tons=45.0,
            received_mass_tons=44.5,
            valorized_mass_tons=0.0,
            treatment_operation="D1",
            virgin_emission_factor_tco2_per_ton=0.180,
            recycling_process_emission_tco2_per_ton=0.040,
        ),
    )


def audit_waste_transfer_notes(
    notes: Sequence[WasteTransferNoteLER],
    max_discrepancy_tolerance_pct: float = DEFAULT_MASS_DISCREPANCY_TOLERANCE_PCT,
    min_valorization_target_pct: float = LEY_7_2022_MIN_VALORIZATION_PCT,
) -> WasteAuditSummary:
    """
    Executes vectorized regulatory and mass-balance auditing on a sequence of LER transfer notes.

    Args:
        notes: Non-empty sequence of validated WasteTransferNoteLER records.
        max_discrepancy_tolerance_pct: Maximum allowed relative mass difference (%) between
            origin dispatch weight and authorized treatment plant scale weight.
        min_valorization_target_pct: Statutory minimum valorization percentage (70% under Ley 7/2022).

    Returns:
        WasteAuditSummary with mass conservation metrics, compliance flags, and Scope 3 avoided tCO2e.
    """
    if not notes:
        raise ValueError("Cannot audit an empty sequence of WasteTransferNoteLER records.")
    if max_discrepancy_tolerance_pct < 0.0:
        raise ValueError("max_discrepancy_tolerance_pct must be non-negative.")
    if not (0.0 <= min_valorization_target_pct <= 100.0):
        raise ValueError("min_valorization_target_pct must be between 0 and 100.")

    dispatched = np.array([n.dispatched_mass_tons for n in notes], dtype=np.float64)
    received = np.array([n.received_mass_tons for n in notes], dtype=np.float64)
    valorized = np.array([n.valorized_mass_tons for n in notes], dtype=np.float64)
    ef_virgin = np.array([n.virgin_emission_factor_tco2_per_ton for n in notes], dtype=np.float64)
    ef_recycled = np.array(
        [n.recycling_process_emission_tco2_per_ton for n in notes], dtype=np.float64
    )

    total_dispatched = float(np.sum(dispatched))
    total_received = float(np.sum(received))
    total_valorized = float(np.sum(valorized))

    # Mass balance discrepancy (positive means lost in transit; negative means scale over-read)
    discrepancy_tons_per_note = np.abs(dispatched - received)
    discrepancy_pct_per_note = (discrepancy_tons_per_note / dispatched) * 100.0

    total_discrepancy_tons = float(np.sum(dispatched - received))
    total_leakage_pct = (total_discrepancy_tons / total_dispatched) * 100.0

    # Statutory Ley 7/2022 Art. 30.2 >=70% valorization rate applies strictly to NON-HAZARDOUS CDW
    non_haz_mask = np.array([not n.is_hazardous for n in notes], dtype=bool)
    non_haz_dispatched = float(np.sum(dispatched[non_haz_mask]))
    non_haz_valorized = float(np.sum(valorized[non_haz_mask]))
    if non_haz_dispatched > 0.0:
        valorization_rate_pct = (non_haz_valorized / non_haz_dispatched) * 100.0
    else:
        valorization_rate_pct = 0.0
    ley_7_2022_compliant = bool(
        non_haz_dispatched > 0.0 and valorization_rate_pct >= min_valorization_target_pct
    )

    # Scope 3 avoided emissions: net credit from replacing virgin materials minus recycling energy
    net_unit_saving = ef_virgin - ef_recycled
    avoided_tco2e = float(np.sum(valorized * net_unit_saving))

    # Flag notes with excessive scale discrepancy or hazardous material requiring special protocol
    flagged_ids = []
    for idx, note in enumerate(notes):
        if discrepancy_pct_per_note[idx] > max_discrepancy_tolerance_pct or note.is_hazardous:
            flagged_ids.append(note.note_id)

    return WasteAuditSummary(
        total_notes_processed=len(notes),
        total_dispatched_tons=round(total_dispatched, 3),
        total_received_tons=round(total_received, 3),
        total_valorized_tons=round(total_valorized, 3),
        mass_discrepancy_tons=round(total_discrepancy_tons, 3),
        mass_leakage_pct=round(total_leakage_pct, 3),
        valorization_rate_pct=round(valorization_rate_pct, 2),
        ley_7_2022_compliant=ley_7_2022_compliant,
        net_scope3_avoided_tco2e=round(avoided_tco2e, 3),
        flagged_note_ids=tuple(flagged_ids),
    )
