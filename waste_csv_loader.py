"""
Waste CSV Loader: Ingestion & Parsing of Real Construction Waste Albaranes (Códigos LER).
Author: Juan Diego Ussa Aponte (Valencia, Spain)
Architecture: Zero Placeholders, Robust Encoding/Delimiter Detection, Strict Schema Enforcement.

Regulatory Reference:
- Ley 7/2022 de residuos y suelos contaminados para una economía circular (Art. 30.2 CDW >=70% valorization).
- Real Decreto 553/2020 por el que se regula el traslado de residuos en el territorio del Estado.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, List, Sequence, TextIO, Tuple, Union

from domain_models import WasteTransferNoteLER

# Flexible column aliases to support heterogeneous ERP / bascula outputs across Spain
COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "note_id": ("note_id", "id", "albaran_id", "di_id", "documento_identificacion", "num_albaran"),
    "project_site": ("project_site", "obra", "site", "proyecto", "centro_trabajo", "origen_obra"),
    "ler_code": ("ler_code", "codigo_ler", "ler", "ewl_code", "cod_ler", "residuo_ler"),
    "material_name": ("material_name", "material", "descripcion", "descripcion_residuo", "concepto"),
    "dispatched_mass_tons": (
        "dispatched_mass_tons",
        "peso_salida_t",
        "dispatched_tons",
        "peso_origen_t",
        "masa_despachada_t",
        "peso_bruto_salida",
    ),
    "received_mass_tons": (
        "received_mass_tons",
        "peso_destino_t",
        "received_tons",
        "peso_bascula_t",
        "masa_recibida_t",
        "peso_neto_destino",
    ),
    "valorized_mass_tons": (
        "valorized_mass_tons",
        "peso_valorizado_t",
        "valorized_tons",
        "masa_valorizada_t",
        "cantidad_reciclada_t",
    ),
    "treatment_operation": (
        "treatment_operation",
        "operacion_tratamiento",
        "operacion_r_d",
        "treatment_code",
        "r_d_code",
        "tratamiento",
    ),
    "virgin_emission_factor_tco2_per_ton": (
        "virgin_emission_factor_tco2_per_ton",
        "factor_emision_virgen",
        "ef_virgin",
        "fe_virgen_tco2_t",
    ),
    "recycling_process_emission_tco2_per_ton": (
        "recycling_process_emission_tco2_per_ton",
        "factor_emision_reciclaje",
        "ef_recycled",
        "fe_reciclaje_tco2_t",
    ),
}


def _parse_float(val: Union[str, float, int], field_name: str, row_idx: int) -> float:
    """Parses a float value supporting both '.' and ',' decimal separators."""
    if isinstance(val, (float, int)):
        return float(val)
    if not isinstance(val, str) or not val.strip():
        raise ValueError(f"Row {row_idx}: Empty or missing numeric value for field '{field_name}'.")
    clean_val = val.strip().replace(" ", "").replace(",", ".")
    try:
        return float(clean_val)
    except ValueError as exc:
        raise ValueError(
            f"Row {row_idx}: Could not parse '{val}' as a valid float for field '{field_name}'."
        ) from exc


def _map_row_columns(header_row: Sequence[str]) -> Dict[str, int]:
    """Maps recognized schema field names to CSV column indexes."""
    normalized_headers = [h.strip().lower() for h in header_row]
    col_map: Dict[str, int] = {}

    for field_name, aliases in COLUMN_ALIASES.items():
        found_idx = -1
        for alias in aliases:
            if alias.lower() in normalized_headers:
                found_idx = normalized_headers.index(alias.lower())
                break
        if found_idx == -1:
            raise ValueError(
                f"Missing required CSV column for '{field_name}'. "
                f"Accepted aliases: {list(aliases)}. Found headers: {header_row}"
            )
        col_map[field_name] = found_idx

    return col_map


def parse_waste_transfer_notes_from_rows(
    rows: Iterable[Sequence[str]],
) -> Tuple[WasteTransferNoteLER, ...]:
    """
    Parses an iterable of tokenized CSV rows into validated WasteTransferNoteLER instances.
    Enforces strict zero-placeholder domain contracts and provides actionable row error messages.
    """
    iterator = iter(rows)
    try:
        header = next(iterator)
    except StopIteration:
        raise ValueError("Cannot parse waste transfer notes: CSV content is empty.")

    col_map = _map_row_columns(header)
    notes: List[WasteTransferNoteLER] = []

    for row_idx, row in enumerate(iterator, start=2):
        # Ignore empty lines
        if not row or all(not cell.strip() for cell in row):
            continue

        if len(row) <= max(col_map.values()):
            raise ValueError(
                f"Row {row_idx}: Incomplete row with {len(row)} columns, expected at least {max(col_map.values()) + 1}."
            )

        note_id = row[col_map["note_id"]].strip()
        project_site = row[col_map["project_site"]].strip()
        ler_code = row[col_map["ler_code"]].strip()
        material_name = row[col_map["material_name"]].strip()
        treatment_operation = row[col_map["treatment_operation"]].strip().upper()

        dispatched = _parse_float(row[col_map["dispatched_mass_tons"]], "dispatched_mass_tons", row_idx)
        received = _parse_float(row[col_map["received_mass_tons"]], "received_mass_tons", row_idx)
        valorized = _parse_float(row[col_map["valorized_mass_tons"]], "valorized_mass_tons", row_idx)
        ef_virgin = _parse_float(
            row[col_map["virgin_emission_factor_tco2_per_ton"]],
            "virgin_emission_factor_tco2_per_ton",
            row_idx,
        )
        ef_recycled = _parse_float(
            row[col_map["recycling_process_emission_tco2_per_ton"]],
            "recycling_process_emission_tco2_per_ton",
            row_idx,
        )

        try:
            note = WasteTransferNoteLER(
                note_id=note_id,
                project_site=project_site,
                ler_code=ler_code,
                material_name=material_name,
                dispatched_mass_tons=dispatched,
                received_mass_tons=received,
                valorized_mass_tons=valorized,
                treatment_operation=treatment_operation,
                virgin_emission_factor_tco2_per_ton=ef_virgin,
                recycling_process_emission_tco2_per_ton=ef_recycled,
            )
            notes.append(note)
        except ValueError as exc:
            raise ValueError(f"Row {row_idx} validation failed: {exc}") from exc

    if not notes:
        raise ValueError("CSV contained a header row but no valid data rows.")

    return tuple(notes)


def load_waste_transfer_notes_from_csv(
    source: Union[str, Path, TextIO, BinaryIO],
    encoding: str = "utf-8",
) -> Tuple[WasteTransferNoteLER, ...]:
    """
    Loads and parses WasteTransferNoteLER records from a CSV file path, string buffer, or file-like stream.
    Automatically detects delimiters (comma vs semicolon) and strips BOM markers.
    """
    if isinstance(source, (str, Path)) and not isinstance(source, io.StringIO):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Waste CSV file not found at path: {path.resolve()}")
        # Check for UTF-8-sig vs UTF-8 vs Latin-1
        raw_bytes = path.read_bytes()
        try:
            content_str = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            content_str = raw_bytes.decode("latin-1")
        stream = io.StringIO(content_str)
    elif isinstance(source, io.BytesIO):
        raw_bytes = source.getvalue()
        try:
            content_str = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError:
            content_str = raw_bytes.decode("latin-1")
        stream = io.StringIO(content_str)
    elif hasattr(source, "read"):
        content = source.read()
        if isinstance(content, bytes):
            try:
                content_str = content.decode("utf-8-sig")
            except UnicodeDecodeError:
                content_str = content.decode("latin-1")
            stream = io.StringIO(content_str)
        else:
            stream = io.StringIO(content)
    else:
        # String content passed directly
        stream = io.StringIO(str(source))

    # Sniff delimiter (supporting comma ',' and Spanish Excel semicolon ';')
    sample = stream.read(4096)
    stream.seek(0)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        delimiter = dialect.delimiter
    except Exception:
        # Fallback to comma if detection fails
        delimiter = ";" if ";" in sample and sample.count(";") > sample.count(",") else ","

    reader = csv.reader(stream, delimiter=delimiter)
    return parse_waste_transfer_notes_from_rows(reader)


def export_sample_waste_csv(output_path: Union[str, Path]) -> Path:
    """
    Exports an authoritative sample CSV file representing real-world construction and
    demolition waste (RCD) transfer notes from sustainable projects in Valencia.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        [
            "note_id",
            "project_site",
            "ler_code",
            "material_name",
            "dispatched_mass_tons",
            "received_mass_tons",
            "valorized_mass_tons",
            "treatment_operation",
            "virgin_emission_factor_tco2_per_ton",
            "recycling_process_emission_tco2_per_ton",
        ],
        [
            "DI-VLC-2026-001",
            "Eco-Barrio La Pinada (Paterna) - Fase Urbanizacion",
            "17 01 01",
            "Hormigon estructural triturado (Arido reciclado)",
            "420.0",
            "416.5",
            "395.0",
            "R5",
            "0.145",
            "0.022",
        ],
        [
            "DI-VLC-2026-002",
            "Woodea Valencia Timber Hub - Estructura CLT",
            "17 02 01",
            "Madera estructural limpia y recortes CLT",
            "95.0",
            "94.2",
            "91.0",
            "R3",
            "0.720",
            "0.065",
        ],
        [
            "DI-VLC-2026-003",
            "Eco-Barrio La Pinada (Paterna) - Cimentacion",
            "17 04 05",
            "Acero corrugado y perfiles metalicos recuperados",
            "130.0",
            "129.1",
            "126.5",
            "R4",
            "1.850",
            "0.410",
        ],
        [
            "DI-VLC-2026-004",
            "Rehabilitacion Sostenible Ciutat Vella",
            "17 09 04",
            "Residuos mezclados de construccion y demolicion",
            "160.0",
            "144.0",
            "82.0",
            "R5",
            "0.210",
            "0.055",
        ],
        [
            "DI-VLC-2026-005",
            "Adecuacion Nave Industrial Paterna",
            "17 05 04",
            "Tierras y piedras de excavacion no contaminadas",
            "210.0",
            "208.0",
            "175.0",
            "R10",
            "0.085",
            "0.015",
        ],
        [
            "DI-VLC-2026-006",
            "Rehabilitacion Sostenible Ciutat Vella",
            "17 08 02",
            "Materiales de construccion a base de yeso (Rechazo)",
            "45.0",
            "44.5",
            "0.0",
            "D1",
            "0.180",
            "0.040",
        ],
    ]

    with open(out, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    return out


if __name__ == "__main__":
    import sys
    from cocircular_waste_auditor import audit_waste_transfer_notes

    default_sample = Path(__file__).parent / "sample_valencia_waste_transfers.csv"
    if not default_sample.exists():
        export_sample_waste_csv(default_sample)
        print(f"[+] Created sample waste CSV at: {default_sample}")

    target = sys.argv[1] if len(sys.argv) > 1 else str(default_sample)
    print(f"[*] Ingesting and auditing waste transfer notes from: {target}")
    parsed_notes = load_waste_transfer_notes_from_csv(target)
    summary = audit_waste_transfer_notes(parsed_notes)

    print(f"[+] Processed {summary.total_notes_processed} notes.")
    print(f"    - Dispatched Mass: {summary.total_dispatched_tons:,.2f} t")
    print(f"    - Received Mass:   {summary.total_received_tons:,.2f} t (Discrepancy: {summary.mass_leakage_pct:.2f}%)")
    print(f"    - Valorization:    {summary.valorization_rate_pct:.2f}% (Ley 7/2022 >=70%: {summary.ley_7_2022_compliant})")
    print(f"    - Avoided Scope 3: {summary.net_scope3_avoided_tco2e:,.2f} tCO2e")
    print(f"    - Flagged Notes:   {list(summary.flagged_note_ids)}")
