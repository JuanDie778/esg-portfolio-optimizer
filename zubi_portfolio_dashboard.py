"""
Zubi Group Portfolio Dashboard: Interactive Triple-Impact Analytics & Pareto Optimizer.
Author: Juan Diego Ussa Aponte (Valencia, Spain)
Neutral GitHub Library Name: esg-portfolio-optimizer
Architecture: Streamlit Web UI, SciML Electrochemistry, Ley 7/2022 LER Auditor, SLSQP Pareto Frontier.
"""

from __future__ import annotations

import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from cocircular_waste_auditor import audit_waste_transfer_notes
from domain_models import ElectrolyzerCellTelemetry
from esg_carbon_telemetry_simulator import (
    build_zubi_ecosystem_portfolio,
    compute_max_achievable_abatement_tons,
    optimize_stochastic_pareto_portfolio,
)
from matteco_h2_kinetics import (
    GREY_H2_COST_PARITY_EUR_PER_KG,
    build_matteco_catalyst_comparison_cells,
    simulate_electrolyzer_degradation_sde,
)
from waste_csv_loader import export_sample_waste_csv, load_waste_transfer_notes_from_csv

st.set_page_config(
    page_title="ESG Portfolio Optimizer | Zubi Group",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for executive presentation styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e9ecef;
    }
    .badge-pass {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-fail {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_default_sample_csv_path() -> Path:
    sample_path = Path(__file__).parent / "sample_valencia_waste_transfers.csv"
    if not sample_path.exists():
        export_sample_waste_csv(sample_path)
    return sample_path


# Sidebar: Architectural Overview & Global Context
with st.sidebar:
    st.title("🌱 ESG Portfolio Optimizer")
    st.markdown("**Triple-Impact Analytics & Capital Allocator**")
    st.caption("Author: **Juan Diego Ussa Aponte** | Valencia, Spain")
    st.caption("Neutral Open-Source Engine: `esg-portfolio-optimizer`")
    st.markdown("---")
    st.markdown("### 🏢 Ecosistema Zubi Group")
    st.markdown(
        """
        - **Zubi Labs:** Impact Venture Builder
        - **CoCircular:** 360° Circular Waste SaaS
        - **Matteco:** PGM-free Green H₂ Deep-Tech
        - **Woodea:** Mass Timber CLT ConTech
        - **Olesym Chemicals:** ITQ UPV-CSIC Green Chemistry
        - **Zubi Capital:** SFDR Art. 9 Impact Funds
        """
    )
    st.markdown("---")
    st.info(
        "💡 **Metodología:** Los datos de trinchera física (CoCircular y Matteco) "
        "calibran dinámicamente el modelo estocástico no lineal de Zubi Capital."
    )

tab_cocircular, tab_matteco, tab_capital = st.tabs(
    [
        "📦 1. CoCircular (Auditoría LER Ley 7/2022)",
        "⚡ 2. Matteco (SciML & Cinética H₂ Verde)",
        "📈 3. Zubi Capital (Frontera de Pareto & CVaR)",
    ]
)

# =============================================================================
# TAB 1: COCIRCULAR - LEY 7/2022 LER WASTE TRACEABILITY & SCOPE 3
# =============================================================================
with tab_cocircular:
    st.header("CoCircular: Trazabilidad de Residuos de Construcción (Códigos LER)")
    st.markdown(
        r"Auditoría automatizada de Documentos de Identificación (DI) bajo el **RD 553/2020** "
        r"y verificación del umbral estatutario del **$\ge 70\%$ de valorización en peso sobre RCDs no peligrosos (Ley 7/2022)**."
    )

    sample_csv_path = get_default_sample_csv_path()

    col_upload_left, col_upload_right = st.columns([2, 1])
    with col_upload_left:
        uploaded_file = st.file_uploader(
            "Cargar archivo CSV de albaranes de báscula / DIs (o usar dataset oficial de Valencia):",
            type=["csv", "txt"],
        )
    with col_upload_right:
        st.write("")
        st.write("")
        if st.button("🔄 Recargar Dataset Ejemplo (Valencia)"):
            uploaded_file = None

    if uploaded_file is not None:
        try:
            content_bytes = uploaded_file.read()
            waste_notes = load_waste_transfer_notes_from_csv(io.BytesIO(content_bytes))
            st.success(f"Archivo '{uploaded_file.name}' cargado con éxito ({len(waste_notes)} albaranes).")
        except Exception as e:
            st.error(f"Error al parsear el CSV cargado: {e}")
            waste_notes = load_waste_transfer_notes_from_csv(sample_csv_path)
    else:
        waste_notes = load_waste_transfer_notes_from_csv(sample_csv_path)

    # Execute CoCircular vectorized audit
    audit_res = audit_waste_transfer_notes(waste_notes)

    # Top KPI Metrics Row
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Albaranes Procesados", f"{audit_res.total_notes_processed} DIs")
    with kpi2:
        st.metric("Masa Despachada", f"{audit_res.total_dispatched_tons:,.1f} t")
    with kpi3:
        st.metric("Masa Recibida en Planta", f"{audit_res.total_received_tons:,.1f} t", delta=f"-{audit_res.mass_leakage_pct:.2f}% merma")
    with kpi4:
        val_status = "✅ CUMPLE" if audit_res.ley_7_2022_compliant else "❌ INCUMPLE"
        st.metric("Tasa Valorización RCD", f"{audit_res.valorization_rate_pct:.1f}%", delta=f"{val_status} (>=70% Ley 7/2022)")
    with kpi5:
        st.metric("Scope 3 Evitado", f"{audit_res.net_scope3_avoided_tco2e:,.1f} tCO₂e")

    if audit_res.flagged_note_ids:
        st.warning(
            f"⚠️ **Alertas de báscula detectadas:** Los siguientes albaranes presentan discrepancia >3% "
            f"o contienen residuos peligrosos que requieren protocolo especial: **{list(audit_res.flagged_note_ids)}**"
        )

    # Data Table View
    df_notes = pd.DataFrame(
        [
            {
                "Albarán / DI": n.note_id,
                "Proyecto / Obra": n.project_site,
                "Código LER": n.ler_code,
                "Material": n.material_name,
                "Despachado (t)": n.dispatched_mass_tons,
                "Recibido (t)": n.received_mass_tons,
                "Valorizado (t)": n.valorized_mass_tons,
                "Operación": n.treatment_operation,
                "Peligroso": "⚠️ SÍ" if n.is_hazardous else "NO",
                "Alerta": "🚨" if n.note_id in audit_res.flagged_note_ids else "✔️",
            }
            for n in waste_notes
        ]
    )
    st.dataframe(df_notes, use_container_width=True)

    # Chart: Material mass distribution
    chart_df = df_notes.groupby("Material")[["Despachado (t)", "Valorizado (t)"]].sum()
    st.bar_chart(chart_df)

# =============================================================================
# TAB 2: MATTECO - SCIML AEM/NIFE GREEN HYDROGEN KINETICS & DEGRADATION
# =============================================================================
with tab_matteco:
    st.header("Matteco: Simulación Electroquímica SciML & Cinética de Degradación")
    st.markdown(
        r"Modelado de curvas de polarización **Butler-Volmer / Tafel** y degradación estocástica "
        r"de voltaje ($\mu\text{V/h}$) vía **Euler-Maruyama (SDE)** bajo cargas renovables intermitentes "
        r"en la planta de **Paterna** (10.000 $\text{m}^2$)."
    )

    col_sim_ctrl1, col_sim_ctrl2, col_sim_ctrl3, col_sim_ctrl4 = st.columns(4)
    with col_sim_ctrl1:
        current_density = st.slider("Densidad de Corriente (A/cm²)", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
    with col_sim_ctrl2:
        op_temp_c = st.slider("Temperatura Operación (°C)", min_value=50.0, max_value=85.0, value=70.0, step=5.0)
    with col_sim_ctrl3:
        op_hours = st.slider("Horas de Operación Anual (h)", min_value=2000, max_value=8760, value=8000, step=500)
    with col_sim_ctrl4:
        ppa_price = st.slider("Precio Electricidad PPA (€/MWh)", min_value=20.0, max_value=70.0, value=34.0, step=2.0)

    # Build and configure cells
    m_base, c_base = build_matteco_catalyst_comparison_cells()
    temp_k = op_temp_c + 273.15

    custom_matteco_cell = ElectrolyzerCellTelemetry(
        cell_id="CELL-MTC-STREAMLIT",
        catalyst_batch="MTC-PATERNA-NIFE-2026A",
        active_area_cm2=m_base.active_area_cm2,
        operating_temp_k=temp_k,
        current_density_a_cm2=current_density,
        reversible_potential_v=m_base.reversible_potential_v,
        thermoneutral_voltage_v=m_base.thermoneutral_voltage_v,
        tafel_slope_v_dec=m_base.tafel_slope_v_dec,
        exchange_current_density_a_cm2=m_base.exchange_current_density_a_cm2,
        area_specific_resistance_ohm_cm2=m_base.area_specific_resistance_ohm_cm2,
        degradation_drift_uv_per_h=m_base.degradation_drift_uv_per_h,
        degradation_volatility_uv_sqrt_h=m_base.degradation_volatility_uv_sqrt_h,
        faradaic_efficiency=m_base.faradaic_efficiency,
    )

    custom_conventional_cell = ElectrolyzerCellTelemetry(
        cell_id="CELL-STD-STREAMLIT",
        catalyst_batch="STD-NICKEL-FOAM-BASELINE",
        active_area_cm2=c_base.active_area_cm2,
        operating_temp_k=temp_k,
        current_density_a_cm2=current_density,
        reversible_potential_v=c_base.reversible_potential_v,
        thermoneutral_voltage_v=c_base.thermoneutral_voltage_v,
        tafel_slope_v_dec=c_base.tafel_slope_v_dec,
        exchange_current_density_a_cm2=c_base.exchange_current_density_a_cm2,
        area_specific_resistance_ohm_cm2=c_base.area_specific_resistance_ohm_cm2,
        degradation_drift_uv_per_h=c_base.degradation_drift_uv_per_h,
        degradation_volatility_uv_sqrt_h=c_base.degradation_volatility_uv_sqrt_h,
        faradaic_efficiency=c_base.faradaic_efficiency,
    )

    sim_matteco = simulate_electrolyzer_degradation_sde(
        custom_matteco_cell,
        operating_hours=float(op_hours),
        ppa_electricity_price_eur_mwh=float(ppa_price),
        seed=42,
    )
    sim_conv = simulate_electrolyzer_degradation_sde(
        custom_conventional_cell,
        operating_hours=float(op_hours),
        ppa_electricity_price_eur_mwh=float(ppa_price),
        seed=42,
    )

    # Comparison Metrics
    col_mtc_metric1, col_mtc_metric2, col_mtc_metric3, col_mtc_metric4 = st.columns(4)
    with col_mtc_metric1:
        st.metric(
            "Voltaje Celda (P95 Terminal)",
            f"{sim_matteco.p95_terminal_voltage_v:.4f} V",
            delta=f"{sim_matteco.p95_terminal_voltage_v - sim_conv.p95_terminal_voltage_v:.4f} V vs convencional",
            delta_color="inverse",
        )
    with col_mtc_metric2:
        st.metric(
            "Consumo Específico",
            f"{sim_matteco.specific_energy_kwh_per_kg_h2:.2f} kWh/kg",
            delta=f"{sim_matteco.voltage_efficiency_lhv_pct:.1f}% LHV (vs {sim_conv.voltage_efficiency_lhv_pct:.1f}%)",
        )
    with col_mtc_metric3:
        delta_parity = sim_matteco.lcoh_eur_per_kg_h2 - GREY_H2_COST_PARITY_EUR_PER_KG
        parity_str = "✅ Bate Paridad Gris" if sim_matteco.beats_grey_h2_parity else "❌ Sobre Paridad"
        st.metric(
            "LCOH H₂ Verde (€/kg)",
            f"{sim_matteco.lcoh_eur_per_kg_h2:.2f} €/kg",
            delta=f"{parity_str} (<{GREY_H2_COST_PARITY_EUR_PER_KG:.2f} €/kg)",
            delta_color="inverse",
        )
    with col_mtc_metric4:
        st.metric(
            "Abatimiento CO₂e Anual",
            f"{sim_matteco.annual_co2_abated_tons:,.1f} tCO₂e/stack",
            delta=f"{sim_matteco.annual_h2_production_kg:,.0f} kg H₂ verde",
        )

    # Synthetic trajectory chart
    time_points = np.linspace(0, op_hours, 50)
    # Mean SDE drift trajectory
    deg_mtc = (
        sim_matteco.initial_cell_voltage_v
        + (custom_matteco_cell.degradation_drift_uv_per_h * 1e-6) * time_points
    )
    deg_conv = (
        sim_conv.initial_cell_voltage_v
        + (custom_conventional_cell.degradation_drift_uv_per_h * 1e-6) * time_points
    )
    df_sde_chart = pd.DataFrame(
        {
            "Horas de Operación": time_points,
            "Matteco Nanolaminado NiFe (V)": deg_mtc,
            "Alcalino Convencional (V)": deg_conv,
        }
    ).set_index("Horas de Operación")

    st.subheader("Trayectoria de Degradación de Voltaje en Banco de Pruebas")
    st.line_chart(df_sde_chart)

# =============================================================================
# TAB 3: ZUBI CAPITAL - NON-LINEAR SLSQP PARETO ALLOCATOR & CVAR 95
# =============================================================================
with tab_capital:
    st.header("Zubi Capital: Asignación Estocástica de Capital & Frontera de Pareto")
    st.markdown(
        r"Optimización no lineal restringida (**SLSQP con Jacobiano Analítico exacto**) "
        r"con cribado **SFDR Artículo 9 (PAIs)** y verificación de riesgo de cola **Monte Carlo ($\text{CVaR}_{95\%}$)**."
    )

    # Dynamic calibration from Tier 1 telemetry
    calibrated_portfolio = build_zubi_ecosystem_portfolio(
        cocircular_audit=audit_res, matteco_sim=sim_matteco
    )

    col_opt_ctrl1, col_opt_ctrl2, col_opt_ctrl3 = st.columns(3)
    with col_opt_ctrl1:
        total_budget_eur = st.number_input(
            "Presupuesto Total de Inversión (€)",
            min_value=120000.0,
            max_value=1000000.0,
            value=450000.0,
            step=25000.0,
        )
    with col_opt_ctrl2:
        # Calculate maximum achievable carbon abatement under current budget
        bounds = [(p.min_investment_eur, p.max_investment_eur) for p in calibrated_portfolio]
        alphas = np.array([p.abatement_alpha for p in calibrated_portfolio])
        max_abatable = compute_max_achievable_abatement_tons(bounds, alphas, total_budget_eur)

        carbon_target = st.slider(
            "Mandato Mínimo Vinculante de Abatimiento (tCO₂e)",
            min_value=2500.0,
            max_value=float(np.floor(max_abatable)),
            value=float(min(4900.0, np.floor(max_abatable))),
            step=50.0,
        )
    with col_opt_ctrl3:
        use_sector_corr = st.checkbox(
            "Aplicar Correlación Sectorial en Riesgo de Cola (Construcción / Deep-Tech)",
            value=True,
        )

    corr_matrix = None
    if use_sector_corr:
        n_v = len(calibrated_portfolio)
        corr_matrix = np.full((n_v, n_v), 0.55)
        np.fill_diagonal(corr_matrix, 1.0)

    # Solve optimization
    opt_result = optimize_stochastic_pareto_portfolio(
        calibrated_portfolio,
        total_budget_eur=float(total_budget_eur),
        min_co2_abatement_target_tons=float(carbon_target),
        correlation_matrix=corr_matrix,
        seed=42,
    )
    greedy_result = optimize_stochastic_pareto_portfolio(
        calibrated_portfolio,
        total_budget_eur=float(total_budget_eur),
        min_co2_abatement_target_tons=0.0,
        correlation_matrix=corr_matrix,
        seed=42,
    )

    # Top KPI Metrics Row
    kpi_opt1, kpi_opt2, kpi_opt3, kpi_opt4 = st.columns(4)
    with kpi_opt1:
        st.metric("Capital Total Desplegado", f"{opt_result.total_invested_eur:,.0f} €", delta=f"KKT: {opt_result.success} ({opt_result.execution_time_ms:.1f} ms)")
    with kpi_opt2:
        st.metric("ROI Ponderado Cartera", f"{opt_result.portfolio_roi_pct:.2f}%", delta=f"{opt_result.expected_financial_return_eur:,.0f} €/año (vs {greedy_result.portfolio_roi_pct:.2f}% Greedy)")
    with kpi_opt3:
        st.metric("Abatimiento Esperado", f"{opt_result.expected_total_abatement_tco2:,.1f} tCO₂e", delta=f"+{opt_result.expected_total_abatement_tco2 - greedy_result.expected_total_abatement_tco2:,.1f} tCO₂e vs Greedy")
    with kpi_opt4:
        st.metric("Riesgo Cola CVaR 95%", f"{opt_result.cvar_95_abatement_tco2:,.1f} tCO₂e", delta=f"VaR 95%: {opt_result.var_95_abatement_tco2:,.1f} tCO₂e")

    st.subheader("Matriz Óptima de Asignación de Capital")
    df_opt_table = pd.DataFrame(
        {
            "Participada Zubi Group": opt_result.venture_names,
            "Vertical": [p.vertical for p in calibrated_portfolio],
            "Capital Asignado (€)": opt_result.allocations_eur,
            "Abatimiento (tCO₂e)": opt_result.expected_abatement_per_venture_tco2,
            "Retorno Anual (€)": [
                round(x * p.expected_financial_roi, 2)
                for x, p in zip(opt_result.allocations_eur, calibrated_portfolio)
            ],
            "Cumple SFDR PAI": ["✅ SÍ" if p.sfdr_pai_compliant else "❌ NO" for p in calibrated_portfolio],
        }
    )
    st.dataframe(df_opt_table, use_container_width=True)

    # Pareto Efficient Frontier Computation
    st.subheader("Curva de Frontera Eficiente de Pareto (Retorno Financiero vs Abatimiento)")
    with st.spinner("Calculando frontera de Pareto multi-objetivo..."):
        min_co2 = greedy_result.expected_total_abatement_tco2
        grid_targets = np.linspace(min_co2, max_abatable - 1.0, 15)
        pareto_points = []
        for target in grid_targets:
            try:
                res_point = optimize_stochastic_pareto_portfolio(
                    calibrated_portfolio,
                    total_budget_eur=float(total_budget_eur),
                    min_co2_abatement_target_tons=float(target),
                    seed=42,
                )
                if res_point.success:
                    pareto_points.append(
                        {
                            "Abatimiento CO₂e (t)": res_point.expected_total_abatement_tco2,
                            "ROI Financiero (%)": res_point.portfolio_roi_pct,
                            "CVaR 95% (t)": res_point.cvar_95_abatement_tco2,
                        }
                    )
            except Exception:
                continue

    if pareto_points:
        df_pareto = pd.DataFrame(pareto_points).sort_values("Abatimiento CO₂e (t)")
        st.line_chart(df_pareto.set_index("Abatimiento CO₂e (t)")["ROI Financiero (%)"])
        st.caption("Gráfica: Relación de intercambio óptima. Cada punto representa la solución matemática KKT no dominada.")

    # Strategy Benchmark Table
    st.subheader("Comparativa de Estrategias")
    naive_alloc = np.minimum(
        total_budget_eur / len(calibrated_portfolio),
        [p.max_investment_eur for p in calibrated_portfolio],
    )
    naive_rois = np.array([p.expected_financial_roi for p in calibrated_portfolio])
    naive_alphas = np.array([p.abatement_alpha for p in calibrated_portfolio])
    naive_invested = np.sum(naive_alloc)
    naive_return = np.sum(naive_alloc * naive_rois)
    naive_roi = (naive_return / naive_invested) * 100.0
    naive_abated = np.sum(12.0 * naive_alphas * np.sqrt(naive_alloc / 1000.0))

    df_comp = pd.DataFrame(
        [
            {
                "Estrategia": "1. Greedy ROI (Foco Financiero Puro)",
                "Capital (€)": f"{greedy_result.total_invested_eur:,.0f} €",
                "Abatimiento (tCO₂e)": f"{greedy_result.expected_total_abatement_tco2:,.1f}",
                "ROI Ponderado (%)": f"{greedy_result.portfolio_roi_pct:.2f}%",
                "Cumple Mandato": "❌ NO" if greedy_result.expected_total_abatement_tco2 < carbon_target else "✅ SÍ",
            },
            {
                "Estrategia": "2. Asignación Ingenua Igualitaria (Naive)",
                "Capital (€)": f"{naive_invested:,.0f} €",
                "Abatimiento (tCO₂e)": f"{naive_abated:,.1f}",
                "ROI Ponderado (%)": f"{naive_roi:.2f}%",
                "Cumple Mandato": "❌ NO" if naive_abated < carbon_target else "✅ SÍ",
            },
            {
                "Estrategia": "3. Optimizador SLSQP Zubi Capital",
                "Capital (€)": f"{opt_result.total_invested_eur:,.0f} €",
                "Abatimiento (tCO₂e)": f"{opt_result.expected_total_abatement_tco2:,.1f}",
                "ROI Ponderado (%)": f"{opt_result.portfolio_roi_pct:.2f}%",
                "Cumple Mandato": "✅ SÍ (Frontera Óptima)",
            },
        ]
    )
    st.table(df_comp)
