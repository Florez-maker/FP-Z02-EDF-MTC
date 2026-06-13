import io
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import traceback
import streamlit as st
import re
import geopandas as gpd
from PIL import Image

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ════════════════════════════════════════════════════

VARIABLES_SUELO_MAP = {
    "pH": {"label": "pH (1:1)", "unit": "", "desc": "Acidez del suelo"},
    "Cea": {"label": "Conductividad eléctrica (dS/m)", "unit": "dS/m", "desc": "Salinidad del suelo"},
    "CE": {"label": "Conductividad eléctrica (dS/m)", "unit": "dS/m", "desc": "Conductividad eléctrica del suelo"},
    "MO": {"label": "Materia orgánica (%)", "unit": "%", "desc": "Contenido de materia orgánica"},
    "P": {"label": "Fósforo disponible (mg/kg)", "unit": "mg/kg", "desc": "Fósforo disponible (Bray II)"},
    "Ca": {"label": "Calcio intercambiable (cmolc/kg)", "unit": "cmolc/kg", "desc": "Calcio intercambiable"},
    "Mg": {"label": "Magnesio intercambiable (cmolc/kg)", "unit": "cmolc/kg", "desc": "Magnesio intercambiable"},
    "K": {"label": "Potasio intercambiable (cmolc/kg)", "unit": "cmolc/kg", "desc": "Potasio intercambiable"},
    "Na": {"label": "Sodio intercambiable (cmolc/kg)", "unit": "cmolc/kg", "desc": "Sodio intercambiable"},
    "K_Na": {"label": "Relación K/Na", "unit": "", "desc": "Relación Potasio/Sodio - Indicador de balance nutricional"},
    "Mg_K": {"label": "Relación Mg/K", "unit": "", "desc": "Relación Magnesio/Potasio - Indicador de balance catiónico"},
    "Ca_K": {"label": "Relación Ca/K", "unit": "", "desc": "Relación Calcio/Potasio - Indicador de balance catiónico"},
    "Al": {"label": "Aluminio intercambiable (cmolc/kg)", "unit": "cmolc/kg", "desc": "Aluminio intercambiable"},
    "CIC": {"label": "CIC (cmolc/kg)", "unit": "cmolc/kg", "desc": "Capacidad de intercambio catiónico"},
    "CICE": {"label": "CIC efectiva (cmolc/kg)", "unit": "cmolc/kg", "desc": "CIC efectiva"},
    "Acid_int": {"label": "Acidez intercambiable (cmolc/kg)", "unit": "cmolc/kg", "desc": "Acidez intercambiable"},
    "Sat_Ca": {"label": "Saturación Ca (%)", "unit": "%", "desc": "Saturación de calcio"},
    "Sat_Mg": {"label": "Saturación Mg (%)", "unit": "%", "desc": "Saturación de magnesio"},
    "Sat_K": {"label": "Saturación K (%)", "unit": "%", "desc": "Saturación de potasio"},
    "Sat_Na": {"label": "Saturación Na (%)", "unit": "%", "desc": "Saturación de sodio"},
    "Sat_Al": {"label": "Saturación Al (%)", "unit": "%", "desc": "Saturación de aluminio"},
    "Ca_Mg": {"label": "Relación Ca/Mg", "unit": "", "desc": "Relación Calcio/Magnesio - Indicador de balance catiónico"},
    "Ca_Mg__K": {"label": "Relación (Ca+Mg)/K", "unit": "", "desc": "Relación (Calcio+Magnesio)/Potasio - Indicador de balance de bases"},
    "PSI": {"label": "PSI - Porcentaje de Sodio Intercambiable (%)", "unit": "%", "desc": "Porcentaje de Sodio Intercambiable - Indicador de sodicidad"},
    "RAS": {"label": "RAS - Relación de Adsorción de Sodio", "unit": "", "desc": "Relación de Adsorción de Sodio - Indicador de riesgo de sodificación"},
    "B": {"label": "Boro (mg/kg)", "unit": "mg/kg", "desc": "Boro disponible"},
    "Cu": {"label": "Cobre (mg/kg)", "unit": "mg/kg", "desc": "Cobre disponible"},
    "Fe": {"label": "Hierro (mg/kg)", "unit": "mg/kg", "desc": "Hierro disponible"},
    "Mn": {"label": "Manganeso (mg/kg)", "unit": "mg/kg", "desc": "Manganeso disponible"},
    "Zn": {"label": "Zinc (mg/kg)", "unit": "mg/kg", "desc": "Zinc disponible"},
    "S": {"label": "Azufre (mg/kg)", "unit": "mg/kg", "desc": "Azufre disponible"},
    "A": {"label": "Arena (%)", "unit": "%", "desc": "Contenido de arena"},
    "L": {"label": "Limo (%)", "unit": "%", "desc": "Contenido de limo"},
    "Ar": {"label": "Arcilla (%)", "unit": "%", "desc": "Contenido de arcilla"},
    "Fe_Mn": {"label": "Relación Fe/Mn", "unit": "", "desc": "Relación Hierro/Manganeso - Indicador de disponibilidad de micronutrientes"},
}

RANGOS_BMA = {
    "pH": (4.5, 5.0),
    "MO": (2.0, 4.0),
    "Cea": (2.0, 4.0),
    "CE": (2.0, 4.0),
    "CIC": (10.0, 20.0),
    "K": (0.2, 0.4),
    "P": (15.0, 20.0),
    "B": (0.25, 0.50),
    "Fe": (15.0, 30.0),
}

DIMENSIONES_CORR = {
    "Reacción del suelo, salinidad y CIC": ["pH","Cea","CE","CIC","CICE","Acid_int","Al","Sat_Al","Sat_Na","PSI","RAS"],
    "Bases intercambiables, MO y relaciones": ["MO","P","Ca","Mg","K","Na","K_Na","Mg_K","Ca_K","Sat_Ca","Sat_Mg","Sat_K","Ca_Mg","Ca_Mg__K"],
    "Micronutrientes y textura": ["B","Cu","Fe","Mn","Zn","S","Fe_Mn","A","L","Ar"],
}

st.set_page_config(
    page_title="Dashboard Suelos Palma",
    page_icon="🌴",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {
    "primary": "#1b60a7",
    "success": "#2ca02c",
    "danger": "#d62728",
    "warning": "#F1C40F",
    "info": "#E74C3C",
    "bg": "#F0F4F8",
}

def sanitize_key(s):
    return str(s).replace(" ", "_").replace("/", "_").replace("-", "_").replace(".", "_").replace("(", "").replace(")", "")

# ════════════════════════════════════════════════════
# CSS PERSONALIZADO
# ════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #0A3D62 0%, #1A6B3C 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; font-weight: 700; }
    .main-header p  { margin: 0.3rem 0 0; opacity: 0.8; font-size: 0.9rem; }

    /* KPI cards */
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #1b60a7;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    .kpi-label { font-size: 0.72rem; font-weight: 600; color: #7A8899;
                 text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #1C2B3A; line-height: 1.1; }
    .kpi-sub   { font-size: 0.72rem; color: #7A8899; margin-top: 2px; }

    /* Sección separadora */
    .section-title {
        font-size: 1rem; font-weight: 700; color: #0A3D62;
        border-bottom: 2px solid #1A6B3C;
        padding-bottom: 0.3rem; margin: 1.2rem 0 0.8rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #F0F4F8; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600; font-size: 0.85rem;
    }

    /* Upload zone */
    .upload-zone {
        border: 2px dashed #1b60a7; border-radius: 10px;
        padding: 2rem; text-align: center; background: #f0f7ff;
    }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# 1. CARGA Y PREPARACIÓN DE DATOS
# ════════════════════════════════════════════════════

COMMON_LAT_COLS = ["lat", "latitude", "latitud", "y", "latitud_gps"]
COMMON_LON_COLS = ["lon", "lng", "longitude", "longitud", "x", "longitud_gps"]
COMMON_DEPTH_COLS = ["prof", "profundidad", "depth"]
COMMON_FARMS = ["finca", "farm", "hacienda"]
COMMON_LOTES = ["lote", "lot", "parcela", "plot"]

def _find_col_ci(cols, candidates):
    cols_low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in cols_low: return cols_low[cand.lower()]
    for c in cols:
        for cand in candidates:
            if cand.lower() in str(c).lower(): return c
    return None

def _clean_col_name(col):
    s = str(col).strip()
    s = re.sub(r"[\/\(\)]", " ", s)
    s = re.sub(r"\s+", "_", s)
    s = s.replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u")
    return s

@st.cache_data(show_spinner=False)
def cargar_suelos(file_bytes: bytes, file_name: str = "") -> tuple:
    """Lee y normaliza un Excel o CSV de suelos. Retorna (df, gdf, colmap)."""
    is_csv = isinstance(file_name, str) and file_name.lower().endswith(".csv")
    if is_csv:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine="python", dtype=object)
        except Exception:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=",", dtype=object)
    else:
        df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")

    orig_cols = list(df.columns)
    new_cols = [_clean_col_name(c) for c in orig_cols]
    df.columns = new_cols

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")].copy()

    canonical = {}
    cols_list = list(df.columns)

    lat_col = _find_col_ci(cols_list, COMMON_LAT_COLS)
    lon_col = _find_col_ci(cols_list, COMMON_LON_COLS)
    prof_col = _find_col_ci(cols_list, COMMON_DEPTH_COLS)
    finca_col = _find_col_ci(cols_list, COMMON_FARMS)
    lote_col = _find_col_ci(cols_list, COMMON_LOTES)

    rename_map = {}
    if lat_col: rename_map[lat_col] = "Latitud"; canonical["lat"]="Latitud"
    if lon_col: rename_map[lon_col] = "Longitud"; canonical["lon"]="Longitud"
    if prof_col: rename_map[prof_col] = "Prof"; canonical["prof"]="Prof"
    if finca_col: rename_map[finca_col] = "Finca"; canonical["finca"]="Finca"
    if lote_col: rename_map[lote_col] = "Lote"; canonical["lote"]="Lote"

    if rename_map: df = df.rename(columns=rename_map)

    # Para CSV, lat/lon/prof son opcionales (puede no tener coordenadas)
    required_strict = ["Finca", "Lote"]
    required_geo    = ["Longitud", "Latitud", "Prof"]
    missing_strict  = [c for c in required_strict if c not in df.columns]
    if missing_strict:
        raise ValueError(f"Faltan columnas requeridas: {missing_strict}. Columnas detectadas: {list(df.columns)}")
    missing_geo = [c for c in required_geo if c not in df.columns]
    if missing_geo:
        # Añadir columnas vacías para que el resto del pipeline no falle
        for c in missing_geo:
            df[c] = np.nan

    df["Latitud"] = pd.to_numeric(df["Latitud"].astype(str).str.replace(",", "."), errors="coerce")
    df["Longitud"] = pd.to_numeric(df["Longitud"].astype(str).str.replace(",", "."), errors="coerce")
    df["Prof"] = pd.to_numeric(df["Prof"].astype(str).str.replace(",", "."), errors="coerce")

    prof_num = df["Prof"]
    prof_cat = pd.Series(index=df.index, dtype="object")
    prof_cat.loc[prof_num <= 20] = "0-20 cm"
    prof_cat.loc[(prof_num > 20) & (prof_num.notna())] = "20-40 cm"
    df["Prof_Cat"] = prof_cat

    if "Mg" in df.columns and "K" in df.columns and "Mg_K" not in df.columns:
        df["Mg_K"] = np.where(pd.to_numeric(df["K"], errors="coerce") > 0,
                              pd.to_numeric(df["Mg"], errors="coerce") / pd.to_numeric(df["K"], errors="coerce"),
                              np.nan)
    if "Ca" in df.columns and "K" in df.columns and "Ca_K" not in df.columns:
        df["Ca_K"] = np.where(pd.to_numeric(df["K"], errors="coerce") > 0,
                              pd.to_numeric(df["Ca"], errors="coerce") / pd.to_numeric(df["K"], errors="coerce"),
                              np.nan)
    if "Na" in df.columns and "CIC" in df.columns and "PSI" not in df.columns:
        df["PSI"] = np.where(pd.to_numeric(df["CIC"], errors="coerce") > 0,
                             pd.to_numeric(df["Na"], errors="coerce") / pd.to_numeric(df["CIC"], errors="coerce") * 100,
                             np.nan)
    if set(["Na", "Ca", "Mg"]).issubset(df.columns) and "RAS" not in df.columns:
        ca = pd.to_numeric(df["Ca"], errors="coerce"); mg = pd.to_numeric(df["Mg"], errors="coerce")
        df["RAS"] = np.where((ca + mg) > 0, pd.to_numeric(df["Na"], errors="coerce") / np.sqrt((ca + mg) / 2), np.nan)
    if "Fe" in df.columns and "Mn" in df.columns and "Fe_Mn" not in df.columns:
        df["Fe_Mn"] = np.where(pd.to_numeric(df["Mn"], errors="coerce") > 0,
                               pd.to_numeric(df["Fe"], errors="coerce") / pd.to_numeric(df["Mn"], errors="coerce"),
                               np.nan)
    if "CE" not in df.columns and "Cea" in df.columns:
        df["CE"] = df["Cea"]

    gdf = None
    try:
        gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df["Longitud"], df["Latitud"]), crs="EPSG:4326")
    except Exception:
        gdf = None

    return df, gdf, canonical

# ════════════════════════════════════════════════════
# 2. FUNCIONES DE GRÁFICOS
# ════════════════════════════════════════════════════

def _step_colorscale(cmin, t1, t2, cmax, low="#2c7bb6", mid="#ffffbf", high="#d7191c"):
    if cmax == cmin: return [(0.0, mid), (1.0, mid)]
    p1 = (t1 - cmin) / (cmax - cmin); p2 = (t2 - cmin) / (cmax - cmin)
    p1 = max(0.0, min(1.0, p1)); p2 = max(0.0, min(1.0, p2))
    return [(0.0, low), (p1, low), (p1, mid), (p2, mid), (p2, high), (1.0, high)]

def plot_distplot(data, metric=None) -> go.Figure:
    if isinstance(data, pd.DataFrame):
        if metric is None:
            return go.Figure(layout=go.Layout(title="No se indicó la columna para el DataFrame."))
        vals = pd.to_numeric(data[metric].astype(str).str.replace(",", "."), errors="coerce").dropna(); title = metric
    else:
        vals = pd.to_numeric(pd.Series(data).astype(str).str.replace(",", "."), errors="coerce").dropna(); title = metric if metric else (getattr(data, "name", "Variable"))

    if vals.shape[0] < 2:
        fig = go.Figure(); fig.add_trace(go.Histogram(x=vals, marker_color=COLORS["primary"], nbinsx=max(1, min(10, len(vals)))))
        fig.update_layout(height=260, margin=dict(t=10, l=0, r=0, b=0), xaxis_title=title, yaxis_title="Conteo"); return fig

    if vals.nunique() < 2:
        single_val = float(vals.iloc[0]); fig = go.Figure(); fig.add_trace(go.Histogram(x=vals, marker_color=COLORS["primary"], nbinsx=1, opacity=0.7))
        fig.add_vline(x=single_val, line_dash="dash", line_color=COLORS["info"], annotation_text=f"Valor único: {single_val:.3f}", annotation_position="top right")
        fig.update_layout(height=260, margin=dict(t=10, l=0, r=0, b=0), xaxis_title=title, yaxis_title="Conteo"); return fig

    try:
        kde_x = np.linspace(vals.min(), vals.max(), 200); kde_y = gaussian_kde(vals)(kde_x)
    except Exception:
        kde_x, kde_y = None, None

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=vals, histnorm="probability density", name="Histograma", marker_color=COLORS["primary"], opacity=0.6, nbinsx=20))
    if kde_x is not None: fig.add_trace(go.Scatter(x=kde_x, y=kde_y, mode="lines", name="KDE", line=dict(color=COLORS["primary"], width=2)))
    mean_val = float(vals.mean()); median_val = float(vals.median())
    fig.add_vline(x=mean_val, line_dash="dash", line_color=COLORS["info"], annotation_text=f"Media: {mean_val:.2f}", annotation_position="top right")
    fig.add_vline(x=median_val, line_dash="dot", line_color=COLORS["warning"], annotation_text=f"Mediana: {median_val:.2f}", annotation_position="bottom right")
    fig.update_layout(height=310, margin=dict(t=10, l=0, r=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, xaxis_title=title, yaxis_title="Densidad")
    return fig

def _stats_por_lote(df: pd.DataFrame, var: str) -> pd.DataFrame:
    d = df[["Lote", var]].replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty: return pd.DataFrame(columns=["Lote", "N", "Media", "Mediana", "DE", "CV (%)", "Mín", "Máx"])
    stats = (d.groupby("Lote")[var].agg(N="count", Media="mean", Mediana="median", DE="std", Mín="min", Máx="max").reset_index())
    stats["CV (%)"] = np.where(stats["Media"] != 0, stats["DE"] / stats["Media"] * 100, np.nan)
    return stats

def plot_boxplot_por_lote(df: pd.DataFrame, var: str):
    if var not in df.columns: return go.Figure(layout=go.Layout(title="Variable no disponible para boxplot."))
    d = df[["Lote", var]].replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty: return go.Figure(layout=go.Layout(title="Sin datos para esta variable."))
    orden_lotes = d.groupby("Lote")[var].median().sort_values().index.tolist()
    traces, medians = [], []
    for lote in orden_lotes:
        vals = d.loc[d["Lote"] == lote, var].astype(float).values
        traces.append(go.Box(y=vals.tolist(), name=lote, boxpoints="outliers", marker=dict(color=COLORS["primary"]), jitter=0.2, pointpos=-1.8))
        medians.append((lote, np.median(vals)))
    med_df = pd.DataFrame(medians, columns=["Lote", "Mediana"])
    fig = go.Figure(data=traces)
    fig.add_trace(go.Scatter(x=med_df["Lote"], y=med_df["Mediana"], mode="markers+lines", marker=dict(color=COLORS["success"], size=8), name="Mediana", hovertemplate="Lote: %{x}<br>Mediana: %{y:.2f}<extra></extra>"))
    label = VARIABLES_SUELO_MAP.get(var, {}).get("label", var)
    fig.update_layout(title=f"Distribución de {label} por lote", height=460, margin=dict(t=40, l=0, r=0, b=120), template="plotly_white", showlegend=True)
    fig.update_xaxes(tickangle=45); fig.update_yaxes(title=label)
    return fig

def plot_boxplot_por_profundidad(df: pd.DataFrame, var: str):
    if var not in df.columns or "Prof_Cat" not in df.columns: return go.Figure(layout=go.Layout(title="Variable o profundidad no disponible."))
    d = df[["Prof_Cat", var]].replace([np.inf, -np.inf], np.nan).dropna()
    if d.empty: return go.Figure(layout=go.Layout(title="Sin datos para esta variable."))
    orden_profundidad = [c for c in sorted(d["Prof_Cat"].unique(), key=lambda s: str(s)) if pd.notna(c)]
    traces = []
    for prof in orden_profundidad:
        vals = d.loc[d["Prof_Cat"] == prof, var].astype(float).values
        traces.append(go.Box(y=vals.tolist(), name=str(prof), jitter=0.2, boxpoints="outliers", marker=dict(color=COLORS["primary"])))
    label = VARIABLES_SUELO_MAP.get(var, {}).get("label", var)
    fig = go.Figure(data=traces)
    fig.update_layout(title=f"Distribución de {label} por profundidad", height=420, margin=dict(t=40, l=0, r=0, b=0), template="plotly_white", showlegend=False)
    fig.update_yaxes(title=label)
    return fig

def tabla_estadisticos_globales(df: pd.DataFrame, var: str):
    if var not in df.columns: return go.Figure()
    serie = pd.to_numeric(df[var].replace([np.inf, -np.inf], np.nan), errors="coerce").dropna()
    if serie.empty: return go.Figure()
    n_obs = len(serie); promedio = serie.mean(); mediana = serie.median(); desv_std = serie.std()
    minimo = serie.min(); maximo = serie.max(); cv = (desv_std / promedio * 100) if promedio != 0 else np.nan
    skewness = serie.skew(); kurtosis = serie.kurtosis()
    stats_data = pd.DataFrame({
        "Parámetro": ["Num. Observaciones", "Promedio", "Mediana", "Desviación estándar", "Mínimo", "Máximo", "Coeficiente de Variación (%)", "Skewness (Asimetría)", "Kurtosis"],
        "Valor": [f"{n_obs}", f"{promedio:.3f}", f"{mediana:.3f}", f"{desv_std:.3f}", f"{minimo:.3f}", f"{maximo:.3f}", f"{cv:.3f}", f"{skewness:.3f}", f"{kurtosis:.3f}"],
    })
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(stats_data.columns), fill_color=COLORS["primary"], font=dict(color="white", size=11), align="left"),
        cells=dict(values=[stats_data[c] for c in stats_data.columns], fill_color=[["#f7f9fc" if i % 2 == 0 else "white" for i in range(len(stats_data))]], align="left", font=dict(size=10)),
    )])
    fig.update_layout(margin=dict(t=10, l=0, r=0, b=0), height=360, paper_bgcolor="rgba(0,0,0,0)")
    return fig

def plot_topn_ranking_by_var(df, var, top_n=10):
    if var not in df.columns or "Lote" not in df.columns: return go.Figure(layout=go.Layout(title="No hay datos para ranking."))
    df_v = df[["Lote", var]].replace([np.inf, -np.inf], np.nan).dropna()
    if df_v.empty: return go.Figure(layout=go.Layout(title="Sin datos numéricos para ranking."))
    df_v[var] = pd.to_numeric(df_v[var].astype(str).str.replace(",", "."), errors="coerce")
    agg_df = df_v.groupby("Lote")[var].mean().reset_index(name="valor")   # mean aggregation as requested
    agg_df = agg_df.sort_values("valor", ascending=False).reset_index(drop=True)
    top_df = agg_df.head(top_n).copy()
    top_df = top_df.sort_values("valor", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=top_df["Lote"], y=top_df["valor"], mode="markers+lines", marker=dict(color=COLORS["success"], size=8), line=dict(color=COLORS["success"], width=2), name=f"Top {top_n} por {var}"))
    fig.update_layout(title=f"Ranking Top {top_n} por {var} (promedio por lote)", xaxis_title="Lote", yaxis_title=VARIABLES_SUELO_MAP.get(var, {}).get("label", var), height=420, margin=dict(t=40, l=20, r=20, b=120), template="plotly_white")
    fig.update_xaxes(tickangle=45)
    return fig

# ════════════════════════════════════════════════════
# 3. SECCIONES DEL DASHBOARD
# ════════════════════════════════════════════════════

def seccion_kpis(df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    n = df.shape[0]
    n_fincas = df["Finca"].nunique() if "Finca" in df.columns else 0
    n_lotes = df["Lote"].nunique() if "Lote" in df.columns else 0
    c1.markdown(f"<div class='kpi-card'><div class='kpi-label'>Registros</div><div class='kpi-value'>{n}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi-card'><div class='kpi-label'>Fincas</div><div class='kpi-value'>{n_fincas}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi-card'><div class='kpi-label'>Lotes</div><div class='kpi-value'>{n_lotes}</div></div>", unsafe_allow_html=True)

def render_tab_map_overview(df_vis, lat_col, lon_col, selected_var, map_mode, lote_col, finca_col, prof_col):

    st.markdown('<div class="section-title">🧾 Resumen rápido</div>', unsafe_allow_html=True)

    if selected_var and selected_var in df_vis.columns:
        serie = pd.to_numeric(df_vis[selected_var].astype(str).str.replace(",", "."), errors="coerce").dropna()
        if serie.empty:
            st.info(f"No hay valores numéricos para `{selected_var}` después del filtrado.")
        else:
            mean_val = float(serie.mean())
            median_val = float(serie.median())
            std_val = float(serie.std())
            min_val = float(serie.min())
            max_val = float(serie.max())
            pct_missing = int(df_vis[selected_var].isna().sum())

            # Métricas compactas
            st.markdown(f"- Media: **{mean_val:.2f}** · Mediana: **{median_val:.2f}**")
            st.markdown(
                f"- Rango: **{min_val:.2f} — {max_val:.2f}** · Std: **{std_val:.2f}** · Faltantes: **{pct_missing}**")

            # Percentiles
            p10, p25, p75, p90 = [float(x) for x in np.percentile(serie, [10, 25, 75, 90])]
            st.markdown(f"- P10 / P25 / P75 / P90: **{p10:.2f} / {p25:.2f} / {p75:.2f} / {p90:.2f}**")

            # Descarga de muestra (no modifica datos)
            try:
                sample = df_map_summary[
                    [lat_col, lon_col] + ([selected_var] if selected_var in df_map_summary.columns else [])].head(1000)
                sample_csv = sample.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Descargar muestra (mapa) 1k filas", sample_csv,
                                   file_name=sanitize_key(f"map_sample_{selected_var}.csv"),
                                   mime="text/csv",
                                   key=sanitize_key(f"dl_map_sample_{selected_var}"))
            except Exception:
                pass

    st.markdown(
        "<small><em>Nota:</em> las capas de densidad son estimaciones (kernel); interpretarlas con cautela.</small>",
        unsafe_allow_html=True)

    st.markdown('<div class="section-title">🗺️ Mapa y resumen espacial</div>', unsafe_allow_html=True)
    col_map, col_side = st.columns([3,1])
    with col_map:
        if not (lat_col and lon_col):
            st.info("No hay columnas lat/lon seleccionadas.")
            return
        try:
            df_map = df_vis.dropna(subset=[lat_col, lon_col]).copy()
            df_map[lat_col] = pd.to_numeric(df_map[lat_col], errors="coerce"); df_map[lon_col] = pd.to_numeric(df_map[lon_col], errors="coerce")
            df_map = df_map.dropna(subset=[lat_col, lon_col])
            if df_map.shape[0] == 0:
                st.warning("No hay puntos georreferenciados válidos después del filtrado."); return
            df_map_plot = df_map.copy()
            size_arg = None
            if selected_var and selected_var in df_map_plot.columns:
                if df_map_plot[selected_var].isna().any():
                    median_val = float(df_map_plot[selected_var].median(skipna=True)) if df_map_plot[selected_var].notna().any() else 1.0
                    df_map_plot = df_map_plot.assign(__size_temp = pd.to_numeric(df_map_plot[selected_var].fillna(median_val), errors="coerce"))
                    size_arg = "__size_temp"
                else:
                    size_arg = selected_var

            if map_mode == "Puntos":
                kwargs = dict(lat=lat_col, lon=lon_col,
                            hover_name=(lote_col if lote_col in df_map_plot.columns else None),
                            hover_data=[finca_col, lote_col, prof_col, selected_var] if selected_var else [finca_col, lote_col, prof_col],
                            zoom=10, height=520, color_continuous_scale="Spectral")
                if size_arg is not None: kwargs.update(dict(size=size_arg, size_max=6))
                color_arg = selected_var if (selected_var in df_map_plot.columns) else None
                fig_map_pts = px.scatter_mapbox(df_map_plot, color=color_arg, **kwargs)

                fig_map_pts.update_layout(
                    hovermode="closest",
                    mapbox_style="open-street-map",
                    margin=dict(l=0,r=0,t=0,b=0)
                )

                padding = 0.08
                fig_map_pts.update_layout(
                    mapbox_bounds={
                        "west": float(df_map_plot["Longitud"].min() - padding),
                        "east": float(df_map_plot["Longitud"].max() + padding),
                        "south": float(df_map_plot["Latitud"].min() - padding),
                        "north": float(df_map_plot["Latitud"].max() + padding),
                    }
                )

                st.plotly_chart(fig_map_pts, width="stretch", key=sanitize_key(f"map_points_{selected_var}_{map_mode}"))
            else:
                d = df_map_plot.dropna(subset=[selected_var]) if selected_var else df_map_plot
                if d.shape[0] < 3:
                    st.warning("Pocos puntos con valores para densidad.")
                else:
                    fig_map_den = px.density_mapbox(d,
                                                    lat=lat_col, lon=lon_col,
                                                    z=selected_var if selected_var in d.columns else None,
                                                    radius=12, center={"lat": float(d[lat_col].median()),
                                                    "lon": float(d[lon_col].median())},
                                                    color_continuous_scale="Spectral",
                                                    zoom=10, height=520)

                    fig_map_den.update_layout(
                        hovermode="closest",
                        mapbox_style="carto-positron",
                        mapbox=dict(
                        bearing=0,
                        center=go.layout.mapbox.Center(
                            lat=float(d["Latitud"].median()),
                            lon=float(d["Longitud"].median()),
                        ),
                        pitch=0),
                        margin=dict(l=0,r=0,t=0,b=0),
                        )

                    padding = 0.08
                    fig_map_den.update_layout(
                        mapbox_bounds={
                            "west": float(d["Longitud"].min() - padding),
                            "east": float(d["Longitud"].max() + padding),
                            "south": float(d["Latitud"].min() - padding),
                            "north": float(d["Latitud"].max() + padding),
                        }
                    )
                    st.plotly_chart(fig_map_den, width="stretch", key=sanitize_key(f"map_density_{selected_var}_{map_mode}"))
        except Exception:
            st.error("Error al generar mapa/visualización:"); st.exception(traceback.format_exc())

    with col_side:
        if selected_var:
            serie = pd.to_numeric(df_vis[selected_var].astype(str).str.replace(",", "."), errors="coerce").dropna()
            if serie.empty:
                st.warning(f"No hay valores numéricos para `{selected_var}` después del filtrado.")
            else:
                fig_box_single = go.Figure()
                fig_box_single.add_trace(go.Box(y=serie, name=selected_var, boxpoints="outliers", marker=dict(color=COLORS["primary"]), jitter=0.15, pointpos=-1.8))
                med_val = float(serie.median()); mean_val = float(serie.mean())
                fig_box_single.add_trace(go.Scatter(x=[selected_var], y=[med_val], mode="markers+lines", marker=dict(color=COLORS["success"], size=8), name="Mediana", hovertemplate=f"Mediana: {med_val:.3f}<extra></extra>"))
                fig_box_single.update_layout(title=f"Distribución de {selected_var}", showlegend=False, margin=dict(t=0,l=0,r=0,b=0), height=280, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_box_single, width="stretch", key=sanitize_key(f"map_overview_box_{selected_var}"))
                st.plotly_chart(plot_distplot(serie, metric=selected_var), width="stretch", key=sanitize_key(f"map_overview_dist_{selected_var}"))

def render_tab_distribution(df_vis, selected_var, lote_col):
    st.markdown('<div class="section-title">📊 Distribución y Boxplots</div>', unsafe_allow_html=True)

    # fallback sizes similar a producción (si no están definidas globalmente)
    RANKING_SIZES_LOCAL = st.session_state.get("RANKING_SIZES", [5, 10, 15, 20])

    def generate_topn_df(df, var, n, lote_column):
        if var is None or var not in df.columns:
            return pd.DataFrame()
        df_tmp = df.dropna(subset=[var])
        lote_col_local = lote_column if lote_column in df_tmp.columns else "Lote"
        if lote_col_local not in df_tmp.columns:
            possible = [c for c in df_tmp.columns if any(k in c.lower() for k in ("lote","parcela","plot","lot"))]
            lote_col_local = possible[0] if possible else None
            if lote_col_local is None:
                df_tmp = df_tmp.reset_index().rename(columns={"index": "Lote"})
                lote_col_local = "Lote"
        rank_df = df_tmp.groupby(lote_col_local)[var].mean().reset_index().rename(columns={var: "mean_val", lote_col_local: "Lote"})
        rank_df = rank_df.sort_values("mean_val", ascending=False).head(n)
        return rank_df

    top_left, top_right = st.columns([3, 1])   # Boxplot | Stats
    bot_left, bot_right = st.columns([3, 1])   # Ranking | Controls

    # --------- BOXPLOT (sin resaltar medias por lote; con línea de media global) ----------
    with top_left:
        if selected_var is None:
            st.info("Selecciona una variable numérica en la barra lateral.")
        else:
            st.markdown("#### Boxplot por Lote / Profundidad")
            try:
                if lote_col and lote_col in df_vis.columns:
                    fig_box = plot_boxplot_por_lote(df_vis, selected_var)
                elif "Prof_Cat" in df_vis.columns:
                    fig_box = plot_boxplot_por_profundidad(df_vis, selected_var)
                else:
                    fig_box = plot_boxplot_por_lote(df_vis, selected_var)
            except Exception as e:
                st.error("Error generando boxplot base:")
                st.exception(e)
                fig_box = None

            try:
                if fig_box is not None:
                    global_mean = df_vis[selected_var].dropna().mean() if selected_var in df_vis.columns else None
                    if global_mean is not None and not np.isnan(global_mean):
                        fig_box.add_hline(
                            y=global_mean,
                            line_dash="dash",
                            line_color=COLORS["info"],
                            line_width=3,
                            annotation_text=f"Media global: {global_mean:.2f}",
                            annotation_position="top right",
                            annotation_font_size=11
                        )
                    fig_box.update_layout(height=420, margin=dict(t=30, b=120))
                    st.plotly_chart(fig_box, use_container_width=True, key=sanitize_key(f"box_lote_tab_{selected_var}"))
                else:
                    st.info("No se pudo generar el boxplot.")
            except Exception as e:
                st.error("Error al superponer media global:")
                st.exception(e)

    # --------- ESTADÍSTICAS GLOBALES ----------
    with top_right:
        st.markdown("#### Estadísticas Globales")
        try:
            stats_obj = tabla_estadisticos_globales(df_vis, selected_var)
            if isinstance(stats_obj, pd.DataFrame):
                st.table(stats_obj.reset_index(drop=True))
            elif isinstance(stats_obj, (go.Figure,)):
                st.plotly_chart(stats_obj, use_container_width=True, key=sanitize_key(f"stats_global_tab_{selected_var}"))
            else:
                try:
                    df_stats = pd.DataFrame(stats_obj)
                    st.table(df_stats)
                except Exception:
                    st.write(stats_obj)
        except Exception as e:
            st.error("Error generando estadísticas globales:")
            st.exception(e)

    # --------- CONTROLES (estilo producción) ----------
    with bot_right:
        st.markdown("#### Controles")
        # Top-N con select_slider (estilo producción)
        top_n = st.select_slider("Tamaño del ranking:", options=RANKING_SIZES_LOCAL, value=10, key="topn_tab")

        # Preparar lista de variables numéricas disponibles como fallback
        options_ordered = st.session_state.get("numeric_cols", []).copy()
        if not options_ordered:
            options_ordered = [selected_var] if selected_var else []

        # Usar únicamente la variable seleccionada en la barra lateral como variable de ranking.
        if selected_var and selected_var in df_vis.columns:
            rank_var = selected_var
            st.markdown(f"Variable para ranking: **{rank_var}**")
        else:
            # fallback: primera variable numérica disponible
            rank_var = options_ordered[0] if options_ordered else None
            st.markdown(f"Variable para ranking: **{rank_var or '—'}** (fallback)")

        st.markdown("Nota: el ranking utiliza promedio (mean) por lote.")
        st.markdown("---")

        # Pre-cálculo del Top-N para habilitar descarga inmediata si hay datos
        df_export = generate_topn_df(df_vis, rank_var, top_n, lote_col)
        if df_export is None or df_export.empty:
            st.info("No hay datos suficientes para generar el ranking con la selección actual.")
        else:
            csv = df_export.to_csv(index=False).encode("utf-8")
            dl_name = sanitize_key(f"topn_{rank_var}_{top_n}.csv")
            st.download_button("⬇️ Descargar Top‑N (CSV)", csv, file_name=dl_name, mime="text/csv", key=sanitize_key(f"dl_topn_{rank_var}_{top_n}"))

    # --------- RANKING TOP‑N (gráfico) ----------
    with bot_left:
        st.markdown("#### Ranking Top‑N por lote (promedio)")
        try:
            df_rank = generate_topn_df(df_vis, rank_var, top_n, lote_col)
            if df_rank.empty:
                st.info("No hay suficientes datos para generar el ranking.")
            else:
                df_rank_sorted = df_rank.sort_values("mean_val", ascending=True).reset_index(drop=True)

                fig_rank = go.Figure()
                fig_rank.add_trace(go.Bar(
                    x=df_rank_sorted["mean_val"],
                    y=df_rank_sorted["Lote"],
                    orientation='h',
                    marker=dict(color=COLORS["success"], line=dict(color="#1f7a1f", width=1.2)),
                    text=df_rank_sorted["mean_val"].round(3),
                    textposition="outside",
                    name=f"Top {top_n}"
                ))

                fig_rank.update_layout(
                    height=360,
                    margin=dict(t=30, b=30, l=150, r=30),
                    xaxis=dict(title=f"{rank_var}"),
                    yaxis=dict(title="Lote", automargin=True, autorange="reversed",
                               tickfont=dict(size=11)),
                    showlegend=False,
                    bargap=0.25
                )

                if len(df_rank_sorted) > 15:
                    fig_rank.update_traces(textposition="inside", insidetextanchor="middle")
                    fig_rank.update_layout(height=30 * len(df_rank_sorted))

                st.plotly_chart(fig_rank, use_container_width=True, key=sanitize_key(f"rank_topn_{rank_var}_{top_n}"))
        except Exception as e:
            st.error("Error generando gráfico de ranking:")
            st.exception(e)

def render_tab_correlations(df_vis):
    st.markdown('<div class="section-title">🔗 Correlaciones</div>', unsafe_allow_html=True)

    numeric_cols = st.session_state.get("numeric_cols")
    if not numeric_cols:
        numeric_cols = df_vis.select_dtypes(include=[np.number]).columns.tolist()
        st.session_state["numeric_cols"] = numeric_cols

    grupos_validos = []
    grupos_cols = {}
    for gname, cols in DIMENSIONES_CORR.items():
        avail = [c for c in cols if c in df_vis.columns and pd.api.types.is_numeric_dtype(df_vis[c])]
        if avail:
            grupos_validos.append(gname)
            grupos_cols[gname] = avail

    if grupos_validos:
        group_options = grupos_validos + ["Custom"]
        default_group = grupos_validos[0]
    else:
        group_options = ["Custom"]
        default_group = "Custom"

    if "corr_group_select" not in st.session_state:
        st.session_state["corr_group_select"] = default_group

    def _apply_group_callback():
        sel = st.session_state.get("corr_group_select", default_group)
        if sel == "Custom":
            st.session_state["corr_cols_tab"] = st.session_state.get("corr_cols_tab", numeric_cols.copy())
        else:
            st.session_state["corr_cols_tab"] = grupos_cols.get(sel, []).copy()

    st.selectbox("Grupo de suelos (selección automática):", options=group_options,
                 index=0 if default_group == group_options[0] else 0,
                 key="corr_group_select", on_change=_apply_group_callback)

    if "corr_cols_tab" not in st.session_state:
        if default_group != "Custom" and default_group in grupos_cols:
            st.session_state["corr_cols_tab"] = grupos_cols[default_group].copy()
        else:
            st.session_state["corr_cols_tab"] = numeric_cols[:min(len(numeric_cols), 12)].copy()

    corr_cols = st.multiselect(
        "Columnas para el heatmap de correlación (ajusta si quieres):",
        options=numeric_cols,
        default=st.session_state.get("corr_cols_tab", []),
        key="corr_cols_tab"
    )

    if not corr_cols or len(corr_cols) < 2:
        st.info("Selecciona al menos 2 columnas del grupo o usa 'Custom' para elegir manualmente.")
        return

    sub_df = df_vis[corr_cols].select_dtypes(include=[np.number]).dropna(how="all")
    if sub_df.shape[0] < 3:
        st.info(f"No hay suficientes filas válidas ({sub_df.shape[0]}) para calcular correlación con las columnas seleccionadas.")
        return

    corr = sub_df.corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1, text_auto=".2f",
                    labels=dict(x="Variables", y="Variables", color="Correlación"))
    fig.update_layout(height=520, margin=dict(t=30, b=30, l=10, r=10))
    st.plotly_chart(fig, width="stretch", key=sanitize_key("correlation_heatmap_selected"))

    st.download_button("Descargar matriz de correlación (CSV)", corr.round(3).to_csv().encode("utf-8"),
                       file_name=sanitize_key("correlation_matrix_selected.csv"), mime="text/csv",
                       key=sanitize_key("dl_corr_selected"))

def seccion_tabla_suelos(df, df_by_year, years_desc, year_sel):

    st.markdown('<div class="section-title">📋 Tabla de suelos: filtrado y descarga</div>', unsafe_allow_html=True)

    show_all = st.checkbox("Mostrar todos los años", value=False, key="suelos_show_all")
    if show_all:
        df_show = df.copy()
    else:
        if isinstance(df_by_year, dict):
            df_show = df_by_year.get(year_sel, pd.DataFrame()).copy()
        elif hasattr(df_by_year, "columns"):
            ano_cols = [c for c in df_by_year.columns if any(k in c.lower() for k in ("ano","año","year"))]
            if ano_cols:
                ycol = ano_cols[0]
                df_show = df_by_year[df_by_year[ycol].astype(str) == str(year_sel)].copy()
            else:
                df_show = df_by_year.copy()
        else:
            df_show = df.copy()

    if df_show is None or df_show.empty:
        st.warning("No hay datos de suelos para mostrar con los filtros actuales.")
        return pd.DataFrame()

    cols = list(df_show.columns)
    def find_col(cands):
        for cand in cands:
            for c in cols:
                if cand in c.lower():
                    return c
        return None

    finca_col   = find_col(["finca", "farm"])
    zona_col    = find_col(["zona", "zone", "sector", "area"])
    lote_col    = find_col(["lote", "parcela", "parcel"])
    material_col= find_col(["material", "materiale", "tipo"])
    muestra_col = find_col(["muestra", "sample"])
    ano_col     = find_col(["ano", "año", "year"])

    c1, c2, c3 = st.columns(3)
    def opts_for(col, label_all="Todos"):
        if col and col in df_show.columns:
            vals = sorted(df_show[col].dropna().astype(str).unique().tolist())
            return [label_all] + vals
        return [label_all]

    finca_opts = opts_for(finca_col, "Todas")
    zona_opts  = opts_for(zona_col, "Todas")
    material_opts = opts_for(material_col, "Todos")

    finca_sel = c1.selectbox("Filtrar por Finca:", finca_opts, key="suelos_finca")
    zona_sel  = c2.selectbox("Filtrar por Zona:", zona_opts, key="suelos_zona")
    mat_sel   = c3.selectbox("Filtrar por Material:", material_opts, key="suelos_material")

    s1, s2 = st.columns([2,1])
    search_label = "Buscar Lote/Parcela" if lote_col else ("Buscar Muestra" if muestra_col else "Buscar texto")
    search_txt = s1.text_input(search_label, value="", key="suelos_search")
    show_missing_ph = s2.checkbox("Mostrar filas con pH faltante", value=False, key="suelos_missing_ph")

    df_filt = df_show.copy()
    try:
        if finca_col and finca_sel and finca_sel not in ("Todas",):
            df_filt = df_filt[df_filt[finca_col].astype(str) == finca_sel]
        if zona_col and zona_sel and zona_sel not in ("Todas",):
            df_filt = df_filt[df_filt[zona_col].astype(str) == zona_sel]
        if material_col and mat_sel and mat_sel not in ("Todos", "Todos"):
            df_filt = df_filt[df_filt[material_col].astype(str) == mat_sel]
        if search_txt:
            if lote_col:
                df_filt = df_filt[df_filt[lote_col].astype(str).str.contains(search_txt, case=False, na=False)]
            elif muestra_col:
                df_filt = df_filt[df_filt[muestra_col].astype(str).str.contains(search_txt, case=False, na=False)]
            else:
                mask = pd.Series(False, index=df_filt.index)
                for c in df_filt.select_dtypes(include=["object", "string"]).columns:
                    mask = mask | df_filt[c].astype(str).str.contains(search_txt, case=False, na=False)
                df_filt = df_filt[mask]
        if show_missing_ph:
            ph_col = find_col(["ph"])
            if ph_col:
                df_filt = df_filt[df_filt[ph_col].isna() | (df_filt[ph_col].astype(str).str.strip()=="")]
            else:
                st.info("No detecté columna 'pH' en los datos.")
    except Exception as e:
        st.error(f"Error aplicando filtros: {e}")

    sort_cols = []
    if ano_col: sort_cols.append(ano_col)
    if finca_col: sort_cols.append(finca_col)
    if lote_col: sort_cols.append(lote_col)
    if sort_cols:
        existing = [c for c in sort_cols if c in df_filt.columns]
        if existing:
            df_filt = df_filt.sort_values(existing).reset_index(drop=True)

    with st.expander("Ver tabla (primeras 500 filas)"):
        try:
            st.dataframe(df_filt.head(500), use_container_width=True)
        except Exception:
            st.error("No se pudo renderizar la tabla (posible problema de tipos/anchura).")

    try:
        csv = df_filt.to_csv(index=False).encode("utf-8")
        dl_name = f"suelos_filtrados_{str(year_sel)}.csv" if not show_all else "suelos_filtrados_todos_los_anos.csv"
        st.download_button("⬇️ Descargar CSV (filtrado)", csv, file_name=dl_name, mime="text/csv", key="dl_suelos_filtrados")
    except Exception:
        st.error("Error preparando descarga CSV.")

    return df_filt

# ════════════════════════════════════════════════════
# 4. SIDEBAR
# ════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        try:
            img = Image.open("logo_sidebar.png")
            st.image(img, width=260)
        except Exception:
            st.markdown("## 🌴 Dashboard Suelos")

        st.markdown("---")

        st.markdown("### 📂 Cargar datos")
        uploaded = st.file_uploader(
            "Sube tu archivo Excel o CSV",
            type=["xlsx", "xls", "csv"],
            help="El archivo debe contener columnas de finca, lote, latitud, longitud, profundidad y variables de suelo.",
        )

        st.markdown("---")
        st.markdown("### ℹ️ Columnas requeridas")
        st.markdown("""
        | Interno           | Variantes aceptadas |
        |-------------------|---------------------|
        | finca             | finca, farm, hacienda |
        | lote              | lote, parcela, plot, codigo_lote |
        | latitud           | lat, latitude, latitud |
        | longitud          | lon, longitude, longitud |
        | profundidad       | prof, profundidad, prof_cm, depth |
        | pH                | ph |
        | materia_organica  | mo, materia_organica, organic_matter |
        | CIC               | cic, capacidad_intercambio_cationes |
        | Ca                | ca, calcio |
        | Mg                | mg, magnesio |
        | K                 | k, potasio |
        | Na                | na, sodio |
        | P                 | p, fosforo |
        | Fe                | fe, hierro |
        | Mn                | mn, manganeso |
        | Zn                | zn, zinc |
        | textura           | arena, limo, arcilla, textura |
        | fecha / año       | fecha, date, ano, año |
        """)

    return uploaded

# ════════════════════════════════════════════════════
# 5. MAIN
# ════════════════════════════════════════════════════

def main():
    uploaded = render_sidebar()

    st.markdown(f"""
    <div class="main-header">
        <h1>🌴 Dashboard Suelos — Farmprecision</h1>
        <p>Mapa, densidad, correlaciones y análisis por variable</p>
    </div>
    """, unsafe_allow_html=True)

    if uploaded is None:
        st.markdown("""
        <div class="upload-zone">
            <h3>📂 Sube tu archivo Excel o CSV para comenzar</h3>
            <p>Usa el panel lateral para cargar el archivo de suelos.</p>
            <p><strong>Formatos soportados:</strong> .xlsx · .xls · .csv</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 📌 ¿Qué puedes analizar?")
        c1, c2, c3 = st.columns(3)
        c1.info("🗺️ Mapa y densidad espacial")
        c2.info("📊 Boxplots por lote/profundidad y Rankings")
        c3.info("🔗 Correlaciones y tabla filtrada")
        return

    # Leer bytes una sola vez y cachear por nombre+tamaño
    file_bytes = uploaded.read()
    file_name  = uploaded.name

    with st.spinner("⏳ Procesando datos..."):
        try:
            df, gdf, detected_colmap = cargar_suelos(file_bytes, file_name)
        except ValueError as e:
            st.error(f"❌ Error al cargar el archivo:\n\n{e}")
            return
        except Exception as e:
            st.error(f"❌ Error inesperado: {e}")
            return

    if df.empty:
        st.warning("⚠️ El archivo no contiene datos válidos después del filtrado.")
        return

    colmap = detected_colmap.copy()
    for k in ("lat","lon","prof","finca","lote"):
        if k not in colmap:
            if k == "lat": cand = _find_col_ci(list(df.columns), COMMON_LAT_COLS)
            elif k == "lon": cand = _find_col_ci(list(df.columns), COMMON_LON_COLS)
            elif k == "prof": cand = _find_col_ci(list(df.columns), COMMON_DEPTH_COLS)
            elif k == "finca": cand = _find_col_ci(list(df.columns), COMMON_FARMS)
            elif k == "lote": cand = _find_col_ci(list(df.columns), COMMON_LOTES)
            else: cand = None
            if cand:
                pretty = cand
                if cand.lower().replace("_","") in ("latitud","latitude","lat"): pretty = "Latitud"
                if cand.lower().replace("_","") in ("longitud","longitude","lon","lng","x"): pretty = "Longitud"
                if cand.lower().replace("_","") in ("prof","profundidad","depth"): pretty = "Prof"
                if cand.lower().replace("_","") in ("finca","farm","hacienda"): pretty = "Finca"
                if cand.lower().replace("_","") in ("lote","lot","parcela","plot"): pretty = "Lote"
                colmap[k] = pretty
    st.session_state["colmap_suelos"] = colmap

    lat_col = colmap.get("lat", _find_col_ci(list(df.columns), COMMON_LAT_COLS))
    lon_col = colmap.get("lon", _find_col_ci(list(df.columns), COMMON_LON_COLS))
    prof_col = colmap.get("prof", _find_col_ci(list(df.columns), COMMON_DEPTH_COLS))
    finca_col = colmap.get("finca", _find_col_ci(list(df.columns), COMMON_FARMS))
    lote_col = colmap.get("lote", _find_col_ci(list(df.columns), COMMON_LOTES))

    gdf2 = None
    if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
        try:
            gdf2 = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs="EPSG:4326")
        except Exception:
            gdf2 = None

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in [lat_col, lon_col]]
    st.session_state["numeric_cols"] = numeric_cols

    ordered_numeric = numeric_cols.copy()
    if "pH" in ordered_numeric:
        ordered_numeric.remove("pH"); ordered_numeric = ["pH"] + ordered_numeric

    selected_var = st.sidebar.selectbox("Variable principal", options=ordered_numeric, index=0 if ordered_numeric else None, key="selected_var_sidebar")
    map_mode = st.sidebar.selectbox("Modo mapa", options=["Puntos", "Densidad"], key="map_mode_sidebar")

    st.sidebar.markdown("Filtros globales")
    finca_vals = sorted(df[finca_col].dropna().astype(str).unique().tolist()) if finca_col and finca_col in df.columns else []
    filter_finca = st.sidebar.multiselect("Finca", options=["Todas"] + finca_vals, default=["Todas"]) if finca_vals else None
    lote_vals = sorted(df[lote_col].dropna().astype(str).unique().tolist()) if lote_col and lote_col in df.columns else []
    filter_lote = st.sidebar.multiselect("Lote", options=["Todos"] + lote_vals, default=["Todos"]) if lote_vals else None

    df_vis = df.copy()
    if filter_finca and "Todas" not in filter_finca:
        df_vis = df_vis[df_vis[finca_col].astype(str).isin(filter_finca)]
    if filter_lote and "Todos" not in filter_lote:
        df_vis = df_vis[df_vis[lote_col].astype(str).isin(filter_lote)]

    st.session_state["df_suelos"] = df
    st.session_state["gdf_suelos"] = gdf2
    st.session_state["available_soil_vars"] = [k for k in VARIABLES_SUELO_MAP.keys() if
                                               k in df.columns and pd.api.types.is_numeric_dtype(df[k])]
    st.sidebar.write(f"Filas (filtradas): {df_vis.shape[0]}")

    seccion_kpis(df_vis)
    st.markdown("---")

    possible_year_cols = [c for c in df_vis.columns if any(k in c.lower() for k in ("ano", "año", "year"))]
    if possible_year_cols:
        year_col = possible_year_cols[0]
        # normalizar a entero cuando sea posible
        years = pd.to_numeric(df_vis[year_col], errors="coerce").dropna().astype(int).unique().tolist()
        years = sorted([int(y) for y in years])
        years_desc = sorted(years, reverse=True)
        df_by_year = {int(y): df_vis[df_vis[year_col].astype(float).astype(int) == int(y)].copy() for y in years}
        # selector en sidebar para la tabla (por defecto el año más reciente)
        year_sel = st.sidebar.selectbox("Año para tabla de suelos", options=years_desc, index=0,
                                        key="suelos_table_year")
    else:
        year_col = None
        years_desc = []
        df_by_year = df_vis
        year_sel = None

    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Mapa y Distribución",
        "📊 Análisis por lote",
        "🔗 Correlaciones",
        "📋 Tabla & Descarga",
    ])

    with tab1:
        render_tab_map_overview(df_vis, lat_col, lon_col, selected_var, map_mode, lote_col, finca_col, prof_col)

    with tab2:
        render_tab_distribution(df_vis, selected_var, lote_col)

    with tab3:
        render_tab_correlations(df_vis)

    with tab4:
        seccion_tabla_suelos(df_vis, df_by_year, years_desc, year_sel)

if __name__ == "__main__":
    main()