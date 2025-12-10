import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------
# Config general
# -----------------------------
st.set_page_config(
    page_title="IDELA/IDAT - Dashboard Gerencial",
    page_icon="📊",
    layout="wide"
)

st.title("📊 IDELA/IDAT Colombia | Visualizador Gerencial y Descriptivo")
st.caption(
    "Panel descriptivo por dimensiones, prop_items, enfoques de aprendizaje y calidad de datos."
)

# -----------------------------
# Utilidades
# -----------------------------
DIM_COLS = [
    "dim_socioemocional",
    "dim_matematica",
    "dim_lectura",
    "dim_motricidad",
]

INDEX_COLS = [
    "score_idela_total",
    "funciones_ejecutivas",
    "enfoques_aprendizaje",
    "persistencia",
]

ID_COLS = ["id", "nombre", "institucion", "grado", "profesor", "evaluador"]

def safe_cols(df, cols):
    return [c for c in cols if c in df.columns]

def get_prop_cols(df):
    return sorted([c for c in df.columns if re.match(r"^prop_item_\d+$", c)])

def get_obs_cols(df):
    return sorted([c for c in df.columns if re.match(r"^OBS_[a-z]$", c)])

def coerce_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def categorize_score(x):
    # Banda simple gerencial (ajustable)
    if pd.isna(x):
        return "Sin dato"
    if x < 0.33:
        return "Bajo"
    if x < 0.66:
        return "Medio"
    return "Alto"

def radar_chart(means_dict, title="Perfil por dimensiones"):
    labels = list(means_dict.keys())
    values = list(means_dict.values())
    # Cerrar el polígono
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            name="Promedio"
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        title=title,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

@st.cache_data
def load_data_from_upload(upload):
    return pd.read_excel(upload)

@st.cache_data
def load_data_from_path(path):
    return pd.read_excel(path)

# -----------------------------
# Sidebar - carga y filtros
# -----------------------------
st.sidebar.header("🔧 Fuente de datos")

upload = st.sidebar.file_uploader(
    "Carga tu base (xlsx)",
    type=["xlsx"]
)

if upload is not None:
    df = load_data_from_upload(upload)
    source_label = "Archivo cargado"
else:
    # Ruta por defecto
    default_path = "cal_persona.xlsx"
    try:
        df = load_data_from_path(default_path)
        source_label = f"Archivo local: {default_path}"
    except Exception:
        st.info("Sube un archivo .xlsx para comenzar.")
        st.stop()

st.sidebar.caption(source_label)

# -----------------------------
# Preparación de columnas clave
# -----------------------------
prop_cols = get_prop_cols(df)
obs_cols = get_obs_cols(df)

dim_cols = safe_cols(df, DIM_COLS)
index_cols = safe_cols(df, INDEX_COLS)
id_cols = safe_cols(df, ID_COLS)

numeric_cols = dim_cols + index_cols + prop_cols + obs_cols + ["porcentaje_vacios"]
numeric_cols = safe_cols(df, numeric_cols)
df = coerce_numeric(df, numeric_cols)

# -----------------------------
# Filtros
# -----------------------------
st.sidebar.header("🎛️ Filtros")

def multiselect_filter(label, col):
    if col in df.columns:
        options = sorted([x for x in df[col].dropna().unique()])
        return st.sidebar.multiselect(label, options, default=options)
    return None

inst_sel = multiselect_filter("Institución", "institucion")
grado_sel = multiselect_filter("Grado", "grado")
prof_sel = multiselect_filter("Profesor", "profesor")
eval_sel = multiselect_filter("Evaluador", "evaluador")

df_f = df.copy()

if inst_sel is not None:
    df_f = df_f[df_f["institucion"].isin(inst_sel)]
if grado_sel is not None:
    df_f = df_f[df_f["grado"].isin(grado_sel)]
if prof_sel is not None:
    df_f = df_f[df_f["profesor"].isin(prof_sel)]
if eval_sel is not None:
    df_f = df_f[df_f["evaluador"].isin(eval_sel)]

# -----------------------------
# Tabs principales
# -----------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📌 Panorama Ejecutivo",
        "🧭 Dimensiones a Profundidad",
        "🧩 Análisis por Ítems",
        "🌱 Enfoques & Observación",
        "🧪 Calidad de Datos",
    ]
)

# =========================================================
# TAB 1 - Panorama Ejecutivo
# =========================================================
with tab1:
    st.subheader("Panorama global del grupo filtrado")

    c1, c2, c3, c4 = st.columns(4)
    total_n = len(df_f)

    # KPIs principales
    with c1:
        st.metric("Niños evaluados", f"{total_n}")
    with c2:
        if "score_idela_total" in df_f.columns:
            st.metric("Promedio total", f"{df_f['score_idela_total'].mean():.3f}")
        else:
            st.metric("Promedio total", "N/D")
    with c3:
        if "porcentaje_vacios" in df_f.columns:
            st.metric("Vacíos promedio", f"{df_f['porcentaje_vacios'].mean():.1f}%")
        else:
            st.metric("Vacíos promedio", "N/D")
    with c4:
        # indicador de consistencia simple
        weird_props = []
        for c in prop_cols:
            mx = df_f[c].max(skipna=True)
            if pd.notna(mx) and mx > 1:
                weird_props.append(c)
        st.metric("Alertas prop_item", f"{len(weird_props)}")

    st.divider()

    # Cards por dimensión
    st.markdown("### Promedios por dimensiones e índices")

    cols_show = dim_cols + [c for c in index_cols if c != "score_idela_total"]
    if "score_idela_total" in df_f.columns:
        cols_show = ["score_idela_total"] + cols_show

    if cols_show:
        kcols = st.columns(min(6, len(cols_show)))
        for i, col in enumerate(cols_show):
            with kcols[i % len(kcols)]:
                st.metric(col, f"{df_f[col].mean():.3f}")
    else:
        st.info("No se detectaron columnas de dimensiones/índices en la base.")

    st.divider()

    # Distribución del total
    if "score_idela_total" in df_f.columns:
        st.markdown("### Distribución del puntaje total")
        fig = px.histogram(
            df_f,
            x="score_idela_total",
            nbins=20,
            title="Distribución de score_idela_total"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Comparativos por institución y grado
    st.markdown("### Comparativos gerenciales")

    group_opts = [c for c in ["institucion", "grado", "profesor", "evaluador"] if c in df_f.columns]
    group_by = st.selectbox("Comparar por", group_opts) if group_opts else None

    metric_opts = [c for c in ["score_idela_total"] + dim_cols + index_cols if c in df_f.columns]
    metric_sel = st.selectbox("Métrica", metric_opts) if metric_opts else None

    if group_by and metric_sel:
        tmp = (
            df_f.groupby(group_by)[metric_sel]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig = px.bar(
            tmp,
            x=group_by,
            y=metric_sel,
            title=f"Promedio de {metric_sel} por {group_by}"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(tmp, use_container_width=True)

# =========================================================
# TAB 2 - Dimensiones a Profundidad
# =========================================================
with tab2:
    st.subheader("Lectura profunda por dimensiones")

    if not dim_cols and not index_cols:
        st.info("No se detectaron dimensiones/índices.")
    else:
        dim_pool = dim_cols + index_cols
        dim_pool = [c for c in dim_pool if c in df_f.columns]

        dim_sel = st.selectbox("Selecciona dimensión/índice", dim_pool)

        c1, c2 = st.columns([1, 1])

        with c1:
            fig = px.box(
                df_f,
                y=dim_sel,
                x="institucion" if "institucion" in df_f.columns else None,
                title=f"Distribución de {dim_sel} (por institución)"
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig = px.histogram(
                df_f,
                x=dim_sel,
                nbins=20,
                title=f"Distribución global de {dim_sel}"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.markdown("### Perfil agregado (radar)")

        # Radar por agrupación
        radar_group_opts = [c for c in ["institucion", "grado"] if c in df_f.columns]
        radar_group = st.selectbox("Armar perfil por", radar_group_opts) if radar_group_opts else None

        if radar_group:
            groups = sorted(df_f[radar_group].dropna().unique())
            group_sel = st.selectbox("Selecciona grupo", groups)

            df_g = df_f[df_f[radar_group] == group_sel]

            radar_dims = [c for c in DIM_COLS if c in df_f.columns]
            if "funciones_ejecutivas" in df_f.columns:
                radar_dims += ["funciones_ejecutivas"]
            if "enfoques_aprendizaje" in df_f.columns:
                radar_dims += ["enfoques_aprendizaje"]

            means = {c: float(df_g[c].mean()) for c in radar_dims if c in df_g.columns}
            if means:
                fig = radar_chart(means, title=f"Perfil de {group_sel} ({radar_group})")
                st.plotly_chart(fig, use_container_width=True)

        st.divider()

        st.markdown("### Ranking de brechas")

        rank_group_opts = [c for c in ["institucion", "grado", "profesor"] if c in df_f.columns]
        rank_group = st.selectbox("Rankear por", rank_group_opts) if rank_group_opts else None

        if rank_group:
            tmp = (
                df_f.groupby(rank_group)[dim_sel]
                .agg(["mean", "count"])
                .reset_index()
                .sort_values("mean", ascending=True)
            )
            st.caption("Ordenado de menor a mayor promedio.")
            st.dataframe(tmp, use_container_width=True)

# =========================================================
# TAB 3 - Análisis por Ítems
# =========================================================
with tab3:
    st.subheader("Análisis por prop_item (logro por bloque)")

    if not prop_cols:
        st.info("No se detectaron columnas prop_item_XX.")
    else:
        item_sel = st.selectbox("Selecciona prop_item", prop_cols)

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            st.metric("Promedio", f"{df_f[item_sel].mean():.3f}")
        with c2:
            st.metric("Mediana", f"{df_f[item_sel].median():.3f}")
        with c3:
            st.metric("N con dato", f"{df_f[item_sel].notna().sum()}")

        fig = px.histogram(
            df_f,
            x=item_sel,
            nbins=20,
            title=f"Distribución de {item_sel}"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        # Heatmap de ítems por grupo
        heat_group_opts = [c for c in ["institucion", "grado"] if c in df_f.columns]
        heat_group = st.selectbox("Mapa de calor por", heat_group_opts) if heat_group_opts else None

        if heat_group:
            tmp = df_f.groupby(heat_group)[prop_cols].mean().reset_index()
            tmp_m = tmp.melt(id_vars=[heat_group], var_name="item", value_name="promedio")

            fig = px.imshow(
                tmp.set_index(heat_group)[prop_cols],
                aspect="auto",
                title=f"Mapa de calor de logro por {heat_group}"
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(tmp, use_container_width=True)

        st.divider()

        st.markdown("### Top fortalezas y cuellos de botella (global)")

        global_means = df_f[prop_cols].mean().sort_values()
        low5 = global_means.head(5).reset_index()
        low5.columns = ["prop_item", "promedio"]

        high5 = global_means.tail(5).sort_values(ascending=False).reset_index()
        high5.columns = ["prop_item", "promedio"]

        c1, c2 = st.columns(2)
        with c1:
            st.caption("5 ítems más bajos")
            st.dataframe(low5, use_container_width=True)
        with c2:
            st.caption("5 ítems más altos")
            st.dataframe(high5, use_container_width=True)

# =========================================================
# TAB 4 - Enfoques & Observación
# =========================================================
with tab4:
    st.subheader("Enfoques de aprendizaje, persistencia y observación general")

    left, right = st.columns([1, 1])

    with left:
        # Índices
        idx_show = [c for c in ["enfoques_aprendizaje", "persistencia", "funciones_ejecutivas"] if c in df_f.columns]
        if idx_show:
            st.markdown("### Índices agregados")
            for c in idx_show:
                st.metric(c, f"{df_f[c].mean():.3f}")

            fig = px.box(
                df_f,
                y=idx_show[0],
                x="institucion" if "institucion" in df_f.columns else None,
                title=f"Distribución de {idx_show[0]} por institución"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No se detectaron índices de enfoques/persistencia/funciones ejecutivas.")

    with right:
        # OBS
        if obs_cols:
            st.markdown("### Observación general (OBS)")
            obs_means = df_f[obs_cols].mean().reset_index()
            obs_means.columns = ["obs", "promedio"]

            fig = px.bar(
                obs_means,
                x="obs",
                y="promedio",
                title="Promedios de OBS"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(obs_means, use_container_width=True)
        else:
            st.info("No se detectaron columnas OBS_a...OBS_g.")

# =========================================================
# TAB 5 - Calidad de Datos
# =========================================================
with tab5:
    st.subheader("Control de calidad de aplicación y consistencia en cálculos")

    # Vacíos
    if "porcentaje_vacios" in df_f.columns:
        c1, c2 = st.columns([1, 2])

        with c1:
            st.metric("Vacíos promedio", f"{df_f['porcentaje_vacios'].mean():.1f}%")
            thresh = st.slider("Umbral de alerta (%)", 10, 80, 40)

        with c2:
            fig = px.histogram(
                df_f,
                x="porcentaje_vacios",
                nbins=20,
                title="Distribución de porcentaje_vacios"
            )
            st.plotly_chart(fig, use_container_width=True)

        alert_df = df_f[df_f["porcentaje_vacios"] >= thresh]
        st.markdown("### Registros que superan el umbral")
        st.dataframe(alert_df[id_cols + ["porcentaje_vacios"]] if id_cols else alert_df, use_container_width=True)
    else:
        st.info("No se encontró la columna porcentaje_vacios.")

    st.divider()

    # Alertas prop_item > 1
    st.markdown("### Alertas de prop_items fuera de rango (esperado 0-1)")
    weird = []
    for c in prop_cols:
        mx = df_f[c].max(skipna=True)
        if pd.notna(mx) and mx > 1:
            weird.append((c, mx))

    if weird:
        weird_df = pd.DataFrame(weird, columns=["prop_item", "max_observado"]).sort_values("max_observado", ascending=False)
        st.warning("Se detectaron prop_items con valores mayores a 1. Esto suele indicar un problema de denominador o cálculo.")
        st.dataframe(weird_df, use_container_width=True)
    else:
        st.success("No se detectaron prop_items fuera de rango.")

    st.divider()

    # Vista rápida de base
    st.markdown("### Vista general de la base filtrada")
    st.dataframe(df_f, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.caption(
    "Sugerencia: interpreta primero tendencias por dimensiones y luego baja a prop_items para explicar el 'por qué' pedagógico."
)