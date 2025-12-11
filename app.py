import re
import numpy as np
import pandas as pd
import streamlit as st

# =============================
# Plotly opcional (no obligatorio)
# =============================
PLOTLY_OK = True
try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    PLOTLY_OK = False


# =============================
# Config general
# =============================
st.set_page_config(
    page_title="IDELA/IDAT - Dashboard Gerencial.",
    page_icon="📊",
    layout="wide"
)

st.title("📊 IDELA/IDAT Colombia | Panel Gerencial Descriptivo")
st.caption(
    "Lectura ejecutiva con profundidad técnica: dimensiones, índices, prop_items, observación y calidad de datos."
)

# Sidebar: estado de librerías
st.sidebar.markdown("### 🧰 Estado de librerías")
st.sidebar.write("Plotly:", "✅ Activo" if PLOTLY_OK else "❌ No instalado (modo estable sin Plotly)")


# =============================
# Constantes esperadas por tu base
# =============================
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


# =============================
# Utilidades de columnas
# =============================
def safe_cols(df, cols):
    return [c for c in cols if c in df.columns]

def get_prop_cols(df):
    # prop_item_1, prop_item_2 ...
    cols = [c for c in df.columns if re.match(r"^prop_item_\d+$", str(c))]
    # orden numérico
    def key_fn(x):
        m = re.search(r"(\d+)$", x)
        return int(m.group(1)) if m else 10**9
    return sorted(cols, key=key_fn)

def get_obs_cols(df):
    # OBS_a ... OBS_g
    return sorted([c for c in df.columns if re.match(r"^OBS_[a-z]$", str(c))])

def coerce_numeric(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# =============================
# Gráficos robustos (Plotly o fallback)
# =============================
def show_hist(df, col, title, nbins=20):
    series = df[col].dropna()
    if series.empty:
        st.info(f"Sin datos para {col}.")
        return

    if PLOTLY_OK:
        fig = px.histogram(df, x=col, nbins=nbins, title=title)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # fallback simple: histograma manual
        counts, edges = np.histogram(series, bins=nbins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        chart_df = pd.DataFrame({col: centers, "conteo": counts}).set_index(col)
        st.subheader(title)
        st.bar_chart(chart_df["conteo"])

def show_bar(df, x, y, title):
    if df.empty:
        st.info("No hay datos para graficar.")
        return

    if PLOTLY_OK:
        fig = px.bar(df, x=x, y=y, title=title)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # fallback: usa dataframe indexado
        st.subheader(title)
        tmp = df[[x, y]].dropna()
        if tmp.empty:
            st.info("No hay datos suficientes para el gráfico.")
            return
        tmp = tmp.set_index(x)
        st.bar_chart(tmp[y])

def show_box(df, x, y, title):
    series = df[y].dropna() if y in df.columns else pd.Series(dtype=float)
    if series.empty:
        st.info(f"Sin datos para {y}.")
        return

    if PLOTLY_OK:
        fig = px.box(df, x=x if x else None, y=y, title=title)
        st.plotly_chart(fig, use_container_width=True)
    else:
        # fallback: resumen estadístico por grupo
        st.subheader(title)
        if x and x in df.columns:
            tmp = df.groupby(x)[y].describe()[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
            st.dataframe(tmp, use_container_width=True)
        else:
            st.dataframe(series.describe(), use_container_width=True)

def radar_chart_plotly(means_dict, title="Perfil por dimensiones"):
    labels = list(means_dict.keys())
    values = list(means_dict.values())
    if not labels:
        return None

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
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title=title,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


# =============================
# Carga de datos
# =============================
@st.cache_data
def load_data_from_upload(upload):
    return pd.read_excel(upload)

@st.cache_data
def load_data_from_path(path):
    return pd.read_excel(path)


st.sidebar.header("🔧 Fuente de datos")

upload = st.sidebar.file_uploader("Carga tu base (xlsx)", type=["xlsx"])

if upload is not None:
    df = load_data_from_upload(upload)
    st.sidebar.caption("Usando archivo cargado.")
else:
    default_path = "cal_persona.xlsx"
    try:
        df = load_data_from_path(default_path)
        st.sidebar.caption(f"Usando archivo local: {default_path}")
    except Exception:
        st.info("Sube un archivo .xlsx para comenzar.")
        st.stop()


# =============================
# Preparación de columnas
# =============================
prop_cols = get_prop_cols(df)
obs_cols = get_obs_cols(df)
dim_cols = safe_cols(df, DIM_COLS)
index_cols = safe_cols(df, INDEX_COLS)
id_cols = safe_cols(df, ID_COLS)

numeric_cols = dim_cols + index_cols + prop_cols + obs_cols + ["porcentaje_vacios"]
numeric_cols = safe_cols(df, numeric_cols)
df = coerce_numeric(df, numeric_cols)


# =============================
# Filtros
# =============================
st.sidebar.header("🎛️ Filtros")

def multiselect_filter(label, col):
    if col in df.columns:
        options = sorted([x for x in df[col].dropna().unique()])
        if not options:
            return None
        return st.sidebar.multiselect(label, options, default=options)
    return None

inst_sel = multiselect_filter("Institución", "institucion")
grado_sel = multiselect_filter("Grado", "grado")
prof_sel = multiselect_filter("Profesor", "profesor")
eval_sel = multiselect_filter("Evaluador", "evaluador")

df_f = df.copy()
if inst_sel is not None and "institucion" in df_f.columns:
    df_f = df_f[df_f["institucion"].isin(inst_sel)]
if grado_sel is not None and "grado" in df_f.columns:
    df_f = df_f[df_f["grado"].isin(grado_sel)]
if prof_sel is not None and "profesor" in df_f.columns:
    df_f = df_f[df_f["profesor"].isin(prof_sel)]
if eval_sel is not None and "evaluador" in df_f.columns:
    df_f = df_f[df_f["evaluador"].isin(eval_sel)]


# =============================
# Funciones gerenciales
# =============================
def compute_group_means(df_in, group_col, metrics):
    tmp = (
        df_in.groupby(group_col)[metrics]
        .mean()
        .reset_index()
    )
    return tmp

def top_bottom_items_global(df_in, prop_cols, n=5):
    if not prop_cols:
        return None, None
    means = df_in[prop_cols].mean().sort_values()
    low = means.head(n).reset_index()
    low.columns = ["prop_item", "promedio"]
    high = means.tail(n).sort_values(ascending=False).reset_index()
    high.columns = ["prop_item", "promedio"]
    return low, high

def detect_weird_prop_items(df_in, prop_cols):
    weird = []
    for c in prop_cols:
        mx = df_in[c].max(skipna=True)
        mn = df_in[c].min(skipna=True)
        if pd.notna(mx) and mx > 1:
            weird.append((c, "max>1", float(mx)))
        if pd.notna(mn) and mn < 0:
            weird.append((c, "min<0", float(mn)))
    if weird:
        return pd.DataFrame(weird, columns=["prop_item", "alerta", "valor_observado"])
    return pd.DataFrame(columns=["prop_item", "alerta", "valor_observado"])

def exec_summary_text(df_in):
    lines = []
    n = len(df_in)
    lines.append(f"**Población analizada:** {n} estudiantes.")

    # Total
    if "score_idela_total" in df_in.columns:
        mean_total = df_in["score_idela_total"].mean()
        lines.append(f"**Promedio global (score total):** {mean_total:.3f}.")
    else:
        lines.append("**Promedio global (score total):** No disponible en la base.")

    # Dimensiones
    available_dims = [c for c in DIM_COLS if c in df_in.columns]
    if available_dims:
        dim_means = df_in[available_dims].mean().sort_values()
        weakest = dim_means.index[0]
        strongest = dim_means.index[-1]
        lines.append(
            f"**Dimensión más rezagada en el grupo filtrado:** {weakest} "
            f"({dim_means[weakest]:.3f})."
        )
        lines.append(
            f"**Dimensión más fuerte en el grupo filtrado:** {strongest} "
            f"({dim_means[strongest]:.3f})."
        )

    # Índices clave
    for idx in ["funciones_ejecutivas", "enfoques_aprendizaje", "persistencia"]:
        if idx in df_in.columns:
            lines.append(f"**{idx}:** {df_in[idx].mean():.3f}.")

    # Calidad
    if "porcentaje_vacios" in df_in.columns:
        pv = df_in["porcentaje_vacios"].mean()
        lines.append(f"**Calidad de registro (vacíos promedio):** {pv:.1f}%.")
        if pv >= 40:
            lines.append("⚠️ Riesgo operativo: el nivel de vacíos sugiere revisar consistencia de aplicación y captura.")
        elif pv >= 25:
            lines.append("🟡 Atención: hay señales moderadas de incompletitud.")
        else:
            lines.append("✅ Registro globalmente consistente.")

    # Ítems extremos
    low, high = top_bottom_items_global(df_in, prop_cols, n=3)
    if low is not None and not low.empty:
        lines.append("**Cuellos de botella (3 prop_items más bajos):** " +
                     ", ".join([f"{r.prop_item} ({r.promedio:.2f})" for r in low.itertuples()]) + ".")
    if high is not None and not high.empty:
        lines.append("**Fortalezas destacadas (3 prop_items más altos):** " +
                     ", ".join([f"{r.prop_item} ({r.promedio:.2f})" for r in high.itertuples()]) + ".")

    # Alertas prop items
    weird_df = detect_weird_prop_items(df_in, prop_cols)
    if not weird_df.empty:
        lines.append("⚠️ Se detectaron prop_items fuera del rango esperado 0-1. Revisar fórmulas de cálculo.")

    return lines


# =============================
# Tabs del dashboard
# =============================
tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🧾 Resumen Ejecutivo",
        "📌 Panorama",
        "🧭 Dimensiones",
        "🧩 Ítems",
        "🌱 Enfoques & OBS",
        "🧪 Calidad",
    ]
)

# =========================================================
# TAB 0 - Resumen Ejecutivo
# =========================================================
with tab0:
    st.subheader("Resumen ejecutivo automático del grupo filtrado")

    if len(df_f) == 0:
        st.info("No hay registros con los filtros actuales.")
    else:
        bullets = exec_summary_text(df_f)
        for b in bullets:
            st.markdown(f"- {b}")

        st.divider()
        st.markdown("### Lectura gerencial por institución (si aplica)")

        if "institucion" in df_f.columns and "score_idela_total" in df_f.columns:
            metrics_for_inst = ["score_idela_total"] + [c for c in DIM_COLS if c in df_f.columns]
            inst_tbl = compute_group_means(df_f, "institucion", metrics_for_inst)
            inst_tbl = inst_tbl.sort_values("score_idela_total", ascending=True)

            st.caption("Ordenado de menor a mayor desempeño total.")
            st.dataframe(inst_tbl, use_container_width=True)

            show_bar(
                inst_tbl,
                "institucion",
                "score_idela_total",
                "Promedio total por institución"
            )
        else:
            st.info("No se puede construir comparación institucional (falta 'institucion' o 'score_idela_total').")

        st.divider()
        st.markdown("### Prioridades sugeridas")

        # Prioridades simples basadas en la dimensión más baja
        available_dims = [c for c in DIM_COLS if c in df_f.columns]
        if available_dims:
            dim_means = df_f[available_dims].mean().sort_values()
            priorities = dim_means.head(2)
            st.markdown(
                "Las **dos dimensiones con mayor prioridad de intervención** en el grupo seleccionado son:\n"
                + "\n".join([f"- **{k}** (promedio {v:.3f})" for k, v in priorities.items()])
            )
        else:
            st.info("No hay dimensiones disponibles para sugerir prioridades.")


# =========================================================
# TAB 1 - Panorama
# =========================================================
with tab1:
    st.subheader("Panorama global del grupo filtrado")

    c1, c2, c3, c4 = st.columns(4)
    total_n = len(df_f)

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
        weird_df = detect_weird_prop_items(df_f, prop_cols)
        st.metric("Alertas prop_item", f"{len(weird_df)}")

    st.divider()

    st.markdown("### Promedios por dimensiones e índices")

    cols_show = []
    if "score_idela_total" in df_f.columns:
        cols_show.append("score_idela_total")
    cols_show += [c for c in dim_cols if c in df_f.columns]
    cols_show += [c for c in index_cols if c in df_f.columns and c != "score_idela_total"]

    if cols_show:
        kcols = st.columns(min(6, len(cols_show)))
        for i, col in enumerate(cols_show):
            with kcols[i % len(kcols)]:
                st.metric(col, f"{df_f[col].mean():.3f}")
    else:
        st.info("No se detectaron columnas de dimensiones/índices en la base.")

    st.divider()

    if "score_idela_total" in df_f.columns:
        st.markdown("### Distribución del puntaje total")
        show_hist(df_f, "score_idela_total", "Distribución de score_idela_total")

    st.divider()
    st.markdown("### Comparativos gerenciales")

    group_opts = [c for c in ["institucion", "grado", "profesor", "evaluador"] if c in df_f.columns]
    metric_opts = [c for c in ["score_idela_total"] + dim_cols + index_cols if c in df_f.columns]

    if group_opts and metric_opts:
        group_by = st.selectbox("Comparar por", group_opts)
        metric_sel = st.selectbox("Métrica", metric_opts)

        tmp = (
            df_f.groupby(group_by)[metric_sel]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
        )
        show_bar(tmp, group_by, metric_sel, f"Promedio de {metric_sel} por {group_by}")
        st.dataframe(tmp, use_container_width=True)
    else:
        st.info("No hay campos suficientes para comparativos.")


# =========================================================
# TAB 2 - Dimensiones
# =========================================================
with tab2:
    st.subheader("Lectura profunda por dimensiones e índices")

    dim_pool = [c for c in (dim_cols + index_cols) if c in df_f.columns]

    if not dim_pool:
        st.info("No se detectaron dimensiones/índices.")
    else:
        dim_sel = st.selectbox("Selecciona dimensión/índice", dim_pool)

        c1, c2 = st.columns([1, 1])

        with c1:
            xgroup = "institucion" if "institucion" in df_f.columns else None
            show_box(df_f, xgroup, dim_sel, f"Distribución de {dim_sel}" + (" por institución" if xgroup else ""))

        with c2:
            show_hist(df_f, dim_sel, f"Distribución global de {dim_sel}")

        st.divider()
        st.markdown("### Perfil por dimensiones (radar si hay Plotly)")

        radar_group_opts = [c for c in ["institucion", "grado"] if c in df_f.columns]
        if radar_group_opts:
            radar_group = st.selectbox("Armar perfil por", radar_group_opts, key="radar_group")
            groups = sorted(df_f[radar_group].dropna().unique())
            if groups:
                group_sel = st.selectbox("Selecciona grupo", groups, key="radar_group_sel")
                df_g = df_f[df_f[radar_group] == group_sel]

                radar_dims = [c for c in DIM_COLS if c in df_f.columns]
                if "funciones_ejecutivas" in df_f.columns:
                    radar_dims += ["funciones_ejecutivas"]
                if "enfoques_aprendizaje" in df_f.columns:
                    radar_dims += ["enfoques_aprendizaje"]

                means = {c: float(df_g[c].mean()) for c in radar_dims if c in df_g.columns}

                if PLOTLY_OK and means:
                    fig = radar_chart_plotly(means, title=f"Perfil de {group_sel} ({radar_group})")
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    # fallback: tabla de perfil
                    if means:
                        prof_df = pd.DataFrame({"dimension": list(means.keys()), "promedio": list(means.values())}) \
                            .sort_values("promedio")
                        st.dataframe(prof_df, use_container_width=True)
                        show_bar(prof_df, "dimension", "promedio", "Perfil promedio (fallback)")
        else:
            st.info("No hay campos de grupo para construir perfiles.")


# =========================================================
# TAB 3 - Ítems
# =========================================================
with tab3:
    st.subheader("Análisis por prop_item (logro por bloque)")

    if not prop_cols:
        st.info("No se detectaron columnas prop_item_XX.")
    else:
        item_sel = st.selectbox("Selecciona prop_item", prop_cols)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Promedio", f"{df_f[item_sel].mean():.3f}")
        with c2:
            st.metric("Mediana", f"{df_f[item_sel].median():.3f}")
        with c3:
            st.metric("N con dato", f"{df_f[item_sel].notna().sum()}")

        show_hist(df_f, item_sel, f"Distribución de {item_sel}")

        st.divider()
        st.markdown("### Fortalezas y cuellos de botella (global)")

        low5, high5 = top_bottom_items_global(df_f, prop_cols, n=5)
        c1, c2 = st.columns(2)

        with c1:
            st.caption("5 ítems más bajos")
            st.dataframe(low5 if low5 is not None else pd.DataFrame(), use_container_width=True)

        with c2:
            st.caption("5 ítems más altos")
            st.dataframe(high5 if high5 is not None else pd.DataFrame(), use_container_width=True)

        st.divider()
        st.markdown("### Comparativo por grupo")

        heat_group_opts = [c for c in ["institucion", "grado"] if c in df_f.columns]
        if heat_group_opts:
            heat_group = st.selectbox("Comparar prop_items por", heat_group_opts)
            tmp = df_f.groupby(heat_group)[prop_cols].mean().reset_index()

            # fallback a tabla siempre disponible
            st.dataframe(tmp, use_container_width=True)

            # gráfico simple de un item seleccionado por grupo
            tmp_one = tmp[[heat_group, item_sel]].sort_values(item_sel, ascending=False)
            show_bar(tmp_one, heat_group, item_sel, f"{item_sel} promedio por {heat_group}")
        else:
            st.info("No hay campos de grupo disponibles para el comparativo.")


# =========================================================
# TAB 4 - Enfoques & OBS
# =========================================================
with tab4:
    st.subheader("Enfoques de aprendizaje, persistencia y observación general")

    left, right = st.columns([1, 1])

    with left:
        idx_show = [c for c in ["enfoques_aprendizaje", "persistencia", "funciones_ejecutivas"] if c in df_f.columns]
        if idx_show:
            st.markdown("### Índices agregados")
            for c in idx_show:
                st.metric(c, f"{df_f[c].mean():.3f}")

            # distribución del principal
            main_idx = idx_show[0]
            show_hist(df_f, main_idx, f"Distribución de {main_idx}")
        else:
            st.info("No se detectaron índices de enfoques/persistencia/funciones ejecutivas.")

    with right:
        if obs_cols:
            st.markdown("### Observación general (OBS)")

            obs_means = df_f[obs_cols].mean().reset_index()
            obs_means.columns = ["obs", "promedio"]

            show_bar(obs_means, "obs", "promedio", "Promedios de OBS")
            st.dataframe(obs_means, use_container_width=True)
        else:
            st.info("No se detectaron columnas OBS_a...OBS_g.")


# =========================================================
# TAB 5 - Calidad
# =========================================================
with tab5:
    st.subheader("Control de calidad de aplicación y consistencia")

    # Vacíos
    if "porcentaje_vacios" in df_f.columns:
        c1, c2 = st.columns([1, 2])

        with c1:
            st.metric("Vacíos promedio", f"{df_f['porcentaje_vacios'].mean():.1f}%")
            thresh = st.slider("Umbral de alerta (%)", 10, 80, 40)

        with c2:
            show_hist(df_f, "porcentaje_vacios", "Distribución de porcentaje_vacios")

        alert_df = df_f[df_f["porcentaje_vacios"] >= thresh]
        st.markdown("### Registros por encima del umbral")

        if id_cols:
            cols = id_cols + ["porcentaje_vacios"]
            cols = [c for c in cols if c in df_f.columns]
            st.dataframe(alert_df[cols], use_container_width=True)
        else:
            st.dataframe(alert_df, use_container_width=True)
    else:
        st.info("No se encontró la columna porcentaje_vacios.")

    st.divider()

    # Alertas de rango en prop_items
    st.markdown("### Alertas de prop_items fuera de rango esperado (0-1)")
    weird_df = detect_weird_prop_items(df_f, prop_cols)

    if weird_df.empty:
        st.success("No se detectaron prop_items fuera de rango.")
    else:
        st.warning("Se detectaron prop_items con valores atípicos. Revisar fórmulas/denominadores.")
        st.dataframe(weird_df.sort_values("valor_observado", ascending=False), use_container_width=True)

    st.divider()
    st.markdown("### Vista de base filtrada")
    st.dataframe(df_f, use_container_width=True)


# =============================
# Pie gerencial
# =============================
st.caption(
    "Sugerencia de uso en comité: 1) Resumen Ejecutivo, 2) Dimensiones, 3) Ítems para explicar causas, 4) Calidad para decisiones operativas."
)
