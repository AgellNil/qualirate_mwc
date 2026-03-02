import os
import streamlit as st
import pandas as pd
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
from datetime import datetime, timedelta
import numpy as np

# ==========================
# CONFIGURACIÓN INICIAL
# ==========================

st.set_page_config(
    page_title="QualiRate - Anàlisi de Ressenyes",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Personalizado para mejor diseño
st.markdown("""
    <style>
    /* Colores principales */
    :root {
        --primary-color: #FF6B6B;
        --success-color: #51CF66;
        --warning-color: #FFD93D;
        --danger-color: #FF6B6B;
        --info-color: #4ECDC4;
        --dark-bg: #1a1a1a;
        --light-bg: #f8f9fa;
    }
    
    /* Fuentes mejoradas */
    * {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header personalizado */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    /* Cards mejoradas */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 4px solid #FF6B6B;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    .metric-card.positive {
        border-left-color: #51CF66;
    }
    
    .metric-card.warning {
        border-left-color: #FFD93D;
    }
    
    /* Sección de análisis */
    .analysis-section {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    
    /* Expanders mejorados */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 12px;
        border-left: 3px solid #FF6B6B;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #e9ecef;
    }
    
    /* Botones personalizados */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 107, 107, 0.3);
    }
    
    /* Sidebar */
    .sidebar-section {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    
    /* Dividers */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #ddd, transparent);
        margin: 2rem 0;
    }
    
    /* Info boxes */
    .info-box {
        background: #e7f5ff;
        border-left: 4px solid #4ECDC4;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    
    /* Stats display */
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FF6B6B;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #666;
        font-weight: 500;
    }
    
    /* Intervalo Puntuació - título minimalista */
    .interval-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 0.5rem;
        text-align: left;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
        }
        
        .metric-card {
            padding: 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Configuración Perplexity API
PPLX_API_KEY = st.secrets.get("PPLX_API_KEY")

pplx_client = OpenAI(
    api_key=PPLX_API_KEY,
    base_url="https://api.perplexity.ai"
)
PPLX_MODEL = "sonar-pro"

# Paraulas a filtrar
PALABRAS_FILTRAR = {"de", "del", "bon", "buena", "bona", "bien", "a", "buen", "bueno", "el", "dar", "ir", "la", "un","mal","malo","dir","decir","bé","be"}

# ==========================
# FUNCIONES AUXILIARES
# ==========================
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep='\t')
    df.columns = [c.strip() for c in df.columns]
    
    required_cols = ["new_score", "text", "processed_text_original", "consensus"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Falta la columna obligatòria: {col}")
    
    df["new_score"] = pd.to_numeric(df["new_score"], errors="coerce")
    df = df.dropna(subset=["new_score"])
    
    if "new_socre_righ" in df.columns:
        df["new_score_right"] = pd.to_numeric(df["new_socre_righ"], errors="coerce")
    elif "new_score_right" in df.columns:
        df["new_score_right"] = pd.to_numeric(df["new_score_right"], errors="coerce")
    
    if "new_score_left" in df.columns:
        df["new_score_left"] = pd.to_numeric(df["new_score_left"], errors="coerce")
    
    if "publishedAtDate" in df.columns:
        df["publishedAtDate"] = pd.to_datetime(df["publishedAtDate"], errors="coerce")
    
    return df

def get_top_words(texts, n_words=15, filter_words=None):
    if filter_words is None:
        filter_words = set()
    all_words = []
    for text in texts:
        if pd.notna(text) and str(text).strip():
            words = [w for w in str(text).split() if w not in filter_words]
            all_words.extend(words)
    word_freq = Counter(all_words)
    return word_freq.most_common(n_words)

def get_words_with_min_ressenyes(df, min_ressenyes=20, filter_words=None):
    if filter_words is None:
        filter_words = set()
    word_ressenyes = {}
    for idx, text in enumerate(df["nouns_verbs_ca"]):
        if pd.notna(text) and str(text).strip():
            words_in_review = set([w for w in str(text).split() if w not in filter_words])
            for word in words_in_review:
                if word not in word_ressenyes:
                    word_ressenyes[word] = 0
                word_ressenyes[word] += 1
    
    filtered_words = sorted([word for word, count in word_ressenyes.items() if count >= min_ressenyes])
    return filtered_words, word_ressenyes

def filter_ressenyes_by_word(df, word):
    mask = df["processed_text_ca"].apply(
        lambda x: word in str(x).split() if pd.notna(x) else False
    )
    return df[mask]

def get_perplexity_summary(ressenyes_df, word, sentiment_type="positius"):
    if len(ressenyes_df) == 0:
        return f"No hay ressenyes {sentiment_type} que contengan la palabra '{word}'."
    
    start_date = st.session_state.get("start_date", "")
    end_date = st.session_state.get("end_date", "")
    review_ids = str(sorted(ressenyes_df.index.tolist()[:10]))
    cache_key = f"pplx_summary_{word}_{sentiment_type}_{start_date}_{end_date}_{hash(review_ids)}"
    
    if "perplexity_cache" not in st.session_state:
        st.session_state.perplexity_cache = {}
    
    if cache_key in st.session_state.perplexity_cache:
        return st.session_state.perplexity_cache[cache_key]
    
    sample = ressenyes_df.head(50)
    context_ressenyes = ""
    for idx, row in sample.iterrows():
        txt = str(row["text"]) if pd.notna(row["text"]) and str(row["text"]).strip() else "[Sense comentari]"
        context_ressenyes += f"- {txt}\n"
    
    
    system_prompt = (
    f"Ets un analista d'experiència de client. "
    f"Analitza **NOMÉS** aquestes ressenyes {sentiment_type} d'un restaurant que mencionen '{word}'. "
    f"Crea un resum molt breu i directe (màxim 2-3 línies) sobre què opinen els clients "
    f"relacionat específicament amb '{word}'.\n\n"
    "REGLES IMPORTANTS:\n"
    "- Basa't EXCLUSIVAMENT en les ressenyes que et proporciono aquí baix.\n"
    "- NO busquis informació a Internet.\n"
    "- NO facis referència a productes, empreses o serveis externs.\n"
    "- NO incloguis referències numèriques com [1], [2], [web:1], etc.\n"
    "- NO mencionis 'segons les ressenyes', 'ressenyes proporcionades', etc.\n"
    "- Escriu de manera directa i natural.\n"
    "- Respon EXCLUSIVAMENT en català.\n"
    "- Ves directe al punt, sense introduccions.\n\n"
    f"Ressenyes del restaurant a analitzar:\n{context_ressenyes}"
)
    
    user_question = f"Resumeix en 2-3 línies què opinen els clients sobre '{word}'"
    
    try:
        response = pplx_client.chat.completions.create(
            model=PPLX_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_question},
            ],
            temperature=0.3,
            max_tokens=200,
        )
        
        summary = response.choices[0].message.content.strip()
        st.session_state.perplexity_cache[cache_key] = summary
        return summary
    except Exception as e:
        error_msg = f"Error en generar resum: {str(e)}"
        st.session_state.perplexity_cache[cache_key] = error_msg
        return error_msg

def apply_date_filter(df, start_date, end_date):
    if "publishedAtDate" not in df.columns:
        return df
    mask = (df["publishedAtDate"].dt.date >= start_date) & (df["publishedAtDate"].dt.date <= end_date)
    return df[mask]


def create_interval_gauge(left_value, right_value):
    fig = go.Figure()
    
    def get_color(value):
        if value <= 3.0: return f'rgb(255,{int((value/3.0)*255)},0)'
        else: return f'rgb({int(255*(1-((value-3.0)/2.0)))},255,0)'
    
    max_val = 5
    num_segments = 100
    seg_w = max_val / num_segments
    
    # Dibujar la barra de color sólido
    for i in range(num_segments):
        mid = i * seg_w + seg_w/2
        fig.add_trace(go.Bar(
            x=[seg_w], y=[1], orientation='h', base=i*seg_w,
            marker=dict(color=get_color(mid), line=dict(width=0)),
            opacity=1.0, 
            showlegend=False, hoverinfo='skip'
        ))
    
    # RECTÁNGULO NEGRO con yref='paper'
    fig.add_shape(
        type="rect",
        x0=left_value, x1=right_value,
        y0=0.6, y1=1,  # 0 y 1 en coordenadas paper = altura completa del área
        xref='x', yref='paper',  # ← CAMBIO CLAVE: paper en vez de y
        line=dict(color="black", width=4),
        fillcolor="rgba(0,0,0,0)",
        layer="above"
    )
    
    fig.update_layout(
        barmode='stack', 
        height=100, 
        margin=dict(l=10, r=10, t=10, b=30),
        xaxis=dict(
            range=[0, max_val], 
            tickvals=[0, 1, 2, 3, 4, 5], 
            showgrid=False,
            fixedrange=True
        ),
        yaxis=dict(visible=False, range=[0, 1], fixedrange=True), 
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig


# ==========================
# CARGAR DATOS
# ==========================
df_original = load_data("gmaps_reviews_with_newrating_consensus_totsompops.csv")

# Inicializar estado
if "start_date" not in st.session_state:
    if "publishedAtDate" in df_original.columns:
        st.session_state.start_date = df_original["publishedAtDate"].min().date()
        st.session_state.end_date = df_original["publishedAtDate"].max().date()
    else:
        st.session_state.start_date = datetime.now().date() - timedelta(days=90)
        st.session_state.end_date = datetime.now().date()

# ==========================
# HEADER PRINCIPAL
# ==========================
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.image("logo_totsompops.png", width=500)
with col_title:
    st.markdown('<div class="main-header">QualiRate - Totsompops</div>', unsafe_allow_html=True)
st.markdown("Què pensen els teus clients?", unsafe_allow_html=False)
st.markdown("---")

# ==========================
# FILTROS GENERALES EN LA BARRA LATERAL
# ==========================
with st.sidebar:
    st.markdown("### 🎯 Filtres Generals")    
    if "publishedAtDate" in df_original.columns:
        min_date = df_original["publishedAtDate"].min().date()
        max_date = df_original["publishedAtDate"].max().date()
        
        st.markdown("#### 📅 Rang de Dates")
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            start_date = st.date_input(
                "Des de",
                value=st.session_state.start_date,
                min_value=min_date,
                max_value=max_date,
                key="date_from"
            )
            st.session_state.start_date = start_date
        
        with col_date2:
            end_date = st.date_input(
                "Fins a",
                value=st.session_state.end_date,
                min_value=min_date,
                max_value=max_date,
                key="date_to"
            )
            st.session_state.end_date = end_date
        
        if st.button("🔄 Reiniciar dates", use_container_width=True):
            st.session_state.start_date = min_date
            st.session_state.end_date = max_date
            st.rerun()
        
        if start_date > end_date:
            st.error("⚠️ La data d'inici no pot ser posterior a la data de fi.")
            st.stop()
        
        # Aplicar filtro de fechas
        df = apply_date_filter(df_original, start_date, end_date)
        
        if len(df) == 0:
            st.error("⚠️ No hi ha ressenyes en el rang seleccionat.")
            st.stop()
    else:
        st.warning("No hi ha columna de dates disponible.")
        df = df_original
    
    st.markdown("---")
    
    # Información del filtro
    total_original = len(df_original)
    total_filtrado = len(df)
    porcentaje = (total_filtrado / total_original * 100) if total_original > 0 else 0
    
    st.markdown("### 📊 Dades Cargades")
    st.metric("Ressenyes en període", total_filtrado)
    st.metric("Ressenyes totals", total_original)
    st.metric("Cobertura", f"{porcentaje:.1f}%")
    
    st.markdown("---")
    st.markdown("### ⚙️ Eines")
    
    if st.button("🗑️ Esborrar memòria cau IA", use_container_width=True):
        if "perplexity_cache" in st.session_state:
            st.session_state.perplexity_cache = {}
        if "top_positive_words" in st.session_state:
            st.session_state.top_positive_words = None
        if "top_negative_words" in st.session_state:
            st.session_state.top_negative_words = None
        st.sidebar.success("✅ Memòria cau esborrada")
        st.rerun()

# ==========================
# NAVEGACIÓN
# ==========================
page = st.selectbox(
    "📍 Selecciona una pàgina",
    ["📊 Tauler Principal", "🔍 Anàlisi per Paraules"],
    label_visibility="collapsed"
)

st.markdown("---")

# ==========================
# PÁGINA 1: DASHBOARD PRINCIPAL
# ==========================
if page == "📊 Tauler Principal":
    
    # MÉTRICAS PRINCIPALES
    st.markdown("## 📈 Mètriques Clau")
    
    avg_score = df["new_score"].mean()+0.5
    consensus = df["consensus"].iloc[0] if len(df) > 0 else 0
    total_ressenyes = len(df)
    positivos = len(df[df["new_score"] > 3])
    negativos = len(df[df["new_score"] <= 3])
    
    # Primera fila - Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="stat-label">⭐ Puntuació QualiRate</div>
            <div class="stat-value">{:.2f}</div>
            <div class="stat-label" style="font-size: 0.8rem;">de 5.0</div>
        </div>
        """.format(avg_score), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="stat-label">📝 Total Ressenyes</div>
            <div class="stat-value">{}</div>
            <div class="stat-label" style="font-size: 0.8rem;">en el període</div>
        </div>
        """.format(total_ressenyes), unsafe_allow_html=True)
    
    with col3:
        pct_pos = (positivos / total_ressenyes * 100) if total_ressenyes > 0 else 0
        st.markdown("""
        <div class="metric-card positive">
            <div class="stat-label">👍 Positius (>3)</div>
            <div class="stat-value">{}</div>
            <div class="stat-label" style="font-size: 0.8rem;">{:.1f}%</div>
        </div>
        """.format(positivos, pct_pos), unsafe_allow_html=True)
    
    with col4:
        pct_neg = (negativos / total_ressenyes * 100) if total_ressenyes > 0 else 0
        st.markdown("""
        <div class="metric-card warning">
            <div class="stat-label">👎 Negatius (≤3)</div>
            <div class="stat-value">{}</div>
            <div class="stat-label" style="font-size: 0.8rem;">{:.1f}%</div>
        </div>
        """.format(negativos, pct_neg), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ============================================
    # INTERVALO SCORE - FORMATO MINIMALISTA
    # ============================================
    has_intervals = "new_score_left" in df.columns and "new_score_right" in df.columns
    
    if has_intervals:
        interval_left = df["new_score_left"].mean()+0.5
        interval_right = df["new_score_right"].mean()+0.5
        if interval_right>=5:
            interval_right=5
            
        
        # Título simple
        st.markdown('<div class="interval-title">📊 Interval Puntuació QualiRate</div>', unsafe_allow_html=True)
        
        # Solo la barra con escala numérica
        fig_interval = create_interval_gauge(interval_left, interval_right)
        st.plotly_chart(fig_interval, use_container_width=True, key="interval_gauge_main")
    
    st.markdown("---")
    
    # ============================================
    # Gráfico temporal
    # ============================================
    st.markdown("## 📅 Evolució Temporal")
    
    if "publishedAtDate" in df.columns:
        temporal_df = df.dropna(subset=["publishedAtDate"]).sort_values("publishedAtDate")
        if len(temporal_df) > 0:
            fig_temporal = px.scatter(
                temporal_df,
                x="publishedAtDate",
                y="new_score",
                title="Evolució temporal de ressenyes",
                labels={"publishedAtDate": "Data", "new_score": "Puntuació"},
                color="new_score",
                color_continuous_scale="RdYlGn",
                opacity=0.7
            )
            
            fig_temporal.add_trace(
                go.Scatter(
                    x=temporal_df["publishedAtDate"],
                    y=temporal_df["new_score"].rolling(window=10, center=True).mean(),
                    mode="lines",
                    name="Tendència (10)",
                    line=dict(color="blue", width=3, dash="dash")
                )
            )
            
            fig_temporal.update_layout(
                height=400,
                yaxis_range=[0, 5.5],
                hovermode="x unified",
                showlegend=True
            )
            st.plotly_chart(fig_temporal, use_container_width=True)
    
    st.markdown("---")
    
    # TOP PALABRAS CON IA
    st.markdown("## 💬 Top 5 Paraules Més Freqüents")
    
    if not PPLX_API_KEY or PPLX_API_KEY == "TU_API_KEY_AQUI":
        st.warning("⚠️ Configura la teva PPLX_API_KEY per activar anàlisi amb IA")
    else:
        positive_ressenyes = df[df["new_score"] > 3]["processed_text_ca"]
        negative_ressenyes = df[df["new_score"] <= 3]["processed_text_ca"]
        
        cache_key_dates = f"top_words_{st.session_state.start_date}_{st.session_state.end_date}"
        if "top_words_cache_key" not in st.session_state or st.session_state.top_words_cache_key != cache_key_dates:
            st.session_state.top_words_cache_key = cache_key_dates
            st.session_state.top_positive_words = None
            st.session_state.top_negative_words = None
        
        if st.session_state.top_positive_words is None:
            st.session_state.top_positive_words = get_top_words(positive_ressenyes, n_words=5, filter_words=PALABRAS_FILTRAR)
        if st.session_state.top_negative_words is None:
            st.session_state.top_negative_words = get_top_words(negative_ressenyes, n_words=5, filter_words=PALABRAS_FILTRAR)
        
        col_pos, col_neg = st.columns(2)
        
        with col_pos:
            st.markdown("### 👍 Reviews Positius")
            if len(positive_ressenyes) > 0:
                top_positive = st.session_state.top_positive_words
                positive_df = df[df["new_score"] > 3]
                
                for idx, (word, freq) in enumerate(top_positive, 1):
                    with st.expander(f"**{idx}. {word}** ({freq} mencions)"):
                        word_ressenyes = filter_ressenyes_by_word(positive_df, word)
                        summary = get_perplexity_summary(word_ressenyes, word, "positius")
                        st.success(summary)
            else:
                st.info("No hi ha ressenyes positives")
        
        with col_neg:
            st.markdown("### 👎 Reviews Negatius")
            if len(negative_ressenyes) > 0:
                top_negative = st.session_state.top_negative_words
                negative_df = df[df["new_score"] <= 3]
                
                for idx, (word, freq) in enumerate(top_negative, 1):
                    with st.expander(f"**{idx}. {word}** ({freq} mencions)"):
                        word_ressenyes = filter_ressenyes_by_word(negative_df, word)
                        summary = get_perplexity_summary(word_ressenyes, word, "negatius")
                        st.warning(summary)
            else:
                st.info("No hi ha ressenyes negatives")
    
    st.markdown("---")
    
        # Q&A con IA
    st.markdown("## 🤖 Assistent de QualiRate")
    
    if not PPLX_API_KEY or PPLX_API_KEY == "TU_API_KEY_AQUI":
        st.warning("⚠️ Configura la teva PPLX_API_KEY per activar aquesta secció")
    else:
        scope = st.radio(
            "Sobre quin tipus de ressenyes?",
            ["Totes", "Només positives (>3)", "Només negatives (≤3)"],
            horizontal=True
        )
        
        if scope == "Totes":
            df_scope = df.copy()
        elif scope == "Només positives (>3)":
            df_scope = df[df["new_score"] > 3].copy()
        else:
            df_scope = df[df["new_score"] <= 3].copy()

        # 🔹 Ordenar por fecha e incluir fecha en el contexto
        max_ressenyes_context = 2000
        if "publishedAtDate" in df_scope.columns:
            df_scope = df_scope.dropna(subset=["publishedAtDate"]).sort_values("publishedAtDate")
            sample = df_scope.tail(max_ressenyes_context)
        else:
            sample = df_scope.head(max_ressenyes_context)

        context_text = ""
        for _, row in sample.iterrows():
            score = row["new_score"]
            txt = str(row["text"]) if pd.notna(row["text"]) else ""
            if "publishedAtDate" in row and pd.notna(row["publishedAtDate"]):
                date_str = row["publishedAtDate"].strftime("%Y-%m-%d")
                context_text += f"- [Data: {date_str}] [Puntuació: {score:.2f}] {txt}\n"
            else:
                context_text += f"- [Puntuació: {score:.2f}] {txt}\n"
        
        user_question = st.text_area(
            "La teva pregunta",
            height=100,
            placeholder="Ex: Quins són els principals problemes?"
        )
        
        if st.button("❓ Preguntar a l'assistent"):
            if user_question.strip():
                with st.spinner("Consultant IA..."):
                    try:
                        system_prompt = (
                            "Eres un analista de experiencia de cliente. "
                            "Te paso un listado de ressenyes de un restaurante con su FECHA y su score. "
                            "Cada línea tiene el formato: [Data: AAAA-MM-DD] [Puntuació: X.XX] texto del review. "
                            "Responde basándote EXCLUSIVAMENTE en esos ressenyes, "
                            "Sé claro, conciso y en español.\n\n"
                            f"Ressenyes:\n{context_text}"
                        )
                        
                        response = pplx_client.chat.completions.create(
                            model=PPLX_MODEL,
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_question},
                            ],
                            temperature=0.3,
                            max_tokens=800,
                        )
                        
                        answer = response.choices[0].message.content
                        st.info(answer)
                    except Exception as e:
                        st.error(f"Error: {e}")


# ==========================
# PÁGINA 2: ANÁLISIS POR PALABRAS
# ==========================
elif page == "🔍 Anàlisi per Paraules":
    
    st.markdown("## 🔍 🔍 Anàlisi per Paraules Clau")
    st.info("Selecciona fins a 4 paraules per veure evolució temporal i sentiment")
    
    available_words, word_review_count = get_words_with_min_ressenyes(df, min_ressenyes=20, filter_words=PALABRAS_FILTRAR)
    
    if len(available_words) == 0:
        st.warning("No hay palabras con ≥20 ressenyes en el període seleccionado")
        st.stop()
    
    st.success(f"✅ {len(available_words)} paraules disponibles")
    
    if "selected_words" not in st.session_state:
        st.session_state.selected_words = []
    
    col_add, col_info = st.columns([1, 4])
    with col_add:
        if len(st.session_state.selected_words) < 4:
            if st.button("➕ Afegir paraula"):
                st.session_state.selected_words.append(None)
                st.rerun()
    with col_info:
        if len(st.session_state.selected_words) > 0:
            st.caption(f"Paraules afegides: {len(st.session_state.selected_words)}/4")
    
    st.markdown("---")
    
    if len(st.session_state.selected_words) > 0:
        st.markdown("### Selecciona paraules:")
        cols = st.columns(min(len(st.session_state.selected_words), 2))
        
        for i in range(len(st.session_state.selected_words)):
            with cols[i % 2]:
                already_selected = [w for w in st.session_state.selected_words if w is not None]
                available_for_this = ["Seleccionar..."] + [w for w in available_words if w not in already_selected or w == st.session_state.selected_words[i]]
                default_index = 0
                if st.session_state.selected_words[i] in available_for_this:
                    default_index = available_for_this.index(st.session_state.selected_words[i])
                
                word = st.selectbox(
                    f"Paraula {i+1}",
                    options=available_for_this,
                    index=default_index,
                    key=f"word_select_{i}"
                )
                
                col_remove, col_empty = st.columns([1, 3])
                with col_remove:
                    if st.button("🗑️", key=f"remove_{i}"):
                        st.session_state.selected_words.pop(i)
                        st.rerun()
                
                if word != "Seleccionar...":
                    st.session_state.selected_words[i] = word
                else:
                    st.session_state.selected_words[i] = None
    
    st.markdown("---")
    
    valid_words = [w for w in st.session_state.selected_words if w is not None]
    
    if len(valid_words) == 0:
        st.info("Selecciona almenys una paraula per veure l'anàlisi")
    else:
        for i, word in enumerate(valid_words, 1):
            num_ressenyes = word_review_count.get(word, 0)
            
            with st.expander(f"📊 {i}. **{word}** — {num_ressenyes} ressenyes", expanded=True):
                filtered_df = filter_ressenyes_by_word(df, word)
                
                if len(filtered_df) == 0:
                    st.warning(f"No hay ressenyes con '{word}'")
                    continue
                
                # Métricas
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                avg_word = filtered_df["new_score"].mean()
                pos_word = len(filtered_df[filtered_df["new_score"] > 3])
                neg_word = len(filtered_df[filtered_df["new_score"] <= 3])
                
                with col_m1:
                    st.metric("Total", len(filtered_df))
                with col_m2:
                    st.metric("Mitjana", f"{avg_word:.2f}")
                with col_m3:
                    st.metric("Positius", pos_word)
                with col_m4:
                    st.metric("Negatius", neg_word)
                
                st.markdown("---")
                
                # Intervalo de confianza - MISMO FORMATO MINIMALISTA
                has_intervals = "new_score_left" in filtered_df.columns and "new_score_right" in filtered_df.columns
                if has_intervals:
                    interval_left = filtered_df["new_score_left"].mean()
                    interval_right = filtered_df["new_score_right"].mean()
                    st.markdown('<div class="interval-title">📊 Intervalo Puntuació QualiRate</div>', unsafe_allow_html=True)
                    fig_interval = create_interval_gauge(interval_left, interval_right)
                    st.plotly_chart(fig_interval, use_container_width=True, key=f"interval_{i}")
                    st.markdown("---")
                
                # Análisis IA
                if PPLX_API_KEY and PPLX_API_KEY != "TU_API_KEY_AQUI":
                    st.markdown("#### 🤖 Análisis")
                    col_pos_ai, col_neg_ai = st.columns(2)
                    
                    with col_pos_ai:
                        st.markdown("**Opinió Positiva**")
                        positive_word_df = filtered_df[filtered_df["new_score"] > 3]
                        if len(positive_word_df) > 0:
                            summary_pos = get_perplexity_summary(positive_word_df, word, "positius")
                            st.success(summary_pos)
                        else:
                            st.info("Sin ressenyes positivos")
                    
                    with col_neg_ai:
                        st.markdown("**Opinió Negativa**")
                        negative_word_df = filtered_df[filtered_df["new_score"] <= 3]
                        if len(negative_word_df) > 0:
                            summary_neg = get_perplexity_summary(negative_word_df, word, "negatius")
                            st.warning(summary_neg)
                        else:
                            st.info("Sin ressenyes negativos")
                    
                    st.markdown("---")
                
                # Temporal
                st.markdown("#### 📅 Evolución")
                if "publishedAtDate" in filtered_df.columns:
                    temporal_df = filtered_df.dropna(subset=["publishedAtDate"]).sort_values("publishedAtDate")
                    if len(temporal_df) > 0:
                        fig = px.scatter(
                            temporal_df,
                            x="publishedAtDate",
                            y="new_score",
                            color="new_score",
                            color_continuous_scale="RdYlGn",
                            hover_data=["text"],
                            title=f"Evolució: {word}"
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=temporal_df["publishedAtDate"],
                                y=temporal_df["new_score"].rolling(window=5, center=True).mean(),
                                mode="lines",
                                name="Tendència",
                                line=dict(color="blue", width=2, dash="dash")
                            )
                        )
                        
                        fig.update_layout(height=400, yaxis_range=[0, 5.5], hovermode="x unified")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Sense dades de data")
                else:
                    st.warning("Sense columna de dates")
