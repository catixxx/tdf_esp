import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

# 🌸 CONFIGURACIÓN DE PÁGINA
st.set_page_config(
    page_title="TF-IDF Femenino Brillante",
    page_icon="💖",
    layout="wide"
)

# 💅 ESTILO PERSONALIZADO BRILLANTE
page_style = """
<style>
/* Fondo suave con brillo */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 10% 20%, #ffe6f9 0%, #fff9ff 40%, #f8e1ff 100%);
    font-family: 'Poppins', sans-serif;
}

/* Encabezado principal */
h1 {
    color: #c46cd6;
    text-align: center;
    font-weight: 800;
    font-size: 2.3em;
    text-shadow: 0 0 10px rgba(210, 160, 255, 0.6);
}

/* Subtítulos */
h2, h3, h4 {
    color: #a84ccf !important;
}

/* Botones */
div.stButton > button:first-child {
    background: linear-gradient(90deg, #ff9de2, #a18aff);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6em 1.2em;
    font-size: 16px;
    font-weight: 600;
    box-shadow: 0px 0px 10px rgba(255, 180, 255, 0.6);
    transition: all 0.3s ease-in-out;
}
div.stButton > button:first-child:hover {
    background: linear-gradient(90deg, #ffc4e9, #c6b0ff);
    transform: scale(1.05);
    box-shadow: 0px 0px 15px rgba(255, 200, 255, 0.8);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #fff0fa 0%, #f4e4ff 100%);
    border-right: 2px solid #e6b6ff;
}
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2 {
    color: #b050c9;
    text-align: center;
}

/* Textos, inputs y áreas */
textarea, input {
    background-color: #fffdfd !important;
    border: 2px solid #f3baff !important;
    border-radius: 10px !important;
    color: #7d4e8a !important;
}

/* DataFrame */
[data-testid="stDataFrame"] {
    background-color: rgba(255, 255, 255, 0.8);
    border-radius: 15px;
    padding: 10px;
}

/* Mensajes */
.stSuccess {
    background-color: #f6e3ff !important;
    color: #7a3692 !important;
    border-radius: 10px;
}
.stInfo {
    background-color: #e5e1ff !important;
    color: #4a3c8a !important;
    border-radius: 10px;
}
.stWarning {
    background-color: #ffe5f0 !important;
    color: #aa3b6a !important;
    border-radius: 10px;
}

/* Pie */
footer, .stCaptionContainer {
    text-align: center;
    color: #b26ccc;
    font-style: italic;
}
</style>
"""
st.markdown(page_style, unsafe_allow_html=True)

# 🌸 TÍTULO
st.title("💖 Analizador TF-IDF Brillante en Español")

# 🐾 Documentos de ejemplo
default_docs = """El perro ladra fuerte en el parque.
El gato maúlla suavemente durante la noche.
El perro y el gato juegan juntos en el jardín.
Los niños corren y se divierten en el parque.
La música suena muy alta en la fiesta.
Los pájaros cantan hermosas melodías al amanecer."""

# 🪞 Stemmer
stemmer = SnowballStemmer("spanish")

def tokenize_and_stem(text):
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    stems = [stemmer.stem(t) for t in tokens]
    return stems

# 🌈 Diseño en columnas
col1, col2 = st.columns([2, 1])

with col1:
    text_input = st.text_area("📝 Documentos (uno por línea):", default_docs, height=150)
    question = st.text_input("💭 Escribe tu pregunta:", "¿Dónde juegan el perro y el gato?")

with col2:
    st.markdown("### 🌷 Preguntas sugeridas")
    buttons = [
        "¿Dónde juegan el perro y el gato?",
        "¿Qué hacen los niños en el parque?",
        "¿Cuándo cantan los pájaros?",
        "¿Dónde suena la música alta?",
        "¿Qué animal maúlla durante la noche?"
    ]
    for q in buttons:
        if st.button(q, use_container_width=True):
            st.session_state.question = q
            st.rerun()

# Actualizar pregunta si se seleccionó
if 'question' in st.session_state:
    question = st.session_state.question

# 🌟 Botón principal de análisis
if st.button("✨ Analizar con magia TF-IDF ✨", type="primary", use_container_width=True):
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]
    
    if len(documents) < 1:
        st.error("⚠️ Ingresa al menos un documento.")
    elif not question.strip():
        st.error("⚠️ Escribe una pregunta válida.")
    else:
        vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, min_df=1)
        X = vectorizer.fit_transform(documents)
        
        # 📊 Mostrar matriz TF-IDF
        st.markdown("### 💎 Matriz TF-IDF")
        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))]
        )
        st.dataframe(df_tfidf.round(3), use_container_width=True)
        
        # 🔍 Calcular similitud
        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()
        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]
        
        # 💬 Mostrar resultados
        st.markdown("### 🌟 Resultado del análisis")
        st.markdown(f"**Tu pregunta:** {question}")
        
        if best_score > 0.01:
            st.success(f"💖 **Respuesta:** {best_doc}")
            st.info(f"📈 Similitud: {best_score:.3f}")
        else:
            st.warning(f"🌸 **Respuesta (baja confianza):** {best_doc}")
            st.info(f"📉 Similitud: {best_score:.3f}")

# 🌙 Pie
st.markdown("---")

