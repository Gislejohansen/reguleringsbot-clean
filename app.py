import streamlit as st
from langchain_openai import ChatOpenAI
import folium
from streamlit_folium import st_folium
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import os
import tempfile

# 🔐 OpenAI API-nøkkel
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Reguleringsbot", layout="wide")
st.title("🏗️ Reguleringsbot – Chat og kart over Tromsø")

# 📍 Velg område / plan
områdevalg = st.sidebar.selectbox(
    "Velg reguleringsplan eller område:",
    ("Plan A – Sentrum", "Plan B – Tomasjord", "Plan C – Workinnmarka")
)

# 📂 Last opp ny PDF (valgfritt)
uploaded_file = st.sidebar.file_uploader("📄 Last opp ny reguleringsplan (PDF)", type="pdf")

# 🎯 Sett PDF-sti og koordinater basert på valg
områdeinfo = {
    "Plan A – Sentrum": {
        "pdf": "plan_sentrum.pdf",
        "koordinater": [69.6496, 18.9560]
    },
    "Plan B – Tomasjord": {
        "pdf": "plan_tomasjord.pdf",
        "koordinater": [69.6800, 19.0300]
    },
    "Plan C – Workinnmarka": {
        "pdf": "plan_workinnmarka.pdf",
        "koordinater": [69.6500, 18.9000]
    }
}

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name
else:
    pdf_path = områdeinfo[områdevalg]["pdf"]

# 🧠 Lag RAG-modell for chat
@st.cache_resource
def setup_bot(pdf_file_path):
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vectordb = FAISS.from_documents(chunks, OpenAIEmbeddings())
    return RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-3.5-turbo"), retriever=vectordb.as_retriever())

qa = setup_bot(pdf_path)

# 🧠 Lag RAG-modell for analyse (kombinerer reguleringsplan + kommuneplan)
@st.cache_resource
def setup_sammenligning(pdf_file_path):
    # Last inn reguleringsplan
    loader = PyPDFLoader(pdf_file_path)
    reg_docs = loader.load()

    # Last inn kommuneplaner
    kp_docs = []
    for fil in ["Planer/kommuneplanens_samfunnsdel_2020.pdf", "Planer/kpa.pdf"]:
        kp_docs.extend(PyPDFLoader(fil).load())

    # Slå sammen
    alle_docs = reg_docs + kp_docs

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(alle_docs)
    vectordb = FAISS.from_documents(chunks, OpenAIEmbeddings())
    return RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-3.5-turbo"), retriever=vectordb.as_retriever())

# 📍 KART
koordinater = områdeinfo[områdevalg]["koordinater"]
m = folium.Map(location=koordinater, zoom_start=13)

folium.Marker(
    location=koordinater,
    popup=f"<b>{områdevalg}</b><br>Still spørsmål i chatten!",
    tooltip="Klikk for mer info"
).add_to(m)

# ➕ Kartlag
folium.TileLayer("OpenStreetMap", name="Standard").add_to(m)
folium.TileLayer("Stamen Terrain", name="Topografisk", attr="Stamen").add_to(m)
folium.TileLayer("Stamen Toner", name="Reguleringsplan", attr="Stamen").add_to(m)
folium.TileLayer(
    tiles="http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    attr="Google",
    name="Satellitt"
).add_to(m)
folium.LayerControl().add_to(m)

# 📐 Layout: venstre = kart, høyre = chatbot
col1, col2 = st.columns([1.5, 1])

# 🗺️ VENSTRE: Kart
with col1:
    st.subheader("🌍 Kart over Tromsø")
    st_folium(m, width=700, height=500)
    st.write(f"📍 Du ser nå på: **{områdevalg}**")

# 💬 HØYRE: Chatbot med historikk og forslag
with col2:
    st.subheader("🤖 Spør AI om reguleringsplanen")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # 💡 Spørsmålsforslag
    forslag = [
        "Hva sier planen om byggehøyder?",
        "Hva er formålet med reguleringen?",
        "Finnes det krav til uteområder?",
        "Hva er regulert til næring?",
        "Hvordan påvirker planen nabolaget?"
    ]
    st.markdown("**Eksempler på spørsmål:**")
    for spm in forslag:
        if st.button(spm):
            st.session_state.input_q = spm

    # 🔤 Inntastingsfelt
    user_input = st.text_input("Skriv inn spørsmål:", key="input_q")

    if user_input:
        with st.spinner("Tenker..."):
            response = qa.invoke(user_input)
            st.session_state.chat_history.append((user_input, response))

    # 🧾 Vis som chatbobler
    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(f"**Du:** {q}")
        with st.chat_message("assistant"):
            st.markdown(f"**Svar:** {a}")

    st.markdown("---")
    st.subheader("📊 Analyse: Er planen i tråd med kommunens mål?")

    if st.button("Analyser mot kommuneplanen"):
        with st.spinner("Sammenligner med kommuneplanens mål..."):
            analyse_prompt = """
Du har tilgang til både reguleringsplanen og Tromsø kommunes overordnede mål (kommuneplan og KPA).
Vurder i hvilken grad denne reguleringsplanen er i tråd med:
- bærekraftig utvikling
- arealstrategi
- krav til grøntområder
- byggehøyder og fortetting
- andre relevante føringer
Svar tydelig og konkret.
"""
            analyse_chain = setup_sammenligning(pdf_path)
            vurdering = analyse_chain.run(analyse_prompt)
            st.success("Analyse fullført")
            st.markdown(f"**AI-vurdering:**\n\n{vurdering}")

# 📥 Eksport som tekstfil
if st.session_state.chat_history:
    full_chat = "\n\n".join([f"Spørsmål: {q}\nSvar: {a}" for q, a in st.session_state.chat_history])
    st.download_button("📄 Last ned samtalen", full_chat, file_name="chat_samtale.txt")

# 💡 Foreslå analyseidé
st.sidebar.markdown("---")
st.sidebar.header("💡 Foreslå analyseidé")

kategori = st.sidebar.selectbox("Velg datasett eller tema:", [
    "Matrikkeldata",
    "Brønnøysundregisteret",
    "Skattedata",
    "Befolkningsdata",
    "Grunnboken",
    "Andre"
])

analyseforslag = st.sidebar.text_area(
    "Beskriv hva du ønsker at vi skal analysere eller undersøke:",
    height=150,
    placeholder="Eks: Kan vi koble eiendomsskatt med tomtestørrelser for å avsløre skjevheter?"
)

if st.sidebar.button("Send inn forslag"):
    if analyseforslag.strip():
        if "innsendte_forslag" not in st.session_state:
            st.session_state.innsendte_forslag = []
        st.session_state.innsendte_forslag.append((kategori, analyseforslag))
        st.sidebar.success("✅ Forslaget er registrert – takk!")
    else:
        st.sidebar.warning("Skriv inn et forslag før du sender.")

with st.expander("📝 Se innsendte forslag (midlertidig lagret)"):
    if "innsendte_forslag" in st.session_state:
        for idx, (kat, txt) in enumerate(st.session_state.innsendte_forslag, 1):
            st.markdown(f"**{idx}. {kat}**\n\n{txt}")
    else:
        st.info("Ingen forslag enda.")
