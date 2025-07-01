
from pathlib import Path


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

områdevalg = st.sidebar.selectbox(
    "Velg reguleringsplan eller område:",
    ("Plan A – Sentrum", "Plan B – Tomasjord", "Plan C – Workinnmarka")
)

uploaded_file = st.sidebar.file_uploader("📄 Last opp ny reguleringsplan (PDF)", type="pdf")

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

def setup_bot(pdf_file_path):
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vectordb = FAISS.from_documents(chunks, OpenAIEmbeddings())
    return RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-3.5-turbo"), retriever=vectordb.as_retriever())

qa = setup_bot(pdf_path)

def last_inn_tekst(pdf_fil):
    loader = PyPDFLoader(pdf_fil)
    docs = loader.load()
    tekst = "\\n\\n".join([doc.page_content for doc in docs])
    return tekst

koordinater = områdeinfo[områdevalg]["koordinater"]
m = folium.Map(location=koordinater, zoom_start=13)

folium.Marker(
    location=koordinater,
    popup=f"<b>{områdevalg}</b><br>Still spørsmål i chatten!",
    tooltip="Klikk for mer info"
).add_to(m)

folium.TileLayer("OpenStreetMap", name="Standard").add_to(m)
folium.TileLayer("Stamen Terrain", name="Topografisk", attr="Stamen").add_to(m)
folium.TileLayer("Stamen Toner", name="Reguleringsplan", attr="Stamen").add_to(m)
folium.TileLayer(
    tiles="http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    attr="Google",
    name="Satellitt"
).add_to(m)
folium.LayerControl().add_to(m)

col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("🌍 Kart over Tromsø")
    st_folium(m, width=700, height=500)
    st.write(f"📍 Du ser nå på: **{områdevalg}**")

with col2:
    st.subheader("🤖 Spør AI om reguleringsplanen")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

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

    user_input = st.text_input("Skriv inn spørsmål:", key="input_q")

    if user_input:
        with st.spinner("Tenker..."):
            response = qa.invoke(user_input)
            st.session_state.chat_history.append((user_input, response))

    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(f"**Du:** {q}")
        with st.chat_message("assistant"):
            st.markdown(f"**Svar:** {a}")

    st.markdown("---")
    st.subheader("📊 Analyse: Er planen i tråd med kommunens mål?")

    if st.button("Analyser mot kommuneplanen"):
        with st.spinner("Laster og analyserer dokumenter..."):
            regtekst = last_inn_tekst(pdf_path)
            kpatekst = last_inn_tekst("Planer/kpa.pdf")
            samftekst = last_inn_tekst("Planer/kommuneplanens_samfunnsdel_2020.pdf")

            full_prompt = f"""Du er arealplanlegger og journalist. Du har fått tilgang til følgende reguleringsplan:

--- REGULERINGSPLAN ---
{regtekst}

--- KOMMUNEPLANENS AREALDEL (KPA) ---
{kpatekst}

--- KOMMUNEPLANENS SAMFUNNSDEL ---
{samftekst}

Basert på dette, vurder:
1. Er reguleringsplanen i tråd med bærekraftsmålene i samfunnsplanen?
2. Følger den føringene i KPA, særlig med tanke på grøntområder, høyder og fortetting?
3. Hvilke avvik finnes, og hvordan kan disse vinkles journalistisk?

Svar tydelig og konkret.
\"""

            llm = ChatOpenAI(model="gpt-3.5-turbo")
            vurdering = llm.invoke(full_prompt)

        st.success("Analyse fullført")
        st.markdown(f"**AI-vurdering:**\\n\\n{vurdering}")

