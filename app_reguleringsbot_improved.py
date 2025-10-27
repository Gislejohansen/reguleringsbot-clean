import os
import tempfile
import base64
import re
from typing import Dict, List, Optional, Tuple

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

import folium
from streamlit_folium import st_folium

import pandas as pd
import geopandas as gpd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from pdf2image import convert_from_path

# ------------------------------------------------------------
# Page setup (must be called once)
# ------------------------------------------------------------
st.set_page_config(page_title="Samleportal for data", layout="wide")

# ------------------------------------------------------------
# Configuration and helpers
# ------------------------------------------------------------
DEFAULT_MODEL_NAME = st.secrets.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL_NAME = st.secrets.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Poppler path handling (Windows-specific path should not break on Linux)
POPPLER_PATH = (
    st.secrets.get("POPPLER_PATH")
    or os.getenv("POPPLER_PATH")
    or None  # None means let pdf2image try system defaults; we handle exceptions below
)

# ------------------------------------------------------------
# Caching: data loading
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_bedrifter() -> pd.DataFrame:
    try:
        return pd.read_csv("bedrifter_tromso.csv")
    except Exception as exc:
        st.warning(f"Kunne ikke laste 'bedrifter_tromso.csv': {exc}")
        return pd.DataFrame(columns=["NAVN", "LAT", "LON"])  # minimal schema


@st.cache_data(show_spinner=False)
def load_adresser() -> pd.DataFrame:
    try:
        df = pd.read_csv("matrikkel_adresse_latlon.csv", sep=";", on_bad_lines="skip")
        if len(df.columns) == 1:
            df = pd.read_csv("matrikkel_adresse_latlon.csv", sep=",", on_bad_lines="skip")
        return df
    except Exception as exc:
        st.warning(f"Kunne ikke laste 'matrikkel_adresse_latlon.csv': {exc}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_teig_gdf() -> gpd.GeoDataFrame:
    try:
        return gpd.read_file("teig.gml")
    except Exception as exc:
        st.info("Fant ikke 'teig.gml' eller kunne ikke lese filen. Planpolygoner vises ikke.")
        return gpd.GeoDataFrame()


# ------------------------------------------------------------
# Caching: LLM and embeddings
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_llm() -> Optional[ChatOpenAI]:
    if not OPENAI_API_KEY:
        st.warning("Ingen OpenAI API-nøkkel funnet i 'st.secrets' eller miljøvariabler. AI-funksjoner er deaktivert.")
        return None
    try:
        return ChatOpenAI(model=DEFAULT_MODEL_NAME)
    except Exception as exc:
        st.warning(f"Klarte ikke å initialisere LLM: {exc}")
        return None


@st.cache_resource(show_spinner=False)
def get_embeddings() -> Optional[OpenAIEmbeddings]:
    if not OPENAI_API_KEY:
        return None
    try:
        return OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)
    except Exception as exc:
        st.warning(f"Klarte ikke å initialisere embeddings: {exc}")
        return None


# ------------------------------------------------------------
# PDF helpers and QA chain (with caching per pdf_path)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_pdf_docs(pdf_path: str) -> List:
    loader = PyPDFLoader(pdf_path)
    return loader.load()


@st.cache_data(show_spinner=False)
def hent_avsnitt(pdf_path: str, maks_tegn: int = 3000) -> str:
    docs = load_pdf_docs(pdf_path)
    tekst = "\n\n".join([d.page_content for d in docs])
    return tekst[:maks_tegn]


def get_qa_for_pdf(pdf_path: str) -> Optional[RetrievalQA]:
    if "qa_cache" not in st.session_state:
        st.session_state.qa_cache = {}

    cache_key = pdf_path
    if cache_key in st.session_state.qa_cache:
        return st.session_state.qa_cache[cache_key]

    llm = get_llm()
    embeddings = get_embeddings()
    if llm is None or embeddings is None:
        return None

    # Build vectorstore
    try:
        docs = load_pdf_docs(pdf_path)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        vectordb = FAISS.from_documents(chunks, embeddings)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=True,
        )
        st.session_state.qa_cache[cache_key] = qa
        return qa
    except Exception as exc:
        st.warning(f"Klarte ikke å sette opp QA-kjede: {exc}")
        return None


# ------------------------------------------------------------
# Domain helpers
# ------------------------------------------------------------
def extract_coordinates_from_pdf(pdf_path: str, docs: Optional[List] = None) -> Optional[List[float]]:
    if docs is None:
        docs = load_pdf_docs(pdf_path)
    text = "\n".join([doc.page_content for doc in docs])

    match = re.search(r"(\d{2}\.\d+)[,\s]+(\d{1,2}\.\d+)", text)
    if match:
        return [float(match.group(1)), float(match.group(2))]

    # Try geocoding based on address terms
    address_match = re.search(r"(Storgata|gate|vegen|gata|Tomasjord|Workinnmarka)[^\n,]*", text, re.IGNORECASE)
    if address_match:
        geolocator = Nominatim(user_agent="reguleringsbot")
        try:
            location = geolocator.geocode("Tromsø, " + address_match.group(0), timeout=6)
            if location:
                return [location.latitude, location.longitude]
        except (GeocoderTimedOut, GeocoderServiceError):
            st.info("Geokoding feilet eller timet ut. Du kan angi koordinater manuelt i sidepanelet.")
    return None


def extract_gnr_bnr_from_pdf(pdf_path: str, docs: Optional[List] = None) -> Tuple[Optional[str], Optional[str]]:
    if docs is None:
        docs = load_pdf_docs(pdf_path)
    text = "\n".join([doc.page_content for doc in docs])
    match = re.search(r"G[\s\-]?N[\s\-]?R[\s:]*([0-9]+)[,\s]+B[\s\-]?N[\s\-]?R[\s:]*([0-9]+)", text, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)
    return None, None


def gnr_bnr_to_coords_csv(adresser_df: pd.DataFrame, gnr: str, bnr: str) -> Optional[List[float]]:
    if adresser_df.empty:
        return None
    try:
        match = adresser_df[
            (adresser_df["gardsnummer"] == int(gnr)) & (adresser_df["bruksnummer"] == int(bnr))
        ]
        if not match.empty:
            return [match.iloc[0]["latitude"], match.iloc[0]["longitude"]]
    except Exception:
        return None
    return None


# ------------------------------------------------------------
# UI: Sidebar navigation
# ------------------------------------------------------------
sidevalg = st.sidebar.radio("Velg side:", ["Hovedside", "Reguleringsbot"]) 

# ------------------------------------------------------------
# Hovedside
# ------------------------------------------------------------
if sidevalg == "Hovedside":
    st.title("Samleportal for data")
    st.markdown(
        """
        Velkommen til Samleportal for data!

        Her finner du ulike moduler for analyse, visualisering og AI-tjenester knyttet til reguleringsplaner, eiendomsdata og mer.
        
        Dette er en app for å enkelt kunne navigere seg gjennom de forskjellige kildene til data vi har tilgjengelig, med funksjoner som:
        
        1) Ulike kartlag for Tromsø (og Norge)
        2) dokumentopplastning av reguleringsplaner som automatisk fører deg til planens lokasjon på kartet 
        3) AI-chat og oppsummering av planer 
        4) Analyse av planer 
        5) Visualisering av boligpriser og bedrifter i Tromsø
        6) Foreslå egne analyser og få tilbakemelding
        
        """
    )

# ------------------------------------------------------------
# Reguleringsbot
# ------------------------------------------------------------
if sidevalg == "Reguleringsbot":
    # CSS for chat og kart
    st.markdown(
        """
        <style>
        .scrollbox {
            max-height: 300px;
            overflow-y: auto;
            padding: 1rem;
            background-color: #fdfdfd;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .leaflet-control-layers {
            z-index: 9999 !important;
            position: absolute !important;
            top: 10px !important;
            right: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Velkomstmelding (vises én gang)
    if "show_welcome" not in st.session_state:
        st.session_state.show_welcome = True

    if st.session_state.show_welcome:
        st.markdown(
            """
            ## 👋 Velkommen til Reguleringsbot!
            
            Her er hva du kan gjøre:
            - **Chat med AI**: Still spørsmål om reguleringsplanen i tekstboksen til høyre.
            - **Utforsk kartet**: Se reguleringsplaner, boligpriser og bedrifter i Tromsø på kartet.
            - **Analyser planer**: Få AI-vurdering av om planen følger kommunens mål.
            - **Foreslå analyser**: Send inn egne analyseforslag i sidepanelet.
            
            Trykk på "Lukk" når du er klar til å bruke appen.
            """
        )
        if st.button("Lukk", key="close_welcome"):
            st.session_state.show_welcome = False

    st.title("🏗️ Reguleringsbot – Chat og kart over Tromsø")

    # Sidebar-valg
    områdevalg = st.sidebar.selectbox(
        "Velg reguleringsplan eller område:",
        ("Plan A – Sentrum", "Plan B – Tomasjord", "Plan C – Workinnmarka"),
    )

    uploaded_file = st.sidebar.file_uploader("📄 Last opp ny reguleringsplan (PDF)", type="pdf")

    områdeinfo: Dict[str, Dict] = {
        "Plan A – Sentrum": {"pdf": "plan_sentrum.pdf", "koordinater": [69.6496, 18.9560]},
        "Plan B – Tomasjord": {"pdf": "plan_tomasjord.pdf", "koordinater": [69.6800, 19.0300]},
        "Plan C – Workinnmarka": {"pdf": "plan_workinnmarka.pdf", "koordinater": [69.6500, 18.9000]},
    }

    # Default pdf path (sikker default, brukes også når kart ikke vises)
    pdf_path: str = områdeinfo[områdevalg]["pdf"]

    # Data
    bedrifter_df = load_bedrifter()
    adresser_df = load_adresser()
    teig_gdf = load_teig_gdf()

    # Kart visning toggle
    if "vis_kart" not in st.session_state:
        st.session_state["vis_kart"] = False

    # Toggle-knapper
    col_toggle_a, col_toggle_b = st.sidebar.columns(2)
    if col_toggle_a.button("Vis kart"):
        st.session_state["vis_kart"] = True
    if col_toggle_b.button("Skjul kart"):
        st.session_state["vis_kart"] = False

    # Håndter kart- og bildeoppsett
    koordinater: Optional[List[float]] = None

    if st.session_state["vis_kart"]:
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                pdf_path = tmp.name

            # Gjenbruk dokumenter
            docs = load_pdf_docs(pdf_path)

            # Prøv GNR/BNR først
            gnr, bnr = extract_gnr_bnr_from_pdf(pdf_path, docs=docs)
            coords = None
            if gnr and bnr:
                coords = gnr_bnr_to_coords_csv(adresser_df, gnr, bnr)
            if not coords:
                coords = extract_coordinates_from_pdf(pdf_path, docs=docs)

            # Lagre første side som bilde (grasiøs fallback hvis poppler mangler)
            try:
                convert_kwargs = dict(first_page=1, last_page=1, fmt="png")
                if POPPLER_PATH:
                    convert_kwargs["poppler_path"] = POPPLER_PATH
                images = convert_from_path(pdf_path, **convert_kwargs)
                if images:
                    img = images[0]
                    img_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                    img.save(img_temp.name, format="PNG")
                    with open(img_temp.name, "rb") as f:
                        img_bytes = f.read()
                    st.session_state["planbilde_b64"] = base64.b64encode(img_bytes).decode()
                else:
                    st.session_state["planbilde_b64"] = None
            except Exception as e:
                st.info("Kunne ikke generere bilde fra PDF (mangler ofte Poppler). Fortsetter uten bilde.")
                st.session_state["planbilde_b64"] = None

            # Manuell fallback om ingen koordinater
            if not coords:
                st.sidebar.markdown("### 📍 Velg kart-posisjon for opplastet plan")
                lat = st.sidebar.number_input("Breddegrad (lat)", value=69.65, format="%.6f")
                lon = st.sidebar.number_input("Lengdegrad (lon)", value=18.95, format="%.6f")
                coords = [lat, lon]

            st.session_state["kart_koordinater"] = coords
            koordinater = coords
        else:
            pdf_path = områdeinfo[områdevalg]["pdf"]
            st.session_state["kart_koordinater"] = områdeinfo[områdevalg]["koordinater"]
            koordinater = st.session_state["kart_koordinater"]
            st.session_state["planbilde_b64"] = None
    else:
        koordinater = None
        st.session_state["planbilde_b64"] = None

    # Layout: kart og chat
    col1, col2 = st.columns([2, 1])

    # Sidebar: kartlagvalg
    kartlagvalg = st.sidebar.selectbox(
        "Velg standard kartlag:",
        [
            "OpenStreetMap",
            "Stamen Terrain",
            "Stamen Toner",
            "Satellitt (Google)",
        ],
        index=0,
        key="kartlagvalg",
    )

    # Kart-faner
    with col1:
        tabs = st.tabs(["📍 Reguleringsplan", "💰 Boligpriser", "🏢 Bedrifter i Tromsø", "🏡 Fremtidig boligbygging"]) 

        # --- Reguleringsplan ---
        with tabs[0]:
            default_coords = [69.6496, 18.9560]
            kart_coords = st.session_state.get("kart_koordinater", default_coords)
            zoom = 17 if st.session_state.get("vis_kart") and kart_coords != default_coords else 13

            m = folium.Map(location=kart_coords, zoom_start=zoom, tiles=None)

            # Marker for planområde
            if st.session_state.get("vis_kart") and kart_coords != default_coords:
                popup_html = f"<b>{områdevalg}</b><br>Still spørsmål i chatten!"
                if st.session_state.get("planbilde_b64"):
                    popup_html += f"<br><img src='data:image/png;base64,{st.session_state['planbilde_b64']}' width='300'>"
                folium.Marker(
                    location=kart_coords,
                    popup=folium.Popup(popup_html, max_width=350),
                    tooltip="Planområde",
                ).add_to(m)

            # Visualiser planpolygoner
            if st.session_state.get("vis_kart") and not teig_gdf.empty:
                def _popup_html(row: pd.Series) -> str:
                    gnr_val = row.get("GNR", row.get("gardsnummer", ""))
                    bnr_val = row.get("BNR", row.get("bruksnummer", ""))
                    return f"GNR: {gnr_val}<br>BNR: {bnr_val}" if gnr_val and bnr_val else "Planpolygon"

                for _, row in teig_gdf.iterrows():
                    folium.GeoJson(
                        row["geometry"],
                        name="Planpolygon",
                        style_function=lambda x: {
                            "fillColor": "#3388ff",
                            "color": "#3388ff",
                            "weight": 2,
                            "fillOpacity": 0.3,
                        },
                        highlight_function=lambda x: {
                            "weight": 4,
                            "color": "#ff7800",
                            "fillOpacity": 0.5,
                        },
                        tooltip=_popup_html(row),
                        popup=folium.Popup(_popup_html(row), max_width=250),
                    ).add_to(m)

            # Kartlag
            tile_layers = {
                "OpenStreetMap": folium.TileLayer("OpenStreetMap", name="OpenStreetMap"),
                "Stamen Terrain": folium.TileLayer("Stamen Terrain", attr="Stamen Design", name="Stamen Terrain"),
                "Stamen Toner": folium.TileLayer("Stamen Toner", attr="Stamen Design", name="Stamen Toner"),
                "Satellitt (Google)": folium.TileLayer(
                    tiles="http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", attr="Google", name="Satellitt (Google)")
            }
            # Legg til lag: velg standard som aktiv ved å legge den til sist
            for navn, lag in tile_layers.items():
                if navn != kartlagvalg:
                    lag.add_to(m)
            tile_layers[kartlagvalg].add_to(m)
            folium.LayerControl().add_to(m)
            st_folium(m, width=900, height=600, key=f"kart_{kartlagvalg}")

        # --- Boligpriser ---
        with tabs[1]:
            st.subheader("🏘️ Boliger etter prisklasse (demo)")
            boliger = [
                {"adresse": "Storgata 1", "pris": 1900000, "koordinater": [69.6501, 18.9550]},
                {"adresse": "Håkon Gamles gate 4", "pris": 3200000, "koordinater": [69.6480, 18.9600]},
                {"adresse": "Strandvegen 20", "pris": 4800000, "koordinater": [69.6520, 18.9400]},
                {"adresse": "Grønnegata 17", "pris": 6200000, "koordinater": [69.6460, 18.9650]},
            ]
            prisklasser = ["Alle", "Under 2 mill", "2–4 mill", "4–6 mill", "Over 6 mill"]
            valgt = st.selectbox("📊 Velg priskategori:", prisklasser)

            def filtrer(prisklasse: str) -> List[Dict]:
                if prisklasse == "Alle":
                    return boliger
                if prisklasse == "Under 2 mill":
                    return [b for b in boliger if b["pris"] < 2000000]
                if prisklasse == "2–4 mill":
                    return [b for b in boliger if 2000000 <= b["pris"] <= 4000000]
                if prisklasse == "4–6 mill":
                    return [b for b in boliger if 4000000 < b["pris"] <= 6000000]
                if prisklasse == "Over 6 mill":
                    return [b for b in boliger if b["pris"] > 6000000]
                return []

            filtrerte = filtrer(valgt)
            if koordinater:
                kart = folium.Map(location=koordinater, zoom_start=13)
                for bolig in filtrerte:
                    folium.Marker(
                        location=bolig["koordinater"],
                        popup=f"{bolig['adresse']}<br>{bolig['pris']:,} kr",
                        tooltip=bolig["adresse"],
                        icon=folium.Icon(color="green", icon="home"),
                    ).add_to(kart)
                folium.TileLayer("OpenStreetMap").add_to(kart)
                folium.TileLayer("Stamen Terrain", attr="Stamen Design").add_to(kart)
                folium.TileLayer("Stamen Toner", attr="Stamen Design").add_to(kart)
                folium.TileLayer(tiles="http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", attr="Google", name="Satellitt").add_to(kart)
                st_folium(kart, width=900, height=600)
                st.caption(f"🟢 Viser {len(filtrerte)} boliger i valgt kategori.")
            else:
                st.info("Kartet vises først når du trykker på 'Vis kart' i sidepanelet.")

        # --- Bedrifter i Tromsø ---
        with tabs[2]:
            st.subheader("🏢 Bedrifter i Tromsø")
            søk = st.text_input("🔎 Søk etter bedrift (skriv hele eller deler av navnet):")

            if søk.strip():
                filtrerte_bedrifter = bedrifter_df[bedrifter_df["NAVN"].str.contains(søk, case=False, na=False)]
                st.markdown(f"🔍 Fant {len(filtrerte_bedrifter)} treff")

                if koordinater:
                    kart = folium.Map(location=koordinater, zoom_start=12)
                    for _, rad in filtrerte_bedrifter.iterrows():
                        folium.Marker(
                            location=[rad.get("LAT", 0.0), rad.get("LON", 0.0)],
                            popup=f"<b>{rad.get('NAVN', '')}</b>",
                            icon=folium.Icon(color="blue", icon="briefcase", prefix="fa"),
                        ).add_to(kart)
                    folium.TileLayer("OpenStreetMap").add_to(kart)
                    folium.TileLayer("Stamen Terrain", attr="Stamen Design").add_to(kart)
                    folium.TileLayer("Stamen Toner", attr="Stamen Design").add_to(kart)
                    folium.TileLayer(tiles="http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", attr="Google", name="Satellitt").add_to(kart)
                    st_folium(kart, width=900, height=600)
                else:
                    st.info("Kartet vises først når du trykker på 'Vis kart' i sidepanelet.")
            else:
                st.info("Skriv inn et søkeord for å vise bedrifter på kartet.")

        # --- Fremtidig boligbygging ---
        with tabs[3]:
            st.subheader("🏡 Fremtidig boligbygging")
            st.info("Her kan du vise eller analysere fremtidige boligprosjekter i Tromsø. (Innhold kan tilpasses)")

    # 💬 Chat og analyse
    with col2:
        st.subheader("🔎 Fulltekst-søk i PDF")
        term = st.text_input("Søk i plan:")
        if term and pdf_path:
            try:
                docs = load_pdf_docs(pdf_path)
                treff: List[Tuple[int, str]] = []
                for i, d in enumerate(docs, 1):
                    for m in re.finditer(re.escape(term), d.page_content, re.IGNORECASE):
                        s = max(m.start() - 40, 0)
                        e = min(m.end() + 40, len(d.page_content))
                        snippet = d.page_content[s:e].replace("\n", " ")
                        treff.append((i, snippet))
                if treff:
                    for side, utdrag in treff:
                        with st.expander(f"Side {side}"):
                            st.write(f"...{utdrag}...")
                else:
                    st.info("Ingen treff i PDF.")
            except Exception as exc:
                st.warning(f"Kunne ikke søke i PDF: {exc}")

        st.subheader("📝 Oppsummer plan")
        lengde = st.selectbox("Lengde:", ["kort", "very_kort"], key="summary_length")

        def summarize_plan(path: str, lengde_val: str = "kort") -> Optional[str]:
            llm = get_llm()
            if llm is None:
                return None
            text = hent_avsnitt(path, maks_tegn=3000)
            prompt = (
                "Du er en erfaren arealplanlegger. "
                + ("Gi meg en veldig kort oppsummering (2 setninger):\n\n" if lengde_val == "very_kort" else "Gi meg en kort oppsummering (3–4 setninger):\n\n")
                + text
            )
            try:
                return llm.invoke(prompt).content
            except Exception as exc:
                st.warning(f"Oppsummering feilet: {exc}")
                return None

        if st.button("Oppsummer plan", key="btn_summary"):
            summary = summarize_plan(pdf_path, lengde)
            if summary:
                st.markdown(f"> {summary}")

        st.subheader("🤖 Spør AI om reguleringsplanen")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        forslag = [
            "Hva sier planen om byggehøyder?",
            "Hva er formålet med reguleringen?",
            "Finnes det krav til uteområder?",
            "Hva er regulert til næring?",
            "Hvordan påvirker planen nabolaget?",
        ]
        st.markdown("**Eksempler på spørsmål:**")
        col_btns = st.columns(1)
        for i, spm in enumerate(forslag):
            if st.button(spm, key=f"forslag_{i}"):
                st.session_state.input_q = spm

        user_input = st.text_input("Skriv inn spørsmål:", key="input_q", value=st.session_state.get("input_q", ""))

        if user_input:
            with st.spinner("Tenker..."):
                qa = get_qa_for_pdf(pdf_path)
                if qa is None:
                    st.info("AI er ikke tilgjengelig. Sjekk API-nøkkelen i 'st.secrets' eller miljøvariabler.")
                else:
                    try:
                        response = qa.invoke({"query": user_input})
                        svar = response.get("result", "")
                        kilder = response.get("source_documents", [])
                        st.session_state.chat_history.append((user_input, svar, kilder))
                    except Exception as exc:
                        st.warning(f"Spørring feilet: {exc}")

        if st.session_state.chat_history:
            st.markdown("### 🗣️ Chat-historikk")
            for q, a, kilder in st.session_state.chat_history:
                st.markdown(f"**Du:** {q}")
                st.markdown(f"**Svar:** {a}")
                with st.expander("📄 Kilder i planen"):
                    for i, kilde in enumerate(kilder, 1):
                        utdrag = getattr(kilde, "page_content", "")[:600]
                        st.markdown(f"**Kilde {i}:**\n\n{utdrag}...")

            # Klargjør CSV (kun spørsmål og svar)
            try:
                df_chat = pd.DataFrame(st.session_state.chat_history, columns=["Spørsmål", "Svar", "Kilder"]).drop(columns=["Kilder"])  
                csv = df_chat.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Last ned chat-historikk (CSV)",
                    data=csv,
                    file_name="chat_history.csv",
                    mime="text/csv",
                )
            except Exception:
                pass

        st.subheader("📊 Analyse: Er planen i tråd med kommunens mål?")
        if st.button("Analyser mot kommuneplanen"):
            with st.spinner("Laster og analyserer dokumenter..."):
                try:
                    regtekst = hent_avsnitt(pdf_path)
                except Exception as exc:
                    st.warning(f"Kunne ikke lese reguleringsplan: {exc}")
                    regtekst = ""

                def safe_hent(path: str) -> str:
                    try:
                        return hent_avsnitt(path)
                    except Exception:
                        return "(Kunne ikke lese dokumentet)"

                kpatekst = safe_hent("Planer/kpa.pdf")
                samftekst = safe_hent("Planer/kommuneplanens_samfunnsdel_2020.pdf")

                llm = get_llm()
                if llm is None:
                    st.info("AI er ikke tilgjengelig. Sjekk API-nøkkel.")
                else:
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
"""
                    try:
                        vurdering = llm.invoke(full_prompt).content
                        st.markdown("### 📋 AI-vurdering")
                        st.markdown(
                            f"<div class='scrollbox'>{vurdering.replace(chr(10), '<br>')}</div>",
                            unsafe_allow_html=True,
                        )
                    except Exception as exc:
                        st.warning(f"Analyse feilet: {exc}")

    # 📥 Brukerforslag
    st.sidebar.markdown("---")
    st.sidebar.header("💡 Foreslå analyseidé")
    kategori = st.sidebar.selectbox(
        "Velg datasett eller tema:", [
            "Matrikkeldata",
            "Brønnøysundregisteret",
            "Skattedata",
            "Befolkningsdata",
            "Grunnboken",
            "Andre",
        ],
    )
    analyseforslag = st.sidebar.text_area(
        "Hva ønsker du at vi skal analysere?",
        height=150,
        placeholder="Eks: Kan vi koble eiendomsskatt med tomtestørrelser?",
    )
    if st.sidebar.button("Send inn forslag"):
        if analyseforslag.strip():
            if "innsendte_forslag" not in st.session_state:
                st.session_state.innsendte_forslag = []
            st.session_state.innsendte_forslag.append((kategori, analyseforslag))
            st.sidebar.success("✅ Forslaget er registrert!")
        else:
            st.sidebar.warning("Skriv inn et forslag før du sender.")

    with st.expander("📝 Dine innsendte forslag"):
        if "innsendte_forslag" in st.session_state:
            for idx, (kat, txt) in enumerate(st.session_state.innsendte_forslag, 1):
                st.markdown(f"**{idx}. {kat}**\n\n{txt}")
        else:
            st.info("Ingen forslag registrert enda.")