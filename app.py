import os
import tempfile
import base64
import re

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
from pdf2image import convert_from_path

# --- Konfigurasjon Poppler for Windows ---
POPPLER_PATH = r"C:\Users\gijo858\OneDrive - Polaris Media\Skrivebord\Release-24.08.0-0\poppler-24.08.0\Library\bin"

# Funksjoner for QA-bot og tekstuttrekk

def setup_bot(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vectordb = FAISS.from_documents(chunks, OpenAIEmbeddings())
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True
    )

def hent_avsnitt(path, maks_tegn=3000):
    loader = PyPDFLoader(path)
    docs = loader.load()
    tekst = "\n\n".join([d.page_content for d in docs])
    return tekst[:maks_tegn]

# API-nøkkel
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
llm = ChatOpenAI(model="gpt-3.5-turbo")

# 📄 Les inn bedriftsdata
bedrifter_df = pd.read_csv("bedrifter_tromso.csv")
# 📄 Les inn matrikkel-adresse-data (CSV) med lat/lon
try:
    adresser_df = pd.read_csv(
        "matrikkel_adresse_latlon.csv",
        sep=';',
        on_bad_lines='skip'
    )
    if len(adresser_df.columns) == 1:
        adresser_df = pd.read_csv(
            "matrikkel_adresse_latlon.csv",
            sep=',',
            on_bad_lines='skip'
        )
except Exception as e:
    st.warning(f"Kunne ikke laste inn matrikkel_adresse_latlon.csv: {e}")
    adresser_df = pd.DataFrame()

# Sidekonfigurasjon
st.set_page_config(page_title="Reguleringsbot", layout="wide")

# Dark/bright mode toggle
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

toggle_label = "Bytt til mørk modus" if not st.session_state.dark_mode else "Bytt til lys modus"
if st.sidebar.button(toggle_label):
    st.session_state.dark_mode = not st.session_state.dark_mode
    st.experimental_rerun()

bg = "#000000" if st.session_state.dark_mode else "#FFFFFF"
text = "#FFFFFF" if st.session_state.dark_mode else "#000000"
scroll_bg = "#333333" if st.session_state.dark_mode else "#fdfdfd"
border_color = "#555555" if st.session_state.dark_mode else "#ddd"

# CSS for chat og kart
st.markdown(
    f"""
    <style>
    .scrollbox {{
        max-height: 300px;
        overflow-y: auto;
        padding: 1rem;
        background-color: {scroll_bg};
        border: 1px solid {border_color};
        border-radius: 5px;
        color: {text};
    }}
    .leaflet-control-layers {{
        z-index: 9999 !important;
        position: absolute !important;
        top: 10px !important;
        right: 10px !important;
    }}
    .stApp {{
        background-color: {bg};
        color: {text};
    }}
    </style>
    """, unsafe_allow_html=True
)

# Velkomstmelding
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
    ("Plan A – Sentrum", "Plan B – Tomasjord", "Plan C – Workinnmarka")
)

uploaded_file = st.sidebar.file_uploader(
    "📄 Last opp ny reguleringsplan (PDF)", type="pdf"
)

områdeinfo = {
    "Plan A – Sentrum": {"pdf": "plan_sentrum.pdf", "koordinater": [69.6496, 18.9560]},
    "Plan B – Tomasjord": {"pdf": "plan_tomasjord.pdf", "koordinater": [69.6800, 19.0300]},
    "Plan C – Workinnmarka": {"pdf": "plan_workinnmarka.pdf", "koordinater": [69.6500, 18.9000]},
}

# Les inn GML med planpolygone
teig_gdf = gpd.read_file("teig.gml")

# Helper-funksjoner for koordinater og GNR/BNR

def extract_coordinates_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text = "\n".join([doc.page_content for doc in docs])
    # Forsøk å finne koordinater i teksten
    match = re.search(r"(\d{2}\.\d+)[,\s]+(\d{1,2}\.\d+)", text)
    if match:
        return [float(match.group(1)), float(match.group(2))]
    # Forsøk geokoding basert på adresse
    address_match = re.search(r'(Storgata|gate|vegen|gata|Tomasjord|Workinnmarka)[^\n,]*', text, re.IGNORECASE)
    if address_match:
        geolocator = Nominatim(user_agent="reguleringsbot")
        location = geolocator.geocode("Tromsø, " + address_match.group(0))
        if location:
            return [location.latitude, location.longitude]
    return None


def extract_gnr_bnr_from_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text = "\n".join([doc.page_content for doc in docs])
    match = re.search(r"G[\s\-]?N[\s\-]?R[\s:]*([0-9]+)[,\s]+B[\s\-]?N[\s\-]?R[\s:]*([0-9]+)", text, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)
    return None, None


def gnr_bnr_to_coords_csv(gnr, bnr):
    match = adresser_df[
        (adresser_df['gardsnummer'] == int(gnr)) &
        (adresser_df['bruksnummer'] == int(bnr))
    ]
    if not match.empty:
        return [match.iloc[0]['latitude'], match.iloc[0]['longitude']]
    return None


def gnr_bnr_to_coords(gnr, bnr):
    return gnr_bnr_to_coords_csv(gnr, bnr)

# Kart visning toggle
if "vis_kart" not in st.session_state:
    st.session_state["vis_kart"] = False

if st.sidebar.button("Vis kart for valgt plan/område"):
    st.session_state["vis_kart"] = True

# Håndter kart- og bildeoppsett
if st.session_state["vis_kart"]:
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        # Prøv GNR/BNR først
        gnr, bnr = extract_gnr_bnr_from_pdf(pdf_path)
        coords = None
        if gnr and bnr:
            coords = gnr_bnr_to_coords(gnr, bnr)
        if not coords:
            coords = extract_coordinates_from_pdf(pdf_path)

        # Lagre første side som bilde
        try:
            images = convert_from_path(
                pdf_path,
                first_page=1,
                last_page=1,
                fmt='png',
                poppler_path=POPPLER_PATH
            )
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
            st.warning(f"Kunne ikke generere bilde fra PDF: {e}")
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
        "Satellitt (Google)"
    ],
    index=0,
    key="kartlagvalg"
)

"""
Replace map tabs with a dropdown menu: Instead of showing tabs side by side,
allow the user to pick a map view from a selectbox. Only the chosen view is
rendered. This keeps the interface compact and better suited for many map
options.
"""

# Kart-faner
with col1:
    fanevalg = st.selectbox(
        "Velg kart:",
        [
            "📍 Reguleringsplan",
            "💰 Boligpriser",
            "🏢 Bedrifter i Tromsø",
            "🏡 Fremtidig boligbygging",
            "🔄 Boligområder med størst turnover",
        ],
    )

    # --- Reguleringsplan ---
    if fanevalg == "📍 Reguleringsplan":
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
                tooltip="Planområde"
            ).add_to(m)

        # Visualiser planpolygoner
        if st.session_state.get("vis_kart") and not teig_gdf.empty:
            def _popup_html(row):
                gnr = row.get('GNR', row.get('gardsnummer', ''))
                bnr = row.get('BNR', row.get('bruksnummer', ''))
                return f"GNR: {gnr}<br>BNR: {bnr}" if gnr and bnr else "Planpolygon"

            for _, row in teig_gdf.iterrows():
                folium.GeoJson(
                    row["geometry"],
                    name="Planpolygon",
                    style_function=lambda x: {'fillColor': '#3388ff','color': '#3388ff','weight': 2,'fillOpacity': 0.3},
                    highlight_function=lambda x: {'weight': 4,'color': '#ff7800','fillOpacity': 0.5},
                    tooltip=_popup_html(row),
                    popup=folium.Popup(_popup_html(row), max_width=250)
                ).add_to(m)

        # Kartlag
        tile_layers = {
            "OpenStreetMap": {"tiles": "OpenStreetMap", "name": "OpenStreetMap"},
            "Stamen Terrain": {"tiles": "Stamen Terrain", "attr": "Stamen Design", "name": "Stamen Terrain"},
            "Stamen Toner": {"tiles": "Stamen Toner", "attr": "Stamen Design", "name": "Stamen Toner"},
            "Satellitt (Google)": {
                "tiles": "http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                "attr": "Google",
                "name": "Satellitt (Google)"
            }
        }
        for navn, params in tile_layers.items():
            folium.TileLayer(**params, show=(navn == kartlagvalg)).add_to(m)
        folium.LayerControl().add_to(m)
        st_folium(m, width=900, height=600, key=f"kart_{kartlagvalg}")

    # --- Boligpriser ---
    elif fanevalg == "💰 Boligpriser":
        st.subheader("🏘️ Boliger etter prisklasse (demo)")
        boliger = [
            {"adresse": "Storgata 1", "pris": 1900000, "koordinater": [69.6501, 18.9550]},
            {"adresse": "Håkon Gamles gate 4", "pris": 3200000, "koordinater": [69.6480, 18.9600]},
            {"adresse": "Strandvegen 20", "pris": 4800000, "koordinater": [69.6520, 18.9400]},
            {"adresse": "Grønnegata 17", "pris": 6200000, "koordinater": [69.6460, 18.9650]},
        ]
        prisklasser = ["Alle", "Under 2 mill", "2–4 mill", "4–6 mill", "Over 6 mill"]
        valgt = st.selectbox("📊 Velg priskategori:", prisklasser)

        def filtrer(prisklasse):
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
                    icon=folium.Icon(color="green", icon="home")
                ).add_to(kart)
            folium.TileLayer("OpenStreetMap").add_to(kart)
            folium.TileLayer("Stamen Terrain", attr="Stamen Design").add_to(kart)
            folium.TileLayer("Stamen Toner", attr="Stamen Design").add_to(kart)
            folium.TileLayer(
                tiles="http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                attr="Google", name="Satellitt"
            ).add_to(kart)
            folium.LayerControl().add_to(kart)
            st_folium(kart, width=900, height=600)
            st.caption(f"🟢 Viser {len(filtrerte)} boliger i valgt kategori.")
        else:
            st.info("Kartet vises først når du trykker på 'Vis kart for valgt plan/område' i sidepanelet.")

    # --- Bedrifter i Tromsø ---
    elif fanevalg == "🏢 Bedrifter i Tromsø":
        st.subheader("🏢 Bedrifter i Tromsø")
        søk = st.text_input("🔎 Søk etter bedrift (skriv hele eller deler av navnet):")

        if søk.strip():
            filtrerte_bedrifter = bedrifter_df[
                bedrifter_df["NAVN"].str.contains(søk, case=False, na=False)
            ]
            st.markdown(f"🔍 Fant {len(filtrerte_bedrifter)} treff")

            if koordinater:
                kart = folium.Map(location=koordinater, zoom_start=12)
                for _, rad in filtrerte_bedrifter.iterrows():
                    folium.Marker(
                        location=[rad["LAT"], rad["LON"]],
                        popup=f"<b>{rad['NAVN']}</b>",
                        icon=folium.Icon(color="blue", icon="briefcase", prefix="fa")
                    ).add_to(kart)
                folium.TileLayer("OpenStreetMap").add_to(kart)
                folium.TileLayer("Stamen Terrain", attr="Stamen Design").add_to(kart)
                folium.TileLayer("Stamen Toner", attr="Stamen Design").add_to(kart)
                folium.TileLayer(
                    tiles="http://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
                    attr="Google", name="Satellitt"
                ).add_to(kart)
                folium.LayerControl().add_to(kart)
                st_folium(kart, width=900, height=600)
            else:
                st.info("Kartet vises først når du trykker på 'Vis kart for valgt plan/område' i sidepanelet.")
        else:
            st.info("Skriv inn et søkeord for å vise bedrifter på kartet.")

    # --- Fremtidig boligbygging ---
    elif fanevalg == "🏡 Fremtidig boligbygging":
        st.subheader("🏡 Fremtidig boligbygging")
        st.info(
            "Her kan du vise eller analysere fremtidige boligprosjekter i Tromsø. (Innhold kan tilpasses)"
        )

    # --- Boligområder med størst turnover ---
    elif fanevalg == "🔄 Boligområder med størst turnover":
        st.subheader("🔄 Boligområder med størst turnover")
        st.info(
            "Her kan du se hvilke områder som har høyest omsetning av boliger. (Innhold kan tilpasses)"
        )

# 💬 Chat og analyse
with col2:
    st.subheader("📊 Analyse: Er planen i tråd med kommunens mål?")
    if st.button("Analyser mot kommuneplanen"):
        with st.spinner("Laster og analyserer dokumenter..."):
            st.session_state.regtekst = hent_avsnitt(pdf_path)
            st.session_state.kpatekst = hent_avsnitt("Planer/kpa.pdf")
            st.session_state.samftekst = hent_avsnitt("Planer/kommuneplanens_samfunnsdel_2020.pdf")

            full_prompt = f"""Du er arealplanlegger og journalist. Du har fått tilgang til følgende reguleringsplan:

--- REGULERINGSPLAN ---
{st.session_state.regtekst}

--- KOMMUNEPLANENS AREALDEL (KPA) ---
{st.session_state.kpatekst}

--- KOMMUNEPLANENS SAMFUNNSDEL ---
{st.session_state.samftekst}

Basert på dette, vurder:
1. Er reguleringsplanen i tråd med bærekraftsmålene i samfunnsplanen?
2. Følger den føringene i KPA, særlig med tanke på grøntområder, høyder og fortetting?
3. Hvilke avvik finnes, og hvordan kan disse vinkles journalistisk?

Svar tydelig og konkret.
"""
            vurdering = llm.invoke(full_prompt).content
            st.markdown("### 📋 AI-vurdering")
            st.markdown(
                f"<div class='scrollbox'>{vurdering.replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True
            )

    st.subheader("🔎 Fulltekst-søk i PDF")
    term = st.text_input("Søk i plan:")
    if term and pdf_path:
        docs = PyPDFLoader(pdf_path).load()
        treff = []
        for i, d in enumerate(docs, 1):
            for m in re.finditer(re.escape(term), d.page_content, re.IGNORECASE):
                s = max(m.start()-40, 0); e = min(m.end()+40, len(d.page_content))
                snippet = d.page_content[s:e].replace("\n", " ")
                treff.append((i, snippet))
        if treff:
            for side, utdrag in treff:
                with st.expander(f"Side {side}"):
                    st.write(f"...{utdrag}...")
        else:
            st.info("Ingen treff i PDF.")

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
    for i, spm in enumerate(forslag):
        if st.button(spm, key=f"forslag_{i}"):
            st.session_state.input_q = spm

    user_input = st.text_input(
        "Skriv inn spørsmål:", key="input_q", value=st.session_state.get("input_q", "")
    )

    if user_input:
        with st.spinner("Tenker..."):
            qa = setup_bot(pdf_path)
            response = qa.invoke({"query": user_input})
            svar = response["result"]
            kilder = response["source_documents"]
            st.session_state.chat_history.append((user_input, svar, kilder))

    if st.session_state.chat_history:
        st.markdown("### 🗣️ Chat-historikk")
        for q, a, kilder in st.session_state.chat_history:
            st.markdown(f"**Du:** {q}")
            st.markdown(f"**Svar:** {a}")
            with st.expander("📄 Kilder i planen"):
                for i, kilde in enumerate(kilder, 1):
                    utdrag = kilde.page_content[:600]
                    st.markdown(f"**Kilde {i}:**\n\n{utdrag}...")

# Funksjon for å hente tekst-avsnitt til analyse
 

# 📥 Brukerforslag
st.sidebar.markdown("---")
st.sidebar.header("💡 Foreslå analyseidé")
kategori = st.sidebar.selectbox(
    "Velg datasett eller tema:", [
        "Matrikkeldata", "Brønnøysundregisteret", "Skattedata",
        "Befolkningsdata", "Grunnboken", "Andre"
    ]
)
analyseforslag = st.sidebar.text_area(
    "Hva ønsker du at vi skal analysere?",
    height=150,
    placeholder="Eks: Kan vi koble eiendomsskatt med tomtestørrelser?"
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