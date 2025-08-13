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
+# ------------------------------------------------------------
+# Page setup (must be called once)
+# ------------------------------------------------------------
+st.set_page_config(page_title="Samleportal for data", layout="wide")
+
+# ------------------------------------------------------------
+# Configuration and helpers
+# ------------------------------------------------------------
+DEFAULT_MODEL_NAME = st.secrets.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
+OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
+EMBEDDING_MODEL_NAME = st.secrets.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
+
+# Poppler path handling (Windows-specific path should not break on Linux)
+POPPLER_PATH = (
+    st.secrets.get("POPPLER_PATH")
+    or os.getenv("POPPLER_PATH")
+    or None  # None means let pdf2image try system defaults; we handle exceptions below
+)
+
+# ------------------------------------------------------------
+# Caching: data loading
+# ------------------------------------------------------------
+        if "chat_history" in st.session_state:
+            chat_history = st.session_state.chat_history
+            for chat in chat_history:
+                st.markdown(f"**{chat['role']}**: {chat['content']}")
+
+        # Ny melding
+        user_input = st.text_input("Skriv inn ditt spørsmål her:")
+        if st.button("Send"):
+            if user_input.strip():
+                # Legg til i chat-historikk
+                chat_history.append({"role": "user", "content": user_input})
+                st.session_state.chat_history = chat_history
+
+                # AI-svar (asynkront)
+                with st.spinner("AI skriver svar..."):
+                    llm = get_llm()
+                    if llm:
+                        try:
+                            response = llm.invoke(user_input)
+                            answer = response.content
+
+                            # Legg til AI-svar i historikk
+                            chat_history.append({"role": "assistant", "content": answer})
+                            st.session_state.chat_history = chat_history
+
+                            # Vis AI-svar
+                            st.markdown(f"**AI**: {answer}")
+                        except Exception as e:
+                            st.warning(f"AI kunne ikke svare: {e}")
+                    else:
+                        st.warning("LLM er ikke tilgjengelig.")
+            else:
+                st.warning("Skriv inn et spørsmål før du sender.")
+
+    # 📥 Brukerforslag
+    st.sidebar.markdown("---")
+    st.sidebar.header("💡 Foreslå analyseidé")
+    kategori = st.sidebar.selectbox(
+        "Velg datasett eller tema:", [
+            "Matrikkeldata",
+            "Brønnøysundregisteret",
+            "Skattedata",
+            "Befolkningsdata",
        st.subheader("AI-chat")
        chat_history = []
        # Vis tidligere samtaler
        if "chat_history" in st.session_state:
            chat_history = st.session_state.chat_history
            for chat in chat_history:
                st.markdown(f"**{chat['role']}**: {chat['content']}")

        # Ny melding
        user_input = st.text_input("Skriv inn ditt spørsmål her:")
        if st.button("Send"):
            if user_input.strip():
                # Legg til i chat-historikk
                chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history = chat_history

                # AI-svar (asynkront)
                llm = get_llm()
                if llm:
                    with st.spinner("AI skriver svar..."):
                        try:
                            response = llm.invoke(user_input)
                            answer = response.content
                            # Legg til AI-svar i historikk
                            chat_history.append({"role": "assistant", "content": answer})
                            st.session_state.chat_history = chat_history
                            # Vis AI-svar
                            st.markdown(f"**AI**: {answer}")
                        except Exception as e:
                            st.warning(f"AI kunne ikke svare: {e}")
                else:
                    st.warning("LLM er ikke tilgjengelig.")
            else:
                st.warning("Skriv inn et spørsmål før du sender.")

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