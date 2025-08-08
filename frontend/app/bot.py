import os
from typing import Any

import requests
import streamlit as st


BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="APH-IF", layout="wide")
st.title("APH-IF • Frontend")


def backend_health() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/healthz", timeout=2)
        return r.ok
    except Exception:
        return False


ok = backend_health()
pill = "🟢 Backend OK" if ok else "🔴 Backend Down"
st.sidebar.markdown(f"**Status:** {pill}")

query_text = st.text_input("Your question")
if st.button("Ask") and query_text:
    with st.spinner("Querying backend…"):
        try:
            resp = requests.post(f"{BACKEND_URL}/query", json={"query": query_text}, timeout=10)
            if resp.ok:
                data: Any = resp.json()
                st.write(data.get("answer", "No answer"))
            else:
                st.error(f"Backend error: {resp.status_code}")
        except Exception as exc:
            st.error(f"Request failed: {exc}")


