import streamlit as st
import datetime as dt

st.set_page_config(page_title="TB debug app", layout="centered")

st.title("✅ Streamlit is working")
st.write("If you see this text, `test_app.py` is running correctly.")

st.markdown("---")
st.write("Timestamp (to be sure this is fresh):")
st.code(str(dt.datetime.now()), language="text")
