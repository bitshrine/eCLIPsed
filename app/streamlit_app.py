import streamlit as st
from utils.setup import *
from utils.console import *

st.title("Hello World!")

with st_stdout("info"):
    fetch_LELSD_requirements()