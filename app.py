import streamlit as st
import pandas as pd
import time
from datetime import datetime

ts = time.time()
date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")


df = pd.read_csv('attendence/attendence_' +date + '.csv')

st.dataframe(df.style.highlight_max(axis=0))