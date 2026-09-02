import streamlit as st

pg = st.navigation([
    st.Page("pages/1_Canlı_Gösterge.py", title="Canlı Gösterge", icon="⚡", default=True),
    st.Page("pages/2_Analiz_Tahmini.py", title="Analiz Tahmini", icon="📊"),
])
pg.run()
