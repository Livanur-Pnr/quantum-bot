import streamlit as st

pg = st.navigation([
    st.Page("pages/1_Analiz_Tahmini.py", title="Analiz Tahmini", icon="⚡", default=True),
    st.Page("pages/2_Canlı_Gösterge.py", title="Canlı Gösterge", icon="📊"),
    st.Page("pages/3_Haber_Analizi.py", title="Haber Analizi", icon="📰"),
])
pg.run()
