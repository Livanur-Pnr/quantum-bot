"""
📰 Haber Analizi — CryptoPanic haber akışı + Gemini AI ile Türkçe özet ve
LONG/SHORT eğilim değerlendirmesi.

ÖNEMLİ: Bu panel bir OLASILIK değerlendirmesi sunar, kesin bir alım/satım
sinyali değildir. Diğer botlardaki "kararlılık, aşırı kesinlik yok" felsefesi
burada da geçerlidir.
"""

import streamlit as st
import requests
import json
import re
import html
import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Haber Analizi",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

PREMIUM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
    .stApp { background-color: #0b0e14; color: #d1d4dc; font-family: 'Inter', sans-serif; }
    section[data-testid="stSidebar"] { background-color: #131722; border-right: 1px solid #2a2e39; }
    [data-testid="stHeader"] { background: transparent !important; }
    footer, [data-testid="stStatusWidget"], [data-testid="stAppRunningIndicator"],
    [data-testid="stDecoration"], #stDecoration,
    [data-testid="stAppDeployButton"], .stDeployButton { display: none !important; }
</style>
"""
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

st.title("📰 Haber Analizi")
st.caption("CryptoPanic haber akışı → Gemini AI ile Türkçe özet ve olası LONG/SHORT eğilim değerlendirmesi.")

if "news_summary_cache" not in st.session_state:
    st.session_state.news_summary_cache = {}

# ─────────────────────────────────────────────────────────────────
# SIDEBAR — KONTROLLER
# ─────────────────────────────────────────────────────────────────
FILTER_MAP = {
    "Genel (en yeni)": None,
    "Yükselenler (rising)": "rising",
    "Önemli (important)": "important",
    "Olumlu (bullish)": "bullish",
    "Olumsuz (bearish)": "bearish",
}

with st.sidebar:
    st.markdown("## 📰 Haber Kontrolleri")
    coin_filter = st.text_input(
        "Coin filtrele (örn: BTC,ETH)",
        value="",
        help="Boş bırakırsan genel kripto haber akışı gösterilir.",
    )
    news_filter_label = st.selectbox("Haber türü", list(FILTER_MAP.keys()), index=0)
    max_news = st.slider("Gösterilecek haber sayısı", min_value=3, max_value=15, value=8)
    refresh_seconds = st.slider("Otomatik yenileme (saniye)", min_value=60, max_value=600, value=180, step=30)

    st.markdown("---")
    if not CRYPTOPANIC_API_KEY:
        st.warning("⚠️ `CRYPTOPANIC_API_KEY` bulunamadı. `.env` dosyanıza ekleyin.")
    if not GEMINI_API_KEY:
        st.warning("⚠️ `GEMINI_API_KEY` bulunamadı. `.env` dosyanıza ekleyin.")
    if CRYPTOPANIC_API_KEY and GEMINI_API_KEY:
        st.success("✅ Her iki API anahtarı da tanımlı.")


# ─────────────────────────────────────────────────────────────────
# CRYPTOPANIC — HABER ÇEKİMİ
# ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=90, show_spinner=False)
def fetch_cryptopanic_news(api_key: str, currencies: str, filter_key, limit: int):
    if not api_key:
        return [], "CRYPTOPANIC_API_KEY tanımlı değil."

    params = {"auth_token": api_key, "public": "true"}
    if currencies:
        params["currencies"] = currencies
    if filter_key:
        params["filter"] = filter_key

    # CryptoPanic API'nin plan bazlı ("free"/"v1") uç nokta biçimleri değişebiliyor;
    # birini deneyip başarısız olursa diğerine düşerek dayanıklılık sağlanır.
    endpoints = [
        "https://cryptopanic.com/api/free/v2/posts/",
        "https://cryptopanic.com/api/v1/posts/",
    ]
    last_err = None
    for url in endpoints:
        try:
            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                return data.get("results", [])[:limit], None
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as e:
            last_err = str(e)
    return [], f"CryptoPanic API hatası: {last_err}"


def format_relative_time(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        diff = datetime.now(timezone.utc) - dt
        mins = int(diff.total_seconds() // 60)
        if mins < 1:
            return "az önce"
        if mins < 60:
            return f"{mins} dk önce"
        hours = mins // 60
        if hours < 24:
            return f"{hours} sa önce"
        return f"{hours // 24} gün önce"
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────
# GEMINI — TÜRKÇE ÖZET + YÖN DEĞERLENDİRMESİ
# ─────────────────────────────────────────────────────────────────
def summarize_news_with_gemini(api_key: str, title: str, description: str, source: str) -> dict:
    prompt = f"""Aşağıda bir kripto para haberi var. Görevlerin:

1. Haberin içindeki ÖNEMLİ bilgileri, haber hangi dilde olursa olsun, TÜRKÇE olarak 2-3 kısa cümlelik bir özet halinde ver.
2. Bu haberin piyasada LONG (yükseliş) mi yoksa SHORT (düşüş) yönünde mi bir baskı yaratma ihtimalinin daha yüksek olduğunu değerlendir. Haber nötr, belirsiz veya etkisizse "NOTR" de. Bu KESİN bir tahmin değildir, sadece bir olasılık değerlendirmesidir — asla kesinlik iddia etme, emin olmadığında NOTR de.

Haber Başlığı: {title}
Haber İçeriği/Açıklaması: {description}
Kaynak: {source}

SADECE şu JSON formatında cevap ver, başka hiçbir açıklama ekleme:
{{"ozet": "...", "yon": "LONG veya SHORT veya NOTR", "gerekce": "tek cümlelik kısa gerekçe"}}
"""
    try:
        resp = requests.post(
            GEMINI_ENDPOINT,
            params={"key": api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 300},
            },
            timeout=30,
        )
        if resp.status_code != 200:
            return {"ozet": f"(Gemini API hatası: HTTP {resp.status_code})", "yon": "NOTR", "gerekce": resp.text[:150]}

        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
        parsed = json.loads(text)

        yon = str(parsed.get("yon", "NOTR")).strip().upper()
        if yon not in ("LONG", "SHORT", "NOTR"):
            yon = "NOTR"
        return {
            "ozet": str(parsed.get("ozet", "")).strip(),
            "yon": yon,
            "gerekce": str(parsed.get("gerekce", "")).strip(),
        }
    except Exception as e:
        return {"ozet": f"(Özetleme sırasında hata: {e})", "yon": "NOTR", "gerekce": ""}


# ─────────────────────────────────────────────────────────────────
# PANEL — OTOMATİK YENİLENEN HABER LİSTESİ
# ─────────────────────────────────────────────────────────────────
@st.fragment(run_every=refresh_seconds)
def render_news_panel():
    currencies = coin_filter.strip().upper().replace(" ", "")
    filter_key = FILTER_MAP[news_filter_label]

    posts, err = fetch_cryptopanic_news(CRYPTOPANIC_API_KEY, currencies, filter_key, max_news)

    if err:
        st.error(err)
        return
    if not posts:
        st.info("Şu anda gösterilecek haber bulunamadı.")
        return

    for post in posts:
        post_id = post.get("id")
        title = post.get("title", "") or ""
        source = ((post.get("source") or {}).get("title")) or "Bilinmeyen kaynak"
        url = post.get("url", "") or "#"
        published = format_relative_time(post.get("published_at", ""))
        description = post.get("body") or post.get("description") or title

        if GEMINI_API_KEY and post_id not in st.session_state.news_summary_cache:
            st.session_state.news_summary_cache[post_id] = summarize_news_with_gemini(
                GEMINI_API_KEY, title, description, source
            )

        summary = st.session_state.news_summary_cache.get(post_id) or {
            "ozet": "(Gemini API anahtarı tanımlı olmadığı için özet oluşturulamadı.)",
            "yon": "NOTR",
            "gerekce": "",
        }

        badge_color = {"LONG": "#22c55e", "SHORT": "#ef4444", "NOTR": "#787b86"}[summary["yon"]]
        badge_text = {
            "LONG": "LONG EĞİLİMİ OLASI",
            "SHORT": "SHORT EĞİLİMİ OLASI",
            "NOTR": "NÖTR / BELİRSİZ",
        }[summary["yon"]]

        safe_title = html.escape(title)
        safe_source = html.escape(source)
        safe_published = html.escape(published)
        safe_summary = html.escape(summary["ozet"])
        safe_gerekce = html.escape(summary["gerekce"])
        safe_url = url if url.startswith(("http://", "https://")) else "#"

        st.markdown(f"""
        <div style="background:#131722;border:1px solid #2a2e39;border-left:4px solid {badge_color};
                    border-radius:8px;padding:14px 16px;margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;flex-wrap:wrap;gap:6px;">
            <span style="color:#787b86;font-size:0.75rem;">{safe_source} · {safe_published}</span>
            <span style="background:{badge_color}22;color:{badge_color};padding:2px 10px;border-radius:12px;
                         font-size:0.7rem;font-weight:800;">{badge_text}</span>
          </div>
          <a href="{safe_url}" target="_blank" rel="noopener noreferrer"
             style="color:#d1d4dc;font-weight:700;font-size:0.95rem;text-decoration:none;">{safe_title}</a>
          <p style="color:#b2b5be;font-size:0.85rem;margin-top:8px;margin-bottom:4px;">{safe_summary}</p>
          <p style="color:#787b86;font-size:0.75rem;font-style:italic;margin:0;">
            Gerekçe: {safe_gerekce} — bu kesin bir tahmin değildir, olasılık değerlendirmesidir.
          </p>
        </div>
        """, unsafe_allow_html=True)


render_news_panel()
