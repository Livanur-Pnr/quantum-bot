"""
📰 Haber Analizi — Ücretsiz kripto haber RSS akışları (Cointelegraph, Decrypt) +
Gemini AI ile Türkçe özet ve LONG/SHORT eğilim değerlendirmesi.

NOT: CryptoPanic'in ücretsiz API katmanı kaldırıldığı için (yalnızca ücretli
Growth/Enterprise planları kaldı) haber kaynağı olarak, hiçbir API anahtarı
gerektirmeyen resmi RSS akışları kullanılıyor.

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
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Haber Analizi",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = "gemini-flash-lite-latest"
GEMINI_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# Ücretsiz, anahtar gerektirmeyen resmi RSS akışları.
RSS_FEEDS = {
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
}

PREMIUM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
    .stApp, [data-testid="stAppViewContainer"] { background-color: #eaf6f4 !important; color: #0f2b2e; font-family: 'Inter', sans-serif; }
    section[data-testid="stSidebar"] { background-color: #eaf6f4 !important; border-right: 1px solid rgba(15,43,46,0.08); }
    [data-testid="stHeader"] { background: transparent !important; }
    footer, [data-testid="stStatusWidget"], [data-testid="stAppRunningIndicator"],
    [data-testid="stDecoration"], #stDecoration,
    [data-testid="stAppDeployButton"], .stDeployButton { display: none !important; }

    [data-testid="stAppViewContainer"] h1, [data-testid="stAppViewContainer"] h2, [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p { color: #0f2b2e; }

    /* --- Nav pilleri (diğer sayfalarla tutarlı) --- */
    [data-testid="stSidebarNavLink"] { border-radius:999px !important; margin-bottom:6px !important; }
    [data-testid="stSidebarNavLink"] p, [data-testid="stSidebarNavLink"] span { color:#0f2b2e !important; }
    [data-testid="stSidebarNavLink"]:not([aria-current="page"]) { background:#ffffff !important; }
    [data-testid="stSidebarNavLink"][aria-current="page"] { background:linear-gradient(135deg, #2dd4bf, #14b8a6) !important; }
    [data-testid="stSidebarNavLink"][aria-current="page"] p, [data-testid="stSidebarNavLink"][aria-current="page"] span { color:#ffffff !important; }

    /* --- Sidebar kartı & form alanları --- */
    .sb-section-title { font-size:16px; font-weight:900; color:#0f2b2e; margin:14px 0 10px 0; }
    .st-key-haber_kontrol_card { background:#ffffff; border-radius:18px; padding:16px 16px 8px 16px; box-shadow:0 4px 14px rgba(15,43,46,0.06); margin-bottom:14px; }
    .st-key-haber_kontrol_card [data-testid="stWidgetLabel"] p { font-weight:800 !important; color:#0f2b2e !important; font-size:13px !important; }
    .st-key-haber_kontrol_card [data-testid="stTextInput"] > div > div {
        background:#f1f5f9 !important; border-radius:999px !important; border:1.5px solid transparent !important;
    }
    .st-key-haber_kontrol_card [data-testid="stTextInput"] input { color:#0f2b2e !important; }
    .st-key-haber_kontrol_card [data-testid="stTextInput"] input:focus { border-color:#14b8a6 !important; }
    .st-key-haber_kontrol_card [data-testid="stTextInput"] > div > div:focus-within { border-color:#14b8a6 !important; background:#ffffff !important; box-shadow:0 0 0 2px rgba(20,184,166,0.15) !important; }
    [data-testid="stSidebar"] [data-testid="stSliderTickBarMin"], [data-testid="stSidebar"] [data-testid="stSliderTickBarMax"] { color:#5f7d7a !important; }
    [data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] { background-color:#0e7490 !important; }
    [data-testid="stSidebar"] [data-baseweb="slider"] div[style*="background-color: rgb(255, 75, 75)"] { background:#14b8a6 !important; }

    .gemini-status-pill { display:flex; align-items:center; justify-content:center; gap:8px; padding:10px 16px; border-radius:999px; font-size:13px; font-weight:800; margin-top:14px; }
    .gemini-status-pill.ok { background:linear-gradient(135deg, #14b8a6, #0e7490); color:#ffffff; }
    .gemini-status-pill.missing { background:#fff7ed; border:1px solid #fdba74; color:#9a3412; }

    .news-header-bar { background:#ffffff; border:6px solid #14b8a6; border-radius:20px; padding:18px 24px; margin-bottom:6px; display:flex; align-items:center; gap:14px; }
    .news-header-icon { background:#ffffff; border:3px solid #14b8a6; color:#14b8a6; width:44px; height:44px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:20px; flex-shrink:0; }
    .news-header-title { font-size:30px; font-weight:900; color:#14b8a6; margin:0; }
</style>
"""
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

st.markdown("""
<div class="news-header-bar">
    <div class="news-header-icon">📰</div>
    <div class="news-header-title">Haber Analizi</div>
</div>
""", unsafe_allow_html=True)
st.caption("Cointelegraph & Decrypt (ücretsiz RSS) → Gemini AI ile Türkçe özet ve olası LONG/SHORT eğilim değerlendirmesi.")

if "news_summary_cache" not in st.session_state:
    st.session_state.news_summary_cache = {}

# ─────────────────────────────────────────────────────────────────
# SIDEBAR — KONTROLLER
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    with st.container(key="haber_kontrol_card"):
        st.markdown("""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
            <span style="font-size:19px; font-weight:900; color:#0f2b2e;">Haber Kontrolleri</span>
            <span style="width:20px; height:20px; border-radius:50%; border:1.5px solid #94a3b8; color:#5f7d7a;
                         font-size:11px; font-weight:800; display:flex; align-items:center; justify-content:center;">?</span>
        </div>
        """, unsafe_allow_html=True)
        coin_filter = st.text_input(
            "Coin Filtrele (örn: BTC, ETH)",
            value="",
            placeholder="🔍  Örn: BTC, ETH",
            help="Boş bırakırsan tüm kripto haberleri gösterilir. Başlık/özet içinde geçen coin adına göre filtrelenir.",
        )
        max_news = st.slider("Gösterilecek haber sayısı", min_value=3, max_value=15, value=8, help="Bir seferde en fazla kaç haber gösterilsin.")
        refresh_seconds = st.slider("Otomatik yenileme (saniye)", min_value=60, max_value=600, value=180, step=30, help="Haber listesi kaç saniyede bir otomatik yenilensin.")

    st.caption("Haber kaynağı: " + ", ".join(RSS_FEEDS.keys()))
    if not GEMINI_API_KEY:
        st.markdown("<div class='gemini-status-pill missing'>⚠️ Gemini API anahtarı eksik</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='gemini-status-pill ok'>✅ Gemini API</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# RSS — HABER ÇEKİMİ (ücretsiz, anahtarsız)
# ─────────────────────────────────────────────────────────────────
def _strip_html(raw: str) -> str:
    return re.sub(r"<[^>]+>", " ", raw or "").strip()


@st.cache_data(ttl=90, show_spinner=False)
def fetch_rss_news(currencies: str, limit: int):
    currency_list = [c.strip().upper() for c in currencies.split(",") if c.strip()] if currencies else []
    all_items = []
    errors = []

    for source_name, feed_url in RSS_FEEDS.items():
        try:
            resp = requests.get(feed_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
            for item in root.findall(".//item"):
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub_date = (item.findtext("pubDate") or "").strip()
                description = _strip_html(item.findtext("description") or "")

                if currency_list:
                    haystack = f"{title} {description}".upper()
                    if not any(c in haystack for c in currency_list):
                        continue

                all_items.append({
                    "id": link or title,
                    "title": title,
                    "url": link,
                    "source": source_name,
                    "published_at": pub_date,
                    "description": description[:600],
                })
        except Exception as e:
            errors.append(f"{source_name}: {e}")

    def _sort_key(it):
        try:
            return parsedate_to_datetime(it["published_at"])
        except Exception:
            return datetime.min.replace(tzinfo=timezone.utc)

    all_items.sort(key=_sort_key, reverse=True)

    err_msg = "; ".join(errors) if errors and not all_items else None
    return all_items[:limit], err_msg


def format_relative_time(date_str: str) -> str:
    dt = None
    try:
        dt = parsedate_to_datetime(date_str)
    except Exception:
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except Exception:
            return ""
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
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


# ─────────────────────────────────────────────────────────────────
# GEMINI — TÜRKÇE ÖZET + YÖN DEĞERLENDİRMESİ
# ─────────────────────────────────────────────────────────────────
def summarize_news_with_gemini(api_key: str, title: str, description: str, source: str) -> dict:
    prompt = f"""Aşağıda bir kripto para haberi var. Görevlerin:

1. Haberin içindeki ÖNEMLİ bilgileri, haber hangi dilde olursa olsun, TÜRKÇE olarak 2-3 kısa cümlelik bir özet halinde ver.
2. Bu haberin piyasada LONG (yükseliş) mu yoksa SHORT (düşüş) yönünde mi bir baskı yaratma ihtimalinin daha yüksek olduğunu değerlendir. Haber nötr, belirsiz veya etkisizse "NOTR" de. Bu KESİN bir tahmin değildir, sadece bir olasılık değerlendirmesidir — asla kesinlik iddia etme, emin olmadığında NOTR de.

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
            return {"ozet": f"(Gemini API hatası: HTTP {resp.status_code})", "yon": "NOTR", "gerekce": resp.text[:150], "ok": False}

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
            "ok": True,
        }
    except Exception as e:
        return {"ozet": f"(Özetleme sırasında hata: {e})", "yon": "NOTR", "gerekce": "", "ok": False}


# ─────────────────────────────────────────────────────────────────
# PANEL — OTOMATİK YENİLENEN HABER LİSTESİ
# ─────────────────────────────────────────────────────────────────
@st.fragment(run_every=refresh_seconds)
def render_news_panel():
    posts, err = fetch_rss_news(coin_filter.strip(), max_news)

    if err:
        st.error(err)
    if not posts:
        st.info("Şu anda gösterilecek haber bulunamadı.")
        return

    for post in posts:
        post_id = post["id"]
        title = post["title"]
        source = post["source"]
        url = post["url"]
        published = format_relative_time(post["published_at"])
        description = post["description"] or title

        cached = st.session_state.news_summary_cache.get(post_id)
        if GEMINI_API_KEY and (cached is None or not cached.get("ok")):
            st.session_state.news_summary_cache[post_id] = summarize_news_with_gemini(
                GEMINI_API_KEY, title, description, source
            )

        summary = st.session_state.news_summary_cache.get(post_id) or {
            "ozet": "(Gemini API anahtarı tanımlı olmadığı için özet oluşturulamadı.)",
            "yon": "NOTR",
            "gerekce": "",
        }

        badge_color = {"LONG": "#16a34a", "SHORT": "#dc2626", "NOTR": "#5f7d7a"}[summary["yon"]]
        badge_bg = {"LONG": "#dcfce7", "SHORT": "#fee2e2", "NOTR": "#f1f5f9"}[summary["yon"]]
        card_tint = {"LONG": "rgba(34,197,94,0.07)", "SHORT": "rgba(239,68,68,0.07)", "NOTR": "rgba(100,116,139,0.05)"}[summary["yon"]]
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
        <div style="background:linear-gradient(90deg, {card_tint} 0%, #ffffff 55%);border-left:4px solid {badge_color};box-shadow:0 4px 14px rgba(15,43,46,0.06);
                    border-radius:14px;padding:14px 16px;margin-bottom:12px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;flex-wrap:wrap;gap:6px;">
            <span style="color:#5f7d7a;font-size:0.75rem;">{safe_source} · {safe_published}</span>
            <span style="background:{badge_bg};color:{badge_color};padding:2px 10px;border-radius:12px;
                         font-size:0.7rem;font-weight:800;">{badge_text}</span>
          </div>
          <a href="{safe_url}" target="_blank" rel="noopener noreferrer"
             style="color:#0f2b2e;font-weight:700;font-size:0.95rem;text-decoration:none;">{safe_title}</a>
          <p style="color:#334155;font-size:0.85rem;margin-top:8px;margin-bottom:4px;">{safe_summary}</p>
          <p style="color:#5f7d7a;font-size:0.75rem;font-style:italic;margin:0;">
            Gerekçe: {safe_gerekce} — bu kesin bir tahmin değildir, olasılık değerlendirmesidir.
          </p>
        </div>
        """, unsafe_allow_html=True)


render_news_panel()
