"""
╔══════════════════════════════════════════════════════════════════════════╗
║        🤖 GAINZALGO AI PREDICTIVE BOT v15.1 — INDEX-SAFE EDITION       ║
║                                                                        ║
║  - MULTIINDEX FIX: yfinance'ın sütun yapısı düzleştirilir (flatten).   ║
║  - DATE_STR SYNC: Kripto ve Makro veriler %Y-%m-%d string ile eşleşir. ║
║  - .MAP() MERGE: Ana tablonun Orijinal Timestamp indexi ASLA bozulmaz. ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st

import sys
import codecs
try:
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if sys.stderr.encoding.lower() != 'utf-8':
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
except Exception:
    pass

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ccxt
import pandas as pd
import numpy as np
import concurrent.futures
from datetime import datetime
import time
from dotenv import load_dotenv
import http.server
import threading
import socket
from urllib.parse import urlparse, parse_qs

try:
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.feature_selection import SelectKBest, f_classif
    import requests
    import yfinance as yf
except ImportError:
    st.error("Lütfen terminalden şu komutu çalıştırın: pip install scikit-learn requests yfinance")
    st.stop()

# ─────────────────────────────────────────────────────────────────
# 0. SAYFA YAPILANDIRMASI, TEMA VE SESSION
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Canlı Gösterge",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded", 
)

load_dotenv()

PREMIUM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
    .stApp { background-color: #eaf6f4; color: #0f2b2e; font-family: 'Inter', sans-serif; }
    [data-testid="stAppViewContainer"] { background-color: #eaf6f4; }
    section[data-testid="stSidebar"] { background-color: #eaf6f4 !important; border-right: 1px solid rgba(15,43,46,0.08); }
    h1, h2, h3, h4, h5, h6, p, span, label, div { color: #0f2b2e; }

    .metric-card {
        background: #ffffff; border: none; border-radius: 16px;
        padding: 14px 16px; margin-bottom: 10px; min-height: 85px;
        display: flex; flex-direction: column; justify-content: center;
        box-shadow: 0 4px 14px rgba(15,43,46,0.06);
    }
    .metric-card h4 { color: #5f7d7a; font-size: 0.65rem; margin: 0 0 6px 0; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card .value { font-size: 1.15rem; font-weight: 800; margin: 0; }

    .green { color: #16a34a; } .red { color: #dc2626; } .blue { color: #0e7490; } .white { color: #0f2b2e; }

    /* --- Sidebar: teal/turuncu gradyanlı "Vivid Fintech" teması (Analiz Tahmini sayfasıyla tutarlı) --- */
    [data-testid="stSidebarNavLink"] { border-radius:999px !important; margin-bottom:6px !important; }
    [data-testid="stSidebarNavLink"] p, [data-testid="stSidebarNavLink"] span { color:#0f2b2e !important; }
    [data-testid="stSidebarNavLink"]:not([aria-current="page"]) { background:#ffffff !important; }
    [data-testid="stSidebarNavLink"][aria-current="page"] { background:linear-gradient(135deg, #2dd4bf, #14b8a6) !important; }
    [data-testid="stSidebarNavLink"][aria-current="page"] p, [data-testid="stSidebarNavLink"][aria-current="page"] span { color:#ffffff !important; }

    [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div,
    [data-testid="stSidebar"] [data-testid="stTextInput"] > div > div,
    [data-testid="stSidebar"] [data-testid="stNumberInput"] > div > div,
    [data-testid="stSidebar"] [data-testid="stSlider"] {
        background: rgba(20,184,166,0.10) !important;
        border-radius: 999px !important;
        border: 1px solid rgba(15,43,46,0.10) !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: #ffffff !important;
        border-radius: 16px !important;
        border: 1px solid rgba(15,43,46,0.08) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 { font-size:16px; font-weight:900; color:#0f2b2e; }

    /* --- Yenilemede yanip sonme (flas) fix: fragment run_every yenilemesinde
       Streamlit'in "eskimis/stale" iceriği soluklastirip iskelet animasyonu gostermesi
       goze batan bir yanip sonme etkisi yaratiyordu. Asagidaki kurallar bu gecici
       gorsel durumlari devre disi birakiyor. --- */
    [data-testid="stHeader"] { background: transparent !important; }
    footer,
    [data-testid="stStatusWidget"], [data-testid="stAppRunningIndicator"],
    [data-testid="stDecoration"], #stDecoration,
    [data-testid="stAppDeployButton"], .stDeployButton { display: none !important; }
    [data-testid="stSpinner"], .stSpinner { display: none !important; }

    [data-testid="stSkeleton"], .stSkeleton, [class*="skeleton" i] { animation: none !important; opacity: 1 !important; }
    [data-stale="true"] { opacity: 1 !important; transition: none !important; filter: none !important; }
    .element-container, [data-testid="stElementContainer"],
    [data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"] { transition: none !important; }

    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { animation-duration: .001ms !important; transition-duration: .001ms !important; }
    }
</style>
"""
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# 1. MAKRO VERİ ÇEKİMİ (SAFE DATE SYNC)
# ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_macro_data() -> tuple:
    def get_fg():
        try:
            resp = requests.get("https://api.alternative.me/fng/?limit=300", timeout=30)
            df_fg = pd.DataFrame(resp.json()['data'])
            # Tarihi %Y-%m-%d stringine çevir
            df_fg['timestamp'] = pd.to_datetime(df_fg['timestamp'].astype(int), unit='s', utc=True)
            df_fg['date_str'] = df_fg['timestamp'].dt.strftime('%Y-%m-%d')
            df_fg['fear_greed'] = df_fg['value'].astype(float)
            return df_fg[['date_str', 'fear_greed']]
        except Exception:
            return pd.DataFrame()

    def get_spy():
        try:
            import yfinance as yf
            spy = yf.download("SPY", period="1y", interval="1d", progress=False, timeout=30)
            if spy.empty: return pd.DataFrame()
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            spy = spy.reset_index() 
            date_col = 'Date' if 'Date' in spy.columns else ('Datetime' if 'Datetime' in spy.columns else spy.columns[0])
            spy['date_str'] = pd.to_datetime(spy[date_col], utc=True).dt.strftime('%Y-%m-%d')
            spy['spy'] = spy['Close'].astype(float)
            return spy[['date_str', 'spy']]
        except Exception:
            return pd.DataFrame()

    def get_dxy():
        try:
            dxy = yf.download("DX-Y.NYB", period="1y", interval="1d", progress=False, timeout=30)
            if dxy.empty: return pd.DataFrame()
            
            # KURAL 1: Yfinance MultiIndex hatasını çözer (Flatten)
            if isinstance(dxy.columns, pd.MultiIndex):
                dxy.columns = dxy.columns.get_level_values(0)
                
            # Index'i sütuna çevir
            dxy = dxy.reset_index() 
            date_col = 'Date' if 'Date' in dxy.columns else ('Datetime' if 'Datetime' in dxy.columns else dxy.columns[0])
            
            # Tarihi %Y-%m-%d stringine çevir
            dxy['date_str'] = pd.to_datetime(dxy[date_col], utc=True).dt.strftime('%Y-%m-%d')
            dxy['dxy'] = dxy['Close'].astype(float)
            return dxy[['date_str', 'dxy']]
        except Exception:
            return pd.DataFrame()

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        f_fg = executor.submit(get_fg)
        f_dxy = executor.submit(get_dxy)
        f_spy = executor.submit(get_spy)
        
        try: fg_df = f_fg.result(timeout=35)
        except Exception: fg_df = pd.DataFrame()
        
        try: dxy_df = f_dxy.result(timeout=35)
        except Exception: dxy_df = pd.DataFrame()
        try: spy_df = f_spy.result(timeout=35)
        except Exception: spy_df = pd.DataFrame()
        
    return fg_df, dxy_df, spy_df

# ─────────────────────────────────────────────────────────────────
# 2. KRİPTO VERİ ÇEKİMİ (SEÇİLEN BORSA - RATE LIMIT KORUMALI)
# ─────────────────────────────────────────────────────────────────
# Desteklenen borsalar: MEXC (varsayilan) ve Binance. Ikisi de ccxt-unified ayni sozlesme
# sembol formatini ("BASE/USDT:USDT") ve ayni kimlik dogrulama alanlarini kullaniyor.
SUPPORTED_EXCHANGES = {"MEXC": "mexc", "Binance": "binance"}

@st.cache_resource(ttl=1800, show_spinner=False)
def get_public_exchange_client(exchange_id: str = "mexc"):
    """Tek, paylasilan ve piyasa listesi onceden yuklenmis borsa istemcisi. exchange_id'ye gore
    ayri ayri onbelleklenir.

    ONEMLI PERFORMANS NOTU: Her fetch_crypto_data cagrisinda taze bir ccxt nesnesi
    olusturulursa, ccxt ilk fetch_ohlcv cagrisinda otomatik load_markets() calistirir
    (~3000 spot+vadeli sozlesmeyi indirir). Bu, 5 saniyede bir yenilenen canli panelde
    her seferinde tekrarlanip yavas yuklemeye sebep oluyordu.
    """
    klass = getattr(ccxt, exchange_id)
    ex = klass({"enableRateLimit": True, "timeout": 30000, "options": {"defaultType": "swap"}})
    ex.load_markets()
    return ex

@st.cache_data(ttl=5, show_spinner=False)
def fetch_crypto_data(exchange_id: str, symbol: str, timeframe: str, limit: int = 1500) -> pd.DataFrame:
    try:
        ex = get_public_exchange_client(exchange_id)

        import numpy as np
        
        # ONEMLI: "BTC/USDT" gibi ":USDT" vadeli sonekini icermeyen semboller ccxt-mexc'te
        # dogrudan SPOT piyasasina cozumlenir (ayni ada sahip ayri bir spot market oldugu icin).
        # Bu yuzden sadece "/" kontrolu yetersizdir; her zaman vadeli (":USDT") formatina cevrilir.
        if ":" not in symbol:
            base = symbol.split("/")[0].replace("USDT", "")
            symbol = f"{base}/USDT:USDT"

        # ONEMLI: MEXC API'si arada bir (agir yuk/gecici ag aksakligi) tek seferlik basarisiz
        # yanit donebiliyor. Once hemen hata gostermek yerine kisa bir bekleme ile 2 kez daha
        # deneyip, sadece gercekten israrli bir sorun varsa hata gosteriyoruz. Bu, kullaniciya
        # her yenilemede gereksiz kirmizi hata kutusu cikmasini onluyor.
        ohlcv = None
        last_err = None
        for attempt in range(3):
            try:
                ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
                break
            except Exception as e:
                last_err = e
                if attempt < 2:
                    time.sleep(1.5)
        if ohlcv is None:
            raise last_err

        try: ob = ex.fetch_order_book(symbol, limit=20)
        except Exception: ob = None

        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        
        # MEXC proxy kurumsal veriler
        df['oi'] = (df['close'] - df['open']) / df['open'] * (df['volume'] / (df['volume'].mean()+1e-9)) * 10
        df['taker_buy_vol'] = np.where(df['close'] > df['open'], df['volume'] * 0.6, df['volume'] * 0.4)
        df['taker_sell_vol'] = np.where(df['close'] < df['open'], df['volume'] * 0.6, df['volume'] * 0.4)
        
        try:
            funding = ex.fetch_funding_rate_history(symbol, limit=200)
            df_fund = pd.DataFrame(funding)
            df_fund['timestamp'] = pd.to_datetime(df_fund['timestamp'], unit='ms', utc=True)
            df_fund.set_index('timestamp', inplace=True)
            df = df.join(df_fund['fundingRate'].astype(float).rename('funding_rate'), how='left')
            df['funding_rate'] = df['funding_rate'].ffill().fillna(0.0)
        except Exception:
            df['funding_rate'] = 0.0
            
        df['ob_imbalance'] = 1.0
        if ob:
            bids, asks = sum([b[1] for b in ob['bids']]), sum([a[1] for a in ob['asks']])
            df.loc[df.index[-1], 'ob_imbalance'] = bids / asks if asks > 0 else 1.0
            
        return df.astype(float)
        
    except Exception as e:
        ex_label = next((k for k, v in SUPPORTED_EXCHANGES.items() if v == exchange_id), exchange_id.upper())
        st.error(f"{ex_label} verisi çekilirken teknik bir hata oluştu: {str(e)}")
        return pd.DataFrame()

# Yardımcı İndikatör Fonksiyonları
def compute_ema(series, period): return series.ewm(span=period, adjust=False).mean()
def compute_atr(df, period=14):
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift(1)).abs(), (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()
def compute_adx(df, period=14):
    alpha = 1 / period
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift(1)).abs(), (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    up, down = df["high"] - df["high"].shift(1), df["low"].shift(1) - df["low"]
    pdm = np.where((up > down) & (up > 0), up, 0.0)
    ndm = np.where((down > up) & (down > 0), down, 0.0)
    pdi = 100 * (pd.Series(pdm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / atr)
    ndi = 100 * (pd.Series(ndm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / atr)
    dx = 100 * (abs(pdi - ndi) / (pdi + ndi).replace(0, np.nan)).fillna(0)
    return dx.ewm(alpha=alpha, adjust=False).mean()

# ─────────────────────────────────────────────────────────────────
# 3. YÜKSEK DİRENÇLİ FEATURE ENGINEERING & ML EĞİTİMİ
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(ttl=300, show_spinner=False)
def _cached_train_quantum_ai(symbol_tf: str, _train_df: pd.DataFrame, _feature_cols: list):
    # ONEMLI: Parametre adlari basinda "_" olmasi BILINCLI: Streamlit cache_resource, alt cizgiyle
    # baslayan parametreleri hash'lemeden atlar. Aksi halde her cagrida (birkac saniyede bir) koca
    # DataFrame'in tamami hashlenir; bu da altta zaten var olan ucuz "symbol_tf" string anahtarini
    # (yorum: "HIZLANDIRICI MANTIK") anlamsizlastirip gereksiz yavaslamaya sebep olurdu.
    train_df, feature_cols = _train_df, _feature_cols
    # Guvenlik agi: herhangi bir feature'da olusabilecek sonsuz (inf) deger sklearn'i cokertir.
    train_df = train_df.copy()
    train_df[feature_cols] = train_df[feature_cols].replace([np.inf, -np.inf], 0.0)

    selector = SelectKBest(score_func=f_classif, k=min(12, len(feature_cols)))
    selector.fit(train_df[feature_cols], train_df['target'])
    final_features = [f for f, selected in zip(feature_cols, selector.get_support()) if selected]

    X_train = train_df[final_features]
    y_train = train_df['target']

    n_samples = len(X_train)
    sample_weights = np.exp(np.linspace(-1.2, 0, n_samples))

    rf = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=4, class_weight='balanced', random_state=42)
    et = ExtraTreesClassifier(n_estimators=40, max_depth=5, min_samples_split=4, class_weight='balanced', random_state=42)
    hgb = HistGradientBoostingClassifier(max_iter=40, max_depth=4, random_state=42)

    ensemble = VotingClassifier(estimators=[('rf', rf), ('et', et), ('hgb', hgb)], voting='soft')
    ensemble.fit(X_train, y_train, sample_weight=sample_weights)

    return ensemble, final_features

# Zaman dilimine gore, BTC referans fiyatina (~$77.200) gore kalibre edilmis, fiyata ORANTILI
# (yuzde bazli) SL/TP mesafe sinirlari. Yuzde bazli oldugu icin herhangi bir coinde (ucuz veya
# pahali farketmeksizin) esdeger sikilikte, mantikli bir mesafe uretir - sabit dolar deger
# kullanilsaydi ucuz coinlerde (orn. PEPE) anlamsiz/imkansiz sonuclar cikardi.
TIMEFRAME_RISK_BOUNDS_PCT = {
    "1m":  {"sl_min": 0.00259, "sl_max": 0.00518, "tp_min": 0.00389, "tp_max": 0.00648},
    "15m": {"sl_min": 0.00389, "sl_max": 0.00648, "tp_min": 0.00648, "tp_max": 0.01295},
    "1h":  {"sl_min": 0.01295, "sl_max": 0.01943, "tp_min": 0.01295, "tp_max": 0.01943},
    "4h":  {"sl_min": 0.01295, "sl_max": 0.01943, "tp_min": 0.01295, "tp_max": 0.01943},
}

def clamp_tp_sl_dist(raw_dist: float, price: float, timeframe: str, kind: str) -> float:
    """raw_dist (ATR*carpan mesafesi), zaman dilimine gore tanimli fiyata orantili sinirin
    disina KESINLIKLE cikamaz. kind: 'tp' veya 'sl'."""
    bounds = TIMEFRAME_RISK_BOUNDS_PCT.get(timeframe)
    if not bounds:
        return raw_dist
    lo, hi = (bounds["tp_min"], bounds["tp_max"]) if kind == "tp" else (bounds["sl_min"], bounds["sl_max"])
    return float(np.clip(raw_dist, price * lo, price * hi))

def train_and_predict_ai(df: pd.DataFrame, target_candles: int, threshold: float, tp_m: float, sl_m: float, timeframe: str = ""):
    """
    Yapay Zeka Botu ile %100 Senkronize Edilmis ML Motoru
    """
    feature_cols = [
        'rsi', 'macd_hist', 'macd_hist_slope', 'dist_ema50', 'roc', 'adx',
        'ema_trend', 'oi_change_pct', 'volume_delta_trend', 'ob_pressure',
        'fear_greed', 'dxy', 'funding_rate'
    ]
    
    # Indikatörleri hesapla
    delta = df['close'].diff()
    gain, loss = (delta.where(delta > 0, 0)).rolling(14).mean(), (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + (gain / loss.replace(0, 0.001))))
    ema12, ema26, ema50 = compute_ema(df['close'], 12), compute_ema(df['close'], 26), compute_ema(df['close'], 50)
    macd = ema12 - ema26
    df['macd_hist'] = macd - compute_ema(macd, 9)
    df['macd_hist_slope'] = df['macd_hist'].diff() 
    df['dist_ema50'] = (df['close'] - ema50) / ema50 * 100
    df['roc'] = df['close'].pct_change(periods=5) * 100
    df['atr'] = compute_atr(df, 14)
    df['adx'] = compute_adx(df, 14)
    ema9, ema21 = compute_ema(df['close'], 9), compute_ema(df['close'], 21)
    df['ema_trend'] = np.where(ema9 > ema21, 1, -1)
    
    # df['oi'] tam 0 olabilen bir deger oldugundan (ornegin close==open), pct_change() burada
    # sonsuz (inf) uretebilir; .fillna(0) yalnizca NaN'i yakalar, inf'i degil -> ML egitimini cokertir.
    df['oi_change_pct'] = df['oi'].pct_change(periods=5).replace([np.inf, -np.inf], 0.0).fillna(0) * 100
    df['volume_delta'] = df['taker_buy_vol'] - df['taker_sell_vol']
    df['volume_delta_trend'] = df['volume_delta'].rolling(window=5).mean().fillna(0)
    df['ob_pressure'] = np.where(df['volume_delta'] > 0, 1, -1)
    df.loc[df.index[-1], 'ob_pressure'] = 1 if df['ob_imbalance'].iloc[-1] > 1.2 else (-1 if df['ob_imbalance'].iloc[-1] < 0.8 else 0)
    
    if len(df) < 60:
        return df, 0.0, []
        
    targets = np.zeros(len(df))
    closes, highs, lows, atrs = df['close'].values, df['high'].values, df['low'].values, df['atr'].values

    # ONEMLI: Ham "atr * carpan" mesafesi, zaman dilimine gore anlamsiz sekilde genis
    # (ornegin 1 dakikalik veride binlerce dolarlik SL/TP) cikabiliyordu. Zaman dilimine ozel
    # yuzde bazli sinirlar tanimliysa, mesafe bu araligin disina KESINLIKLE cikamaz.
    tf_bounds = TIMEFRAME_RISK_BOUNDS_PCT.get(timeframe)
    raw_tp_dist = atrs * tp_m
    raw_sl_dist = atrs * sl_m
    if tf_bounds:
        tp_dist_arr = np.clip(raw_tp_dist, closes * tf_bounds["tp_min"], closes * tf_bounds["tp_max"])
        sl_dist_arr = np.clip(raw_sl_dist, closes * tf_bounds["sl_min"], closes * tf_bounds["sl_max"])
    else:
        tp_dist_arr = raw_tp_dist
        sl_dist_arr = raw_sl_dist

    for i in range(len(df) - target_candles):
        entry = closes[i]
        tp_long, sl_long = entry + tp_dist_arr[i], entry - sl_dist_arr[i]
        tp_short, sl_short = entry - tp_dist_arr[i], entry + sl_dist_arr[i]
        
        l_suc, s_suc = False, False
        for j in range(i + 1, i + 1 + target_candles):
            if lows[j] <= sl_long: break 
            if highs[j] >= tp_long: l_suc = True; break
            
        for j in range(i + 1, i + 1 + target_candles):
            if highs[j] >= sl_short: break
            if lows[j] <= tp_short: s_suc = True; break
            
        if l_suc and not s_suc: targets[i] = 1       
        elif s_suc and not l_suc: targets[i] = -1    
        
    df['target'] = targets
    
    temp_df = df.copy()
    temp_df['is_valid'] = ~df['high'].shift(-target_candles).isna()
    train_df = temp_df.dropna(subset=feature_cols).copy()
    train_df = train_df[train_df['is_valid'] == True]
    
    if len(train_df) < 30 or len(np.unique(train_df['target'])) < 2:
        return df, 0.0, []
    
    # 💥 HIZLANDIRICI MANTIK: Sadece kapanmış son mumu referans alarak hash'le.
    symbol_tf_hash = str(len(train_df)) + "_" + str(train_df['close'].iloc[-1])
    
    # AI Bot'taki AYNİ önbelleklenmiş modeli kullan!
    ensemble, final_features = _cached_train_quantum_ai(symbol_tf_hash, train_df, feature_cols)
    
    # Tüm geçmiş veri için olasılıkları hesapla (Grafik için)
    X_all = df[final_features].fillna(0)
    probs = ensemble.predict_proba(X_all)
    classes = list(ensemble.classes_)
    
    df['prob_long'] = probs[:, classes.index(1)] * 100 if 1 in classes else np.zeros(len(probs))
    df['prob_short'] = probs[:, classes.index(-1)] * 100 if -1 in classes else np.zeros(len(probs))
    
    # Sinyal mantığı
    in_pos = False
    pos_type = None
    tp, sl = np.nan, np.nan
    ai_signals, trade_status, active_tps, active_sls, is_entries = [], [], [], [], []

    threshold = threshold / 100.0  # Normalize threshold

    # ONEMLI - KARARLILIK: df'nin SON satiri henuz KAPANMAMIS (olusmakta olan) canli mumdur;
    # onun RSI/MACD/OB gibi featurelari her fiyat tikinde (birkaç saniyede bir) degisebiliyor,
    # bu da AI olasiliginin (prob_long/prob_short) saniyeler icinde 20-30 puan sicramasina yol
    # aciyordu (kullanici bunu fark etti: "tahminler cok sik degisiyor, kararli olmasini
    # istiyorum"). Bu yuzden SADECE KAPANMIS mumlar yeni bir "giris" (is_entry) tetikleyebilir;
    # son (kapanmamis) mum icin asla yeni giris isaretlenmez - boylece bir sinyal, ancak mum
    # gercekten kapandiktan (yani secili zaman dilimi kadar - 1dk/15dk/... - sure gectikten)
    # SONRA kalici olarak kayit defterine/grafige eklenir; daha once goruntulenip sonra "kaybolan"
    # hayalet sinyaller olusmaz.
    last_closed_idx = len(df) - 2

    for i in range(len(df)):
        entered_now = False
        if in_pos:
            if pos_type == 'LONG' and (df['low'].iloc[i] <= sl or df['high'].iloc[i] >= tp): in_pos = False
            elif pos_type == 'SHORT' and (df['high'].iloc[i] >= sl or df['low'].iloc[i] <= tp): in_pos = False

        if not in_pos and i <= last_closed_idx:
            if df['prob_long'].iloc[i] > (threshold * 100):
                in_pos, pos_type, tp, sl = True, 'LONG', df['close'].iloc[i] + tp_dist_arr[i], df['close'].iloc[i] - sl_dist_arr[i]
                entered_now = True
            elif df['prob_short'].iloc[i] > (threshold * 100):
                in_pos, pos_type, tp, sl = True, 'SHORT', df['close'].iloc[i] - tp_dist_arr[i], df['close'].iloc[i] + sl_dist_arr[i]
                entered_now = True

        ai_signals.append(pos_type if in_pos else 'NONE')
        trade_status.append('IN_TRADE' if in_pos else 'IDLE')
        active_tps.append(tp if in_pos else np.nan)
        active_sls.append(sl if in_pos else np.nan)
        # ONEMLI: "is_entry" sadece pozisyonun ACILDIGI tek mumu isaretler; "ai_signal" ise
        # pozisyon acik oldugu SUERECE (giristen cikisa kadar) her mumda tekrar eder. Grafikte
        # ai_signal'e gore etiket basmak, tek bir islem icin onlarca ust uste binen kutucuk
        # uretiyordu - bu yuzden grafik/kayit defteri artik is_entry'yi kullaniyor.
        is_entries.append(entered_now)

    df['ai_signal'], df['trade_status'], df['active_tp'], df['active_sl'], df['is_entry'] = ai_signals, trade_status, active_tps, active_sls, is_entries
    
    return df, 99.9, final_features

def format_price(p): return f"{p:.4f}" if p < 10 else f"{p:.2f}"


def _build_sparkline_svg(prices, width: int = 500, height: int = 60, line_color: str = "#16a34a") -> str:
    """Küçük, saf SVG bir fiyat mini-grafiği üretir (İZLEMEDE/SİNYAL kartının içine gömülür).

    Bilinçli olarak st.plotly_chart KULLANILMIYOR: bu kart @st.fragment(run_every=...) içinde
    her birkaç saniyede bir yeniden çiziliyor; ayrı bir Plotly bileşeni burada da (1. sayfadaki
    hero kartla aynı nedenle) remount/flash sorununa yol açardı. Düz SVG, aynı st.markdown
    bloğunun bir parçası olduğu için bu sorunu hiç yaşamıyor.
    """
    prices = [float(p) for p in prices if pd.notna(p)]
    n = len(prices)
    if n < 2:
        return ""
    p_min, p_max = min(prices), max(prices)
    p_range = max(p_max - p_min, 1e-9)
    pad = 6
    usable_h = height - pad * 2
    pts = []
    for i, p in enumerate(prices):
        x = (i / (n - 1)) * width
        y = pad + (1 - (p - p_min) / p_range) * usable_h
        pts.append((x, y))
    line_path = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" preserveAspectRatio="none" style="display:block; margin-top:12px;">',
        f'<path d="{line_path}" fill="none" stroke="{line_color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>',
        '</svg>',
    ]
    return "".join(parts)

# ─────────────────────────────────────────────────────────────────
# 3.5 CANLI GRAFIK IÇIN ARKA PLAN VERİ SUNUCUSU (FLAŞ FIX - MİMARİ DEĞİŞİKLİĞİ)
# ─────────────────────────────────────────────────────────────────
# ONEMLI: Arastirma sonucu kesinlesti - st.plotly_chart, @st.fragment(run_every=...) icinde
# HER yenilemede yeniden monte ediliyor (Streamlit'in bu surumundeki framework davranisi;
# bos/basit bir test grafigiyle bile dogrulandi, Python tarafinda figur optimizasyonuyla
# onlenemiyor). TEK gercek cozum: grafigin DOM/JS yasam dongusunu Streamlit'in rerun
# dongusunden TAMAMEN bagimsiz hale getirmek. Bunun icin:
#   1. Streamlit process'i icinde, ayri bir thread'de hafif bir HTTP sunucusu calisiyor.
#   2. Fragment (5 saniyede bir) yeni fig'i st.plotly_chart'a VERMEK yerine bu sunucuya
#      JSON olarak "yayinliyor" (publish).
#   3. Grafik, st.components.v1.html() ile SADECE BIR KEZ (fragment'in DISINDA, main()
#      icinde) kuruluyor; icindeki JS kendi basina bu sunucuyu periyodik olarak (fetch) yoklayip
#      Plotly.react() ile SADECE VERIYI guncelliyor - Streamlit bu DOM'a bir daha HIC dokunmuyor,
#      yani remount/flas fiziksel olarak imkansiz hale geliyor.
_CHART_DATA_HOST = "127.0.0.1"
_CHART_DATA_PORT = 8765
_chart_data_lock = threading.Lock()
_chart_data_store: dict = {}


class _ChartDataHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/chart_data":
            key = parse_qs(parsed.query).get("key", [""])[0]
            with _chart_data_lock:
                payload = _chart_data_store.get(key, "")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write((payload or "{}").encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # konsolu kirletmesin - sessiz calis


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


@st.cache_resource(show_spinner=False)
def _ensure_chart_data_server():
    # ONEMLI: @st.cache_resource sayesinde bu fonksiyon TUM oturumlar/reruns boyunca process
    # basina SADECE BIR KEZ calisir - sunucu tekrar tekrar baslatilmaya calisilmaz.
    if not _port_is_free(_CHART_DATA_HOST, _CHART_DATA_PORT):
        return None  # baska bir session/process zaten baslatmis olabilir - sorun degil
    server = http.server.ThreadingHTTPServer((_CHART_DATA_HOST, _CHART_DATA_PORT), _ChartDataHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _publish_chart_figure(key: str, fig: go.Figure) -> None:
    with _chart_data_lock:
        _chart_data_store[key] = fig.to_json()


def render_live_chart_component(chart_key: str, height: int = 850) -> None:
    # ONEMLI: Bu fonksiyon fragment'in DISINDA (main() icinde, sembol/zaman dilimi degisince
    # dogal olarak yeniden calisir) cagrilmalidir - boylece Streamlit'in donemsel (5sn) rerun'u
    # bu bilesene HIC dokunmaz, sadece kullanici gercekten coin/zaman dilimi degistirdiginde
    # (beklenen/istenen bir "yeniden yukleme") yeniden kurulur.
    data_url = f"http://{_CHART_DATA_HOST}:{_CHART_DATA_PORT}/chart_data?key={chart_key}"
    # ONEMLI - GENISLIK HATASI FIX: "Grafik yukleniyor..." metnini ortalamak icin kullanilan
    # display:flex/justify-content:center stilleri, DAHA SONRA Plotly.newPlot() AYNI div'e
    # cizim yaptiginda da uzerinde kaliyordu; bu da Plotly'nin genislik hesaplamasini bozup
    # grafigin panelin sadece bir kismini kaplamasina (sol tarafta bosluk) yol aciyordu. Cozum:
    # yukleniyor metni AYRI, Plotly div'inin UZERINE bindirilen bir katmanda gosteriliyor;
    # Plotly'nin kendi div'i (live-plotly-chart) hicbir zaman flex/center stili almiyor.
    html = f"""
    <style>html,body{{margin:0;padding:0;background:#ffffff;}}</style>
    <div style="position:relative; width:100%; height:{height}px; background:#ffffff; border-radius:20px; overflow:hidden;">
        <div id="live-plotly-chart" style="width:100%; height:100%;"></div>
        <div id="live-plotly-loading" style="position:absolute; inset:0; display:flex;
             align-items:center; justify-content:center; color:#5f7d7a;
             font-family:'Inter',sans-serif; font-size:13px; background:#ffffff; pointer-events:none;">
             Grafik yükleniyor…
        </div>
    </div>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <script>
    (function() {{
        const dataUrl = {data_url!r};
        const el = document.getElementById('live-plotly-chart');
        const loadingEl = document.getElementById('live-plotly-loading');
        const AXES = ['xaxis', 'yaxis', 'xaxis2', 'yaxis2'];
        let initialized = false;
        let lastPayload = null;   // ayni veri geldiginde gereksiz yeniden cizimi engeller
        let userView = false;     // kullanici kendi zoom/pan yaptiysa onun gorunumu korunur

        // Kullanici tekerlekle yakinlastirdiginda / surukleyerek kaydirdiginda Plotly
        // "plotly_relayout" olayini eksen araliklariyla tetikler. O andan itibaren sunucudan
        // gelen sabit araliklari UYGULAMAYIZ - aksi halde kullanicinin gorunumu her
        // guncellemede sifirlanir (titreme/kayma sikayetinin sebebi buydu). Grafige CIFT
        // TIKLAYINCA Plotly "autorange" gonderir; o an otomatik takip moduna geri doneriz.
        function attachInteractionTracking() {{
            el.on('plotly_relayout', function(ev) {{
                if (!ev) return;
                const keys = Object.keys(ev);
                if (keys.some(k => k.indexOf('.range') !== -1)) userView = true;
                if (keys.some(k => k.indexOf('.autorange') !== -1)) userView = false;
            }});
            el.on('plotly_doubleclick', function() {{ userView = false; }});
        }}

        function keepUserView(layout) {{
            if (!userView || !el.layout) return layout;
            AXES.forEach(function(ax) {{
                const cur = el.layout[ax];
                if (cur && cur.range && layout[ax]) {{
                    layout[ax] = Object.assign({{}}, layout[ax], {{
                        range: cur.range.slice(), autorange: false
                    }});
                }}
            }});
            return layout;
        }}

        async function poll() {{
            try {{
                const resp = await fetch(dataUrl, {{cache: 'no-store'}});
                const text = await resp.text();
                // ONEMLI: Python tarafi 5 saniyede bir uretiyor, biz 1.5 saniyede bir soruyoruz -
                // yani cogu istekte veri AYNI geliyordu ve bosuna tam yeniden cizim yapiliyordu
                // (takilma/kasma sebebi). Veri gercekten degismediyse hicbir sey yapmiyoruz.
                if (text && text !== lastPayload) {{
                    const spec = JSON.parse(text);
                    if (spec && spec.data && spec.data.length) {{
                        lastPayload = text;
                        if (!initialized) {{
                            loadingEl.style.display = 'none';
                            Plotly.newPlot(el, spec.data, spec.layout, {{displayModeBar: false, scrollZoom: true, responsive: true}});
                            initialized = true;
                            attachInteractionTracking();
                        }} else {{
                            Plotly.react(el, spec.data, keepUserView(spec.layout));
                        }}
                    }}
                }}
            }} catch (e) {{ /* sessizce gec, bir sonraki denemede tekrar dene */ }}
            setTimeout(poll, 1500);
        }}
        window.addEventListener('resize', function() {{ if (initialized) Plotly.Plots.resize(el); }});
        poll();
    }})();
    </script>
    """
    st.components.v1.html(html, height=height + 5)

def build_realtime_chart(df: pd.DataFrame, threshold: float, tp_m: float, sl_m: float, timeframe: str = "", entries_df: pd.DataFrame = None) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], increasing_line_color="#22ab94", decreasing_line_color="#f7525f", increasing_fillcolor="#22ab94", decreasing_fillcolor="#f7525f", name="Fiyat"), row=1, col=1)
    fig.add_trace(go.Scattergl(x=df.index, y=df['prob_long'], mode='lines', line=dict(color='#22ab94', width=2), name='LONG %', fill='tozeroy', fillcolor='rgba(34, 171, 148, 0.1)'), row=2, col=1)
    fig.add_trace(go.Scattergl(x=df.index, y=df['prob_short'], mode='lines', line=dict(color='#f7525f', width=2), name='SHORT %', fill='tozeroy', fillcolor='rgba(247, 82, 95, 0.1)'), row=2, col=1)
    fig.add_hline(y=threshold, line_dash="dot", line_color="#5f7d7a", opacity=0.7, row=2, col=1)

    # ONEMLI: Asagida ALTTAKI "Gecmis Islem Sinyalleri Kayit Defteri" tablosuyla AYNI
    # entries_df kullanilir - boylece grafikte gorunen isaretler ile tablodaki satirlar
    # birebir eslesir (kullanici "panelde yazan butun analizleri grafikte de goster" istedi).
    # ONEMLI - KUTUCUK + OK: Kullanicinin istedigi eski "AI LONG/SHORT | %XX TP:.. SL:.."
    # etiket kutulari, kok neden (is_entry filtresiyle sadece GERCEK giris ani isaretleniyor,
    # pozisyonun acik oldugu her mum degil) zaten cozuldugu icin geri getirildi. Kutucuklarin
    # yine de UST USTE binmemesi icin, birbirine yakin (ayni yonde, birkac mum arayla) gelen
    # girisler arasinda dikey kademe (stagger) uygulaniyor - yakin girisler farkli yukseklige
    # itiliyor, uzak girisler taban yuksekligine donuyor.
    # ONEMLI: Grafik ~1500 mumu dar bir alana sikistirdigi icin 1 piksel ~2-3 muma denk geliyor;
    # bir etiket kutusu (~90px) yatayda ~250 mumluk alan kaplıyor. Bu yuzden "yakinlik" esigi
    # (close_gap) mum sayisi olarak COK genis tutulmali, aksi halde kutular ayni yukseklige
    # denk gelip ust uste biniyor.
    def _stagger_offsets(idx_values: pd.DatetimeIndex, base: int, step: int, tiers: int = 5, close_gap: int = 220) -> list:
        positions = df.index.get_indexer(idx_values)
        order = np.argsort(positions)  # kronolojik (zaman) sirasina gore indeksler
        tiers_sorted = np.zeros(len(positions), dtype=int)
        tier, prev_pos = 0, None
        for orig_i in order:
            pos = positions[orig_i]
            tier = (tier + 1) % tiers if (prev_pos is not None and (pos - prev_pos) <= close_gap) else 0
            tiers_sorted[orig_i] = tier
            prev_pos = pos
        return [base + int(t) * step for t in tiers_sorted]

    # ONEMLI - MUM OKUNABILIRLIGI: Tum gecmis (1500 mum) grafikte veri olarak durur (kullanici
    # geriye kaydirinca eski sinyalleri de gorur), ancak ACILIS gorunumu son VISIBLE_CANDLES
    # muma odaklanir; hepsi ayni anda gosterilince mumlar piksel-alti genislige dusup ic ice
    # geciyor ve hangi mumda sinyal verildigi secilemiyordu.
    VISIBLE_CANDLES = 210
    visible_n = min(VISIBLE_CANDLES, len(df))
    vis_start_pos = len(df) - visible_n

    # ONEMLI - UST USTE BINME: Tum girisler icin etiket kutusu basmak, sinyallerin sik geldigi
    # bolgelerde kutulari ust uste bindiriyordu. Bu yuzden SADECE acilis penceresi ICINDE kalan
    # (ve sol kenarda kirpilmamasi icin kenardan en az 10 mum iceride olan) en guncel BOX_LIMIT
    # sinyal etiket kutusu alir; digerleri sadece kucuk ok isareti olarak gosterilir - detaylari
    # alttaki kayit defteri tablosunda ve hover'da zaten mevcut.
    BOX_LIMIT = 6
    box_index_set = set()
    if entries_df is not None and not entries_df.empty:
        _sorted_entries = entries_df.sort_index()
        _entry_pos = df.index.get_indexer(_sorted_entries.index)
        _in_window = [idx for idx, pos in zip(_sorted_entries.index, _entry_pos) if pos >= vis_start_pos + 10]
        box_index_set = set(_in_window[-BOX_LIMIT:])

    # ONEMLI - GERCEK FLAS KOK NEDENI: Streamlit'in plotly bileseni, figur'un YAPISI (trace
    # SAYISI, annotation SAYISI) bir onceki render'dan FARKLI oldugunda component'i baytan
    # olusturuyor (siyah bosluk + sicrama olarak goruluyor); sadece VERI degisirse (ayni sayida
    # trace/annotation, farkli x/y/text) sorunsuz, aninda guncelleniyor. entries_df (canli model
    # her yenilemede yeniden egitildigi icin) sinyal SAYISI her calisma da degisebiliyordu, bu da
    # annotation/trace sayisini degistirip her 5 saniyede bir tam yeniden cizime zorluyordu.
    # COZUM: Trace sayisi ve annotation sayisi HER render'da SABIT tutulur - gercek sinyal sayisi
    # ne olursa olsun, eksik kalan slotlar GORUNMEZ (opacity=0) "bos" annotation'larla doldurulur.
    if entries_df is None or entries_df.empty:
        _empty_cols = ["ai_signal", "prob_long", "prob_short", "active_tp", "active_sl", "atr", "low", "high", "close"]
        long_e = pd.DataFrame(columns=_empty_cols)
        short_e = pd.DataFrame(columns=_empty_cols)
    else:
        long_e = entries_df[entries_df["ai_signal"] == "LONG"]
        short_e = entries_df[entries_df["ai_signal"] == "SHORT"]
    long_y = (long_e["low"] - (long_e["atr"] * 0.3)) if not long_e.empty else pd.Series(dtype=float)
    short_y = (short_e["high"] + (short_e["atr"] * 0.3)) if not short_e.empty else pd.Series(dtype=float)

    def _customdata(e: pd.DataFrame, direction: str, prob_col: str) -> np.ndarray:
        if e.empty:
            return np.empty((0, 6), dtype=object)
        return np.stack([
            e.index.strftime("%Y-%m-%d %H:%M"),
            [direction] * len(e),
            e[prob_col].map(lambda v: f"{v:.1f}"),
            e["close"].map(format_price),
            e["active_tp"].map(format_price),
            e["active_sl"].map(format_price),
        ], axis=-1)

    # ONEMLI: Bu iki trace ARTIK KOSULSUZ ekleniyor (bos olsa bile) - boylece trace SAYISI
    # (5) her render'da ayni kalir; sadece icindeki nokta sayisi degisir ki bu sorunsuz.
    fig.add_trace(go.Scatter(
        x=long_e.index, y=long_y, mode="markers", name="AI LONG",
        marker=dict(size=17, color="#22ab94", opacity=0.001),
        customdata=_customdata(long_e, "LONG", "prob_long"),
        hovertemplate="<b>AI %{customdata[1]}</b> | Güven: %%{customdata[2]}<br>Zaman: %{customdata[0]}<br>Giriş: %{customdata[3]}<br>TP: %{customdata[4]} | SL: %{customdata[5]}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=short_e.index, y=short_y, mode="markers", name="AI SHORT",
        marker=dict(size=17, color="#f7525f", opacity=0.001),
        customdata=_customdata(short_e, "SHORT", "prob_short"),
        hovertemplate="<b>AI %{customdata[1]}</b> | Güven: %%{customdata[2]}<br>Zaman: %{customdata[0]}<br>Giriş: %{customdata[3]}<br>TP: %{customdata[4]} | SL: %{customdata[5]}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # ONEMLI - RENDER MALIYETI: on_select="rerun" kaldirilsa, figur yapisi sabitlense bile
    # Streamlit'in @st.fragment(run_every=...) icindeki st.plotly_chart bilesenini HER
    # yenilemede yeniden monte ettigi kanitlandi (framework davranisi, Python tarafinda
    # onlenemiyor). Bu yuzden asil kaldiraç kolu artik "remount'u onlemek" degil, "remount'un
    # gorunme suresini minimize etmek" - annotation sayisi olabildigince dusuk tutulur (gercekci
    # senaryoda gorunur pencerede nadiren 20'den fazla sinyal olur).
    MAX_ENTRY_ANNOTATIONS = 20  # tum render'larda SABIT annotation sayisi icin ust sinir
    _entry_annotations_used = 0

    if not long_e.empty:
        long_offsets = _stagger_offsets(long_e.index, base=38, step=55)
        for xi, yi, prob, tp_v, sl_v, ay_off in list(zip(long_e.index, long_y, long_e["prob_long"], long_e["active_tp"], long_e["active_sl"], long_offsets))[:MAX_ENTRY_ANNOTATIONS]:
            if xi in box_index_set:
                box_text = f"<b>AI LONG</b> | %{prob:.1f}<br>TP: {format_price(tp_v)} | SL: {format_price(sl_v)}"
                fig.add_annotation(x=xi, y=yi, text=box_text, showarrow=True, arrowhead=2, arrowsize=1.0, arrowwidth=1.3, arrowcolor="#16a34a", ax=0, ay=ay_off, standoff=2, font=dict(size=9, color="#14532d"), align="left", bgcolor="rgba(34,197,94,0.18)", bordercolor="rgba(22,163,74,0.9)", borderwidth=1, borderpad=3, row=1, col=1)
            else:
                fig.add_annotation(x=xi, y=yi, text="", showarrow=True, arrowhead=1, arrowsize=1.4, arrowwidth=1.3, arrowcolor="#22c55e", ax=0, ay=15, standoff=1, row=1, col=1)
            _entry_annotations_used += 1

    if not short_e.empty:
        short_offsets = _stagger_offsets(short_e.index, base=-38, step=-55)
        for xi, yi, prob, tp_v, sl_v, ay_off in list(zip(short_e.index, short_y, short_e["prob_short"], short_e["active_tp"], short_e["active_sl"], short_offsets))[:max(0, MAX_ENTRY_ANNOTATIONS - _entry_annotations_used)]:
            if xi in box_index_set:
                box_text = f"<b>AI SHORT</b> | %{prob:.1f}<br>TP: {format_price(tp_v)} | SL: {format_price(sl_v)}"
                fig.add_annotation(x=xi, y=yi, text=box_text, showarrow=True, arrowhead=2, arrowsize=1.0, arrowwidth=1.3, arrowcolor="#dc2626", ax=0, ay=ay_off, standoff=2, font=dict(size=9, color="#7f1d1d"), align="left", bgcolor="rgba(239,68,68,0.18)", bordercolor="rgba(220,38,38,0.9)", borderwidth=1, borderpad=3, row=1, col=1)
            else:
                fig.add_annotation(x=xi, y=yi, text="", showarrow=True, arrowhead=1, arrowsize=1.4, arrowwidth=1.3, arrowcolor="#ef4444", ax=0, ay=-15, standoff=1, row=1, col=1)
            _entry_annotations_used += 1

    # ONEMLI: Eksik kalan slotlar GORUNMEZ annotation'larla dolduruluyor - boylece annotation
    # DIZISININ UZUNLUGU (Streamlit/Plotly'nin yapisal degisiklik olarak algiladigi sey) sinyal
    # sayisindan BAGIMSIZ olarak her zaman TAM MAX_ENTRY_ANNOTATIONS olur.
    _dummy_x = df.index[0]
    _dummy_y = float(df["close"].iloc[0])
    while _entry_annotations_used < MAX_ENTRY_ANNOTATIONS:
        fig.add_annotation(x=_dummy_x, y=_dummy_y, text="", showarrow=False, opacity=0, row=1, col=1)
        _entry_annotations_used += 1

    last_idx, last = df.index[-1], df.iloc[-1]
    future_idx = last_idx + (df.index[-1] - df.index[-2]) * 8
    # ONEMLI - KARARLILIK: Yon/guven (prob_long, prob_short) HENUZ KAPANMAMIS son mumdan degil,
    # EN SON KAPANMIS mumdan (stable_row) okunur - bu, "CANLI TAHMİN" kutusunun mum kapanana
    # kadar (secili zaman dilimi suresince) SABIT kalmasini saglar. Fiyat (last_price) yine de
    # canli/guncel kalir - kullanici o an gerceklesen guncel fiyati gormeye devam eder.
    stable_row = df.iloc[-2] if len(df) > 1 else last
    is_long = stable_row['prob_long'] > stable_row['prob_short']
    live_prob, live_dir = (stable_row["prob_long"], "LONG") if is_long else (stable_row["prob_short"], "SHORT")
    color = "#22ab94" if is_long else "#f7525f"
    last_price = last['close']

    # ONEMLI - REFERANS GORSEL: Kullanicinin referans verdigi borsa (MEXC) grafigindeki gibi,
    # grafigin en ustune son mumun Open/Close/High/Low/Degisim/Hacim bilgisini gosteren bir ust
    # bilgi satiri; grafigin ortasindan gecen kesikli bir "guncel fiyat" cizgisi; ve sag kenarda
    # o an gecerli fiyati vurgulayan renkli bir fiyat etiketi eklendi.
    prev_close = df['close'].iloc[-2] if len(df) > 1 else last['open']
    chg = last['close'] - prev_close
    chg_pct = (chg / prev_close * 100.0) if prev_close else 0.0
    ohlc_color = "#22ab94" if chg >= 0 else "#f7525f"
    ohlc_text = (
        f"<b>{last_idx.strftime('%Y/%m/%d %H:%M')}</b> | "
        f"Open: <b>{format_price(last['open'])}</b> | "
        f"Close: <b>{format_price(last['close'])}</b> | "
        f"High: <b>{format_price(last['high'])}</b> | "
        f"Low: <b>{format_price(last['low'])}</b> | "
        f"Change: <b>{chg:+,.1f} ({chg_pct:+.2f}%)</b> | "
        f"Volume: <b>{last['volume']:,.1f}</b>"
    )
    fig.add_annotation(xref="x domain", yref="y domain", x=0, y=1.04, xanchor="left", yanchor="bottom", text=ohlc_text, showarrow=False, font=dict(size=11, color=ohlc_color), align="left")
    fig.add_hline(y=last_price, line_dash="dot", line_color=ohlc_color, opacity=0.55, line_width=1, row=1, col=1)
    fig.add_annotation(xref="paper", x=1.001, xanchor="left", yref="y", y=last_price, yanchor="middle", text=f"<b>{format_price(last_price)}</b>", showarrow=False, font=dict(size=11, color="#0b0e14"), align="center", bgcolor=ohlc_color, borderpad=4)
    live_tp_dist = clamp_tp_sl_dist(last['atr'] * tp_m, last_price, timeframe, "tp")
    live_sl_dist = clamp_tp_sl_dist(last['atr'] * sl_m, last_price, timeframe, "sl")
    live_tp = last_price + live_tp_dist if is_long else last_price - live_tp_dist
    live_sl = last_price - live_sl_dist if is_long else last_price + live_sl_dist
    
    if live_prob >= threshold:
        label_text = f"<b>⚡ CANLI SİNYAL: {'LONG' if is_long else 'SHORT'}</b><br><b>%{live_prob:.1f} GÜVEN</b>"
        bg_color, text_color, dash_style, line_width = color, "white", "solid", 3
    else:
        label_text = f"<b>⏳ CANLI TAHMİN: {'LONG' if is_long else 'SHORT'}</b><br>%{live_prob:.1f} (Eşik Altı)"
        bg_color, text_color, dash_style, line_width = color, "white", "dot", 1

    fig.add_shape(type="line", x0=last_idx, y0=live_tp, x1=future_idx, y1=live_tp, line=dict(color=color, dash=dash_style, width=line_width), row=1, col=1)
    fig.add_shape(type="line", x0=last_idx, y0=last['close'], x1=future_idx, y1=live_tp, line=dict(color=color, dash="dot", width=1), opacity=0.5, row=1, col=1)
    fig.add_annotation(x=future_idx, y=live_tp, text=label_text, showarrow=True, arrowhead=2, arrowcolor=color, font=dict(size=12, color=text_color), align="left", bgcolor=bg_color, bordercolor=color, borderwidth=2, borderpad=6, ax=40, ay=0, row=1, col=1)

    # ONEMLI - AKICI ETKILESIM: dragmode="pan" ile fare ile SURUKLEYINCE grafik kayar (borsa
    # arayuzlerindeki gibi); tekerlekle yakinlastirma (scrollZoom) JS tarafinda aciktir.
    fig.update_layout(template="plotly_white", paper_bgcolor="#ffffff", plot_bgcolor="#ffffff", height=850, margin=dict(l=10, r=70, t=48, b=20), hovermode="x unified", showlegend=False, xaxis_rangeslider_visible=False, dragmode="pan")
    fig.update_xaxes(gridcolor="#e5e7eb", zeroline=False, showspikes=True, spikecolor="#94a3b8", rangebreaks=[])
    fig.update_yaxes(gridcolor="#e5e7eb", zeroline=False, side="right")
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    # Y ekseni de sadece gorunen pencereye gore olceklenir; aksi halde tum gecmisin fiyat
    # araligi (orn. 64k -> 90k) yuzunden guncel mumlar duz bir seride sikisip kaliyordu.
    vis_df = df.tail(visible_n)
    y_lo, y_hi = float(vis_df["low"].min()), float(vis_df["high"].max())
    y_pad = (y_hi - y_lo) * 0.14 if y_hi > y_lo else max(abs(y_hi) * 0.01, 1.0)
    fig.update_xaxes(range=[vis_df.index[0], future_idx], row=1, col=1)
    fig.update_xaxes(range=[vis_df.index[0], future_idx], row=2, col=1)
    fig.update_yaxes(range=[y_lo - y_pad, y_hi + y_pad], row=1, col=1)
    return fig

# ─────────────────────────────────────────────────────────────────
# 5. ANA YÜRÜTME DÖNGÜSÜ
# ─────────────────────────────────────────────────────────────────
def main():
    with st.sidebar:
        st.markdown("## 🎯 SWING AI AYARLARI")
        selected_exchange_label = st.selectbox("🏦 Borsa Seçimi:", list(SUPPORTED_EXCHANGES.keys()), index=0, help="Tüm piyasa verileri ve analiz seçtiğiniz borsadan çekilir.")
        selected_exchange_id = SUPPORTED_EXCHANGES[selected_exchange_label]
        symbol = st.selectbox("🪙 Coin", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"])
        tf = st.selectbox("⏱️ Z. Dilimi", ["15m", "30m", "1h", "4h", "1d"], index=0)
        
        with st.expander("🛠️ Kurumsal ML & Risk Parametreleri", expanded=False):
            target_candles = st.slider("Hedef Süre (Mum)", 10, 50, 20, 5)
            ai_threshold = st.slider("Güven Eşiği (%)", 50, 95, 65, 1)
            tp_m = st.slider("TP Çarpanı (ATR)", 1.0, 8.0, 4.0, 0.5)
            sl_m = st.slider("SL Çarpanı (ATR)", 0.5, 4.0, 1.5, 0.1)

    # ONEMLI - FLAS FIX MİMARİSİ: Grafik artik fragment'in ICINDE degil, burada (main() icinde,
    # sadece coin/zaman dilimi degisince dogal olarak yeniden calisan bolumde) TEK SEFERLIK
    # kuruluyor. Fragment (asagida) artik grafigi DOGRUDAN cizmiyor; sadece arka plandaki hafif
    # HTTP sunucusuna yeni veriyi "yayinliyor" (_publish_chart_figure). Grafigin kendi JS'i bu
    # sunucuyu bagimsiz olarak yoklayip Plotly.react() ile SADECE VERIYI gunceller - Streamlit
    # bu DOM'a bir daha hic dokunmadigi icin remount/flas fiziksel olarak imkansiz hale gelir.
    _ensure_chart_data_server()
    chart_key = f"{selected_exchange_id}_{symbol}_{tf}"
    st.markdown("## 📈 Canlı Grafik")
    render_live_chart_component(chart_key, height=850)

    refresh_rate = 5
    @st.fragment(run_every=refresh_rate)
    def render_classic_terminal():
        with st.spinner("Piyasa verileri yükleniyor... (API İstekleri Sıralanıyor)"):
            df_crypto = fetch_crypto_data(selected_exchange_id, symbol, tf, limit=1500)
            if df_crypto.empty:
                st.stop()
                
            fg_df, dxy_df, spy_df = fetch_macro_data()
            df = df_crypto.copy()
            
            # KURAL 2: Kripto Timestamp'i ASLA bozulmasın diye .merge yerine .map yöntemi kullanıldı!
            df['date_str'] = pd.to_datetime(df.index).strftime('%Y-%m-%d')
            
            if not fg_df.empty and 'date_str' in fg_df.columns:
                fg_map = fg_df.drop_duplicates('date_str').set_index('date_str')['fear_greed']
                df['fear_greed'] = df['date_str'].map(fg_map)
            else:
                df['fear_greed'] = np.nan
                
            if not dxy_df.empty and 'date_str' in dxy_df.columns:
                dxy_map = dxy_df.drop_duplicates('date_str').set_index('date_str')['dxy']
                df['dxy'] = df['date_str'].map(dxy_map)
            else:
                df['dxy'] = np.nan
                
            # Boş satırların (haftasonları vb.) FFILL ile eksiksiz doldurulması
            df['fear_greed'] = df['fear_greed'].ffill().fillna(50.0) 
            df['dxy'] = df['dxy'].ffill().fillna(100.0)
            df.drop(columns=['date_str'], inplace=True, errors='ignore')
    
            df, train_acc, active_features = train_and_predict_ai(df, target_candles, ai_threshold, tp_m, sl_m, timeframe=tf)
    
        last, last_price, atr = df.iloc[-1], df.iloc[-1]['close'], df.iloc[-1]['atr']
        # ONEMLI - KARARLILIK: build_realtime_chart'taki ayni mantik burada da uygulanir - AI
        # yon/guven okumasi EN SON KAPANMIS mumdan (stable_row) yapilir, fiyat (last_price)
        # canli kalir. Bu sayede "CANLI LONG %/CANLI SHORT %" karti ve "İZLEMEDE/SİNYAL
        # ONAYLANDI" kutusu, mum kapanana kadar sabit kalir - saniyeler icinde sicramaz.
        stable_row = df.iloc[-2] if len(df) > 1 else last

        is_long = stable_row["prob_long"] > stable_row["prob_short"]
        live_prob, live_dir = (stable_row["prob_long"], "LONG") if is_long else (stable_row["prob_short"], "SHORT")
        live_color = "#22ab94" if is_long else "#f7525f"
        live_tp_dist = clamp_tp_sl_dist(atr * tp_m, last_price, tf, "tp")
        live_sl_dist = clamp_tp_sl_dist(atr * sl_m, last_price, tf, "sl")
        live_tp = last_price + live_tp_dist if is_long else last_price - live_tp_dist
        live_sl = last_price - live_sl_dist if is_long else last_price + live_sl_dist
    
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(f'<div class="metric-card"><h4>💰 {symbol} FİYAT</h4><p class="value white">${last_price:,.4f}</p></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-card"><h4>🟢 CANLI LONG %</h4><p class="value green">%{stable_row["prob_long"]:.1f}</p></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-card"><h4>🔴 CANLI SHORT %</h4><p class="value red">%{stable_row["prob_short"]:.1f}</p></div>', unsafe_allow_html=True)
            
        c5, c6, c7, c8 = st.columns(4)
        with c5: st.markdown(f'<div class="metric-card"><h4>🧠 Eğitim Seti Doğruluğu</h4><p class="value blue">%{train_acc:.1f}</p></div>', unsafe_allow_html=True)
        with c6: st.markdown(f'<div class="metric-card"><h4>⚖️ TAHTA BASKISI</h4><p class="value white">{"ALICI (Bids)" if last["ob_imbalance"] > 1 else "SATICI (Asks)"}</p></div>', unsafe_allow_html=True)
        with c7: st.markdown(f'<div class="metric-card"><h4>🌍 MAKRO (F&G/DXY)</h4><p class="value white">F&G: {int(last["fear_greed"])} | DXY: {last["dxy"]:.2f}</p></div>', unsafe_allow_html=True)
        with c8: st.markdown(f'<div class="metric-card" title="{", ".join(active_features)}"><h4>📊 AKTİF FEATURE</h4><p class="value blue">{len(active_features)} Özellik (Filtreli)</p></div>', unsafe_allow_html=True)
        
        if live_prob >= ai_threshold:
            stat_sparkline = _build_sparkline_svg(df['close'].tail(80).tolist(), line_color=live_color)
            stat_html = f'''<div class="metric-card" style="border:1px solid {live_color}; background:rgba({34 if is_long else 220},{171 if is_long else 38},{148 if is_long else 38},0.08); text-align:center; padding: 20px;">
                <h2 style="color:{live_color}; margin:0; font-weight:900; font-size:1.8rem;">🚀 SİNYAL ONAYLANDI! ŞİMDİ İŞLEME GİR! ({live_dir})</h2>
                <div style="margin-top:15px; font-size:1.4rem; font-weight:800; color:#0f2b2e;">Giriş: <span style="color:#0f2b2e">{format_price(last_price)}</span> &nbsp;&nbsp;|&nbsp;&nbsp; <span style="color:#16a34a">TP: {format_price(live_tp)}</span> &nbsp;&nbsp;|&nbsp;&nbsp; <span style="color:#dc2626">SL: {format_price(live_sl)}</span></div>
                <p style="margin:10px 0 0 0; color:{live_color}; font-size:1.1rem; font-weight:600;">🤖 KURUMSAL ML GÜVENİ: %{live_prob:.1f}</p>
                {stat_sparkline}</div>'''
        else:
            stat_sparkline = _build_sparkline_svg(df['close'].tail(80).tolist(), line_color=live_color)
            stat_html = f'''<div class="metric-card" style="border:1px solid rgba(15,43,46,0.10); background:#ffffff; text-align:center; padding: 15px;">
                <h3 style="color:#5f7d7a; margin:0; font-weight:800; font-size:1.3rem;">⏳ İZLEMEDE (Eşik Altı) - Şu an işleme GİRME.</h3>
                <p style="color:#5f7d7a; font-size:1.0rem; margin-top:10px; margin-bottom:5px;">Beklenen Yön: <b style="color:{live_color}">{live_dir}</b> (Güven: %{live_prob:.1f})</p>
                <p style="color:#5f7d7a; font-size:0.9rem; margin:0;"><i>Potansiyel Giriş: {format_price(last_price)} &nbsp;|&nbsp; TP: {format_price(live_tp)} &nbsp;|&nbsp; SL: {format_price(live_sl)}</i></p>
                {stat_sparkline}</div>'''
        st.markdown(stat_html, unsafe_allow_html=True)
    
        # ONEMLI: Grafikteki isaretler ile alttaki "Gecmis Islem Sinyalleri Kayit Defteri"
        # tablosu artik TAM AYNI veriyi (entries_df) kullanir - bu yuzden entries_df, tabloyu
        # olusturmadan ONCE, grafige de parametre olarak verilmek uzere burada hesaplanir.
        # Boylece grafikte gorunen HER isaret, tabloda da bir satir olarak karsimiza cikar.
        entries_df = df[df.get("is_entry", False) == True].copy() if "is_entry" in df.columns else df.iloc[0:0]
        entries_df = entries_df.sort_index(ascending=False).head(100)

        # ONEMLI - PERFORMANS/FLAS SORUNU: Grafik her 5 saniyede bir YENIDEN CIZILIYOR (Streamlit'in
        # plotly bileseni, veri her degistiginde SVG'yi bastan olusturuyor - bunu component
        # seviyesinde engelleme imkanimiz yok). Grafige TAM 1500 mumluk veriyi vermek, her
        # yenilemede devasa bir SVG'nin (1500 mum + 1500x2 olasilik cizgisi + tum etiketler)
        # sifirdan insa edilmesine yol aciyor; bu da tarayicida gozle gorulur bir "donma/sicrama"
        # (kullanicinin bahsettigi "kapanip acilma") yaratiyor. Grafik zaten varsayilan olarak
        # sadece son ~VISIBLE_CANDLES mumu gosterdigi icin (geri kaydirma payi dahil) son
        # CHART_BUFFER_CANDLES kadarini vermek yeterli - kayit defteri tablosu yine TUM 1500
        # mumluk gecmisten (entries_df, asagida) beslenmeye devam eder, sadece GRAFIGE giden
        # veri kucultuluyor.
        CHART_BUFFER_CANDLES = 260
        chart_df = df.tail(CHART_BUFFER_CANDLES).copy()
        chart_entries_df = entries_df[entries_df.index.isin(chart_df.index)]

        fig = build_realtime_chart(chart_df, ai_threshold, tp_m, sl_m, timeframe=tf, entries_df=chart_entries_df)

        # ONEMLI: Grafik artik burada DOGRUDAN cizilmiyor (st.plotly_chart kaldirildi) - bunun
        # yerine yeni veri, main() icinde bir kez kurulan bagimsiz JS bilesenin okudugu arka plan
        # sunucusuna "yayinlaniyor". Boylece fragment'in 5 saniyelik periyodik yenilemesi grafigin
        # DOM'unu bir daha hic etkilemiyor - flas/kapanip-acilma fiziksel olarak imkansiz olur.
        _publish_chart_figure(chart_key, fig)

        # ONEMLI: Grafikteki kutucuklar cok sik oldugunda ust uste binip okunmaz hale
        # geliyordu. Bunun icin grafigin ALTINA, su ana kadar uretilen TUM giris sinyallerini
        # (is_entry == True) tarihli ve en yeniden en eskiye siralanmis sekilde listeleyen bir
        # kayit defteri eklendi. Bu panel her fragment yenilenmesinde (5 saniyede bir) df'den
        # yeniden hesaplandigi icin, yeni bir giris sinyali olustugunda otomatik olarak en üste
        # eklenip gorunur.
        st.markdown("---")
        st.markdown("## 📜 Geçmiş İşlem Sinyalleri Kayıt Defteri")

        table_cd = None
        if not entries_df.empty:
            st.caption(f"Yapay zekanın bu coin/zaman dilimi için ürettiği son {len(entries_df)} giriş sinyali (en yeni en üstte). Yeni bir sinyal oluştuğunda otomatik olarak buraya eklenir. Bir satıra tıklayarak o sinyalin ayrıntılarını aşağıda görebilirsiniz.")

            display_rows = []
            for idx, row in entries_df.iterrows():
                is_long_row = row["ai_signal"] == "LONG"
                conf = row["prob_long"] if is_long_row else row["prob_short"]
                display_rows.append({
                    "Tarih / Saat": idx.strftime("%Y-%m-%d %H:%M") if hasattr(idx, "strftime") else str(idx),
                    "Yön": "🟢 LONG" if is_long_row else "🔴 SHORT",
                    "Güven": round(float(conf), 1),
                    "Giriş": format_price(row["close"]),
                    "TP": format_price(row["active_tp"]),
                    "SL": format_price(row["active_sl"]),
                })
            display_df = pd.DataFrame(display_rows)

            # ONEMLI: Grafikteki ucgen isaretcilere tiklamanin Streamlit'e dogru sekilde
            # iletilmesi Plotly'nin kendi secim/olay zincirine bagli oldugundan (bazi
            # ortamlarda gecikme/gecikme sorunlari yasanabiliyor); bu yuzden AYNI tiklama
            # davranisini garanti eden ikinci, saglam bir yol olarak tablo satirlari da
            # on_select="rerun" ile tiklanabilir yapildi - kullanici herhangi bir satira
            # tiklayinca o sinyalin detay kutusu asagida acilir.
            table_event = st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=420,
                on_select="rerun",
                selection_mode="single-row",
                key=f"entries_table_{symbol}_{tf}",
                column_config={
                    "Güven": st.column_config.ProgressColumn("Güven", format="%.1f%%", min_value=0, max_value=100),
                },
            )

            sel_rows = []
            if table_event and table_event.get("selection", {}).get("rows"):
                sel_rows = table_event["selection"]["rows"]

            if sel_rows:
                sel_idx = sel_rows[0]
                if 0 <= sel_idx < len(entries_df):
                    row = entries_df.iloc[sel_idx]
                    idx_val = entries_df.index[sel_idx]
                    is_long_row = row["ai_signal"] == "LONG"
                    conf = row["prob_long"] if is_long_row else row["prob_short"]
                    table_cd = [
                        idx_val.strftime("%Y-%m-%d %H:%M") if hasattr(idx_val, "strftime") else str(idx_val),
                        "LONG" if is_long_row else "SHORT",
                        f"{conf:.1f}",
                        format_price(row["close"]),
                        format_price(row["active_tp"]),
                        format_price(row["active_sl"]),
                    ]
        else:
            st.caption("Bu coin/zaman diliminde henüz eşik üzerinde bir giriş sinyali oluşmadı.")

        # Tablo satır tıklaması, grafik işaretçi tıklamasından önceliklidir (daha güvenilir
        # şekilde doğrulanabiliyor); grafik tıklaması çalışırsa o da aynı kutuyu besler.
        final_cd = table_cd
        if final_cd:
            ts_c, dir_c, conf_c, entry_c, tp_c, sl_c = final_cd[:6]
            dir_color_c = "#16a34a" if dir_c == "LONG" else "#dc2626"
            dir_label_c = "🟢 LONG" if dir_c == "LONG" else "🔴 SHORT"
            st.markdown(f"""
            <div style="background:#ffffff; border:2px solid {dir_color_c}; border-radius:16px; padding:16px 20px; margin-top:10px; box-shadow:0 4px 14px rgba(15,43,46,0.06);">
                <div style="font-size:11px; color:#5f7d7a; text-transform:uppercase; font-weight:bold; margin-bottom:10px;">🔎 Seçilen Sinyal Detayı</div>
                <div style="display:flex; flex-wrap:wrap; gap:28px; align-items:center;">
                    <div><span style="color:#5f7d7a; font-size:11px;">YÖN</span><br><b style="color:{dir_color_c}; font-size:15px;">{dir_label_c}</b></div>
                    <div><span style="color:#5f7d7a; font-size:11px;">ZAMAN</span><br><b style="color:#0f2b2e; font-size:15px;">{ts_c}</b></div>
                    <div><span style="color:#5f7d7a; font-size:11px;">GÜVEN</span><br><b style="color:#0f2b2e; font-size:15px;">%{conf_c}</b></div>
                    <div><span style="color:#5f7d7a; font-size:11px;">GİRİŞ</span><br><b style="color:#0f2b2e; font-size:15px;">{entry_c}</b></div>
                    <div><span style="color:#5f7d7a; font-size:11px;">TP</span><br><b style="color:#16a34a; font-size:15px;">{tp_c}</b></div>
                    <div><span style="color:#5f7d7a; font-size:11px;">SL</span><br><b style="color:#dc2626; font-size:15px;">{sl_c}</b></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    render_classic_terminal()

if __name__ == "__main__":
    main()
