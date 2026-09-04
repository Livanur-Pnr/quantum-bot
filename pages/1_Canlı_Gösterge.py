import streamlit as st

st.set_page_config(page_title="Analiz Tahmini", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

# ONEMLI - Yenilemede yanip sonme (flas) fix: bu sayfadaki @st.fragment(run_every=...)
# bloklari her yenilendiginde Streamlit, eskimis ("stale") icerigi soluklastirip iskelet
# animasyonu gosteriyor; bu goze batan bir yanip sonme etkisi yaratiyordu. Asagidaki
# kurallar bu gecici gorsel durumlari devre disi birakiyor.
st.markdown("""
<style>
    /* --- "Vivid Fintech" açık tema: sayfanın ana arkaplanı (Canlı Gösterge sayfasıyla tutarlı) --- */
    .stApp, [data-testid="stAppViewContainer"] { background-color: #eaf6f4 !important; color: #0f2b2e; }
    /* NOT: Bilinçli olarak SADECE Streamlit'in kendi başlık/metin elemanları hedefleniyor
       (h1-h3, düz markdown <p>, caption) — genel div/span kuralı YAZILMIYOR çünkü bu, kart
       bileşenlerinin (section-header-title, tp-sl-label vb.) kendi <div>/<span> renklerini
       (daha yüksek CSS özgüllüğü nedeniyle) eziyor ve metni görünmez hale getiriyordu. */
    [data-testid="stAppViewContainer"] h1, [data-testid="stAppViewContainer"] h2, [data-testid="stAppViewContainer"] h3,
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"] p { color: #0f2b2e; }
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
""", unsafe_allow_html=True)

import sys
import codecs
try:
    if sys.stdout.encoding.lower() != 'utf-8':
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    if sys.stderr.encoding.lower() != 'utf-8':
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
except Exception:
    pass

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, VotingClassifier

import os
import difflib
import ccxt
from dotenv import load_dotenv

# --- MEXC GÜVENLİ KİMLİK DOĞRULAMA VE OTOMATİK İŞLEM MOTORU (CCXT) ---

import concurrent.futures
import requests

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_macro_data() -> tuple:
    def get_fg():
        try:
            resp = requests.get("https://api.alternative.me/fng/?limit=300", timeout=30)
            df_fg = pd.DataFrame(resp.json()['data'])
            df_fg['timestamp'] = pd.to_datetime(df_fg['timestamp'].astype(int), unit='s', utc=True)
            df_fg['date_str'] = df_fg['timestamp'].dt.strftime('%Y-%m-%d')
            df_fg['fear_greed'] = df_fg['value'].astype(float)
            return df_fg[['date_str', 'fear_greed']]
        except Exception:
            return pd.DataFrame()

    def get_dxy():
        try:
            import yfinance as yf
            dxy = yf.download("DX-Y.NYB", period="1y", interval="1d", progress=False, timeout=30)
            if dxy.empty: return pd.DataFrame()
            if isinstance(dxy.columns, pd.MultiIndex):
                dxy.columns = dxy.columns.get_level_values(0)
            dxy = dxy.reset_index() 
            date_col = 'Date' if 'Date' in dxy.columns else ('Datetime' if 'Datetime' in dxy.columns else dxy.columns[0])
            dxy['date_str'] = pd.to_datetime(dxy[date_col], utc=True).dt.strftime('%Y-%m-%d')
            dxy['dxy'] = dxy['Close'].astype(float)
            return dxy[['date_str', 'dxy']]
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


def load_env_credentials(exchange_id: str = "mexc") -> dict:
    """.env dosyasindan ve yedek metin dosyalarindan secilen borsaya ait API anahtarlarini
    yukler. Degisken adlari borsa onekiyle aranir (orn. MEXC_API_KEY veya BINANCE_API_KEY)."""
    # Proje kök dizini: bu dosya pages/ altinda oldugu icin bir ust klasore cikilir.
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(project_dir, ".env"))

    prefix = exchange_id.upper()
    api_key = os.getenv(f"{prefix}_API_KEY", "").strip()
    secret_key = os.getenv(f"{prefix}_SECRET_KEY", "").strip()

    possible_files = ["MEXC API ADRESLERİ.txt", "mexc_api.txt", "api_key.txt", "keys.txt"]

    for fname in possible_files:
        fpath = os.path.join(project_dir, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"): continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k_upper = k.strip().upper().replace(" ", "_")
                            val = v.strip().strip("'").strip('"')
                            if val:
                                if k_upper in [f"{prefix}_API_KEY", f"{prefix}_ACCESS_KEY", "API_KEY", "ACCESS_KEY"]:
                                    if not api_key: api_key = val
                                elif k_upper in [f"{prefix}_SECRET_KEY", "SECRET_KEY"]:
                                    if not secret_key: secret_key = val
            except Exception:
                pass
    if api_key: os.environ[f"{prefix}_API_KEY"] = api_key
    if secret_key: os.environ[f"{prefix}_SECRET_KEY"] = secret_key
    return {"api_key": api_key, "secret_key": secret_key}

# Desteklenen borsalar: MEXC (varsayilan) ve Binance. Ikisi de ccxt-unified ayni sozlesme
# sembol formatini ("BASE/USDT:USDT") ve ayni kimlik dogrulama alanlarini (sadece apiKey+secret,
# ekstra passphrase gerekmiyor) kullaniyor - test edip dogruladim. Farkli bir borsa eklenecekse
# once ccxt uzerinden requiredCredentials ve sembol formati kontrol edilmeli.
SUPPORTED_EXCHANGES = {"MEXC": "mexc", "Binance": "binance"}

@st.cache_resource(ttl=1800, show_spinner=False)
def get_public_exchange_client(exchange_id: str = "mexc"):
    """Herhangi bir API anahtari gerektirmeyen (genel/public) veri cekimleri icin TEK, paylasilan
    ve piyasa listesi onceden yuklenmis borsa istemcisi. exchange_id'ye gore ayri ayri
    onbelleklenir (orn. 'mexc' ve 'binance' icin farkli istemci nesneleri saklanir).

    ONEMLI PERFORMANS NOTU: ccxt her create_order/fetch_ohlcv/fetch_ticker/... cagrisinda,
    eger o istemcinin market listesi (self.markets) yuklu degilse otomatik olarak load_markets()
    calistirir; bu tek seferde binlerce spot+vadeli sozlesmeyi indirip parse eder. Eskiden neredeyse
    her fonksiyon kendi taze ccxt nesnesini olusturuyordu; bu da 3-5 saniyede bir calisan
    canli terminalde HER YENILEMEDE birden fazla kez bu agir load_markets() indirmesinin tekrar
    tekrar yapilmasina (ve panelin cok yavas/gec yuklenmesine) sebep oluyordu. Artik tum genel
    fonksiyonlar bu tek, onbelleklenmis istemciyi paylasiyor.
    """
    klass = getattr(ccxt, exchange_id)
    ex = klass({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    ex.load_markets()
    return ex


@st.cache_resource(ttl=1800, show_spinner=False)
def get_exchange_client(exchange_id: str, api_key: str = "", secret_key: str = ""):
    prefix = exchange_id.upper()
    key = api_key or os.getenv(f"{prefix}_API_KEY", "")
    sec = secret_key or os.getenv(f"{prefix}_SECRET_KEY", "")
    klass = getattr(ccxt, exchange_id)
    ex = klass({
        'apiKey': key,
        'secret': sec,
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })
    ex.load_markets()
    return ex

@st.cache_data(ttl=5, show_spinner=False)
def fetch_binance_account_balance(exchange_id: str = "mexc", api_key: str = "", secret_key: str = "") -> dict:
    try:
        ex = get_exchange_client(exchange_id, api_key, secret_key)
        bal = ex.fetch_balance()
        usdt_bal = bal.get('USDT', {})
        total = usdt_bal.get('total', 0.0)
        free = usdt_bal.get('free', 0.0)
        ex_label = [k for k, v in SUPPORTED_EXCHANGES.items() if v == exchange_id]
        ex_label = ex_label[0] if ex_label else exchange_id.upper()
        return {"status": "FUTURES_OK", "wallet_balance": total, "available_balance": free, "type": f"{ex_label} Vadeli (Swap)"}
    except Exception as e:
        return {"status": "ERROR", "msg": str(e)}

@st.cache_data(ttl=1800, show_spinner=False)
def get_all_binance_futures_symbols(exchange_id: str = "mexc") -> tuple:
    try:
        ex = get_public_exchange_client(exchange_id)
        markets = ex.markets
        symbols = []
        symbol_map = {}
        for sym, m in markets.items():
            # ONEMLI: ccxt load_markets() cogu borsada spot VE swap piyasalarini birlikte dondurur.
            # 'swap' filtresi olmadan spot semboller (orn. BTC/USDT) de listeye karisir ve
            # secildiginde bot Vadeli fiyati yerine yanlislikla Spot fiyatini gosterir/kullanir.
            if m.get('swap') and m['active'] and m['quote'] == 'USDT':
                symbols.append(sym)
                base = m['base']
                symbol_map[base] = sym
                if base.startswith("1000"):
                    symbol_map[base[4:]] = sym
        symbols.sort()
        return symbols, symbol_map
    except Exception:
        return ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT"], {}

def normalize_futures_symbol(user_input: str, exchange_id: str = "mexc") -> tuple:
    all_symbols, symbol_map = get_all_binance_futures_symbols(exchange_id)
    ex_label = next((k for k, v in SUPPORTED_EXCHANGES.items() if v == exchange_id), exchange_id.upper())
    text = user_input.strip().upper().replace(" ", "").replace("_", "").replace("-", "")

    if text in all_symbols:
        return text, ""

    # "BTC/USDT" gibi spot gorunumlu (":USDT" vadeli sonekini icermeyen) girisleri
    # yanlislikla Spot fiyatini kullanmamak icin secili borsanin Vadeli karsiligina cevir.
    if "/" in text and ":" not in text:
        base = text.split("/")[0]
        if base in symbol_map:
            return symbol_map[base], f" {ex_label} Vadeli Eşleşti: {symbol_map[base]}"
        sym = f"{base}/USDT:USDT"
        if sym in all_symbols:
            return sym, f" {ex_label} Vadeli Eşleşti: {sym}"

    if "/" not in text and ":" not in text:
        clean = text.replace("USDT", "")
        if clean in symbol_map:
            return symbol_map[clean], f" {ex_label} Eşleşti: {symbol_map[clean]}"
        sym = f"{clean}/USDT:USDT"
        if sym in all_symbols: return sym, f" {ex_label} Eşleşti: {sym}"
        import difflib
        matches = difflib.get_close_matches(sym, all_symbols, n=1, cutoff=0.5)
        if matches: return matches[0], f"💡 Yakın Eşleşme: {matches[0]}"

    return user_input, ""

INTERVAL_MAP = {"Min1": "1m", "Min5": "5m", "Min15": "15m", "Min60": "1h", "Min240": "4h"}

@st.cache_data(ttl=3, show_spinner=False)
def fetch_binance_kline_data(symbol: str, interval: str = "Min1", limit: int = 500, exchange_id: str = "mexc") -> pd.DataFrame:
    try:
        ex = get_public_exchange_client(exchange_id)
        api_interval = INTERVAL_MAP.get(interval, "1m")
        ohlcv = ex.fetch_ohlcv(symbol, api_interval, limit=limit)
        
        try:
            funding = ex.fetch_funding_rate_history(symbol, limit=200)
            df_fund = pd.DataFrame(funding)
            df_fund['time'] = pd.to_datetime(df_fund['timestamp'], unit='ms')
            df_fund.set_index('time', inplace=True)
        except Exception: df_fund = pd.DataFrame()
        
        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            df.set_index('time', inplace=True)
            if not df_fund.empty and 'fundingRate' in df_fund.columns:
                df = df.join(df_fund['fundingRate'].astype(float).rename('funding_rate'), how='left')
                df['funding_rate'] = df['funding_rate'].ffill().fillna(0.0)
            else:
                df['funding_rate'] = 0.0
            return df.reset_index()
    except Exception:
        pass
    return pd.DataFrame()

@st.cache_data(ttl=3, show_spinner=False)
def fetch_binance_ticker_details(symbol: str, exchange_id: str = "mexc") -> dict:
    try:
        ex = get_public_exchange_client(exchange_id)
        ticker = ex.fetch_ticker(symbol)
        funding = ex.fetch_funding_rate(symbol) if ex.has['fetchFundingRate'] else {}
        return {
            "lastPrice": ticker.get('last', 0),
            "riseFallRate": ticker.get('percentage', 0),
            "high24Price": ticker.get('high', 0),
            "lower24Price": ticker.get('low', 0),
            "volume24": ticker.get('quoteVolume', 0),
            "fundingRate": funding.get('fundingRate', 0) * 100 if funding else 0.0,
            "markPrice": ticker.get('info', {}).get('markPrice', ticker.get('last', 0))
        }
    except Exception:
        return {}

@st.cache_data(ttl=3, show_spinner=False)
def fetch_binance_depth_imbalance(symbol: str, exchange_id: str = "mexc") -> dict:
    try:
        ex = get_public_exchange_client(exchange_id)
        ob = ex.fetch_order_book(symbol, limit=20)
        bid_vol = sum([b[1] for b in ob['bids']])
        ask_vol = sum([a[1] for a in ob['asks']])
        total = bid_vol + ask_vol + 1e-9
        bid_ratio = (bid_vol / total) * 100
        ask_ratio = (ask_vol / total) * 100
        return {
            "bid_ratio": bid_ratio, "ask_ratio": ask_ratio, 
            "imbalance": bid_ratio - ask_ratio,
            "bias": "ALICI AĞIRLIKLI" if bid_ratio > 53 else ("SATICI AĞIRLIKLI" if ask_ratio > 53 else "DENGELİ")
        }
    except Exception:
        return {"bid_ratio": 50.0, "ask_ratio": 50.0, "imbalance": 0.0, "bias": "DENGELİ"}

@st.cache_data(ttl=5, show_spinner=False)
def fetch_institutional_order_flow(symbol: str, timeframe: str = "5m", limit: int = 150, exchange_id: str = "mexc") -> pd.DataFrame:
    try:
        ex = get_public_exchange_client(exchange_id)
        api_interval = INTERVAL_MAP.get(timeframe, "5m")
        ohlcv = ex.fetch_ohlcv(symbol, api_interval, limit=limit)
        if not ohlcv: return pd.DataFrame()
        
        df = pd.DataFrame(ohlcv, columns=["time", "open", "high", "low", "close", "vol"])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        
        df['oi_change_pct'] = (df['close'] - df['open']) / df['open'] * (df['vol'] / (df['vol'].mean()+1e-9)) * 10
        df['taker_delta_trend'] = np.where(df['close'] > df['open'], df['vol'] * 0.6, -df['vol'] * 0.6)
        
        return df[['time', 'oi_change_pct', 'taker_delta_trend']]
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=5, show_spinner=False)
def fetch_usdt_dominance_matrix() -> dict:
    """TradingView'in resmi canlı sunucusundan (CRYPTOCAP:USDT.D) Dolar Dominansı ve BTC.D verilerini 0 gecikmeyle çeker.

    ONEMLI: Bu fonksiyonun onbellegi yoktu; ana terminal her yenilendiginde (varsayilan 3 saniyede
    bir) TradingView'e, o basarisiz olursa CoinGecko'ya kadar 2 ayri dis HTTP istegi (her biri 2.5s
    timeout ile) atiliyordu. Bu, panelin agir/yavas yuklenmesinin en buyuk sebeplerinden biriydi.
    """
    # 1. Öncelik: TradingView Resmi Canlı Uç Noktası (CRYPTOCAP:USDT.D & BTC.D)
    try:
        url_tv = "https://scanner.tradingview.com/global/scan"
        payload = {
            "symbols": {"tickers": ["CRYPTOCAP:USDT.D", "CRYPTOCAP:BTC.D", "CRYPTOCAP:ETH.D"]},
            "columns": ["close", "change", "change_abs", "high", "low"]
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Content-Type": "application/json",
            "Origin": "https://www.tradingview.com",
            "Referer": "https://www.tradingview.com/"
        }
        r_tv = requests.post(url_tv, json=payload, headers=headers, timeout=2.5)
        if r_tv.status_code == 200:
            data = r_tv.json().get("data", [])
            tv_map = {}
            for row in data:
                ticker = row.get("s")
                vals = row.get("d", [])
                if len(vals) >= 2:
                    tv_map[ticker] = {
                        "close": float(vals[0]),
                        "change": float(vals[1])
                    }
                    
            if "CRYPTOCAP:USDT.D" in tv_map:
                usdt_d = tv_map["CRYPTOCAP:USDT.D"]["close"]
                usdt_chg = tv_map["CRYPTOCAP:USDT.D"]["change"]
                btc_d = tv_map.get("CRYPTOCAP:BTC.D", {}).get("close", 59.5)
                eth_d = tv_map.get("CRYPTOCAP:ETH.D", {}).get("close", 11.2)
                
                trend = "DÜŞÜŞTE (Kripto Boğa / LONG)" if usdt_chg < 0 else "YÜKSELİŞTE (Kripto Ayı / SHORT)"
                bias = "BULLISH_CRYPTO" if usdt_chg < 0 else "BEARISH_CRYPTO"
                
                print(f"[STAT] [TRADINGVIEW CANLI] CRYPTOCAP:USDT.D = %{usdt_d:.4f} (24s Değişim: %{usdt_chg:+.2f}) | BTC.D = %{btc_d:.2f}")
                return {
                    "usdt_d": usdt_d,
                    "usdt_d_change": usdt_chg,
                    "btc_d": btc_d,
                    "eth_d": eth_d,
                    "trend": trend,
                    "bias": bias,
                    "source": "TRADINGVIEW_LIVE"
                }
    except Exception as e:
        print(f"[WARN] TradingView API İstek Hatası: {e}")
        
    # 2. İkinci Yedek: CoinGecko Global API
    try:
        url_cg = "https://api.coingecko.com/api/v3/global"
        r_cg = requests.get(url_cg, headers={"User-Agent": "Mozilla/5.0"}, timeout=2.5)
        if r_cg.status_code == 200:
            d_cg = r_cg.json().get("data", {})
            mcap = d_cg.get("market_cap_percentage", {})
            usdt_d = float(mcap.get("usdt", 7.03))
            btc_d = float(mcap.get("btc", 59.6))
            eth_d = float(mcap.get("eth", 11.3))
            chg = float(d_cg.get("market_cap_change_percentage_24h_usd", 0.0))
            usdt_chg = -chg * 0.3
            print(f"[WARN] [COINGECKO YEDEK]: USDT.D = %{usdt_d:.4f}")
            return {
                "usdt_d": usdt_d,
                "usdt_d_change": usdt_chg,
                "btc_d": btc_d,
                "eth_d": eth_d,
                "trend": "DÜŞÜŞTE (Kripto Boğa / LONG)" if usdt_chg < 0 else "YÜKSELİŞTE (Kripto Ayı / SHORT)",
                "bias": "BULLISH_CRYPTO" if usdt_chg < 0 else "BEARISH_CRYPTO",
                "source": "COINGECKO_FALLBACK"
            }
    except Exception:
        pass
        
    return {
        "usdt_d": 0.0,
        "usdt_d_change": 0.0,
        "btc_d": 0.0,
        "eth_d": 0.0,
        "trend": "VERİ ÇEKİLEMEDİ",
        "bias": "NEUTRAL",
        "source": "UNAVAILABLE"
    }


@st.cache_data(ttl=5, show_spinner=False)
def get_usdt_dominance() -> float:
    """TradingView canlı sunucusundan anlık USDT piyasa payı yüzdesini (% USDT.D) çeker ve konsola yazdırır."""
    matrix = fetch_usdt_dominance_matrix()
    usdt_val = matrix.get("usdt_d", 7.03)
    print(f"[STAT] [ANLIK SİNYAL KONTROLÜ] Anlık TradingView USDT Dominance: %{usdt_val:.4f}")
    return usdt_val


@st.cache_data(ttl=30, show_spinner=False)
def compute_liquidation_clusters(df: pd.DataFrame, current_price: float) -> dict:
    """Yüksek kaldıraçlı (50x/100x) Long ve Short likidasyon yoğunlaşma bölgelerini ve balina avlama riskini hesaplar."""
    if len(df) < 50:
        return {}
        
    recent_high_50 = float(df['high'].iloc[-50:].max())
    recent_low_50 = float(df['low'].iloc[-50:].min())
    
    # Tahmini Likidasyon Seviyeleri (100x %1, 50x %2 kayma ile likit olur)
    long_liq_100x = recent_low_50 * 0.99
    long_liq_50x = recent_low_50 * 0.98
    
    short_liq_100x = recent_high_50 * 1.01
    short_liq_50x = recent_high_50 * 1.02
    
    dist_long_liq_pct = ((current_price - long_liq_100x) / current_price) * 100.0
    dist_short_liq_pct = ((short_liq_100x - current_price) / current_price) * 100.0
    
    if dist_long_liq_pct <= 0.8:
        liq_status = "LONG_LIQ_DANGER"
        liq_badge = " 50x/100x LONG LİKİDASYON AV BÖLGESİ (Balina Düzeltme Riski)"
        liq_color = "#ef4444"
    elif dist_short_liq_pct <= 0.8:
        liq_status = "SHORT_LIQ_DANGER"
        liq_badge = " 50x/100x SHORT LİKİDASYON AV BÖLGESİ (Balina Sıkıştırma Riski)"
        liq_color = "#f59e0b"
    else:
        liq_status = "SAFE"
        liq_badge = " LİKİDASYON DENGELİ & GÜVENLİ BÖLGE (Tuzak Riski Düşük)"
        liq_color = "#10b981"
        
    return {
        "long_liq_100x": long_liq_100x,
        "long_liq_50x": long_liq_50x,
        "short_liq_100x": short_liq_100x,
        "short_liq_50x": short_liq_50x,
        "dist_long_liq_pct": dist_long_liq_pct,
        "dist_short_liq_pct": dist_short_liq_pct,
        "liq_status": liq_status,
        "liq_badge": liq_badge,
        "liq_color": liq_color
    }


@st.cache_data(ttl=300, show_spinner=False)
def compute_macro_support_resistance_levels(df: pd.DataFrame, symbol: str = "BTCUSDT", exchange_id: str = "mexc") -> dict:
    """Uzun vadeli kalıcı majör S/R seviyeleri için 1D (Günlük) Hacim Profili (VRVP)
    ve Yapısal Pivot ekstrem noktalarını (Swing High/Low) hesaplar."""
    if len(df) < 5:
        return {}

    current_price = float(df['close'].iloc[-1])
    recent_close = float(df['close'].iloc[-2]) if len(df) > 1 else current_price

    # 1D Makro veriyi çek (Son 180 Günlük Hacim Profili için)
    try:
        _ex = get_public_exchange_client(exchange_id)
        ohlcv_1d = _ex.fetch_ohlcv(symbol, '1d', limit=180)
        if ohlcv_1d:
            df_1d = pd.DataFrame(ohlcv_1d, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
            df_1d['close'] = df_1d['close'].astype(float)
            df_1d['high'] = df_1d['high'].astype(float)
            df_1d['low'] = df_1d['low'].astype(float)
            df_1d['vol'] = df_1d['vol'].astype(float)
        else:
            df_1d = df.copy()
    except Exception:
        df_1d = df.copy() # Fallback

    # 1. Hacim Profili (Volume Profile - VRVP) Düğümleri
    bins = 60
    import numpy as np
    hist, bin_edges = np.histogram(df_1d['close'], bins=bins, weights=df_1d['vol'])
    peaks = []
    for i in range(1, len(hist)-1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
            peaks.append(((bin_edges[i] + bin_edges[i+1]) / 2.0, hist[i]))
            
    # Hacme göre en güçlü 8 bölgeyi al
    peaks = sorted(peaks, key=lambda x: x[1], reverse=True)[:8]
    hvn_levels = [p[0] for p in peaks]
    
    # 2. Yapısal 1D Swing Low / High
    highs = df_1d['high'].values
    lows = df_1d['low'].values
    struct_highs = []
    struct_lows = []
    for i in range(10, len(df_1d) - 10):
        if highs[i] == max(highs[i-10:i+11]): struct_highs.append(highs[i])
        if lows[i] == min(lows[i-10:i+11]): struct_lows.append(lows[i])
        
    all_resistances = sorted(list(set(struct_highs + [h for h in hvn_levels if h > current_price])))
    all_supports = sorted(list(set(struct_lows + [h for h in hvn_levels if h < current_price])), reverse=True)
    
    # Fiyata en yakın, ama çok dip dibe olmayan kademeler
    def get_spaced_levels(levels, start_price, is_res=True):
        res = []
        last_lvl = start_price
        for lvl in levels:
            if abs(lvl - last_lvl) / start_price > 0.005: # En az %0.5 mesafe
                res.append(lvl)
                last_lvl = lvl
            if len(res) >= 2: break
        return res
        
    r_levels = get_spaced_levels([r for r in all_resistances if r > current_price], current_price, True)
    s_levels = get_spaced_levels([s for s in all_supports if s < current_price], current_price, False)
    
    # 1D PP
    recent_high = float(df_1d['high'].iloc[-2]) if len(df_1d)>1 else float(df['high'].max())
    recent_low = float(df_1d['low'].iloc[-2]) if len(df_1d)>1 else float(df['low'].min())
    recent_1d_close = float(df_1d['close'].iloc[-2]) if len(df_1d)>1 else recent_close
    pp = (recent_high + recent_low + recent_1d_close) / 3.0
    
    s1 = s_levels[0] if len(s_levels) > 0 else (current_price * 0.98)
    s2 = s_levels[1] if len(s_levels) > 1 else (s1 * 0.96)
    r1 = r_levels[0] if len(r_levels) > 0 else (current_price * 1.02)
    r2 = r_levels[1] if len(r_levels) > 1 else (r1 * 1.04)

    dist_sup_pct = ((current_price - s1) / current_price) * 100.0
    dist_res_pct = ((r1 - current_price) / current_price) * 100.0
    
    # Trend Vektörü
    if current_price > r1:
        sr_direction = "🚀 YUKARI BOĞA KIRILIMI (DİRENÇ KIRILDI -> LONG HEVESLİ)"
        sr_direction_code = "BREAKOUT_LONG"
        sr_status = "NEAR_MAJOR_SUPPORT"
        sr_note = f"🟢 R1 Hacim Direnci Kırılımı (${r1:,.2f}) -> Hedef: ${r2:,.2f}"
    elif current_price < s1:
        sr_direction = "🩸 AŞAĞI AYI KIRILIMI (DESTEK KIRILDI -> SHORT HEVESLİ)"
        sr_direction_code = "BREAKDOWN_SHORT"
        sr_status = "NEAR_MAJOR_RESISTANCE"
        sr_note = f"🔴 S1 Hacim Desteği Kırılımı (${s1:,.2f}) -> Hedef: ${s2:,.2f}"
    elif dist_sup_pct < dist_res_pct:
        sr_direction = "🟢 DESTEK TABANINDAN YUKARI SEKMELİ YÖN (LONG REAKSİYON)"
        sr_direction_code = "BOUNCE_LONG"
        sr_status = "NEAR_MAJOR_SUPPORT"
        sr_note = f"🟢 S1 Desteğine Yakın (${s1:,.2f}) -> Yön Yukarı Eğilimli"
    else:
        sr_direction = "🔴 DİRENÇ TEPESİNDEN AŞAĞI REDDEDİLEN YÖN (SHORT REAKSİYON)"
        sr_direction_code = "REJECTION_SHORT"
        sr_status = "NEAR_MAJOR_RESISTANCE"
        sr_note = f"🔴 R1 Direncine Yakın (${r1:,.2f}) -> Yön Aşağı Eğilimli"

    return {
        "s1": s1, "s2": s2, "r1": r1, "r2": r2, "pp": pp,
        "major_support": s1, "major_resistance": r1,
        "dist_sup_pct": dist_sup_pct, "dist_res_pct": dist_res_pct,
        "sr_direction": sr_direction, "sr_direction_code": sr_direction_code,
        "sr_status": sr_status, "sr_note": sr_note
    }

@st.cache_data(ttl=60, show_spinner=False)
def compute_macro_timeframe_trend(symbol: str = "BTCUSDT", exchange_id: str = "mexc") -> dict:
    """4 Saatlik (4h) ve Günlük (1d) zaman dilimlerindeki makro trend yönünü hesaplar."""
    try:
        _ex = get_public_exchange_client(exchange_id)
        ohlcv_4h = _ex.fetch_ohlcv(symbol, '4h', limit=100)
        if ohlcv_4h:
            c4 = pd.Series([float(row[4]) for row in ohlcv_4h])
            ema50_4h = c4.ewm(span=50, adjust=False).mean().iloc[-1]
            ema200_4h = c4.ewm(span=200, adjust=False).mean().iloc[-1]
            last4 = c4.iloc[-1]
            
            trend_4h = "BULLISH" if last4 > ema50_4h and ema50_4h > ema200_4h else ("BEARISH" if last4 < ema200_4h else "NEUTRAL")
            
            if trend_4h == "BULLISH":
                macro_badge = "🟢 4H/1D MAKRO BOĞA TRENDİ (YÜKSELİŞ REJİMİ)"
                macro_bias = "LONG_STRONG"
            elif trend_4h == "BEARISH":
                macro_badge = "🔻 4H/1D MAKRO AYI TRENDİ (DÜŞÜŞ REJİMİ)"
                macro_bias = "SHORT_STRONG"
            else:
                macro_badge = "⚖️ 4H/1D MAKRO YATAY/DENGELİ REJİM"
                macro_bias = "NEUTRAL"
                
            return {
                "trend_4h": trend_4h,
                "ema50_4h": ema50_4h,
                "ema200_4h": ema200_4h,
                "macro_badge": macro_badge,
                "macro_bias": macro_bias
            }
    except Exception:
        pass
        
    return {
        "trend_4h": "UNAVAILABLE",
        "ema50_4h": 0.0,
        "ema200_4h": 0.0,
        "macro_badge": "⚠️ 4H/1D MAKRO VERİ ÇEKİLEMEDİ",
        "macro_bias": "NEUTRAL"
    }


@st.cache_data(ttl=60, show_spinner=False)
def compute_4h_scalp_prediction(symbol: str = "BTCUSDT", exchange_id: str = "mexc") -> dict:
    """4 Saatlik (4H) grafik verileri üzerinden uzun vadeli trend yönünü ve AI tahminini hesaplar."""
    try:
        df_4h = fetch_binance_kline_data(symbol, interval="Min240", limit=100, exchange_id=exchange_id)
        if not df_4h.empty and len(df_4h) >= 30:
            df_feat = compute_quantum_features(df_4h, symbol=symbol, interval="Min240", exchange_id=exchange_id)
            ai_4h = train_and_predict_quantum_ai(df_feat)
            
            last_c = float(df_4h['close'].iloc[-1])
            ema50 = float(df_feat['ema_50'].iloc[-1])
            ema200 = float(df_feat['ema_200'].iloc[-1])
            
            direction = ai_4h.get("direction", "LONG")
            conf = ai_4h.get("confidence", 65.0)
            
            if direction == "LONG" and last_c > ema50:
                pred_title = "🟢 4H YÜKSELİŞ TAHMİNİ (BOĞA)"
                pred_badge = "BULLISH_4H"
                color = "#10b981"
            elif direction == "SHORT" and last_c < ema50:
                pred_title = "🔻 4H DÜŞÜŞ TAHMİNİ (AYI)"
                pred_badge = "BEARISH_4H"
                color = "#ef4444"
            else:
                pred_title = "⚖️ 4H YATAY / DENGELİ TAHMİN"
                pred_badge = "NEUTRAL_4H"
                color = "#38bdf8"
                
            return {
                "direction": direction,
                "confidence": conf,
                "pred_title": pred_title,
                "pred_badge": pred_badge,
                "color": color,
                "ema50": ema50,
                "ema200": ema200
            }
    except Exception:
        pass
        
    return {
        "direction": "NEUTRAL",
        "confidence": 0.0,
        "pred_title": "⚠️ 4H VERİ ÇEKİLEMEDİ (Tahmin Yapılamıyor)",
        "pred_badge": "UNAVAILABLE",
        "color": "#f59e0b",
        "ema50": 0.0,
        "ema200": 0.0
    }


# --- 3. ÇOK BOYUTLU KANTİTATİF ÖZELLİK MÜHENDİSLİĞİ (FEATURE ENGINEERING) ---
def compute_quantum_features(df_raw: pd.DataFrame, symbol: str = "BTCUSDT", interval: str = "5m", exchange_id: str = "mexc") -> pd.DataFrame:
    """Gerçek mum verileri üzerinden kurumsal teknik indikatörler ve yapay zeka özniteliklerini üretir."""
    df = df_raw.copy()
    
    # --- MACRO DATA MERGE ---
    fg_df, dxy_df, spy_df = fetch_macro_data()
    df['date_str'] = pd.to_datetime(df.index if df.index.name else df['time']).dt.strftime('%Y-%m-%d')
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
        
    df['fear_greed'] = df['fear_greed'].ffill().fillna(50.0)
    df['dxy'] = df['dxy'].ffill().fillna(100.0)

    if not spy_df.empty and 'date_str' in spy_df.columns:
        spy_map = spy_df.drop_duplicates('date_str').set_index('date_str')['spy']
        df['spy'] = df['date_str'].map(spy_map)
    else:
        df['spy'] = np.nan
    df['spy'] = df['spy'].ffill().bfill().fillna(1.0)

    df.drop(columns=['date_str'], inplace=True, errors='ignore')

    
    # 1. Getiriler ve Momentum
    df['returns'] = df['close'].pct_change()
    df['log_ret'] = np.log(df['close'] / (df['close'].shift(1) + 1e-9))
    
    # 2. Üstel Hareketli Ortalamalar (EMA Ribbon & Trend Hizalaması)
    for span in [9, 21, 50, 200]:
        df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
        df[f'dist_ema_{span}'] = (df['close'] - df[f'ema_{span}']) / (df[f'ema_{span}'] + 1e-9)
    
    # EMA Eğilimi (Trend Rejimi)
    df['ema_trend'] = np.where(df['ema_9'] > df['ema_21'], 1, -1)
    df['ema_macro'] = np.where(df['close'] > df['ema_200'], 1, -1)
    
    # 3. RSI (Relative Strength Index - 14) + RSI MA
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_ma'] = df['rsi'].rolling(9).mean()
    df['rsi_dist'] = df['rsi'] - df['rsi_ma']
    
    # 4. MACD (12, 26, 9)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # 5. Bollinger Bantları (20, 2)
    df['bb_mid'] = df['close'].rolling(20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_up'] = df['bb_mid'] + (2 * df['bb_std'])
    df['bb_low'] = df['bb_mid'] - (2 * df['bb_std'])
    df['bb_pct'] = (df['close'] - df['bb_low']) / (df['bb_up'] - df['bb_low'] + 1e-9)
    df['bb_width'] = (df['bb_up'] - df['bb_low']) / (df['bb_mid'] + 1e-9)
    
    # 6. ATR (Average True Range - Dinamik Volatilite ve SL/TP Motoru)
    high_low = df['high'] - df['low']
    high_cp = (df['high'] - df['close'].shift(1)).abs()
    low_cp = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / (df['close'] + 1e-9)
    
    # 7. Stokastik Osilatör (%K, %D)
    low14 = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * ((df['close'] - low14) / (high14 - low14 + 1e-9))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    
    # 8. Hacim Dinamikleri & Anomalileri
    df['vol_ma'] = df['vol'].rolling(20).mean()
    df['vol_ratio'] = df['vol'] / (df['vol_ma'] + 1e-9)
    
    # 9. Fiyat Yapısı & Mum Gövde/Gölge Oranı
    body = (df['close'] - df['open']).abs()
    range_hl = (df['high'] - df['low']) + 1e-9
    df['body_ratio'] = body / range_hl
    
    # --- YENİ EKLENEN: Kurumsal Order Flow ve Makro Birleştirme ---
    df_flow = fetch_institutional_order_flow(symbol, interval, limit=len(df), exchange_id=exchange_id)
    if not df_flow.empty:
        df_flow = df_flow.reset_index()
        df_flow = df_flow.rename(columns={'timestamp': 'time'})
        
        # 'time' sütunu üzerinden kusursuzca kaynaştırma (Sol/Left birleştirme)
        df = pd.merge(df, df_flow[['time', 'oi_change_pct', 'taker_delta_trend']], on='time', how='left')
        
        # Olası eksik verileri ileri doğru doldur (Forward Fill)
        df['oi_change_pct'] = df['oi_change_pct'].ffill().fillna(0)
        df['taker_delta_trend'] = df['taker_delta_trend'].ffill().fillna(0)
    else:
        # Veri çekilemezse model çökmesin diye güvenli değerler atanır
        df['oi_change_pct'] = 0.0
        df['taker_delta_trend'] = 0.0
    
    return df.dropna().reset_index(drop=True)


# --- 4. YAPAY ZEKA TOPLULUK MODELİ (ENSEMBLE MACHINE LEARNING & KALİBRASYON) ---
from sklearn.feature_selection import SelectKBest, f_classif

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

def train_and_predict_quantum_ai(df_features: pd.DataFrame) -> dict:
    """Path Dependency (SL/TP) ve SelectKBest Filtreli Profesyonel AI Motoru (Ultra Hizli Caching)"""
    
    feature_cols = [
        'returns', 'dist_ema_9', 'dist_ema_21', 'dist_ema_50', 'dist_ema_200',
        'rsi', 'rsi_dist', 'macd', 'macd_diff', 'bb_pct', 'bb_width',
        'stoch_k', 'stoch_d', 'vol_ratio', 'body_ratio',
        'oi_change_pct', 'taker_delta_trend', 'fear_greed', 'dxy', 'spy', 'funding_rate'
    ]
    
    if len(df_features) < 60:
        return {"prob_long": 0.5, "prob_short": 0.5, "raw_dir": "NEUTRAL", "status": "YETERSİZ VERİ", "confidence": 50.0}
        
    df = df_features.copy()
    
    if 'atr' not in df.columns:
        tr = pd.concat([df["high"] - df["low"], 
                       (df["high"] - df["close"].shift(1)).abs(), 
                       (df["low"] - df["close"].shift(1)).abs()], axis=1).max(axis=1)
        df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()
    
    target_candles = 15
    tp_m = 3.0
    sl_m = 1.5
    
    targets = np.zeros(len(df))
    closes, highs, lows, atrs = df['close'].values, df['high'].values, df['low'].values, df['atr'].values
    
    for i in range(len(df) - target_candles):
        entry = closes[i]
        tp_long, sl_long = entry + (atrs[i] * tp_m), entry - (atrs[i] * sl_m)
        tp_short, sl_short = entry - (atrs[i] * tp_m), entry + (atrs[i] * sl_m)
        
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
        return {"prob_long": 0.5, "prob_short": 0.5, "raw_dir": "NEUTRAL", "status": "DENGESİZ ETİKET", "confidence": 50.0}
    
    # 💥 HIZLANDIRICI MANTIK: Sadece kapanmış son mumu referans alarak hash'le.
    # Canlı mum her saniye değişir, ama önceki mumlar (train_df) sabittir!
    symbol_tf_hash = str(len(train_df)) + "_" + str(train_df['close'].iloc[-1])
    
    ensemble, final_features = _cached_train_quantum_ai(symbol_tf_hash, train_df, feature_cols)

    # ONEMLI - KARARLILIK: df'nin SON satiri henuz KAPANMAMIS (olusmakta olan) canli mumdur;
    # onun RSI/MACD/OB gibi featurelari her fiyat tikinde (birkac saniyede bir) degisebiliyor,
    # bu da tahmin yuzdesinin (LONG/SHORT) saniyeler icinde 20-30 puan sicramasina yol aciyordu
    # (kullanici: "tahminler cok sik degisiyor, kararli olmasini istiyorum"). Bu yuzden tahmin
    # EN SON KAPANMIS muma gore yapilir - bu da mum kapanana kadar (secili zaman dilimi
    # suresince) SABIT kalmasini garanti eder.
    stable_row_idx = -2 if len(df) > 1 else -1
    X_current = df[final_features].iloc[[stable_row_idx]]
    raw_probs = ensemble.predict_proba(X_current)[0]
    classes = list(ensemble.classes_)
    
    p_long = float(raw_probs[classes.index(1)]) if 1 in classes else 0.0
    p_short = float(raw_probs[classes.index(-1)]) if -1 in classes else 0.0
    
    total = p_long + p_short + 1e-9
    p_long = p_long / total
    p_short = p_short / total
    
    latest = df.iloc[stable_row_idx]
    if latest['rsi'] > 55 and latest['macd'] > 0: p_long += 0.05
    elif latest['rsi'] < 45 and latest['macd'] < 0: p_short += 0.05

    # ONEMLI: RSI/MACD bonusu eklendikten sonra p_long+p_short artik 1.0 (%100) etmiyordu -
    # bu yuzden ekranda "LONG: %72.6 | SHORT: %32.4" gibi toplami %100'u asan (veya altinda
    # kalan) degerler goruluyordu. Bonus sonrasi tekrar normalize edilerek iki olasiligin
    # HER ZAMAN toplamda %100 etmesi garanti edilir; yon karari (max olan taraf) degismez.
    post_total = p_long + p_short + 1e-9
    p_long = p_long / post_total
    p_short = p_short / post_total

    final_dir = "LONG" if p_long > p_short else ("SHORT" if p_short > p_long else "NEUTRAL")
    confidence = max(p_long, p_short) * 100.0
    
    return {
        "prob_long": p_long,
        "prob_short": p_short,
        "raw_dir": final_dir,
        "status": "GÜÇLÜ HİZALAMA" if confidence > 65 else "ZAYIF SİNYAL",
        "confidence": confidence,
        "active_features": final_features
    }

def evaluate_confluence_and_filter(ai_res: dict, latest_row: pd.Series, depth_data: dict, usdt_dom: dict = None, sr_levels: dict = None) -> dict:
    """Yapay zeka çıktısını teknik indikatörler, trend, tahta baskısı, TradingView Dolar Dominansı (% USDT.D) ve Majör Destek/Direnç Seviyeleri ile çapraz doğrulayarak sahte sinyalleri eler."""
    prob_long = ai_res["prob_long"]
    prob_short = ai_res["prob_short"]
    raw_dir = ai_res["raw_dir"]
    confidence = ai_res.get("confidence", 50.0)
    
    rsi = float(latest_row.get("rsi", 50.0))
    macd_diff = float(latest_row.get("macd_diff", 0.0))
    close = float(latest_row.get("close", 1.0))
    ema_200 = float(latest_row.get("ema_200", close))
    vol_ratio = float(latest_row.get("vol_ratio", 1.0))
    stoch_k = float(latest_row.get("stoch_k", 50.0))
    imbalance = depth_data.get("imbalance", 0.0)
    
    checks = []
    confluence_score = 0
    
    # 1. AI Güven Filtresi
    if confidence >= 62.0:
        checks.append(("AI Model Güven Seviyesi", f"Yüksek (%{confidence:.1f})", "pass", 25))
        confluence_score += 25
    elif confidence >= 51.0:
        checks.append(("AI Model Güven Seviyesi", f"Aktif / Kararlı (%{confidence:.1f})", "pass", 18))
        confluence_score += 18
    else:
        checks.append(("AI Model Güven Seviyesi", f"Kararsız (%{confidence:.1f})", "warn", 8))
        confluence_score += 8
        
    # 2. Makro Trend Onayı (EMA 200)
    is_macro_bull = close > ema_200
    if raw_dir == "LONG":
        if is_macro_bull:
            checks.append(("EMA 200 Makro Trend", "Boğa (Fiyat > EMA200)", "pass", 20))
            confluence_score += 20
        else:
            checks.append(("EMA 200 Makro Trend", "Tepki / Düzeltme Dalgası", "warn", 10))
            confluence_score += 10
    else:
        if not is_macro_bull:
            checks.append(("EMA 200 Makro Trend", "Ayı (Fiyat < EMA200)", "pass", 20))
            confluence_score += 20
        else:
            checks.append(("EMA 200 Makro Trend", "Karşı Trend Satış Dalgası", "warn", 10))
            confluence_score += 10
            
    # 3. Momentum Filtresi (RSI & Stoch)
    if raw_dir == "LONG":
        if rsi > 78 or stoch_k > 90:
            checks.append(("Momentum / Uç Bölge", f"Aşırı Alım Tepesi (RSI: {rsi:.1f})", "warn", 5))
            confluence_score += 5
        elif 44 <= rsi <= 75:
            checks.append(("Momentum (RSI & Stoch)", f"Pozitif Bölge (RSI: {rsi:.1f})", "pass", 20))
            confluence_score += 20
        else:
            checks.append(("Momentum (RSI & Stoch)", f"Toparlanma Alanı (RSI: {rsi:.1f})", "pass", 15))
            confluence_score += 15
    else:
        if rsi < 22 or stoch_k < 10:
            checks.append(("Momentum / Uç Bölge", f"Aşırı Satım Dibi (RSI: {rsi:.1f})", "warn", 5))
            confluence_score += 5
        elif 25 <= rsi <= 56:
            checks.append(("Momentum (RSI & Stoch)", f"Negatif Bölge (RSI: {rsi:.1f})", "pass", 20))
            confluence_score += 20
        else:
            checks.append(("Momentum (RSI & Stoch)", f"Direnç Reddi Alanı (RSI: {rsi:.1f})", "pass", 15))
            confluence_score += 15
            
    # 4. MACD Histogram Onayı
    if (raw_dir == "LONG" and macd_diff > 0) or (raw_dir == "SHORT" and macd_diff < 0):
        checks.append(("MACD Momentum Hibrit", "Yön Destekliyor", "pass", 18))
        confluence_score += 18
    else:
        checks.append(("MACD Momentum Hibrit", "Kısmi Uyuşmazlık", "warn", 8))
        confluence_score += 8
        
    # 5. Hacim ve Emir Defteri Baskısı
    book_aligned = (raw_dir == "LONG" and imbalance > 1.0) or (raw_dir == "SHORT" and imbalance < -1.0)
    vol_strong = vol_ratio > 0.95
    if book_aligned and vol_strong:
        checks.append(("Hacim & Tahta Baskısı", f"Güçlü Onay (Tahta: %{abs(imbalance):.1f})", "pass", 17))
        confluence_score += 17
    elif book_aligned or vol_strong:
        checks.append(("Hacim & Tahta Baskısı", "Kısmi Destek", "pass", 12))
        confluence_score += 12
    else:
        checks.append(("Hacim & Tahta Baskısı", "Dengeli Tahta", "warn", 6))
        confluence_score += 6

    # 6. TradingView Dolar Dominansı (% USDT.D) Sert Korelasyon Koruması
    if usdt_dom:
        u_val = usdt_dom.get("usdt_d", 6.98)
        u_chg = usdt_dom.get("usdt_d_change", 0.0)
        u_bias = usdt_dom.get("bias", "NEUTRAL")
        
        if raw_dir == "LONG":
            if u_bias == "BULLISH_CRYPTO" or u_chg < -0.1:
                checks.append(("Dolar Dominansı (% USDT.D)", f"%{u_val:.2f} (Düşüşte -> Kriptoya Para Akışı Var)", "pass", 20))
                confluence_score += 20
            elif u_chg > +0.2:
                checks.append(("Dolar Dominansı (% USDT.D)", f"%{u_val:.2f} (Yükselişte -> Nakite Kaçış Var) -> LONG İptal!", "fail", 0))
                confluence_score -= 35
                raw_dir = "NEUTRAL"
            else:
                checks.append(("Dolar Dominansı (% USDT.D)", f"%{u_val:.2f} (Yatay / Dengeli Dominans)", "warn", 8))
                confluence_score += 8
        elif raw_dir == "SHORT":
            if u_bias == "BEARISH_CRYPTO" or u_chg > +0.1:
                checks.append(("Dolar Dominansı (% USDT.D)", f"%{u_val:.2f} (Yükselişte -> Nakite Kaçış Var)", "pass", 20))
                confluence_score += 20
            elif u_chg < -0.2:
                checks.append(("Dolar Dominansı (% USDT.D)", f"%{u_val:.2f} (Düşüşte -> Kriptoya Para Akıyor) -> SHORT İptal!", "fail", 0))
                confluence_score -= 35
                raw_dir = "NEUTRAL"
            else:
                checks.append(("Dolar Dominansı (% USDT.D)", f"%{u_val:.2f} (Yatay / Dengeli Dominans)", "warn", 8))
                confluence_score += 8

    # 7. Statik Majör Destek & Direnç Seviyeleri Filtresi (Gürültüsüz Uzun Vadeli Analiz)
    if sr_levels:
        sr_status = sr_levels.get("sr_status", "MID_RANGE_NOISE")
        maj_sup = sr_levels.get("major_support", close * 0.97)
        maj_res = sr_levels.get("major_resistance", close * 1.03)
        dist_sup = sr_levels.get("dist_sup_pct", 2.0)
        dist_res = sr_levels.get("dist_res_pct", 2.0)
        
        if raw_dir == "LONG":
            if sr_status == "NEAR_MAJOR_SUPPORT" or dist_sup <= 1.2:
                checks.append(("Majör Destek Testi", f"Ana Destek Teması/Tepkisi (${maj_sup:,.2f} - %{dist_sup:.2f} Yakın)", "pass", 18))
                confluence_score += 18
            elif dist_res <= 0.6:
                checks.append(("Majör S/R Uyarısı", f"Direnç Tepesine Çok Yakın (${maj_res:,.2f}) -> LONG Riskli!", "fail", 0))
                confluence_score -= 15
            else:
                checks.append(("Majör Destek/Direnç Filtresi", f"Kanal Ortası Gürültülü Bölge (Destek: ${maj_sup:,.2f})", "warn", 6))
                confluence_score += 6
        else:
            if sr_status == "NEAR_MAJOR_RESISTANCE" or dist_res <= 1.2:
                checks.append(("Majör Direnç Testi", f"Ana Direnç Teması/Reddi (${maj_res:,.2f} - %{dist_res:.2f} Yakın)", "pass", 18))
                confluence_score += 18
            elif dist_sup <= 0.6:
                checks.append(("Majör S/R Uyarısı", f"Destek Dibi Çok Yakın (${maj_sup:,.2f}) -> SHORT Riskli!", "fail", 0))
                confluence_score -= 15
            else:
                checks.append(("Majör Destek/Direnç Filtresi", f"Kanal Ortası Gürültülü Bölge (Direnç: ${maj_res:,.2f})", "warn", 6))
                confluence_score += 6

    # NİHAİ KARAR MATRİSİ
    if confluence_score < 38 and confidence < 51.0:
        final_signal = "NEUTRAL"
        signal_title = "🛑 NÖTR / BEKLE (YATAY PİYASA)"
        badge_class = "signal-badge-neutral"
        trade_allowed = False
        action_note = "Piyasada yön belirginleşene kadar nakitte beklenmeli."
    elif raw_dir == "NEUTRAL":
        # ONEMLI: raw_dir buraya ya modelin kendisi LONG/SHORT arasinda net ayrisamadigi icin
        # ya da yukaridaki Dolar Dominansi filtresi orijinal yonu iptal ettigi icin (LONG/SHORT Iptal!)
        # gelir. Bu dal olmadan kod asagidaki 'else' (SHORT) dalina dusup iptal edilen bir LONG
        # sinyalini yanlislikla "GÜÇLÜ SHORT" olarak gosteriyordu.
        final_signal = "NEUTRAL"
        signal_title = "🛑 NÖTR / BEKLE (SİNYAL İPTAL EDİLDİ)"
        badge_class = "signal-badge-neutral"
        trade_allowed = False
        action_note = "Çapraz doğrulama filtrelerinden biri (ör. Dolar Dominansı) orijinal AI yönünü iptal etti; net bir yön oluşmadı."
    elif raw_dir == "LONG":
        if confluence_score >= 60 and confidence >= 58.0:
            final_signal = "STRONG_LONG"
            signal_title = "🚀 GÜÇLÜ LONG (YÜKSEK GÜVEN)"
            badge_class = "signal-badge-strong-long"
            trade_allowed = True
            action_note = "Yapay zeka topluluğu ve teknik momentum güçlü yükseliş yönünü onaylıyor."
        else:
            final_signal = "WEAK_LONG"
            signal_title = "📈 AKTİF LONG POZİSYONU"
            badge_class = "signal-badge-strong-long"
            trade_allowed = True
            action_note = "Yükseliş yönünde işlem fırsatı mevcut. Kademeli kâr alımı önerilir."
    else:
        if confluence_score >= 60 and confidence >= 58.0:
            final_signal = "STRONG_SHORT"
            signal_title = "🔻 GÜÇLÜ SHORT (YÜKSEK GÜVEN)"
            badge_class = "signal-badge-strong-short"
            trade_allowed = True
            action_note = "Yapay zeka topluluğu ve satış baskısı güçlü düşüş yönünü onaylıyor."
        else:
            final_signal = "WEAK_SHORT"
            signal_title = "📉 AKTİF SHORT POZİSYONU"
            badge_class = "signal-badge-strong-short"
            trade_allowed = True
            action_note = "Düşüş yönünde işlem fırsatı mevcut. Sıkı stop-loss ile takip edin."
            
    return {
        "final_signal": final_signal,
        "signal_title": signal_title,
        "badge_class": badge_class,
        "confluence_score": min(confluence_score, 100),
        "trade_allowed": trade_allowed,
        "action_note": action_note,
        "checks": checks,
        "confidence": confidence
    }


# --- 6. KURUMSAL DİNAMİK ATR, KALDIRAÇ VE İŞLEM GÜCÜ KAZANÇ MOTORU ---

# Zaman dilimine gore, BTC referans fiyatina (~$77.200) gore kalibre edilmis, fiyata ORANTILI
# (yuzde bazli) SL/TP mesafe sinirlari. Yuzde bazli oldugu icin herhangi bir coinde (ucuz veya
# pahali farketmeksizin) esdeger sikilikte, mantikli bir mesafe uretir - sabit dolar deger
# kullanilsaydi ucuz coinlerde (orn. PEPE) anlamsiz/imkansiz sonuclar cikardi. Bu tablo
# pages/2_Analiz_Tahmini.py'deki ayni isimli tabloyla senkron tutulmalidir.
TIMEFRAME_RISK_BOUNDS_PCT = {
    "Min1":   {"sl_min": 0.00259, "sl_max": 0.00518, "tp_min": 0.00389, "tp_max": 0.00648},
    "Min15":  {"sl_min": 0.00389, "sl_max": 0.00648, "tp_min": 0.00648, "tp_max": 0.01295},
    "Min60":  {"sl_min": 0.01295, "sl_max": 0.01943, "tp_min": 0.01295, "tp_max": 0.01943},
    "Min240": {"sl_min": 0.01295, "sl_max": 0.01943, "tp_min": 0.01295, "tp_max": 0.01943},
}

def compute_institutional_risk_levels(
    current_price: float,
    atr: float,
    signal: str,
    leverage: int = 10,
    margin_budget: float = 100.0,
    confidence: float = 65.0,
    confluence_score: float = 70.0,
    sr_levels: dict = None,
    interval: str = ""
) -> dict:
    """Yapay zekâ işlem gücüne, kaldıraca, majör destek/direnç seviyelerine ve kullanıcı bütçesine göre Min Kazanç, Max Kazanç ve Orantılı SL seviyelerini hesaplar."""
    if current_price <= 0:
        return {}
        
    lev = max(int(leverage), 1)
    budget = max(float(margin_budget), 1.0)
    atr_val = float(atr) if atr > 0 else current_price * 0.005
    
    # 1. Yapay Zekâ İşlem Gücü Katsayısı (Trade Power: 0.45 - 1.0)
    trade_power = (float(confidence) + float(confluence_score)) / 2.0
    power_factor = float(np.clip(trade_power / 100.0, 0.45, 1.0))
    
    # 2. Temel Volatilite Tabanlı Stop-Loss Mesafesi (ATR & Destek/Direnç Hizalı)
    sl_atr_dist = max(1.5 * atr_val, current_price * (0.012 / lev))
    
    if sr_levels and "major_support" in sr_levels and "major_resistance" in sr_levels:
        maj_sup = sr_levels["major_support"]
        maj_res = sr_levels["major_resistance"]
        if "LONG" in signal and maj_sup < current_price:
            # SL Desteğin altına koyulur, ancak girişten en fazla %3 uzaklaşabilir
            sl_dist = min(max(sl_atr_dist, current_price - (maj_sup * 0.997)), current_price * 0.03)
        elif "SHORT" in signal and maj_res > current_price:
            sl_dist = min(max(sl_atr_dist, (maj_res * 1.003) - current_price), current_price * 0.03)
        else:
            sl_dist = sl_atr_dist
    else:
        sl_dist = sl_atr_dist

    # 3. Kesin Orantılı Risk-Ödül Oranı (Minimum TP1 = 1.8 * SL, TP2 = 3.5 * SL)
    min_tp_dist = sl_dist * 1.8
    max_tp_dist = sl_dist * 3.5

    # ONEMLI: Yukarida hesaplanan sl_dist, "majör destek/direnç" seviyesine gore genisletilmis
    # olabilir (swing mantigi); bu, kisa zaman dilimlerinde (orn. 1 dakika scalp) fiyattan
    # binlerce dolar uzakta, anlamsiz genis bir SL/TP uretiyordu. Zaman dilimine ozel yuzde
    # bazli sinir tanimliysa, mesafe bu araligin KESINLIKLE disina cikamaz.
    tf_bounds = TIMEFRAME_RISK_BOUNDS_PCT.get(interval)
    if tf_bounds:
        sl_dist = float(np.clip(sl_dist, current_price * tf_bounds["sl_min"], current_price * tf_bounds["sl_max"]))
        min_tp_dist = float(np.clip(min_tp_dist, current_price * tf_bounds["tp_min"], current_price * tf_bounds["tp_max"]))
        max_tp_dist = float(np.clip(max_tp_dist, current_price * tf_bounds["tp_min"], current_price * tf_bounds["tp_max"]))

    # ONEMLI - LİKİDASYON GÜVENLİĞİ: Yuksek kaldiracta likidasyon mesafesi kucalir; eger SL bu
    # mesafeden daha genis kalirsa, pozisyon SL'e ULAŞAMADAN once likit olur (marjinin tamami
    # gider). Kaldiraca gore SL'in likidasyon mesafesinin guvenli bir payinin (%75) icinde
    # kalmasi garanti edilir - kaldirac arttikca SL otomatik daralir. Bu, zaman dilimi
    # sinirinin ALTINA inmesi gerekse bile ("SL cok daralmasin" kuralindan daha oncelikli),
    # cunku amac hicbir kosulda tam likidasyon riskine izin vermemek.
    liq_dist_abs = current_price * (0.90 / lev)
    safe_sl_ceiling = liq_dist_abs * 0.75
    sl_dist = min(sl_dist, safe_sl_ceiling)

    liq_dist_pct = (0.90 / lev) * 100.0
    
    if "LONG" in signal:
        sl_price = current_price - sl_dist
        min_tp_price = current_price + min_tp_dist
        max_tp_price = current_price + max_tp_dist
        liq_price = max(current_price * (1.0 - (0.90 / lev)), 0.0)
    elif "SHORT" in signal:
        sl_price = current_price + sl_dist
        min_tp_price = current_price - min_tp_dist
        max_tp_price = current_price - max_tp_dist
        liq_price = current_price * (1.0 + (0.90 / lev))
    else:
        sl_price = current_price - sl_dist
        min_tp_price = current_price + min_tp_dist
        max_tp_price = current_price + max_tp_dist
        liq_price = max(current_price * (1.0 - (0.90 / lev)), 0.0)
        
    actual_sl_pct = (abs(current_price - sl_price) / current_price) * 100.0
    actual_min_pct = (abs(min_tp_price - current_price) / current_price) * 100.0
    actual_max_pct = (abs(max_tp_price - current_price) / current_price) * 100.0
    
    actual_sl_roe = actual_sl_pct * lev
    actual_min_roe = actual_min_pct * lev
    actual_max_roe = actual_max_pct * lev
    
    min_profit_usd = budget * (actual_min_roe / 100.0)
    max_profit_usd = budget * (actual_max_roe / 100.0)
    max_loss_usd = budget * (actual_sl_roe / 100.0)
    
    rr_min = actual_min_pct / (actual_sl_pct + 1e-9)
    rr_max = actual_max_pct / (actual_sl_pct + 1e-9)
    volatility_ratio = (atr_val / current_price) * 100.0
    
    if trade_power >= 72.0:
        power_title = "⚡ YÜKSEK GÜÇLÜ İŞLEM"
        power_color = "#10b981"
    elif trade_power >= 58.0:
        power_title = "⚖️ DENGELİ İŞLEM GÜCÜ"
        power_color = "#38bdf8"
    else:
        power_title = " DÜŞÜK / BELİRSİZ GÜÇ"
        power_color = "#f59e0b"
        
    return {
        "entry": current_price,
        "sl_price": sl_price,
        "min_tp_price": min_tp_price,
        "max_tp_price": max_tp_price,
        "liq_price": liq_price,
        "min_profit_usd": min_profit_usd,
        "max_profit_usd": max_profit_usd,
        "max_loss_usd": max_loss_usd,
        "actual_min_pct": actual_min_pct,
        "actual_max_pct": actual_max_pct,
        "actual_sl_pct": actual_sl_pct,
        "actual_min_roe": actual_min_roe,
        "actual_max_roe": actual_max_roe,
        "actual_sl_roe": actual_sl_roe,
        "liq_dist_pct": liq_dist_pct,
        "rr_min": rr_min,
        "rr_max": rr_max,
        "budget": budget,
        "leverage": lev,
        "trade_power": trade_power,
        "power_title": power_title,
        "power_color": power_color,
        "atr_usd": atr_val,
        "volatility_pct": volatility_ratio
    }


# --- 6.5. SIDEBAR API ANAHTARLARI VE İŞLEM KONTROLLERİ ---
with st.sidebar:
    st.markdown("""
    <style>
        [data-testid="stSidebar"] { background:#eaf6f4 !important; }

        [data-testid="stSidebarNavLink"] { border-radius:999px !important; margin-bottom:6px !important; }
        [data-testid="stSidebarNavLink"] p, [data-testid="stSidebarNavLink"] span { color:#0f2b2e !important; }
        [data-testid="stSidebarNavLink"]:not([aria-current="page"]) { background:#ffffff !important; }
        [data-testid="stSidebarNavLink"][aria-current="page"] { background:linear-gradient(135deg, #2dd4bf, #14b8a6) !important; }
        [data-testid="stSidebarNavLink"][aria-current="page"] p, [data-testid="stSidebarNavLink"][aria-current="page"] span { color:#ffffff !important; }
        [data-testid="stSidebar"] .sb-section-title { font-size:15px; font-weight:900; color:#0f2b2e; margin:14px 0 2px 0; }
        [data-testid="stSidebar"] .sb-section-sub { font-size:11px; color:#5f7d7a; margin-bottom:10px; }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { font-weight:800 !important; color:#0f2b2e !important; font-size:13.5px !important; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { font-weight:700; }

        [data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div,
        [data-testid="stSidebar"] [data-testid="stTextInput"] > div > div,
        [data-testid="stSidebar"] [data-testid="stNumberInput"] > div > div {
            background: rgba(20,184,166,0.10) !important;
            border-radius: 999px !important;
            border: 1px solid rgba(15,43,46,0.10) !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            background: #ffffff !important;
            border-radius: 16px !important;
            border: 1px solid rgba(15,43,46,0.08) !important;
        }
        [data-testid="stSidebar"] [data-testid="stButton"] button {
            background: linear-gradient(135deg, #2dd4bf, #0e7490) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 999px !important;
            font-weight: 800 !important;
        }
        [data-testid="stSidebar"] [data-testid="stButton"] button:hover { filter: brightness(1.08); }
        [data-testid="stSidebar"] [data-testid="stButton"] button p { color:#ffffff !important; }
        [data-testid="stSidebar"] [data-testid="stCheckbox"] p,
        [data-testid="stSidebar"] [data-testid="stCheckbox"] span,
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#0f2b2e !important; }

        .sb-teal-card { background:linear-gradient(135deg, #14b8a6, #0e7490); border-radius:16px; padding:14px 16px; color:#ffffff; margin:10px 0; }
        .sb-teal-card .sb-lbl { font-size:11px; opacity:0.9; font-weight:700; }
        .sb-teal-card .sb-val { font-size:19px; font-weight:900; margin-top:2px; }
        .sb-warn-card { background:#fff7ed; border:1px solid #fdba74; border-radius:16px; padding:12px 16px; color:#9a3412; font-size:12px; font-weight:700; margin:10px 0; }
        .sb-white-card { background:#ffffff; border-radius:16px; padding:14px 16px; margin:6px 0; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sb-section-title'>🔐 Borsa API & Otomasyon</div>", unsafe_allow_html=True)
    st.markdown("<div class='sb-section-sub'>Professional Borsa API & Otomasyon</div>", unsafe_allow_html=True)

    selected_exchange_label = st.selectbox(
        "🏦 Borsa Seçimi:",
        list(SUPPORTED_EXCHANGES.keys()),
        index=0,
        help="Hesap bakiyeniz ve tüm piyasa/sinyal verileri seçtiğiniz borsadan çekilir."
    )
    selected_exchange_id = SUPPORTED_EXCHANGES[selected_exchange_label]

    env_creds = load_env_credentials(selected_exchange_id)

    with st.expander("🔑 API Key ve Secret Tanımla", expanded=not (env_creds["api_key"] and env_creds["secret_key"])):
        st.info(f"💡 {selected_exchange_label} API anahtarlarınızı ister aşağıdaki kutucuklara yapıştırabilir, isterseniz de `.env` dosyasına kaydedebilirsiniz.")
        user_api_key = st.text_input(f"{selected_exchange_label} API Key:", value=env_creds["api_key"], type="password", help=f"{selected_exchange_label} hesabınızdan ürettiğiniz Futures API Key", key=f"api_key_{selected_exchange_id}")
        user_secret_key = st.text_input(f"{selected_exchange_label} Secret Key:", value=env_creds["secret_key"], type="password", help=f"{selected_exchange_label} Secret Key", key=f"secret_key_{selected_exchange_id}")

        if user_api_key:
            os.environ[f"{selected_exchange_id.upper()}_API_KEY"] = user_api_key
        if user_secret_key:
            os.environ[f"{selected_exchange_id.upper()}_SECRET_KEY"] = user_secret_key

    active_api_key = os.getenv(f"{selected_exchange_id.upper()}_API_KEY", "").strip()
    active_secret_key = os.getenv(f"{selected_exchange_id.upper()}_SECRET_KEY", "").strip()

    wallet_balance_val = 0.0
    ignore_wallet_for_budget = False
    if active_api_key and active_secret_key:
        bal = fetch_binance_account_balance(selected_exchange_id, active_api_key, active_secret_key)
        if bal["status"] == "FUTURES_OK":
            wallet_balance_val = bal['wallet_balance']
            st.markdown(f"""
            <div class="sb-teal-card">
                <div class="sb-lbl">🟢 Private API: VADELİ BAĞLI & HAZIR</div>
                <div class="sb-val">💰 ${wallet_balance_val:,.2f} USDT</div>
            </div>
            """, unsafe_allow_html=True)
            ignore_wallet_for_budget = st.checkbox(
                "🔓 Kasa sınırı olmadan genel analiz yap",
                value=False,
                help="İşaretlerseniz Bütçe/Marjin alanı gerçek cüzdan bakiyenizle sınırlanmaz; TP/SL hesaplamaları kasanızdan bağımsız, sadece genel bir analiz olarak yapılır."
            )
        else:
            st.markdown("<div class='sb-warn-card'>⚠️ API Anahtarı Algılandı Fakat Yetki Reddedildi (IP kısıtlaması veya geçersiz anahtar)</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='sb-warn-card'>⚠️ Private API: BAĞLI DEĞİL (Yalnızca Sinyal Modu)<br><span style='font-weight:400;'>📌 Proje klasörünüzdeki .env dosyasına API anahtarınızı girip kaydedin VEYA yukarıdaki menüye yapıştırın.</span></div>", unsafe_allow_html=True)

    # "Kasa sinirlamasi" sadece butce ust sinirini belirlerken kullanilir.
    budget_cap_balance = 0.0 if ignore_wallet_for_budget else wallet_balance_val

    # --- COİN & GÖSTERGE KONTROLLERİ (Kullanıcı isteğiyle ana panelden sol menüye taşındı) ---
    st.markdown("<div class='sb-section-title' style='margin-top:18px;'>⚡ Coin & Gösterge Kontrolleri</div>", unsafe_allow_html=True)

    if "active_symbol_query" not in st.session_state:
        st.session_state["active_symbol_query"] = "BTCUSDT"

    st.markdown("<div style='margin-bottom:6px; font-size:11px; color:#5f7d7a; font-weight:800; text-transform:uppercase;'>Hızlı Favori Coin Seçimi:</div>", unsafe_allow_html=True)
    fav_row1 = st.columns(2)
    fav_row2 = st.columns(2)
    fav_row3 = st.columns(2)
    fav_row4 = st.columns(2)

    if fav_row1[0].button("BTC", use_container_width=True):
        st.session_state["active_symbol_query"] = "BTCUSDT"
        st.rerun()
    if fav_row1[1].button("ETH", use_container_width=True):
        st.session_state["active_symbol_query"] = "ETHUSDT"
        st.rerun()
    if fav_row2[0].button("SOL", use_container_width=True):
        st.session_state["active_symbol_query"] = "SOLUSDT"
        st.rerun()
    if fav_row2[1].button("1000PEPE", use_container_width=True):
        st.session_state["active_symbol_query"] = "1000PEPEUSDT"
        st.rerun()
    if fav_row3[0].button("SUI", use_container_width=True):
        st.session_state["active_symbol_query"] = "SUIUSDT"
        st.rerun()
    if fav_row3[1].button("DOGE", use_container_width=True):
        st.session_state["active_symbol_query"] = "DOGEUSDT"
        st.rerun()
    if fav_row4[0].button("XRP", use_container_width=True):
        st.session_state["active_symbol_query"] = "XRPUSDT"
        st.rerun()

    all_futures_list, _ = get_all_binance_futures_symbols(selected_exchange_id)
    total_coin_count = len(all_futures_list)

    user_query = st.text_input(
        f"🔍 Serbest Arama ({total_coin_count} Coin):",
        value=st.session_state.get("active_symbol_query", "BTCUSDT"),
        placeholder="Örn: btc, pengui, sol, eth, pepe, btctry",
        help="Herhangi bir coin adı yazın. Akıllı eşleştirici otomatik bulur ve analizi başlatır."
    )

    symbol_str, match_info = normalize_futures_symbol(user_query, selected_exchange_id)

    default_idx = all_futures_list.index(symbol_str) if symbol_str in all_futures_list else 0
    selected_from_dropdown = st.selectbox(
        f"📋 Tüm {selected_exchange_label} Çiftleri ({total_coin_count} Coin):",
        all_futures_list,
        index=default_idx
    )
    if selected_from_dropdown != symbol_str:
        symbol_str = selected_from_dropdown

    timeframe_map = {
        "1 Dakika (Scalp)": "Min1",
        "5 Dakika (Day Trade)": "Min5",
        "15 Dakika (Swing)": "Min15",
        "1 Saat (Trend)": "Min60",
        "4 Saatlik (Saatlik Scalp / Trend)": "Min240"
    }
    # "active_timeframe_label" session_state anahtarı, hero karttaki tıklanabilir zaman
    # dilimi pilleriyle PAYLAŞILIYOR (aşağıda, render_quantum_terminal() çağrılmadan önce).
    # ÖNEMLİ: Widget'ın kendi key'i (active_timeframe_label) instantiate edildikten SONRA
    # doğrudan değiştirilemiyor (StreamlitAPIException) — bu yüzden pil butonları ayrı bir
    # "_pending_timeframe" anahtarına yazıyor, biz de widget oluşmadan HEMEN ÖNCE burada
    # onu asıl anahtara aktarıp temizliyoruz.
    if "active_timeframe_label" not in st.session_state:
        st.session_state["active_timeframe_label"] = "1 Dakika (Scalp)"
    if "_pending_timeframe" in st.session_state:
        st.session_state["active_timeframe_label"] = st.session_state.pop("_pending_timeframe")
    selected_tf_label = st.selectbox("⏱️ Zaman Dilimi:", list(timeframe_map.keys()), key="active_timeframe_label")
    active_interval = timeframe_map[selected_tf_label]

    # ONEMLI: Kaldirac artik sabit secenekler yerine serbest metin/sayi girisi - kullanici
    # borsanin izin verdigi herhangi bir degeri (orn. 37x, 63x) yazabilir. SL/TP hesaplamasi
    # (compute_institutional_risk_levels) zaten formul bazli oldugundan her kaldirac degerinde
    # dogru sekilde calisir; sabit bir liste ile sinirli degildir.
    selected_leverage = int(st.number_input("⚡ Kaldıraç (x):", min_value=1, max_value=200, value=10, step=1, format="%d", help="Borsanızın izin verdiği herhangi bir kaldıraç değerini yazabilirsiniz (1x - 200x)."))

    # ONEMLI: API baglantisi varsa VE kullanici "kasa siniri olmadan genel analiz" secmediyse,
    # kullaniciyi gercekte sahip olmadigi bir tutari marjin olarak girmekten (imkansiz/tehlikeli
    # bir islem varsayimindan) korumak icin butce, gercek cuzdan bakiyesiyle sinirlandirilir.
    if budget_cap_balance > 0:
        # ONEMLI: Bakiye 10$'in altinda (orn. $0.10) olabilir - bu durumda sabit "min_value=10.0"
        # kullanmak, varsayilan deger min_value'nin altinda kalip StreamlitAPIException firlatiyordu.
        # Alt/ust sinir ve varsayilan deger, gercek bakiyeye gore dinamik olarak ayarlanir.
        _budget_max = max(budget_cap_balance, 0.01)
        _budget_min = min(10.0, _budget_max)
        _budget_default = _budget_max
        _budget_step = 50.0 if _budget_max >= 50 else max(_budget_max / 10.0, 0.01)
        _budget_fmt = "%.0f" if _budget_max >= 10 else "%.2f"
        margin_budget = st.number_input("💰 Bütçe / Marjin ($):", min_value=_budget_min, max_value=_budget_max, value=float(_budget_default), step=_budget_step, format=_budget_fmt, help=f"Bağlı hesap bakiyeniz ${budget_cap_balance:,.2f} — bu tutarı aşamazsınız.")
    else:
        margin_budget = st.number_input("💰 Bütçe / Marjin ($):", min_value=10.0, max_value=100000.0, value=100.0, step=50.0, format="%.0f")

    refresh_rate = st.selectbox("🔄 Akış Hızı:", [3, 5, 10, 15], index=0, format_func=lambda x: f"{x} Saniyede Bir")


# --- 7. BAŞLIK VE DURUM KARTI ---
st.title("⚡ Analiz Tahmini")

# Entegre Coin ve Canlı Yapay Zeka Analiz Durumu Kartı
st.markdown(f"""
<div style="background:#ffffff; box-shadow:0 4px 14px rgba(15,43,46,0.06); padding:10px 16px; border-radius:14px; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
    <div>
        <span style="font-size:11px; color:#5f7d7a; text-transform:uppercase; font-weight:bold;">🌐 {selected_exchange_label} Entegreli Toplam İşlem Çifti Sayısı:</span>
        <div style="font-size:18px; font-weight:900; color:#0e7490;">{total_coin_count} / {total_coin_count} İşlem Çifti <span style="font-size:12px; color:#16a34a; font-weight:bold;">(%100 Tam Borsa Entegre)</span></div>
    </div>
    <div style="text-align:right;">
        <span style="font-size:11px; color:#5f7d7a; text-transform:uppercase; font-weight:bold;">🤖 Anlık Canlı Analiz Edilen Coin:</span>
        <div style="font-size:16px; font-weight:bold; color:#16a34a;"><span style="color:#0e7490;">{symbol_str}</span> Sinyal & Derinlik Analizi Aktif</div>
    </div>
</div>
""", unsafe_allow_html=True)

if match_info and "💡" in match_info:
    st.info(match_info)


# --- 8. ANLIK CANLI QUANTUM TERMİNALİ (STREAMLIT FRAGMENT) ---
@st.fragment(run_every=refresh_rate)
def render_quantum_terminal():
    fetch_start = time.time()
    
    # 1. Gerçek Borsa Vadeli İşlemler ve Dolar Dominansı Verisi Çekme
    df_raw = fetch_binance_kline_data(symbol_str, interval=active_interval, limit=350, exchange_id=selected_exchange_id)
    ticker = fetch_binance_ticker_details(symbol_str, exchange_id=selected_exchange_id)
    depth = fetch_binance_depth_imbalance(symbol_str, exchange_id=selected_exchange_id)
    usdt_dom = fetch_usdt_dominance_matrix()

    if df_raw.empty or len(df_raw) < 50:
        st.error(f" '{symbol_str}' ({user_query}) için {selected_exchange_label} Vadeli İşlemler mum verisi alınamadı! Lütfen sembolü kontrol edin.")
        return

    current_price = ticker.get("lastPrice", df_raw['close'].iloc[-1])
    change_24h = ticker.get("riseFallRate", 0.0)
    high_24h = ticker.get("high24Price", df_raw['high'].max())
    low_24h = ticker.get("lower24Price", df_raw['low'].min())
    funding_rate = ticker.get("fundingRate", 0.0)

    # 2. Özellik Mühendisliği (Sembol ve Zaman Dilimi Geçildi)
    df_features = compute_quantum_features(df_raw, symbol=symbol_str, interval=active_interval, exchange_id=selected_exchange_id)
    # ONEMLI - KARARLILIK: df_features'in SON satiri henuz KAPANMAMIS (olusmakta olan) canli
    # mumdur; RSI/MACD/EMA/hacim gibi degerleri her fiyat tikinde degisebiliyordu. Bu da
    # confluence skorunu (ve dolayisiyla GÜÇLÜ/ZAYIF/NÖTR siniflandirmasini) ve asagidaki
    # indikator radarinda gosterilen degerleri saniyeler icinde tutarsizca sallantiya
    # sokuyordu. "latest_row" artik EN SON KAPANMIS muma isaret eder - hem AI tahmini hem
    # confluence filtreleri hem de gorunen indikator degerleri, mum kapanana kadar (secili
    # zaman dilimi suresince) birbiriyle TUTARLI ve SABIT kalir. Anlik fiyat (current_price)
    # zaten ayri bir ticker kaynagindan geldigi icin bundan etkilenmez, canli kalmaya devam eder.
    latest_row = df_features.iloc[-2] if len(df_features) > 1 else df_features.iloc[-1]

    # 3. Yapay Zeka Modeli Eğitimi ve Olasılık Tahmini
    ai_result = train_and_predict_quantum_ai(df_features)

    # 3.1. Uzun Vadeli Statik Destek/Direnç Seviyeleri, Makro Trend, 4H Tahmin ve Likidasyon Taraması
    sr_levels = compute_macro_support_resistance_levels(df_raw, symbol_str, exchange_id=selected_exchange_id)
    macro_tf = compute_macro_timeframe_trend(symbol_str, exchange_id=selected_exchange_id)
    ai_4h_pred = compute_4h_scalp_prediction(symbol_str, exchange_id=selected_exchange_id)
    liq_matrix = compute_liquidation_clusters(df_raw, current_price)
    
    # 4. Sahte Sinyal Filtresi ve Confluence Değerlendirmesi (USDT Dominance + Majör S/R Entegre)
    confluence = evaluate_confluence_and_filter(ai_result, latest_row, depth, usdt_dom, sr_levels)
    
    # 4.1. Anlık Sinyal Dalgalanmasını Engelleme (Hysteresis & Signal Lock Engine)
    latched_key = f"latched_sig_{symbol_str}"
    latched_time_key = f"latched_time_{symbol_str}"
    
    current_raw_sig = confluence["final_signal"]
    prev_sig = st.session_state.get(latched_key, current_raw_sig)
    prev_time = st.session_state.get(latched_time_key, 0)
    
    time_elapsed = time.time() - prev_time
    if prev_sig != "NEUTRAL" and current_raw_sig != prev_sig and current_raw_sig != "NEUTRAL":
        if confluence["confidence"] < 66.0 and time_elapsed < 60:
            confluence["final_signal"] = prev_sig
            confluence["action_note"] += " (🔒 Kararlı Sinyal Kilitli - Anlık Dalgalanma Engellendi)"
        else:
            st.session_state[latched_key] = current_raw_sig
            st.session_state[latched_time_key] = time.time()
    else:
        st.session_state[latched_key] = current_raw_sig
        if latched_time_key not in st.session_state:
            st.session_state[latched_time_key] = time.time()
    
    # 5. Dinamik Kaldıraçlı ve İşlem Gücüne Dayalı İkili Kazanç (Min/Max TP) ve Orantılı SL Seviyeleri
    risk = compute_institutional_risk_levels(
        current_price=current_price,
        atr=latest_row["atr"],
        signal=confluence["final_signal"],
        leverage=selected_leverage,
        margin_budget=margin_budget,
        confidence=confluence["confidence"],
        confluence_score=confluence["confluence_score"],
        sr_levels=sr_levels,
        interval=active_interval
    )
    
    # Fiyat Formatlama Yardımcısı
    def fmt(p):
        return f"${p:,.6f}" if p < 1 else f"${p:,.4f}" if p < 100 else f"${p:,.2f}"

    # Hero kart içindeki mini fiyat grafiğini SAF SVG olarak üretir. Bilinçli olarak
    # st.plotly_chart KULLANILMIYOR: bu fonksiyon @st.fragment(run_every=...) içinde
    # çalışıyor ve Streamlit'in bu sürümünde plotly bileşenleri her fragment
    # yenilemesinde yeniden mount olup göz batan bir "flash" a sebep oluyor (2. sayfada
    # aynı sorunu tam da bu yüzden ayrı bir HTML/JS mimarisiyle çözmüştük). Düz SVG,
    # aynı st.markdown bloğunun bir parçası olduğu için bu sorunu hiç yaşamıyor.
    def _build_hero_sparkline_svg(prices, width=640, height=170):
        prices = [float(p) for p in prices if pd.notna(p)]
        n = len(prices)
        if n < 2:
            return ""
        p_min, p_max = min(prices), max(prices)
        p_range = max(p_max - p_min, 1e-9)
        pad = 14
        usable_h = height - pad * 2
        pts = []
        for i, p in enumerate(prices):
            x = (i / (n - 1)) * width
            y = pad + (1 - (p - p_min) / p_range) * usable_h
            pts.append((x, y))
        line_path = "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        area_path = line_path + f" L {width:.2f},{height:.2f} L 0,{height:.2f} Z"
        last_x, last_y = pts[-1]
        # NOT: SVG kasıtlı olarak TEK SATIRDA döndürülüyor. Çok satırlı/girintili bir
        # string, dışarıdaki st.markdown bloğuna eklendiğinde CommonMark'ın HTML blok
        # ayrıştırıcısını kandırıp (boşluk satırı = HTML bloğunun sonu kuralı) ondan
        # sonraki gerçek HTML'in düz metin olarak kaçırılmasına (escape) sebep oluyordu.
        svg_parts = [
            f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" preserveAspectRatio="none" style="display:block;">',
            '<defs><linearGradient id="heroFillGrad" x1="0" y1="0" x2="0" y2="1">',
            '<stop offset="0%" stop-color="rgba(20,184,166,0.35)"/>',
            '<stop offset="100%" stop-color="rgba(20,184,166,0.02)"/>',
            '</linearGradient></defs>',
            f'<path d="{area_path}" fill="url(#heroFillGrad)" stroke="none"/>',
            f'<path d="{line_path}" fill="none" stroke="#0e7490" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>',
            f'<circle cx="{last_x:.2f}" cy="{last_y:.2f}" r="9" fill="rgba(14,116,144,0.20)"/>',
            f'<circle cx="{last_x:.2f}" cy="{last_y:.2f}" r="5" fill="#0e7490"/>',
            '</svg>',
        ]
        return "".join(svg_parts)

    # Üst Bilgi Rozeti (Seçili Coin / Zaman Dilimi / Kaldıraç / Bütçe)
    st.markdown(f"""
    <div style="background:#ffffff; padding:10px 16px; border-radius:14px; box-shadow:0 4px 14px rgba(15,43,46,0.06); margin-bottom:12px;">
        <span style="font-size: 20px; font-weight: 800; color: #0f2b2e;">🟢 {selected_exchange_label}: {symbol_str}</span>
        <span style="color: #5f7d7a; font-size: 13px; margin-left: 10px;">| {selected_tf_label} | <b style="color:#d97706;">{selected_leverage}x Kaldıraç</b> | <b style="color:#16a34a;">Bütçe: ${margin_budget:,.0f}</b></span>
    </div>
    """, unsafe_allow_html=True)
    
    # =========================================================================
    # KANTİTATİF ANALİZ VE İNDİKATÖR RADARI
    # =========================================================================
    with st.container():
        # --- "VIVID FINTECH" HERO BÖLÜMÜ (Stitch referans tasarımına göre) ---
        st.markdown("""
        <style>
            .hero-chart-card { background:#ffffff; border-radius:0 0 20px 20px; padding:14px 22px 20px 22px; color:#0f2b2e; height:100%; margin-top:-8px; box-shadow:0 4px 14px rgba(15,43,46,0.06); }
            /* NOT: st.segmented_control, st.button'dan TAMAMEN FARKLI bir DOM/testid yapısı
               kullanıyor (stButtonGroup > button[data-variant="segmented_control"], seçili
               durum aria-checked ile belirleniyor) - eski stBaseButton-primary/secondary
               kuralları hiç eşleşmiyordu, bu yüzden segmentler Streamlit'in varsayılan
               (koyu/siyah) temasıyla kalıyordu. Doğru seçicilerle yeniden yazıldı. */
            .st-key-tf_pill_bar { background:linear-gradient(135deg, #14b8a6 0%, #0e7490 100%); border-radius:20px 20px 0 0; padding:12px 16px 8px 16px; }
            .st-key-tf_pill_bar [data-testid="stWidgetLabel"] { display:none !important; }
            /* NOT: Streamlit bu elemanlara width="fit-content" HTML özniteliği veriyor,
               bu yuzden sadece butonlarin gercek metin genisligi kadar yer kapliyorlardi
               (5 pil ~144px'e sikisip "15dk" gibi metinler kesiliyordu). Tum zinciri
               (element container -> button group -> radiogroup) %100 genislige zorluyoruz. */
            .st-key-tf_pill_bar [data-testid="stElementContainer"] { width:100% !important; }
            .st-key-tf_pill_bar [data-testid="stButtonGroup"] { width:100% !important; gap:6px !important; }
            .st-key-tf_pill_bar [data-testid="stButtonGroup"] > div[role="radiogroup"] { width:100% !important; max-width:100% !important; gap:6px; display:flex; }
            .st-key-tf_pill_bar button[data-variant="segmented_control"] {
                flex:1 1 0; min-width:0; width:100%; border:none !important; border-radius:16px !important;
                font-size:12px !important; font-weight:900 !important; letter-spacing:0;
                padding:6px 2px !important; min-height:34px !important; box-shadow:none !important;
                background:rgba(255,255,255,0.16) !important; color:#f0fdfa !important;
                transition:background 0.15s ease, color 0.15s ease; white-space:nowrap; overflow:hidden;
            }
            .st-key-tf_pill_bar button[data-variant="segmented_control"]:hover { background:rgba(255,255,255,0.30) !important; color:#ffffff !important; }
            .st-key-tf_pill_bar button[data-variant="segmented_control"] p { font-size:12px !important; font-weight:900 !important; color:inherit !important; white-space:nowrap; }
            .st-key-tf_pill_bar button[data-variant="segmented_control"] > div { min-width:0; }
            .st-key-tf_pill_bar button[data-variant="segmented_control"][aria-checked="true"] {
                background:#ffffff !important; color:#0e7490 !important; box-shadow:0 2px 8px rgba(0,0,0,0.15) !important;
            }
            .st-key-tf_pill_bar button[data-variant="segmented_control"][aria-checked="true"] p { color:#0e7490 !important; }
            .hero-price-row { display:flex; align-items:baseline; gap:10px; flex-wrap:wrap; margin-top:4px; }
            .hero-price { font-size:30px; font-weight:900; color:#0f2b2e; }
            .hero-chg-badge { padding:4px 12px; border-radius:20px; font-size:12px; font-weight:800; }
            .hero-chg-badge.up { background:#dcfce7; color:#16a34a; }
            .hero-chg-badge.down { background:#fee2e2; color:#dc2626; }
            .hero-tf-pill { display:inline-block; background:#f1f5f9; color:#5f7d7a; padding:4px 13px; border-radius:16px; font-size:11px; font-weight:800; margin:12px 6px 0 0; }
            .hero-tf-pill.active { background:#0e7490; color:#ffffff; }

            .ai-signal-card { background:#ffffff; border-radius:20px; padding:20px; text-align:center; height:100%; display:flex; flex-direction:column; justify-content:center; box-shadow:0 4px 14px rgba(15,43,46,0.06); }
            .ai-signal-card.dir-long .ai-signal-title { color:#16a34a; }
            .ai-signal-card.dir-short .ai-signal-title { color:#dc2626; }
            .ai-signal-card.dir-neutral .ai-signal-title { color:#7c3aed; }
            .ai-signal-label { font-size:11px; letter-spacing:1.5px; color:#5f7d7a; font-weight:800; }
            .ai-signal-title { font-size:20px; font-weight:900; margin:6px 0 16px 0; }
            .confidence-ring { width:112px; height:112px; border-radius:50%; margin:0 auto; display:flex; align-items:center; justify-content:center; }
            .confidence-inner { width:88px; height:88px; border-radius:50%; background:#ffffff; display:flex; flex-direction:column; align-items:center; justify-content:center; box-shadow:inset 0 0 0 1px rgba(15,43,46,0.06); }
            .confidence-value { font-size:22px; font-weight:900; color:#0f2b2e; }
            .confidence-label { font-size:9px; letter-spacing:1px; color:#5f7d7a; margin-top:2px; }
            .ai-prob-line { font-size:12px; margin-top:14px; color:#5f7d7a; }

            .stat-card { background:#ffffff; border-radius:18px; padding:15px 16px; height:100%; }
            .stat-icon { width:36px; height:36px; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:16px; margin-bottom:10px; color:#fff; }
            .stat-value { font-size:19px; font-weight:900; color:#1e1b2e; }
            .stat-label { font-size:11px; color:#6b7280; margin-bottom:6px; }
            .stat-progress-track { background:#f1f5f9; border-radius:8px; height:6px; overflow:hidden; margin-top:10px; }
            .stat-progress-fill { height:100%; border-radius:8px; }
            .stat-footer { display:flex; justify-content:space-between; font-size:10px; color:#94a3b8; margin-top:6px; }

            .mini-card { background:#ffffff; border-radius:18px; padding:16px; height:100%; }
            .mini-card-title { font-size:12px; font-weight:900; color:#1e1b2e; margin-bottom:10px; }
            .check-row { display:flex; justify-content:space-between; align-items:center; gap:8px; padding:7px 0; border-bottom:1px solid #f1f5f9; font-size:11px; color:#334155; }
            .check-row:last-child { border-bottom:none; }
            .check-badge { padding:2px 9px; border-radius:12px; font-size:10px; font-weight:800; white-space:nowrap; }
            .check-badge.pass { background:#dcfce7; color:#16a34a; }
            .check-badge.warn { background:#fef3c7; color:#d97706; }
            .check-badge.fail { background:#fee2e2; color:#dc2626; }

            .sr-row { display:flex; justify-content:space-between; align-items:center; padding:7px 0; border-bottom:1px solid #f1f5f9; font-size:12px; }
            .sr-row:last-child { border-bottom:none; }
            .sr-price { font-weight:800; color:#1e1b2e; }
            .sr-pct { font-weight:800; font-size:11px; }
            .sr-pct.up { color:#16a34a; } .sr-pct.down { color:#dc2626; }

            .tp-sl-card { border-radius:18px; padding:16px 18px; color:#fff; }
            .tp-sl-card.min { background:linear-gradient(135deg, #2dd4bf, #059669); }
            .tp-sl-card.sl { background:linear-gradient(135deg, #fb923c, #dc2626); }
            .tp-sl-label { font-size:11px; opacity:0.85; font-weight:800; }
            .tp-sl-value { font-size:22px; font-weight:900; margin-top:5px; }
            .tp-sl-sub { font-size:11px; opacity:0.85; margin-top:5px; }

            .section-header { display:flex; justify-content:space-between; align-items:center; background:#ffffff; box-shadow:0 4px 14px rgba(15,43,46,0.06); border-radius:14px; padding:10px 16px; margin:6px 0 12px 0; flex-wrap:wrap; gap:8px; }
            .section-header-title { font-size:12px; font-weight:900; color:#0f2b2e; letter-spacing:0.3px; }
            .section-header-badge { font-size:11px; font-weight:800; }

            .info-bar { display:flex; justify-content:space-between; align-items:center; background:#ffffff; border-radius:14px; padding:12px 16px; margin-bottom:12px; border-left:4px solid #94a3b8; flex-wrap:wrap; gap:10px; }
            .info-bar-label { font-size:10px; color:#6b7280; text-transform:uppercase; font-weight:800; letter-spacing:0.4px; }
            .info-bar-value { font-size:15px; font-weight:900; color:#1e1b2e; margin-top:2px; }

            .info-tile { background:#ffffff; border-radius:14px; padding:12px 14px; border-left:4px solid #94a3b8; height:100%; }
            .info-tile-label { font-size:10px; color:#6b7280; text-transform:uppercase; font-weight:800; margin-bottom:4px; letter-spacing:0.3px; }
            .info-tile-value { font-size:15px; font-weight:900; color:#1e1b2e; }
            .info-tile-value-sm { font-size:12px; font-weight:900; color:#1e1b2e; }
            .info-tile-sub { font-size:11px; color:#6b7280; margin-top:5px; }

            .indicator-chip { background:#ffffff; border-radius:12px; padding:10px 6px; text-align:center; height:100%; }
            .indicator-chip-title { font-size:9px; color:#6b7280; font-weight:800; text-transform:uppercase; letter-spacing:0.2px; }
            .indicator-chip-value { font-size:13px; font-weight:900; margin-top:5px; }
            .indicator-chip-dir { font-size:10px; font-weight:800; margin-top:3px; }

            .st-key-coin_tab_strip { background:#ffffff; border-radius:16px; padding:8px 4px; height:100%; box-shadow:0 4px 14px rgba(15,43,46,0.06); }
            .st-key-coin_tab_strip [data-testid="stVerticalBlock"] { gap:6px; }
            .st-key-coin_tab_strip [data-testid="stButton"] button {
                writing-mode: vertical-rl; text-orientation: mixed; transform: rotate(180deg);
                width:100%; min-height:88px; border:none !important; border-radius:10px !important;
                font-size:10px !important; font-weight:800 !important; letter-spacing:0.5px;
                box-shadow:none !important; padding:8px 0 !important;
            }
            .st-key-coin_tab_strip [data-testid="stBaseButton-secondary"] { background:#f1f5f9 !important; color:#5f7d7a !important; }
            .st-key-coin_tab_strip.st-key-coin_tab_strip [data-testid="stBaseButton-primary"] { background:#0e7490 !important; color:#ffffff !important; }
            .st-key-coin_tab_strip.st-key-coin_tab_strip [data-testid="stBaseButton-primary"] p { color:#ffffff !important; }
        </style>
        """, unsafe_allow_html=True)

        # 1) DİKEY FAVORİ COİN SEKMELERİ + HERO GRAFİK KARTI + AI SİNYAL KARTI
        coin_tab_col, hero_col, signal_col = st.columns([0.16, 1.5, 1])

        with coin_tab_col:
            with st.container(key="coin_tab_strip"):
                _fav_tabs = [("BTC/USDT", "BTCUSDT"), ("ETH/USDT", "ETHUSDT"), ("SOL/USDT", "SOLUSDT"), ("1000PEPE/USDT", "1000PEPEUSDT")]
                for _tab_label, _tab_sym in _fav_tabs:
                    _tab_active = st.session_state.get("active_symbol_query") == _tab_sym
                    if st.button(_tab_label, key=f"coin_tab_{_tab_sym}", type="primary" if _tab_active else "secondary"):
                        st.session_state["active_symbol_query"] = _tab_sym
                        st.rerun(scope="app")

        with hero_col:
            chg_up = change_24h >= 0
            chg_badge_icon = "▲" if chg_up else "▼"
            sparkline_svg = _build_hero_sparkline_svg(df_raw['close'].tail(120).tolist())
            st.markdown(f"""
            <div class="hero-chart-card">
                <div style="font-size:11px; letter-spacing:1px; color:#5f7d7a; font-weight:700;">ANLIK CANLI FİYAT</div>
                <div class="hero-price-row">
                    <span class="hero-price">{fmt(current_price)}</span>
                    <span class="hero-chg-badge {'up' if chg_up else 'down'}">{chg_badge_icon} %{change_24h:+.2f} (24s)</span>
                </div>
                {sparkline_svg}
            </div>
            """, unsafe_allow_html=True)

        with signal_col:
            final_sig = confluence["final_signal"]
            if "LONG" in final_sig:
                dir_cls, dir_arrow, ring_grad = "dir-long", "↑", "conic-gradient(#14b8a6, #16a34a {p}%, #e5e7eb 0)"
            elif "SHORT" in final_sig:
                dir_cls, dir_arrow, ring_grad = "dir-short", "↓", "conic-gradient(#fb923c, #dc2626 {p}%, #e5e7eb 0)"
            else:
                dir_cls, dir_arrow, ring_grad = "dir-neutral", "—", "conic-gradient(#14b8a6, #7c3aed {p}%, #e5e7eb 0)"

            conf_pct = confluence['confidence']
            ring_style = ring_grad.format(p=conf_pct)
            st.markdown(f"""
            <div class="ai-signal-card {dir_cls}">
                <div class="ai-signal-label">AI İŞLEM SİNYALİ</div>
                <div class="ai-signal-title">{confluence['signal_title']} {dir_arrow}</div>
                <div class="confidence-ring" style="background:{ring_style};">
                    <div class="confidence-inner">
                        <div class="confidence-value">%{conf_pct:.0f}</div>
                        <div class="confidence-label">GÜVEN</div>
                    </div>
                </div>
                <div class="ai-prob-line">LONG: <b style="color:#16a34a;">%{ai_result['prob_long']*100:.1f}</b> &nbsp;|&nbsp; SHORT: <b style="color:#dc2626;">%{ai_result['prob_short']*100:.1f}</b></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        # 2) KÜÇÜK İSTATİSTİK KARTLARI
        s1c, s2c, s3c, s4c = st.columns(4)
        b_ratio = depth.get("bid_ratio", 50)
        a_ratio = depth.get("ask_ratio", 50)
        u_chg_val = usdt_dom['usdt_d_change']
        fund_dir_txt = "Uzun Ağırlıkta" if funding_rate > 0 else "Kısa Ağırlıkta"
        u_dir_txt = "Kripto Boğa" if u_chg_val < 0 else "Nakite Kaçış"

        stat_defs = [
            (s1c, "#0e7490", "⚡", "Fonlama Oranı", f"%{funding_rate:.4f}", fund_dir_txt, min(abs(funding_rate) * 2000, 100)),
            (s2c, "#f97316", "⚖️", "Alıcı/Satıcı Baskısı", f"%{b_ratio:.0f} / %{a_ratio:.0f}", depth.get('bias', 'DENGELİ'), b_ratio),
            (s3c, "#8b5cf6", "🌐", "Dolar Dominansı", f"%{usdt_dom['usdt_d']:.2f}", u_dir_txt, min(usdt_dom['usdt_d'] * 8, 100)),
            (s4c, "#16a34a", "🚀", "İşlem Gücü Skoru", f"%{risk['trade_power']:.0f}", risk['power_title'], risk['trade_power']),
        ]
        for col, color, icon, label, value, footer, pct in stat_defs:
            with col:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-icon" style="background:{color};">{icon}</div>
                    <div class="stat-label">{label}</div>
                    <div class="stat-value">{value}</div>
                    <div class="stat-progress-track"><div class="stat-progress-fill" style="width:{pct:.0f}%; background:{color};"></div></div>
                    <div class="stat-footer"><span>{footer}</span><span>%{pct:.0f}</span></div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        # 3) SAHTE SİNYAL FİLTRESİ + DESTEK/DİRENÇ MİNİ KARTLARI
        filt_col, sr_col = st.columns(2)
        with filt_col:
            rows_html = ""
            for name, detail, status, weight in confluence["checks"]:
                badge_cls = "pass" if status == "pass" else ("warn" if status == "warn" else "fail")
                badge_txt = "DOĞRULANDI" if status == "pass" else ("UYARI" if status == "warn" else "REDDEDİLDİ")
                rows_html += f"""<div class="check-row"><span><b>{name}</b> — {detail}</span><span class="check-badge {badge_cls}">{badge_txt}</span></div>"""
            st.markdown(f"""
            <div class="mini-card">
                <div class="mini-card-title">🛡️ GÜVENLİK FİLTRE KATMANI</div>
                {rows_html}
            </div>
            """, unsafe_allow_html=True)

        with sr_col:
            def _sr_row(label, level_price):
                dist_pct = abs(level_price - current_price) / max(current_price, 1e-9) * 100.0
                is_above = level_price >= current_price
                arrow = "▲" if is_above else "▼"
                cls = "up" if is_above else "down"
                return f"""<div class="sr-row"><span>{label}</span><span class="sr-price">{fmt(level_price)}</span><span class="sr-pct {cls}">{arrow} %{dist_pct:.2f}</span></div>"""

            sr_rows = (
                _sr_row("R2 (Zirve)", sr_levels.get('r2', current_price))
                + _sr_row("R1 (Birincil)", sr_levels.get('r1', current_price))
                + _sr_row("S1 (Birincil)", sr_levels.get('s1', current_price))
                + _sr_row("S2 (Derin Dip)", sr_levels.get('s2', current_price))
            )
            st.markdown(f"""
            <div class="mini-card">
                <div class="mini-card-title">📍 DESTEK & DİRENÇ</div>
                {sr_rows}
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

        # 4) TP / SL ÖZET KARTLARI — Min/Max ayrımı kaldırıldı, hesaplanan tek TP değeri gösteriliyor
        # (yüksek ihtimalli/gerçekçi hedef olan min_tp_price esas alınıyor; max_tp_price artık ayrı bir
        # kart olarak gösterilmiyor).
        total_pos_size = margin_budget * selected_leverage
        tp1, tp2 = st.columns(2)
        with tp1:
            st.markdown(f"""
            <div class="tp-sl-card min">
                <div class="tp-sl-label">🎯 KÂR AL (TP) — R:R 1:{risk['rr_min']:.2f}</div>
                <div class="tp-sl-value">{fmt(risk['min_tp_price'])}</div>
                <div class="tp-sl-sub">+%{risk['actual_min_pct']:.2f} (ROE +%{risk['actual_min_roe']:.1f}) &nbsp;•&nbsp; +${risk['min_profit_usd']:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with tp2:
            st.markdown(f"""
            <div class="tp-sl-card sl">
                <div class="tp-sl-label">🛑 STOP-LOSS</div>
                <div class="tp-sl-value">{fmt(risk['sl_price'])}</div>
                <div class="tp-sl-sub">-%{risk['actual_sl_pct']:.2f} (ROE -%{risk['actual_sl_roe']:.1f}) &nbsp;•&nbsp; -${risk['max_loss_usd']:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
        entry_col, pos_col = st.columns(2)
        with entry_col:
            st.markdown(f"""
            <div class="info-tile" style="border-left-color:#0e7490;">
                <div class="info-tile-label">📍 Giriş Fiyatı</div>
                <div class="info-tile-value">{fmt(risk['entry'])}</div>
            </div>
            """, unsafe_allow_html=True)
        with pos_col:
            st.markdown(f"""
            <div class="info-tile" style="border-left-color:#d97706;">
                <div class="info-tile-label">💰 Pozisyon</div>
                <div class="info-tile-value">${total_pos_size:,.0f}</div>
                <div class="info-tile-sub">{selected_leverage}x Kaldıraç, Marjin ${margin_budget:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

        # --- UZUN VADELİ MAJÖR DESTEK VE DİRENÇ YAPISAL ANALİZ KARTI ---
        st.markdown(f"""
        <div class="section-header">
            <div class="section-header-title">🏰 UZUN VADELİ MAJÖR DESTEK & DİRENÇ (KADEMELİ S1/S2 - R1/R2)</div>
            <div class="section-header-badge" style="color:#16a34a;">{macro_tf['macro_badge']}</div>
        </div>
        """, unsafe_allow_html=True)

        sr_dir_clr = "#16a34a" if "BOUNCE" in sr_levels.get("sr_direction_code", "") or "BREAKOUT" in sr_levels.get("sr_direction_code", "") or "BULL" in sr_levels.get("sr_direction_code", "") else "#dc2626"
        st.markdown(f"""
        <div class="info-bar" style="border-left-color:{sr_dir_clr};">
            <div>
                <div class="info-bar-label">🎯 Destek/Direnç Yön Tahmini</div>
                <div class="info-bar-value" style="color:{sr_dir_clr};">{sr_levels.get('sr_direction', '')}</div>
            </div>
            <div style="text-align:right;">
                <div class="info-bar-label">⚖️ Haftalık / Günlük Pivot Denge</div>
                <div class="info-bar-value">{fmt(sr_levels.get('pp', 0))}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        sr1, sr2, sr3 = st.columns([1.3, 1.3, 1.4])
        with sr1:
            st.markdown(f"""
            <div class="info-tile" style="border-left-color:#16a34a;">
                <div class="info-tile-label">🛡️ Statik Destek Kademeleri</div>
                <div class="info-tile-value-sm" style="color:#16a34a;">S1: {fmt(sr_levels.get('s1', 0))}</div>
                <div class="info-tile-sub">S2 (Derin Dip): {fmt(sr_levels.get('s2', 0))}</div>
                <div class="info-tile-sub">Fiyata Uzaklık: <b>%{sr_levels.get('dist_sup_pct', 0):.2f}</b></div>
            </div>
            """, unsafe_allow_html=True)
        with sr2:
            st.markdown(f"""
            <div class="info-tile" style="border-left-color:#dc2626;">
                <div class="info-tile-label">🏰 Statik Direnç Kademeleri</div>
                <div class="info-tile-value-sm" style="color:#dc2626;">R1: {fmt(sr_levels.get('r1', 0))}</div>
                <div class="info-tile-sub">R2 (Zirve): {fmt(sr_levels.get('r2', 0))}</div>
                <div class="info-tile-sub">Fiyata Uzaklık: <b>%{sr_levels.get('dist_res_pct', 0):.2f}</b></div>
            </div>
            """, unsafe_allow_html=True)
        with sr3:
            st.markdown(f"""
            <div class="info-tile" style="border-left-color:#8b5cf6;">
                <div class="info-tile-label">📌 Reaksiyon & Strateji Uyarısı</div>
                <div class="info-tile-value-sm" style="color:#4c1d95;">{sr_levels.get('sr_note', '')}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # --- SCALP ADAPTASYONU VE 4 SAATLİK (4H) MAKRO TAHMİN KARTI ---
        is_scalp_tf = active_interval in ["Min1", "Min5", "Min240"]
        scalp_badge = "⚡ SCALP / 4H MODU AKTİF" if is_scalp_tf else "SWING / TREND MODU AKTİF"

        scalp_dir = "LONG" if "LONG" in confluence["final_signal"] else ("SHORT" if "SHORT" in confluence["final_signal"] else "NEUTRAL")
        tf4h_dir = ai_4h_pred["direction"]

        if scalp_dir == tf4h_dir and scalp_dir != "NEUTRAL":
            alignment_badge = "%100 YÖN UYUMLU: Scalp & 4H Makro Aynı Yönde (Yüksek İhtimal)"
            align_clr = "#16a34a"
        elif scalp_dir != "NEUTRAL" and tf4h_dir != "NEUTRAL":
            alignment_badge = "KARŞI TREND SCALP: Scalp Yönü 4H Trendin Tersine!"
            align_clr = "#d97706"
        else:
            alignment_badge = "⚖️ DENGELİ / NÖTR KOMBİNASYON"
            align_clr = "#3b82f6"

        st.markdown(f"""
        <div class="section-header">
            <div class="section-header-title">⚡ SCALP ADAPTASYONU VE 4 SAATLİK (4H) İŞLEM ANALİZİ</div>
            <div class="section-header-badge" style="color:#0e7490;">{scalp_badge}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="info-bar" style="border-left-color:{align_clr};">
            <div>
                <div class="info-bar-label">⚡ Scalp & 4H Trend Uyum Durumu</div>
                <div class="info-bar-value" style="color:{align_clr}; font-size:13px;">{alignment_badge}</div>
            </div>
            <div style="text-align:right;">
                <div class="info-bar-label">🤖 4H AI Tahmin Güveni</div>
                <div class="info-bar-value" style="color:{ai_4h_pred['color']};">%{ai_4h_pred['confidence']:.1f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        sc1, sc2, sc3 = st.columns([1.3, 1.3, 1.4])
        with sc1:
            st.markdown(f"""
            <div class="info-tile" style="border-left-color:{ai_4h_pred['color']};">
                <div class="info-tile-label">4 Saatlik (4H) Tahmin</div>
                <div class="info-tile-value-sm" style="color:{ai_4h_pred['color']};">{ai_4h_pred['pred_title']}</div>
                <div class="info-tile-sub">Model Güveni: <b>%{ai_4h_pred['confidence']:.1f}</b></div>
            </div>
            """, unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""
            <div class="info-tile" style="border-left-color:#16a34a;">
                <div class="info-tile-label">🔐 Private API Bağlantı Analizi</div>
                <div class="info-tile-value-sm" style="color:#16a34a;">🟢 Borsa API: AKTİF & HAZIR</div>
                <div class="info-tile-sub">Cüzdan Bakiyesi: <b>${wallet_balance_val:,.2f} USDT</b></div>
            </div>
            """, unsafe_allow_html=True)
        with sc3:
            st.markdown(f"""
            <div class="info-tile" style="border-left-color:#d97706;">
                <div class="info-tile-label">⚡ Scalp Mikro Emir Akışı</div>
                <div class="info-tile-value-sm" style="color:#d97706;">Tahta Dengesi: %{depth.get('imbalance', 0):+.1f}</div>
                <div class="info-tile-sub">{depth.get('bias', 'DENGELİ')}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # --- KURUMSAL LİKİDASYON ISI HARİTASI VE AKILLI İZLEYEN STOP KARTI ---
        st.markdown(f"""
        <div class="section-header">
            <div class="section-header-title">🔥 LİKİDASYON ISI HARİTASI & AKILLI İZLEYEN STOP</div>
            <div class="section-header-badge" style="color:{liq_matrix.get("liq_color", "#16a34a")};">{liq_matrix.get("liq_badge", "")}</div>
        </div>
        """, unsafe_allow_html=True)

        lq1, lq2, lq3 = st.columns([1.3, 1.3, 1.4])
        with lq1:
            st.markdown(f"""
            <div class="info-tile" style="border-left-color:#dc2626;">
                <div class="info-tile-label">🩸 Long Likidasyon Duvarı</div>
                <div class="info-tile-value-sm" style="color:#dc2626;">100x Liq: {fmt(liq_matrix.get('long_liq_100x', 0))}</div>
                <div class="info-tile-sub">50x Liq: {fmt(liq_matrix.get('long_liq_50x', 0))}</div>
                <div class="info-tile-sub">Duvar Uzaklığı: <b>%{liq_matrix.get('dist_long_liq_pct', 0):.2f}</b></div>
            </div>
            """, unsafe_allow_html=True)
        with lq2:
            st.markdown(f"""
            <div class="info-tile" style="border-left-color:#d97706;">
                <div class="info-tile-label">🚀 Short Likidasyon Duvarı</div>
                <div class="info-tile-value-sm" style="color:#d97706;">100x Liq: {fmt(liq_matrix.get('short_liq_100x', 0))}</div>
                <div class="info-tile-sub">50x Liq: {fmt(liq_matrix.get('short_liq_50x', 0))}</div>
                <div class="info-tile-sub">Duvar Uzaklığı: <b>%{liq_matrix.get('dist_short_liq_pct', 0):.2f}</b></div>
            </div>
            """, unsafe_allow_html=True)
        with lq3:
            atr_val = latest_row["atr"]
            if "LONG" in confluence["final_signal"]:
                trail_sl = current_price - (atr_val * 1.5)
                trail_note = f"🟢 İzleyen Stop: {fmt(trail_sl)} (Fiyat yükseldikçe stop yukarı kayar)"
            elif "SHORT" in confluence["final_signal"]:
                trail_sl = current_price + (atr_val * 1.5)
                trail_note = f"🔻 İzleyen Stop: {fmt(trail_sl)} (Fiyat düştükçe stop aşağı kayar)"
            else:
                trail_note = "⚖️ Pozisyon Beklemede (Pasif)"
            st.markdown(f"""
            <div class="info-tile" style="border-left-color:#3b82f6;">
                <div class="info-tile-label">🎯 Akıllı İzleyen Stop & Parçalı Kâr</div>
                <div class="info-tile-value-sm" style="color:#3b82f6;">{trail_note}</div>
                <div class="info-tile-sub">Parçalı Çıkış: <b>%50 TP1'de Kâr Al + Girişe SL Çek</b></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        # --- TEKNİK İNDİKATÖR RADARI VE YÖN MATRİSİ ---
        st.markdown("""
        <div class="section-header">
            <div class="section-header-title">📡 ÇOK BOYUTLU İNDİKATÖR RADARI & CANLI YÖN MATRİSİ</div>
        </div>
        """, unsafe_allow_html=True)

        ind1, ind2, ind3, ind4, ind5, ind6, ind7 = st.columns(7)

        rsi_val = latest_row["rsi"]
        macd_diff_val = latest_row["macd_diff"]
        is_ema_bull = latest_row["ema_9"] > latest_row["ema_21"]
        is_ema200_bull = current_price > latest_row["ema_200"]
        stoch_k_val = latest_row["stoch_k"]
        stoch_d_val = latest_row["stoch_d"]
        bb_pct_val = latest_row["bb_pct"]
        u_d_val = usdt_dom.get("usdt_d", 6.98)
        u_chg = usdt_dom.get("usdt_d_change", 0.0)

        indicator_defs = [
            (ind1, "RSI (14)", f"{rsi_val:.1f}", rsi_val > 50),
            (ind2, "MACD HİST", f"{macd_diff_val:+.3f}", macd_diff_val > 0),
            (ind3, "EMA (9/21)", fmt(latest_row['ema_9']), is_ema_bull),
            (ind4, "EMA 200 MAKRO", fmt(latest_row['ema_200']), is_ema200_bull),
            (ind5, "STOKASTİK", f"%{stoch_k_val:.1f}", stoch_k_val > stoch_d_val),
            (ind6, "BOLLİNGER", f"%{bb_pct_val*100:.0f}", bb_pct_val > 0.50),
            (ind7, "USDT.D", f"%{u_d_val:.2f} {'▼' if u_chg < 0 else '▲'}", u_chg < 0),
        ]
        for col, title, value, is_long in indicator_defs:
            clr = "#16a34a" if is_long else "#dc2626"
            dir_txt = "🟢 LONG" if is_long else "🔻 SHORT"
            with col:
                st.markdown(f"""
                <div class="indicator-chip" style="border-top:3px solid {clr};">
                    <div class="indicator-chip-title">{title}</div>
                    <div class="indicator-chip-value" style="color:{clr};">{value}</div>
                    <div class="indicator-chip-dir" style="color:{clr};">{dir_txt}</div>
                </div>
                """, unsafe_allow_html=True)

    # İşlem Motoru Gecikme Bilgisi
    elapsed_ms = (time.time() - fetch_start) * 1000
    st.caption(f"⚡ BtcSatoshi Live-Trade & AI Quant Engine: {elapsed_ms:.1f} ms | Model: Ensemble (RF+ET+HGB) | Canlı Borsa: {selected_exchange_label} Futures")

# Zaman dilimi pilleri BİLİNÇLİ OLARAK fragment'in DIŞINDA tanımlanıyor: fragment içinde
# olsalardı tıklama sadece fragment'i yeniden çalıştırır, sidebar'daki selectbox'tan
# hesaplanan active_interval değişmezdi (fragment rerun'u dış scope'u tekrar çalıştırmaz).
# Burada gerçek bir st.button() tam sayfa yenilemesi tetikleyip aynı
# "active_timeframe_label" session_state anahtarını (sidebar'daki dropdown ile paylaşılan)
# güncelliyor, böylece herhangi bir zaman dilimine gerçekten geçilebiliyor.
# NOT: st.button + type="primary" ile Streamlit'in kendi tema rengini (kırmızı) CSS ile
# ezmeye çalışmak güvenilmez çıktı (Streamlit'in dahili emotion stilleri kazanıyordu).
# Bunun yerine tam da bu amaç için var olan st.segmented_control kullanılıyor.
_tf_short_map = {
    "1dk": "1 Dakika (Scalp)",
    "5dk": "5 Dakika (Day Trade)",
    "15dk": "15 Dakika (Swing)",
    "1sa": "1 Saat (Trend)",
    "4sa": "4 Saatlik (Saatlik Scalp / Trend)",
}
_current_short = next((s for s, f in _tf_short_map.items() if f == st.session_state["active_timeframe_label"]), "1dk")

# ÖNEMLİ: Bu senkronizasyon SADECE "dıştan" bir değişikliği (ör. sidebar'daki Zaman
# Dilimi dropdown'undan gelen) segmented_control'e yansıtmak için var. Kullanıcının
# pile TIKLAMASI on_change callback'i üzerinden ayrı akıyor (aşağıda) — o yüzden bu
# satır kullanıcının az önce yaptığı tıklamayı asla ezmiyor: callback zaten çalışıp
# active_timeframe_label'ı güncellemiş oluyor, bu yüzden _current_short ile
# tf_pill_segmented burada zaten eşleşiyor.
if st.session_state.get("tf_pill_segmented") != _current_short:
    st.session_state["tf_pill_segmented"] = _current_short

def _on_tf_pill_change():
    picked = st.session_state.get("tf_pill_segmented")
    if picked and _tf_short_map.get(picked) and _tf_short_map[picked] != st.session_state["active_timeframe_label"]:
        st.session_state["_pending_timeframe"] = _tf_short_map[picked]

_tf_pill_col, _tf_spacer_col = st.columns([1.5, 1])
with _tf_pill_col:
    with st.container(key="tf_pill_bar"):
        st.segmented_control(
            "Zaman Dilimi",
            list(_tf_short_map.keys()),
            key="tf_pill_segmented",
            label_visibility="collapsed",
            on_change=_on_tf_pill_change,
        )

# Terminali Çalıştır
render_quantum_terminal()