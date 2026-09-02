import streamlit as st

st.set_page_config(page_title="Canlı Gösterge", page_icon="⚡", layout="wide", initial_sidebar_state="expanded")

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


def load_env_credentials() -> dict:
    # Proje kök dizini: bu dosya pages/ altinda oldugu icin bir ust klasore cikilir.
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(project_dir, ".env"))

    api_key = os.getenv("MEXC_API_KEY", "").strip()
    secret_key = os.getenv("MEXC_SECRET_KEY", "").strip()

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
                                if k_upper in ["MEXC_API_KEY", "MEXC_ACCESS_KEY", "API_KEY", "ACCESS_KEY"]:
                                    if not api_key: api_key = val
                                elif k_upper in ["MEXC_SECRET_KEY", "SECRET_KEY"]:
                                    if not secret_key: secret_key = val
            except Exception:
                pass
    if api_key: os.environ["MEXC_API_KEY"] = api_key
    if secret_key: os.environ["MEXC_SECRET_KEY"] = secret_key
    return {"api_key": api_key, "secret_key": secret_key}

load_env_credentials()

@st.cache_resource(ttl=1800, show_spinner=False)
def get_public_mexc_client():
    """Herhangi bir API anahtari gerektirmeyen (genel/public) veri cekimleri icin TEK, paylasilan
    ve piyasa listesi onceden yuklenmis MEXC istemcisi.

    ONEMLI PERFORMANS NOTU: ccxt her create_order/fetch_ohlcv/fetch_ticker/... cagrisinda,
    eger o istemcinin market listesi (self.markets) yuklu degilse otomatik olarak load_markets()
    calistirir; bu tek seferde ~3000 spot+vadeli sozlesmeyi indirip parse eder. Eskiden neredeyse
    her fonksiyon kendi taze ccxt.mexc() nesnesini olusturuyordu; bu da 3-5 saniyede bir calisan
    canli terminalde HER YENILEMEDE birden fazla kez bu agir load_markets() indirmesinin tekrar
    tekrar yapilmasina (ve panelin cok yavas/gec yuklenmesine) sebep oluyordu. Artik tum genel
    fonksiyonlar bu tek, onbelleklenmis istemciyi paylasiyor.
    """
    ex = ccxt.mexc({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    ex.load_markets()
    return ex


@st.cache_resource(ttl=1800, show_spinner=False)
def get_mexc_ccxt_client(api_key="", secret_key=""):
    key = api_key or os.getenv("MEXC_API_KEY", "")
    sec = secret_key or os.getenv("MEXC_SECRET_KEY", "")
    ex = ccxt.mexc({
        'apiKey': key,
        'secret': sec,
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    })
    ex.load_markets()
    return ex

@st.cache_data(ttl=5, show_spinner=False)
def fetch_binance_account_balance(api_key: str = "", secret_key: str = "") -> dict:
    # İsmi aynı bırakıyoruz, içi MEXC
    try:
        ex = get_mexc_ccxt_client(api_key, secret_key)
        bal = ex.fetch_balance()
        usdt_bal = bal.get('USDT', {})
        total = usdt_bal.get('total', 0.0)
        free = usdt_bal.get('free', 0.0)
        return {"status": "FUTURES_OK", "wallet_balance": total, "available_balance": free, "type": "MEXC Vadeli (Swap)"}
    except Exception as e:
        return {"status": "ERROR", "msg": str(e)}

@st.cache_data(ttl=1800, show_spinner=False)
def get_all_binance_futures_symbols() -> tuple:
    try:
        ex = get_public_mexc_client()
        markets = ex.markets
        symbols = []
        symbol_map = {}
        for sym, m in markets.items():
            # ONEMLI: ccxt-mexc load_markets() spot VE swap piyasalarini birlikte dondurur.
            # 'swap' filtresi olmadan spot semboller (orn. BTC/USDT) de listeye karisir ve
            # secildiginde bot MEXC Vadeli fiyati yerine yanlislikla Spot fiyatini gosterir/kullanir.
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

def normalize_futures_symbol(user_input: str) -> tuple:
    all_symbols, symbol_map = get_all_binance_futures_symbols()
    text = user_input.strip().upper().replace(" ", "").replace("_", "").replace("-", "")

    if text in all_symbols:
        return text, ""

    # "BTC/USDT" gibi spot gorunumlu (":USDT" vadeli sonekini icermeyen) girisleri
    # yanlislikla Spot fiyatini kullanmamak icin MEXC Vadeli karsiligina cevir.
    if "/" in text and ":" not in text:
        base = text.split("/")[0]
        if base in symbol_map:
            return symbol_map[base], f" MEXC Vadeli Eşleşti: {symbol_map[base]}"
        sym = f"{base}/USDT:USDT"
        if sym in all_symbols:
            return sym, f" MEXC Vadeli Eşleşti: {sym}"

    if "/" not in text and ":" not in text:
        clean = text.replace("USDT", "")
        if clean in symbol_map:
            return symbol_map[clean], f" MEXC Eşleşti: {symbol_map[clean]}"
        sym = f"{clean}/USDT:USDT"
        if sym in all_symbols: return sym, f" MEXC Eşleşti: {sym}"
        import difflib
        matches = difflib.get_close_matches(sym, all_symbols, n=1, cutoff=0.5)
        if matches: return matches[0], f"💡 Yakın Eşleşme: {matches[0]}"

    return user_input, ""

INTERVAL_MAP = {"Min1": "1m", "Min5": "5m", "Min15": "15m", "Min60": "1h", "Min240": "4h"}

@st.cache_data(ttl=3, show_spinner=False)
def fetch_binance_kline_data(symbol: str, interval: str = "Min1", limit: int = 500) -> pd.DataFrame:
    try:
        ex = get_public_mexc_client()
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
def fetch_binance_ticker_details(symbol: str) -> dict:
    try:
        ex = get_public_mexc_client()
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
def fetch_binance_depth_imbalance(symbol: str) -> dict:
    try:
        ex = get_public_mexc_client()
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
def fetch_institutional_order_flow(symbol: str, timeframe: str = "5m", limit: int = 150) -> pd.DataFrame:
    try:
        ex = get_public_mexc_client()
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
def compute_macro_support_resistance_levels(df: pd.DataFrame, symbol: str = "BTCUSDT") -> dict:
    """Uzun vadeli kalıcı majör S/R seviyeleri için 1D (Günlük) Hacim Profili (VRVP)
    ve Yapısal Pivot ekstrem noktalarını (Swing High/Low) hesaplar."""
    if len(df) < 5:
        return {}
        
    current_price = float(df['close'].iloc[-1])
    recent_close = float(df['close'].iloc[-2]) if len(df) > 1 else current_price
    
    # 1D Makro veriyi çek (Son 180 Günlük Hacim Profili için)
    try:
        _ex = get_public_mexc_client()
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
def compute_macro_timeframe_trend(symbol: str = "BTCUSDT") -> dict:
    """4 Saatlik (4h) ve Günlük (1d) zaman dilimlerindeki makro trend yönünü hesaplar."""
    try:
        _ex = get_public_mexc_client()
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
def compute_4h_scalp_prediction(symbol: str = "BTCUSDT") -> dict:
    """4 Saatlik (4H) grafik verileri üzerinden uzun vadeli trend yönünü ve AI tahminini hesaplar."""
    try:
        df_4h = fetch_binance_kline_data(symbol, interval="Min240", limit=100)
        if not df_4h.empty and len(df_4h) >= 30:
            df_feat = compute_quantum_features(df_4h, symbol=symbol, interval="Min240")
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
def compute_quantum_features(df_raw: pd.DataFrame, symbol: str = "BTCUSDT", interval: str = "5m") -> pd.DataFrame:
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
    df_flow = fetch_institutional_order_flow(symbol, interval, limit=len(df))
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
    
    X_current = df[final_features].tail(1)
    raw_probs = ensemble.predict_proba(X_current)[0]
    classes = list(ensemble.classes_)
    
    p_long = float(raw_probs[classes.index(1)]) if 1 in classes else 0.0
    p_short = float(raw_probs[classes.index(-1)]) if -1 in classes else 0.0
    
    total = p_long + p_short + 1e-9
    p_long = p_long / total
    p_short = p_short / total
    
    latest = df.iloc[-1]
    if latest['rsi'] > 55 and latest['macd'] > 0: p_long += 0.05
    elif latest['rsi'] < 45 and latest['macd'] < 0: p_short += 0.05
        
    p_long = min(max(p_long, 0.0), 1.0)
    p_short = min(max(p_short, 0.0), 1.0)
    
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
def compute_institutional_risk_levels(
    current_price: float,
    atr: float,
    signal: str,
    leverage: int = 10,
    margin_budget: float = 100.0,
    confidence: float = 65.0,
    confluence_score: float = 70.0,
    sr_levels: dict = None
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
    st.markdown("### 🔐 MEXC API & Otomasyon")
    
    env_creds = load_env_credentials()


    
    with st.expander("🔑 API Key ve Secret Tanımla", expanded=not (env_creds["api_key"] and env_creds["secret_key"])):
        st.info("💡 API anahtarlarınızı ister aşağıdaki kutucuklara yapıştırabilir, isterseniz de `.env` dosyasına kaydedebilirsiniz.")
        user_api_key = st.text_input("MEXC API Key:", value=env_creds["api_key"], type="password", help="MEXC hesabınızdan ürettiğiniz Futures API Key")
        user_secret_key = st.text_input("MEXC Secret Key:", value=env_creds["secret_key"], type="password", help="MEXC Secret Key")
        
        if user_api_key:
            os.environ["MEXC_API_KEY"] = user_api_key
        if user_secret_key:
            os.environ["MEXC_SECRET_KEY"] = user_secret_key
            
    active_api_key = os.getenv("MEXC_API_KEY", "").strip()
    active_secret_key = os.getenv("MEXC_SECRET_KEY", "").strip()
    
    wallet_balance_val = 0.0
    if active_api_key and active_secret_key:
        bal = fetch_binance_account_balance(active_api_key, active_secret_key)
        if bal["status"] == "FUTURES_OK":
            wallet_balance_val = bal['wallet_balance']
            st.success(f"🟢 Private API: VADELİ BAĞLI & HAZIR\n\n💰 Vadeli Cüzdan: ${wallet_balance_val:,.2f} USDT")
        else:
            st.warning(" API Anahtarı Algılandı Fakat Yetki Reddedildi (IP kısıtlaması veya geçersiz anahtar)")
    else:
        st.warning(" Private API: BAĞLI DEĞİL (Yalnızca Sinyal Modu)")
        st.caption("📌 Proje klasörünüzdeki `.env` dosyasına API anahtarınızı girip kaydedin VEYA yukarıdaki menüye yapıştırın.")
        
    st.markdown("---")
    st.markdown("### 💵 Sermaye & Kasa Takip Paneli")
    initial_capital = st.number_input("Başlangıç Sermayesi ($):", min_value=10.0, max_value=100000.0, value=100.0, step=50.0, help="Botu başlattığınız kasa bütçesi")
    
    current_kasa = wallet_balance_val if wallet_balance_val > 0 else initial_capital
    net_pnl = current_kasa - initial_capital
    net_return_pct = (net_pnl / (initial_capital + 1e-9)) * 100.0
    pnl_clr = "#10b981" if net_pnl >= 0 else "#ef4444"
    
    st.markdown(f"""
    <div style="background:#0b1120; padding:10px 14px; border-radius:8px; border:1px solid #1e293b; margin-top:6px;">
        <div style="display:flex; justify-content:space-between; font-size:12px; color:#94a3b8;">
            <span>Güncel Anlık Kasa:</span>
            <b style="color:#38bdf8;">${current_kasa:,.2f} USD</b>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:13px; font-weight:bold; margin-top:4px;">
            <span>Net Kâr/Zarar:</span>
            <span style="color:{pnl_clr};">{net_pnl:+,.2f} USD (%{net_return_pct:+.1f})</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    

# --- 7. BAŞLIK VE KONTROL PANELİ ---
st.title("⚡ Canlı Gösterge")

# Favori coin butonları için session state başlat
if "active_symbol_query" not in st.session_state:
    st.session_state["active_symbol_query"] = "BTCUSDT"

# Hızlı Favori Coin Şeridi
st.markdown("<div style='margin-bottom:8px; font-size:12px; color:#94a3b8; font-weight:bold; text-transform:uppercase;'>⚡ Hızlı Favori Coin Seçimi (Tek Tıkla Adaptasyon):</div>", unsafe_allow_html=True)
fav_c1, fav_c2, fav_c3, fav_c4, fav_c5, fav_c6, fav_c7 = st.columns(7)

if fav_c1.button("⚡ BTC (Bitcoin)", use_container_width=True):
    st.session_state["active_symbol_query"] = "BTCUSDT"
    st.rerun()
if fav_c2.button("💎 ETH (Ethereum)", use_container_width=True):
    st.session_state["active_symbol_query"] = "ETHUSDT"
    st.rerun()
if fav_c3.button("🚀 SOL (Solana)", use_container_width=True):
    st.session_state["active_symbol_query"] = "SOLUSDT"
    st.rerun()
if fav_c4.button("🐸 1000PEPE", use_container_width=True):
    st.session_state["active_symbol_query"] = "1000PEPEUSDT"
    st.rerun()
if fav_c5.button("🌊 SUI (Sui)", use_container_width=True):
    st.session_state["active_symbol_query"] = "SUIUSDT"
    st.rerun()
if fav_c6.button("🐕 DOGE (Doge)", use_container_width=True):
    st.session_state["active_symbol_query"] = "DOGEUSDT"
    st.rerun()
if fav_c7.button("⚡ XRP (Ripple)", use_container_width=True):
    st.session_state["active_symbol_query"] = "XRPUSDT"
    st.rerun()

all_futures_list, _ = get_all_binance_futures_symbols()
total_coin_count = len(all_futures_list)

col_search, col_dropdown, col_tf, col_lev, col_budget, col_speed = st.columns([1.8, 1.8, 1.1, 1.0, 1.1, 1.0])

with col_search:
    user_query = st.text_input(
        f"🔍 Serbest Arama ({total_coin_count} Coin):",
        value=st.session_state.get("active_symbol_query", "BTCUSDT"),
        placeholder="Örn: btc, pengui, sol, eth, pepe, btctry",
        help="Herhangi bir coin adı yazın. Akıllı eşleştirici otomatik bulur ve analizi başlatır."
    )

symbol_str, match_info = normalize_futures_symbol(user_query)

with col_dropdown:
    default_idx = all_futures_list.index(symbol_str) if symbol_str in all_futures_list else 0
    selected_from_dropdown = st.selectbox(
        f"📋 Tüm MEXC Çiftleri ({total_coin_count} Coin):",
        all_futures_list,
        index=default_idx
    )
    if selected_from_dropdown != symbol_str:
        symbol_str = selected_from_dropdown

# Entegre Coin ve Canlı Yapay Zeka Analiz Durumu Kartı
st.markdown(f"""
<div style="background:#0b1120; border:1px solid #1e293b; padding:10px 16px; border-radius:8px; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;">
    <div>
        <span style="font-size:11px; color:#94a3b8; text-transform:uppercase; font-weight:bold;">🌐 MEXC Entegreli Toplam İşlem Çifti Sayısı:</span>
        <div style="font-size:18px; font-weight:900; color:#38bdf8;">{total_coin_count} / {total_coin_count} İşlem Çifti <span style="font-size:12px; color:#10b981; font-weight:bold;">(%100 Tam Borsa Entegre)</span></div>
    </div>
    <div style="text-align:right;">
        <span style="font-size:11px; color:#94a3b8; text-transform:uppercase; font-weight:bold;">🤖 Anlık Canlı Analiz Edilen Coin:</span>
        <div style="font-size:16px; font-weight:bold; color:#10b981;"><span style="color:#38bdf8;">{symbol_str}</span> Sinyal & Derinlik Analizi Aktif</div>
    </div>
</div>
""", unsafe_allow_html=True)

with col_tf:
    timeframe_map = {
        "1 Dakika (Scalp)": "Min1",
        "5 Dakika (Day Trade)": "Min5",
        "15 Dakika (Swing)": "Min15",
        "1 Saat (Trend)": "Min60",
        "4 Saatlik (Saatlik Scalp / Trend)": "Min240"
    }
    selected_tf_label = st.selectbox("⏱️ Zaman Dilimi:", list(timeframe_map.keys()), index=0)
    active_interval = timeframe_map[selected_tf_label]

with col_lev:
    leverage_options = [1, 2, 3, 5, 10, 15, 20, 25, 50, 75, 100]
    selected_leverage = st.selectbox("⚡ Kaldıraç:", leverage_options, index=4, format_func=lambda x: f"{x}x")

with col_budget:
    margin_budget = st.number_input("💰 Bütçe / Marjin ($):", min_value=10.0, max_value=100000.0, value=100.0, step=50.0, format="%.0f")

with col_speed:
    refresh_rate = st.selectbox("🔄 Akış Hızı:", [3, 5, 10, 15], index=0, format_func=lambda x: f"{x} Saniyede Bir")

if match_info and "💡" in match_info:
    st.info(match_info)


# --- 8. ANLIK CANLI QUANTUM TERMİNALİ (STREAMLIT FRAGMENT) ---
@st.fragment(run_every=refresh_rate)
def render_quantum_terminal():
    fetch_start = time.time()
    
    # 1. Gerçek MEXC Futures ve Dolar Dominansı Verisi Çekme
    df_raw = fetch_binance_kline_data(symbol_str, interval=active_interval, limit=350)
    ticker = fetch_binance_ticker_details(symbol_str)
    depth = fetch_binance_depth_imbalance(symbol_str)
    usdt_dom = fetch_usdt_dominance_matrix()
    
    if df_raw.empty or len(df_raw) < 50:
        st.error(f" '{symbol_str}' ({user_query}) için MEXC Vadeli İşlemler mum verisi alınamadı! Lütfen sembolü kontrol edin.")
        return
        
    current_price = ticker.get("lastPrice", df_raw['close'].iloc[-1])
    change_24h = ticker.get("riseFallRate", 0.0)
    high_24h = ticker.get("high24Price", df_raw['high'].max())
    low_24h = ticker.get("lower24Price", df_raw['low'].min())
    funding_rate = ticker.get("fundingRate", 0.0)
    
    # 2. Özellik Mühendisliği (Sembol ve Zaman Dilimi Geçildi)
    df_features = compute_quantum_features(df_raw, symbol=symbol_str, interval=active_interval)
    latest_row = df_features.iloc[-1]
    
    # 3. Yapay Zeka Modeli Eğitimi ve Olasılık Tahmini
    ai_result = train_and_predict_quantum_ai(df_features)
    
    # 3.1. Uzun Vadeli Statik Destek/Direnç Seviyeleri, Makro Trend, 4H Tahmin ve Likidasyon Taraması
    sr_levels = compute_macro_support_resistance_levels(df_raw, symbol_str)
    macro_tf = compute_macro_timeframe_trend(symbol_str)
    ai_4h_pred = compute_4h_scalp_prediction(symbol_str)
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
        sr_levels=sr_levels
    )
    
    # Fiyat Formatlama Yardımcısı
    def fmt(p):
        return f"${p:,.6f}" if p < 1 else f"${p:,.4f}" if p < 100 else f"${p:,.2f}"
        
    # Üst Bilgi Rozeti (Seçili Coin / Zaman Dilimi / Kaldıraç / Bütçe)
    st.markdown(f"""
    <div style="background:#0b1120; padding:10px 16px; border-radius:8px; border:1px solid #1e293b; margin-bottom:12px;">
        <span style="font-size: 20px; font-weight: 800; color: #38bdf8;">🟢 MEXC: {symbol_str}</span>
        <span style="color: #94a3b8; font-size: 13px; margin-left: 10px;">| {selected_tf_label} | <b style="color:#f59e0b;">{selected_leverage}x Kaldıraç</b> | <b style="color:#10b981;">Bütçe: ${margin_budget:,.0f}</b></span>
    </div>
    """, unsafe_allow_html=True)
    
    # =========================================================================
    # KANTİTATİF ANALİZ VE İNDİKATÖR RADARI
    # =========================================================================
    with st.container():
        # --- ÜST METRİK ŞERİDİ ---
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        with m1:
            st.markdown("<div class='quantum-card'>", unsafe_allow_html=True)
            st.markdown("<div class='sub-label'>💰 Anlık Canlı Fiyat</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='price-display'>{fmt(current_price)}</div>", unsafe_allow_html=True)
            chg_cls = "val-positive" if change_24h >= 0 else "val-negative"
            st.markdown(f"<div class='{chg_cls}'>24s Değişim: %{change_24h:+.2f}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with m2:
            st.markdown("<div class='quantum-card'>", unsafe_allow_html=True)
            st.markdown("<div class='sub-label'> 24s En Yüksek / Düşük</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 15px; font-weight: bold; color: #10b981;'>Y: {fmt(high_24h)}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 15px; font-weight: bold; color: #ef4444;'>D: {fmt(low_24h)}</div>", unsafe_allow_html=True)
            st.markdown("<div style='color: #64748b; font-size: 11px; margin-top: 2px;'>Aralık Derinliği</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with m3:
            st.markdown("<div class='quantum-card'>", unsafe_allow_html=True)
            st.markdown("<div class='sub-label'>⚡ Fonlama & Piyasa Eğilimi</div>", unsafe_allow_html=True)
            fund_color = "val-negative" if funding_rate > 0.02 else ("val-positive" if funding_rate < -0.01 else "val-neutral")
            st.markdown(f"<div style='font-size: 18px; font-weight: bold;' class='{fund_color}'>%{funding_rate:.4f}</div>", unsafe_allow_html=True)
            fund_note = "Uzun Ağırlıkta" if funding_rate > 0 else "Kısa Ağırlıkta"
            st.markdown(f"<div style='color: #94a3b8; font-size: 11px;'>{fund_note}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with m4:
            st.markdown("<div class='quantum-card'>", unsafe_allow_html=True)
            st.markdown("<div class='sub-label'>⚖️ Tahta Alıcı/Satıcı Baskısı</div>", unsafe_allow_html=True)
            b_ratio = depth.get("bid_ratio", 50)
            a_ratio = depth.get("ask_ratio", 50)
            st.markdown(f"<div style='font-size: 16px; font-weight: bold;'><span style='color:#10b981;'>%{b_ratio:.0f}</span> / <span style='color:#ef4444;'>%{a_ratio:.0f}</span></div>", unsafe_allow_html=True)
            st.progress(b_ratio / 100.0)
            st.markdown(f"<div style='color: #94a3b8; font-size: 11px;'>{depth.get('bias', 'DENGELİ')}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with m5:
            st.markdown("<div class='quantum-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='sub-label'>🌐 Dolar Dominansı (USDT.D)</div>", unsafe_allow_html=True)
            u_val_fmt = f"%{usdt_dom['usdt_d']:.3f}"
            u_chg_val = usdt_dom['usdt_d_change']
            u_dir_badge = "<span style='color:#10b981; font-weight:bold; font-size:12px;'>🟢 LONG YÖNLÜ (▼ Kripto Boğa)</span>" if u_chg_val < 0 else "<span style='color:#ef4444; font-weight:bold; font-size:12px;'>🔻 SHORT YÖNLÜ (▲ Nakite Kaçış)</span>"
            st.markdown(f"<div style='font-size: 18px; font-weight: bold; color: {'#10b981' if u_chg_val < 0 else '#ef4444'};'>{u_val_fmt}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='margin-top:2px;'>{u_dir_badge}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with m6:
            st.markdown("<div class='quantum-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='sub-label'>⚡ İşlem Gücü Katsayısı</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 16px; font-weight: bold; color: {risk['power_color']};'>{risk['power_title']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='color: #94a3b8; font-size: 11px; margin-top: 2px;'>Güç Skoru: %{risk['trade_power']:.1f}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # --- ANA YAPAY ZEKA VE STRATEJİ BÖLÜMÜ ---
        col_ai, col_strategy_preview = st.columns([1.35, 1.65])
        
        with col_ai:
            st.markdown("<div class='quantum-card'>", unsafe_allow_html=True)
            st.markdown("<div class='sub-label'>🤖 YAPAY ZEKA İŞLEM SİNYALİ & DOĞRULAMA</div>", unsafe_allow_html=True)
            
            # Sinyal Rozeti
            st.markdown(f"<div class='{confluence['badge_class']}'>{confluence['signal_title']}</div>", unsafe_allow_html=True)
            
            st.markdown("<div style='margin-top: 15px;'>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 13px; color: #cbd5e1; margin-bottom: 4px;'><b>🧠 AI Topluluk Güven Skoru:</b> %{confluence['confidence']:.1f} (LONG: %{ai_result['prob_long']*100:.1f} | SHORT: %{ai_result['prob_short']*100:.1f})</div>", unsafe_allow_html=True)
            st.progress(ai_result['prob_long'])
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: #0b1120; padding: 12px; border-radius: 8px; border: 1px solid #1e293b; margin-top: 12px;">
                <div style="font-size: 12px; color: #94a3b8;"><b>Doğruluk / Confluence Skoru:</b> <span style="font-size: 15px; font-weight: bold; color: #38bdf8;">%{confluence['confluence_score']}/100</span></div>
                <div style="font-size: 12px; color: #cbd5e1; margin-top: 5px;">💡 <i>{confluence['action_note']}</i></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Onay Listesi (Confluence Checklist)
            st.markdown("<div style='margin-top: 12px;'><div class='sub-label'>🛡️ Sahte Sinyal Filtre Kontrolleri</div>", unsafe_allow_html=True)
            for name, detail, status, weight in confluence["checks"]:
                cls_name = "check-pass" if status == "pass" else ("check-warn" if status == "warn" else "check-fail")
                icon = "" if status == "pass" else ("" if status == "warn" else "")
                st.markdown(f"""
                <div class='checklist-item {cls_name}'>
                    <span>{icon} <b>{name}</b></span>
                    <span style='font-size: 12px;'>{detail}</span>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_strategy_preview:
            st.markdown("<div class='quantum-card'>", unsafe_allow_html=True)
            total_pos_size = margin_budget * selected_leverage
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <div class='sub-label'>🎯 İŞLEM GÜCÜ VE KALDIRAÇLI HEDEF ÖZETİ</div>
                <div style='font-size:12px; color:#38bdf8; font-weight:bold;'>Pozisyon: ${total_pos_size:,.0f} ({selected_leverage}x)</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="background: #0b1120; padding: 8px 12px; border-radius: 8px; border: 1px solid #1e293b; margin-bottom: 12px; display:flex; justify-content:space-between; align-items:center;">
                <span>📍 <b>Giriş Fiyatı:</b> <span style="color:#38bdf8; font-size:16px; font-weight:bold;">{fmt(risk['entry'])}</span></span>
                <span>💰 <b>Marjin:</b> ${margin_budget:,.0f} | <b>Kaldıraç:</b> {selected_leverage}x</span>
            </div>
            """, unsafe_allow_html=True)
            
            # 1. MİN KAZANÇ
            st.markdown(f"""
            <div class='profit-box-min'>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:13px; font-weight:bold; color:#10b981;">🟢 MİN KAZANÇ (Yüksek İhtimalli TP)</span>
                    <span style="background:rgba(16,185,129,0.2); color:#10b981; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:bold;">R:R = 1:{risk['rr_min']:.2f}</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-top:6px;">
                    <div>
                        <div style="font-size:18px; font-weight:bold; color:#f8fafc;">{fmt(risk['min_tp_price'])}</div>
                        <div style="font-size:12px; color:#94a3b8;">Fiyat: <b style="color:#10b981;">+%{risk['actual_min_pct']:.2f}</b> (ROE: <b style="color:#10b981;">+%{risk['actual_min_roe']:.1f}</b>)</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:11px; color:#94a3b8;">Net Min Kazanç:</div>
                        <div style="font-size:20px; font-weight:900; color:#10b981;">+${risk['min_profit_usd']:,.2f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 2. MAX KAZANÇ
            st.markdown(f"""
            <div class='profit-box-max'>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:13px; font-weight:bold; color:#38bdf8;">🚀 MAX KAZANÇ (Zirve Potansiyel TP)</span>
                    <span style="background:rgba(56,189,248,0.2); color:#38bdf8; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:bold;">R:R = 1:{risk['rr_max']:.2f}</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-top:6px;">
                    <div>
                        <div style="font-size:18px; font-weight:bold; color:#f8fafc;">{fmt(risk['max_tp_price'])}</div>
                        <div style="font-size:12px; color:#94a3b8;">Fiyat: <b style="color:#38bdf8;">+%{risk['actual_max_pct']:.2f}</b> (ROE: <b style="color:#38bdf8;">+%{risk['actual_max_roe']:.1f}</b>)</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:11px; color:#94a3b8;">Net Max Kazanç:</div>
                        <div style="font-size:20px; font-weight:900; color:#38bdf8;">+${risk['max_profit_usd']:,.2f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # 3. STOP-LOSS
            st.markdown(f"""
            <div class='loss-box-sl'>
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:13px; font-weight:bold; color:#ef4444;">🛑 ORANTILI STOP-LOSS (SL)</span>
                    <span style="background:rgba(239,68,68,0.2); color:#ef4444; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:bold;">Risk Sınırı</span>
                </div>
                <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-top:6px;">
                    <div>
                        <div style="font-size:18px; font-weight:bold; color:#f8fafc;">{fmt(risk['sl_price'])}</div>
                        <div style="font-size:12px; color:#94a3b8;">Fiyat: <b style="color:#ef4444;">-%{risk['actual_sl_pct']:.2f}</b> (ROE: <b style="color:#ef4444;">-%{risk['actual_sl_roe']:.1f}</b>)</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:11px; color:#94a3b8;">Göze Alınan Risk:</div>
                        <div style="font-size:20px; font-weight:900; color:#ef4444;">-${risk['max_loss_usd']:,.2f}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # --- UZUN VADELİ MAJÖR DESTEK VE DİRENÇ YAPISAL ANALİZ KARTI ---
        st.markdown("<div class='quantum-card'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <div class='sub-label'>🏰 UZUN VADELİ MAJÖR DESTEK & DİRENÇ YAPISAL ANALİZİ (KADEMELİ S1/S2 - R1/R2 MATRİSİ)</div>
            <div style='font-size:12px; font-weight:bold; color:#10b981;'>{macro_tf['macro_badge']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 1. Net Yön Analizi ve Vektör Rozeti
        sr_dir_clr = "#10b981" if "BOUNCE" in sr_levels.get("sr_direction_code", "") or "BREAKOUT" in sr_levels.get("sr_direction_code", "") or "BULL" in sr_levels.get("sr_direction_code", "") else "#ef4444"
        st.markdown(f"""
        <div style="background:#0b1120; padding:10px 16px; border-radius:8px; border:1px solid {sr_dir_clr}; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:11px; color:#94a3b8; text-transform:uppercase; font-weight:bold;">🎯 DESTEK VE DİRENÇ YÖN HESAPLAMASI & TAHMİNİ:</span>
                <div style="font-size:16px; font-weight:900; color:{sr_dir_clr}; margin-top:2px;">{sr_levels.get('sr_direction', '')}</div>
            </div>
            <div style="text-align:right;">
                <span style="font-size:11px; color:#94a3b8; text-transform:uppercase; font-weight:bold;">⚖️ Haftalık / Günlük Pivot Denge:</span>
                <div style="font-size:16px; font-weight:bold; color:#38bdf8;">{fmt(sr_levels.get('pp', 0))}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        sr1, sr2, sr3 = st.columns([1.3, 1.3, 1.4])
        with sr1:
            st.markdown(f"""
            <div style="background:#0b1120; padding:10px 14px; border-radius:8px; border:1px solid #10b981;">
                <div style="font-size:11px; color:#94a3b8;">🛡️ STATİK DESTEK KADEMELERİ</div>
                <div style="font-size:16px; font-weight:bold; color:#10b981; margin-top:3px;">S1 (Birincil): {fmt(sr_levels.get('s1', 0))}</div>
                <div style="font-size:13px; color:#64748b; font-weight:bold;">S2 (Derin Dip): {fmt(sr_levels.get('s2', 0))}</div>
                <div style="font-size:11px; color:#cbd5e1; margin-top:4px;">Fiyata Uzaklık: <b>%{sr_levels.get('dist_sup_pct', 0):.2f}</b></div>
            </div>
            """, unsafe_allow_html=True)
            
        with sr2:
            st.markdown(f"""
            <div style="background:#0b1120; padding:10px 14px; border-radius:8px; border:1px solid #ef4444;">
                <div style="font-size:11px; color:#94a3b8;">🏰 STATİK DİRENÇ KADEMELERİ</div>
                <div style="font-size:16px; font-weight:bold; color:#ef4444; margin-top:3px;">R1 (Birincil): {fmt(sr_levels.get('r1', 0))}</div>
                <div style="font-size:13px; color:#64748b; font-weight:bold;">R2 (Zirve): {fmt(sr_levels.get('r2', 0))}</div>
                <div style="font-size:11px; color:#cbd5e1; margin-top:4px;">Fiyata Uzaklık: <b>%{sr_levels.get('dist_res_pct', 0):.2f}</b></div>
            </div>
            """, unsafe_allow_html=True)
            
        with sr3:
            st.markdown(f"""
            <div style="background:#0b1120; padding:10px 14px; border-radius:8px; border:1px solid #38bdf8;">
                <div style="font-size:11px; color:#94a3b8;">📌 REAKSİYON & STRATEJİ UYARISI</div>
                <div style="font-size:12px; font-weight:bold; color:#38bdf8; margin-top:4px;">{sr_levels.get('sr_note', '')}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

        # --- SCALP ADAPTASYONU VE 4 SAATLİK (4H) MAKRO TAHMİN KARTI ---
        st.markdown("<div class='quantum-card'>", unsafe_allow_html=True)
        is_scalp_tf = active_interval in ["Min1", "Min5", "Min240"]
        scalp_badge = "⚡ SCALP / 4H MODU AKTİF (1m/5m/4h Hızlı Reaksiyon & Dar Marjin)" if is_scalp_tf else " SWING / TREND MODU AKTİF (Dengeli Rejim)"
        
        scalp_dir = "LONG" if "LONG" in confluence["final_signal"] else ("SHORT" if "SHORT" in confluence["final_signal"] else "NEUTRAL")
        tf4h_dir = ai_4h_pred["direction"]
        
        if scalp_dir == tf4h_dir and scalp_dir != "NEUTRAL":
            alignment_badge = " %100 YÖN UYUMLU: 1m/5m SCALP YÖNÜ İLE 4H MAKRO TAHMİNİ BİREBİR AYNI (YÜKSEK KAZANÇ İHTİMALİ)"
            align_clr = "#10b981"
        elif scalp_dir != "NEUTRAL" and tf4h_dir != "NEUTRAL":
            alignment_badge = " KARŞI TREND SCALP: 1m/5m Scalp Yönü 4H Makro Trendin Tersine! (Hızlı Kâr Alıp Çıkın)"
            align_clr = "#f59e0b"
        else:
            alignment_badge = "⚖️ DENGELİ / NÖTR KOMBİNASYON"
            align_clr = "#38bdf8"
            
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <div class='sub-label'>⚡ SCALP ADAPTASYONU VE 4 SAATLİK (4H) İŞLEM ANALİZİ & TAHMİNİ</div>
            <div style='font-size:12px; font-weight:bold; color:#38bdf8;'>{scalp_badge}</div>
        </div>
        
        <div style="background:#0b1120; padding:10px 16px; border-radius:8px; border:1px solid {align_clr}; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <span style="font-size:11px; color:#94a3b8; text-transform:uppercase; font-weight:bold;">⚡ Scalp & 4H Trend Uyum Durumu:</span>
                <div style="font-size:13px; font-weight:900; color:{align_clr}; margin-top:2px;">{alignment_badge}</div>
            </div>
            <div style="text-align:right;">
                <span style="font-size:11px; color:#94a3b8; text-transform:uppercase; font-weight:bold;">🤖 4H AI Tahmin Güveni:</span>
                <div style="font-size:16px; font-weight:bold; color:{ai_4h_pred['color']};">%{ai_4h_pred['confidence']:.1f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        sc1, sc2, sc3 = st.columns([1.3, 1.3, 1.4])
        with sc1:
            st.markdown(f"""
            <div style="background:#0b1120; padding:10px 14px; border-radius:8px; border:1px solid {ai_4h_pred['color']};">
                <div style="font-size:11px; color:#94a3b8;"> 4 SAATLİK (4H) TAHMİN</div>
                <div style="font-size:16px; font-weight:bold; color:{ai_4h_pred['color']}; margin-top:3px;">{ai_4h_pred['pred_title']}</div>
                <div style="font-size:11px; color:#cbd5e1; margin-top:4px;">Model Güveni: <b>%{ai_4h_pred['confidence']:.1f}</b></div>
            </div>
            """, unsafe_allow_html=True)
            
        with sc2:
            st.markdown(f"""
            <div style="background:#0b1120; padding:10px 14px; border-radius:8px; border:1px solid #10b981;">
                <div style="font-size:11px; color:#94a3b8;">🔐 PRİVATE API BAĞLANTI ANALİZİ</div>
                <div style="font-size:14px; font-weight:bold; color:#10b981; margin-top:3px;">🟢 Borsa API: AKTİF & HAZIR</div>
                <div style="font-size:11px; color:#cbd5e1; margin-top:4px;">Cüzdan Bakiyesi: <b>${wallet_balance_val:,.2f} USDT</b></div>
            </div>
            """, unsafe_allow_html=True)
            
        with sc3:
            st.markdown(f"""
            <div style="background:#0b1120; padding:10px 14px; border-radius:8px; border:1px solid #f59e0b;">
                <div style="font-size:11px; color:#94a3b8;">⚡ SCALP MİKRO EMİR AKIŞI</div>
                <div style="font-size:13px; font-weight:bold; color:#f59e0b; margin-top:4px;">Tahta Dengesi: %{depth.get('imbalance', 0):+.1f} ({depth.get('bias', 'DENGELİ')})</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

        # --- KURUMSAL LİKİDASYON ISI HARİTASI VE AKILLI İZLEYEN STOP KARTI ---
        st.markdown("<div class='quantum-card'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
            <div class='sub-label'>🔥 KURUMSAL LİKİDASYON ISI HARİTASI & AKILLI İZLEYEN STOP (TRAILING STOP)</div>
            <div style='font-size:12px; font-weight:bold; color:{liq_matrix.get("liq_color", "#10b981")};'>{liq_matrix.get("liq_badge", "")}</div>
        </div>
        """, unsafe_allow_html=True)
        
        lq1, lq2, lq3 = st.columns([1.3, 1.3, 1.4])
        with lq1:
            st.markdown(f"""
            <div style="background:#0b1120; padding:10px 14px; border-radius:8px; border:1px solid #ef4444;">
                <div style="font-size:11px; color:#94a3b8;">🩸 50x/100x LONG LİKİDASYON DUVARI</div>
                <div style="font-size:16px; font-weight:bold; color:#ef4444; margin-top:3px;">100x Liq: {fmt(liq_matrix.get('long_liq_100x', 0))}</div>
                <div style="font-size:12px; color:#64748b; font-weight:bold;">50x Liq: {fmt(liq_matrix.get('long_liq_50x', 0))}</div>
                <div style="font-size:11px; color:#cbd5e1; margin-top:4px;">Duvar Uzaklığı: <b>%{liq_matrix.get('dist_long_liq_pct', 0):.2f}</b></div>
            </div>
            """, unsafe_allow_html=True)
            
        with lq2:
            st.markdown(f"""
            <div style="background:#0b1120; padding:10px 14px; border-radius:8px; border:1px solid #f59e0b;">
                <div style="font-size:11px; color:#94a3b8;">🚀 50x/100x SHORT LİKİDASYON DUVARI</div>
                <div style="font-size:16px; font-weight:bold; color:#f59e0b; margin-top:3px;">100x Liq: {fmt(liq_matrix.get('short_liq_100x', 0))}</div>
                <div style="font-size:12px; color:#64748b; font-weight:bold;">50x Liq: {fmt(liq_matrix.get('short_liq_50x', 0))}</div>
                <div style="font-size:11px; color:#cbd5e1; margin-top:4px;">Duvar Uzaklığı: <b>%{liq_matrix.get('dist_short_liq_pct', 0):.2f}</b></div>
            </div>
            """, unsafe_allow_html=True)
            
        with lq3:
            atr_val = latest_row["atr"]
            if "LONG" in confluence["final_signal"]:
                trail_sl = current_price - (atr_val * 1.5)
                trail_note = f"🟢 İzleyen Stop Seviyesi: <b>{fmt(trail_sl)}</b> (Fiyat yükseldikçe stop yukarı kayar)"
            elif "SHORT" in confluence["final_signal"]:
                trail_sl = current_price + (atr_val * 1.5)
                trail_note = f"🔻 İzleyen Stop Seviyesi: <b>{fmt(trail_sl)}</b> (Fiyat düştükçe stop aşağı kayar)"
            else:
                trail_note = "⚖️ Pozisyon Beklemede (Akıllı İzleyen Stop Pasif)"
                
            st.markdown(f"""
            <div style="background:#0b1120; padding:10px 14px; border-radius:8px; border:1px solid #38bdf8;">
                <div style="font-size:11px; color:#94a3b8;">🎯 AKILLI İZLEYEN STOP & PARÇALI KÂR</div>
                <div style="font-size:12px; color:#38bdf8; font-weight:bold; margin-top:3px;">{trail_note}</div>
                <div style="font-size:11px; color:#cbd5e1; margin-top:4px;">Parçalı Çıkış: <b>%50 TP1'de Kâr Al + Girişe SL Çek</b></div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

        # --- TEKNİK İNDİKATÖR RADARI VE YÖN MATRİSİ ---
        st.markdown("<div class='quantum-card'>", unsafe_allow_html=True)
        st.markdown("<div class='sub-label'>📡 ÇOK BOYUTLU İNDİKATÖR RADARI & CANLI YÖN MATRİSİ (LONG / SHORT EĞİLİMLERİ)</div>", unsafe_allow_html=True)
        
        ind1, ind2, ind3, ind4, ind5, ind6, ind7 = st.columns(7)
        
        with ind1:
            rsi_val = latest_row["rsi"]
            rsi_dir = "🟢 LONG" if rsi_val > 50 else "🔻 SHORT"
            rsi_clr = "#10b981" if rsi_val > 50 else "#ef4444"
            st.markdown(f"""
            <div class='indicator-box'>
                <div class='indicator-title'>RSI (14)</div>
                <div class='indicator-value' style='color:{rsi_clr};'>{rsi_val:.1f}</div>
                <div class='indicator-status' style='color:{rsi_clr}; font-weight:bold;'>{rsi_dir}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with ind2:
            macd_diff_val = latest_row["macd_diff"]
            macd_dir = "🟢 LONG" if macd_diff_val > 0 else "🔻 SHORT"
            macd_clr = "#10b981" if macd_diff_val > 0 else "#ef4444"
            st.markdown(f"""
            <div class='indicator-box'>
                <div class='indicator-title'>MACD HİST</div>
                <div class='indicator-value' style='color:{macd_clr};'>{macd_diff_val:+.3f}</div>
                <div class='indicator-status' style='color:{macd_clr}; font-weight:bold;'>{macd_dir}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with ind3:
            is_ema_bull = latest_row["ema_9"] > latest_row["ema_21"]
            ema_dir = "🟢 LONG" if is_ema_bull else "🔻 SHORT"
            ema_clr = "#10b981" if is_ema_bull else "#ef4444"
            st.markdown(f"""
            <div class='indicator-box'>
                <div class='indicator-title'>EMA (9/21)</div>
                <div class='indicator-value' style='color:{ema_clr}; font-size: 12px;'>{fmt(latest_row['ema_9'])}</div>
                <div class='indicator-status' style='color:{ema_clr}; font-weight:bold;'>{ema_dir}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with ind4:
            is_ema200_bull = current_price > latest_row["ema_200"]
            ema200_dir = "🟢 LONG" if is_ema200_bull else "🔻 SHORT"
            ema200_clr = "#10b981" if is_ema200_bull else "#ef4444"
            st.markdown(f"""
            <div class='indicator-box'>
                <div class='indicator-title'>EMA 200 MAKRO</div>
                <div class='indicator-value' style='color:{ema200_clr}; font-size: 12px;'>{fmt(latest_row['ema_200'])}</div>
                <div class='indicator-status' style='color:{ema200_clr}; font-weight:bold;'>{ema200_dir}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with ind5:
            stoch_k_val = latest_row["stoch_k"]
            stoch_d_val = latest_row["stoch_d"]
            stoch_dir = "🟢 LONG" if stoch_k_val > stoch_d_val else "🔻 SHORT"
            stoch_clr = "#10b981" if stoch_k_val > stoch_d_val else "#ef4444"
            st.markdown(f"""
            <div class='indicator-box'>
                <div class='indicator-title'>STOKASTİK</div>
                <div class='indicator-value' style='color:{stoch_clr};'>%{stoch_k_val:.1f}</div>
                <div class='indicator-status' style='color:{stoch_clr}; font-weight:bold;'>{stoch_dir}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with ind6:
            bb_pct_val = latest_row["bb_pct"]
            bb_dir = "🟢 LONG" if bb_pct_val > 0.50 else "🔻 SHORT"
            bb_clr = "#10b981" if bb_pct_val > 0.50 else "#ef4444"
            st.markdown(f"""
            <div class='indicator-box'>
                <div class='indicator-title'>BOLLİNGER BANT</div>
                <div class='indicator-value' style='color:{bb_clr};'>%{bb_pct_val*100:.0f}</div>
                <div class='indicator-status' style='color:{bb_clr}; font-weight:bold;'>{bb_dir}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with ind7:
            u_d_val = usdt_dom.get("usdt_d", 6.98)
            u_chg = usdt_dom.get("usdt_d_change", 0.0)
            usdt_dir = "🟢 LONG" if u_chg < 0 else "🔻 SHORT"
            usdt_clr = "#10b981" if u_chg < 0 else "#ef4444"
            st.markdown(f"""
            <div class='indicator-box'>
                <div class='indicator-title'>🌐 USDT.D DOMİNANS</div>
                <div class='indicator-value' style='color:{usdt_clr}; font-size:13px;'>%{u_d_val:.2f} {'▼' if u_chg < 0 else '▲'}</div>
                <div class='indicator-status' style='color:{usdt_clr}; font-weight:bold;'>{usdt_dir}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

    # İşlem Motoru Gecikme Bilgisi
    elapsed_ms = (time.time() - fetch_start) * 1000
    st.caption(f"⚡ BtcSatoshi Live-Trade & AI Quant Engine: {elapsed_ms:.1f} ms | Model: Ensemble (RF+ET+HGB) | Canlı Borsa: MEXC Futures v1 & v2")

# Terminali Çalıştır
render_quantum_terminal()