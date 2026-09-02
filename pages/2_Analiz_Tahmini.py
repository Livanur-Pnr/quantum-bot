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
    page_title="Analiz Tahmini",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded", 
)

if "balance" not in st.session_state: st.session_state.balance = 10000.0 
if "position" not in st.session_state: st.session_state.position = None

load_dotenv()

PREMIUM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800;900&display=swap');
    .stApp { background-color: #0b0e14; color: #d1d4dc; font-family: 'Inter', sans-serif; }
    section[data-testid="stSidebar"] { background-color: #131722; border-right: 1px solid #2a2e39; }
    
    .metric-card {
        background: #131722; border: 1px solid #2a2e39; border-radius: 8px;
        padding: 12px; margin-bottom: 10px; min-height: 85px;
        display: flex; flex-direction: column; justify-content: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-card h4 { color: #787b86; font-size: 0.65rem; margin: 0 0 6px 0; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card .value { font-size: 1.15rem; font-weight: 800; margin: 0; }
    
    .green { color: #22ab94; } .red { color: #f7525f; } .blue { color: #2962ff; } .white { color: #d1d4dc; }
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
# 2. KRİPTO VERİ ÇEKİMİ (MEXC - RATE LIMIT KORUMALI)
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(ttl=1800, show_spinner=False)
def get_public_mexc_client():
    """Tek, paylasilan ve piyasa listesi onceden yuklenmis MEXC istemcisi.

    ONEMLI PERFORMANS NOTU: Her fetch_crypto_data cagrisinda taze bir ccxt.mexc() nesnesi
    olusturulursa, ccxt ilk fetch_ohlcv cagrisinda otomatik load_markets() calistirir
    (~3000 spot+vadeli sozlesmeyi indirir). Bu, 5 saniyede bir yenilenen canli panelde
    her seferinde tekrarlanip yavas yuklemeye sebep oluyordu.
    """
    ex = ccxt.mexc({"enableRateLimit": True, "timeout": 30000, "options": {"defaultType": "swap"}})
    ex.load_markets()
    return ex

@st.cache_data(ttl=5, show_spinner=False)
def fetch_crypto_data(exchange_id: str, symbol: str, timeframe: str, limit: int = 1500) -> pd.DataFrame:
    try:
        # exchange_id görmezden gelinerek her halükarda MEXC Swap kullanılacak
        ex = get_public_mexc_client()

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
        st.error(f"MEXC verisi çekilirken teknik bir hata oluştu: {str(e)}")
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

def train_and_predict_ai(df: pd.DataFrame, target_candles: int, threshold: float, tp_m: float, sl_m: float):
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
    ai_signals, trade_status, active_tps, active_sls = [], [], [], []
    
    threshold = threshold / 100.0  # Normalize threshold
    
    for i in range(len(df)):
        if in_pos:
            if pos_type == 'LONG' and (df['low'].iloc[i] <= sl or df['high'].iloc[i] >= tp): in_pos = False
            elif pos_type == 'SHORT' and (df['high'].iloc[i] >= sl or df['low'].iloc[i] <= tp): in_pos = False
            
        if not in_pos:
            if df['prob_long'].iloc[i] > (threshold * 100):
                in_pos, pos_type, tp, sl = True, 'LONG', df['close'].iloc[i] + (df['atr'].iloc[i] * tp_m), df['close'].iloc[i] - (df['atr'].iloc[i] * sl_m)
            elif df['prob_short'].iloc[i] > (threshold * 100):
                in_pos, pos_type, tp, sl = True, 'SHORT', df['close'].iloc[i] - (df['atr'].iloc[i] * tp_m), df['close'].iloc[i] + (df['atr'].iloc[i] * sl_m)
                
        ai_signals.append(pos_type if in_pos else 'NONE')
        trade_status.append('IN_TRADE' if in_pos else 'IDLE')
        active_tps.append(tp if in_pos else np.nan)
        active_sls.append(sl if in_pos else np.nan)
        
    df['ai_signal'], df['trade_status'], df['active_tp'], df['active_sl'] = ai_signals, trade_status, active_tps, active_sls
    
    return df, 99.9, final_features

def format_price(p): return f"{p:.4f}" if p < 10 else f"{p:.2f}"

def build_realtime_chart(df: pd.DataFrame, threshold: float, tp_m: float, sl_m: float) -> go.Figure:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"], increasing_line_color="#22ab94", decreasing_line_color="#f7525f", increasing_fillcolor="#22ab94", decreasing_fillcolor="#f7525f", name="Fiyat"), row=1, col=1)
    fig.add_trace(go.Scattergl(x=df.index, y=df['prob_long'], mode='lines', line=dict(color='#22ab94', width=2), name='LONG %', fill='tozeroy', fillcolor='rgba(34, 171, 148, 0.1)'), row=2, col=1)
    fig.add_trace(go.Scattergl(x=df.index, y=df['prob_short'], mode='lines', line=dict(color='#f7525f', width=2), name='SHORT %', fill='tozeroy', fillcolor='rgba(247, 82, 95, 0.1)'), row=2, col=1)
    fig.add_hline(y=threshold, line_dash="dot", line_color="#d1d4dc", opacity=0.7, row=2, col=1)

    recent_df = df.tail(300) 
    for idx, row in recent_df[recent_df["ai_signal"] == "LONG"].iterrows():
        if idx != df.index[-1]:
            text = f"<b>AI LONG</b> | %{row['prob_long']:.1f}<br>TP: {format_price(row['active_tp'])} | SL: {format_price(row['active_sl'])}"
            fig.add_annotation(x=idx, y=row["low"]-(row["atr"]*0.3), text=text, showarrow=True, arrowhead=2, arrowcolor="#22ab94", ax=0, ay=40, font=dict(size=9, color="#d1d4dc"), bgcolor="rgba(34,171,148,0.15)", bordercolor="rgba(34,171,148,0.8)", borderwidth=1, row=1, col=1)
    for idx, row in recent_df[recent_df["ai_signal"] == "SHORT"].iterrows():
        if idx != df.index[-1]:
            text = f"<b>AI SHORT</b> | %{row['prob_short']:.1f}<br>TP: {format_price(row['active_tp'])} | SL: {format_price(row['active_sl'])}"
            fig.add_annotation(x=idx, y=row["high"]+(row["atr"]*0.3), text=text, showarrow=True, arrowhead=2, arrowcolor="#f7525f", ax=0, ay=-40, font=dict(size=9, color="#d1d4dc"), bgcolor="rgba(247,82,95,0.15)", bordercolor="rgba(247,82,95,0.8)", borderwidth=1, row=1, col=1)

    last_idx, last = df.index[-1], df.iloc[-1]
    future_idx = last_idx + (df.index[-1] - df.index[-2]) * 8 
    is_long = last['prob_long'] > last['prob_short']
    live_prob, live_dir = (last["prob_long"], "LONG") if is_long else (last["prob_short"], "SHORT")
    color = "#22ab94" if is_long else "#f7525f"
    last_price = last['close']
    live_tp = last_price + (last['atr'] * tp_m) if is_long else last_price - (last['atr'] * tp_m)
    live_sl = last_price - (last['atr'] * sl_m) if is_long else last_price + (last['atr'] * sl_m)
    
    if live_prob >= threshold:
        label_text = f"<b>⚡ CANLI SİNYAL: {'LONG' if is_long else 'SHORT'}</b><br><b>%{live_prob:.1f} GÜVEN</b>"
        bg_color, text_color, dash_style, line_width = color, "white", "solid", 3
    else:
        label_text = f"<b>⏳ CANLI TAHMİN: {'LONG' if is_long else 'SHORT'}</b><br>%{live_prob:.1f} (Eşik Altı)"
        bg_color, text_color, dash_style, line_width = "rgba(0,0,0,0.6)", color, "dot", 1

    fig.add_shape(type="line", x0=last_idx, y0=live_tp, x1=future_idx, y1=live_tp, line=dict(color=color, dash=dash_style, width=line_width), row=1, col=1)
    fig.add_shape(type="line", x0=last_idx, y0=last['close'], x1=future_idx, y1=live_tp, line=dict(color=color, dash="dot", width=1), opacity=0.5, row=1, col=1)
    fig.add_annotation(x=future_idx, y=live_tp, text=label_text, showarrow=True, arrowhead=2, arrowcolor=color, font=dict(size=12, color=text_color), align="left", bgcolor=bg_color, bordercolor=color, borderwidth=2, borderpad=6, ax=40, ay=0, row=1, col=1)

    fig.update_layout(template="plotly_dark", paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14", height=850, margin=dict(l=10, r=40, t=10, b=20), hovermode="x unified", showlegend=False, xaxis_rangeslider_visible=False)
    fig.update_xaxes(gridcolor="#1c212d", zeroline=False, showspikes=True, spikecolor="#2a2e39", rangebreaks=[])
    fig.update_yaxes(gridcolor="#1c212d", zeroline=False, side="right") 
    fig.update_yaxes(range=[0, 100], row=2, col=1) 
    return fig

# ─────────────────────────────────────────────────────────────────
# 5. ANA YÜRÜTME DÖNGÜSÜ
# ─────────────────────────────────────────────────────────────────
def main():
    with st.sidebar:
        st.markdown("## 🎯 SWING AI AYARLARI")
        symbol = st.selectbox("🪙 Coin", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT"])
        tf = st.selectbox("⏱️ Z. Dilimi", ["5m", "15m", "30m", "1h", "4h", "1d"], index=1)
        
        with st.expander("🛠️ Kurumsal ML & Risk Parametreleri", expanded=False):
            target_candles = st.slider("Hedef Süre (Mum)", 10, 50, 20, 5)
            ai_threshold = st.slider("Güven Eşiği (%)", 50, 95, 65, 1)
            tp_m = st.slider("TP Çarpanı (ATR)", 1.0, 8.0, 4.0, 0.5)
            sl_m = st.slider("SL Çarpanı (ATR)", 0.5, 4.0, 1.5, 0.1)

        st.markdown("---")
        # ONEMLI: Fragment (otomatik yenilenen bolum) icinde dogrudan "with st.sidebar:"
        # acilirsa Streamlit hata verir (StreamlitAPIException) - cunku o pozisyon tam
        # (fragment disi) calistirmada rezerve edilmemis olur. Bunun yerine burada, tam
        # calistirma sirasinda bos bir yer (.empty()) ayirip, fragment icinde sadece bu
        # yerin icini dolduruyoruz.
        paper_trade_slot = st.empty()

    refresh_rate = 5
    @st.fragment(run_every=refresh_rate)
    def render_classic_terminal():
        with st.spinner("Piyasa verileri yükleniyor... (API İstekleri Sıralanıyor)"):
            df_crypto = fetch_crypto_data("mexc", symbol, tf, limit=1500)
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
    
            df, train_acc, active_features = train_and_predict_ai(df, target_candles, ai_threshold, tp_m, sl_m)
    
        last, last_price, atr = df.iloc[-1], df.iloc[-1]['close'], df.iloc[-1]['atr']
        
        with paper_trade_slot.container():
            st.markdown("## 💸 SANAL İŞLEM (PAPER TRADING)")
            if st.session_state.position is None:
                st.markdown(f"**💰 Cüzdan Bakiyesi:** <span style='color:#22ab94'>${st.session_state.balance:,.2f}</span>", unsafe_allow_html=True)
                with st.form("paper_trade_form"):
                    trade_qty = st.number_input("Miktar (Coin Adedi)", min_value=0.001, value=0.1, step=0.01)
                    st.markdown(f"<small>İşlem Bedeli: ${(trade_qty * last_price):,.2f}</small>", unsafe_allow_html=True)
                    c_b, c_s = st.columns(2)
                    if c_b.form_submit_button("🟢 AL (LONG)"):
                        if trade_qty * last_price <= st.session_state.balance:
                            st.session_state.position = {'side': 'LONG', 'entry': last_price, 'qty': trade_qty, 'symbol': symbol}
                            st.rerun()
                    if c_s.form_submit_button("🔴 SAT (SHORT)"):
                        if trade_qty * last_price <= st.session_state.balance:
                            st.session_state.position = {'side': 'SHORT', 'entry': last_price, 'qty': trade_qty, 'symbol': symbol}
                            st.rerun()
            else:
                pos = st.session_state.position
                live_pnl = (last_price - pos['entry']) * pos['qty'] if pos['side'] == 'LONG' else (pos['entry'] - last_price) * pos['qty']
                # ONEMLI: Ayni markdown cagrisinda 2+ tane duz "$" (dolar) isareti olursa,
                # Streamlit aralarini LaTeX matematik modu sanip HTML/markdown'i bozuyor.
                # Bu yuzden dolar isaretleri "\$" olarak kacisli (escaped) kullaniliyor.
                st.markdown(f"**Açık İşlem:** {pos['side']} {pos['symbol']}<br>**Giriş:** \\${pos['entry']:,.2f}<br>**Canlı PnL:** <span style='color:{'#22ab94' if live_pnl>=0 else '#f7525f'}; font-size:1.2rem; font-weight:bold'>\\${live_pnl:,.2f}</span>", unsafe_allow_html=True)
                if st.button("❌ Pozisyonu Kapat", use_container_width=True):
                    st.session_state.balance += live_pnl
                    st.session_state.position = None
                    st.rerun()
        
        is_long = last["prob_long"] > last["prob_short"]
        live_prob, live_dir = (last["prob_long"], "LONG") if is_long else (last["prob_short"], "SHORT")
        live_color = "#22ab94" if is_long else "#f7525f"
        live_tp = last_price + (atr * tp_m) if is_long else last_price - (atr * tp_m)
        live_sl = last_price - (atr * sl_m) if is_long else last_price + (atr * sl_m)
    
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.markdown(f'<div class="metric-card"><h4>💰 {symbol} FİYAT</h4><p class="value white">${last_price:,.4f}</p></div>', unsafe_allow_html=True)
        with c2: st.markdown(f'<div class="metric-card"><h4>🟢 CANLI LONG %</h4><p class="value green">%{last["prob_long"]:.1f}</p></div>', unsafe_allow_html=True)
        with c3: st.markdown(f'<div class="metric-card"><h4>🔴 CANLI SHORT %</h4><p class="value red">%{last["prob_short"]:.1f}</p></div>', unsafe_allow_html=True)
        with c4: st.markdown(f'<div class="metric-card" style="border-color:#2962ff"><h4>💸 SANAL PORTFÖY</h4><p class="value white">${st.session_state.balance:,.0f}</p></div>', unsafe_allow_html=True)
            
        c5, c6, c7, c8 = st.columns(4)
        with c5: st.markdown(f'<div class="metric-card"><h4>🧠 Eğitim Seti Doğruluğu</h4><p class="value blue">%{train_acc:.1f}</p></div>', unsafe_allow_html=True)
        with c6: st.markdown(f'<div class="metric-card"><h4>⚖️ TAHTA BASKISI</h4><p class="value white">{"ALICI (Bids)" if last["ob_imbalance"] > 1 else "SATICI (Asks)"}</p></div>', unsafe_allow_html=True)
        with c7: st.markdown(f'<div class="metric-card"><h4>🌍 MAKRO (F&G/DXY)</h4><p class="value white">F&G: {int(last["fear_greed"])} | DXY: {last["dxy"]:.2f}</p></div>', unsafe_allow_html=True)
        with c8: st.markdown(f'<div class="metric-card" title="{", ".join(active_features)}"><h4>📊 AKTİF FEATURE</h4><p class="value blue">{len(active_features)} Özellik (Filtreli)</p></div>', unsafe_allow_html=True)
        
        if live_prob >= ai_threshold:
            stat_html = f'''<div class="metric-card" style="border-color:{live_color}; background:rgba({34 if is_long else 247},{171 if is_long else 82},{148 if is_long else 95},0.15); text-align:center; padding: 20px;">
                <h2 style="color:{live_color}; margin:0; font-weight:900; font-size:1.8rem;">🚀 SİNYAL ONAYLANDI! ŞİMDİ İŞLEME GİR! ({live_dir})</h2>
                <div style="margin-top:15px; font-size:1.4rem; font-weight:800; color:#d1d4dc;">Giriş: <span style="color:#ffffff">{format_price(last_price)}</span> &nbsp;&nbsp;|&nbsp;&nbsp; <span style="color:#22ab94">TP: {format_price(live_tp)}</span> &nbsp;&nbsp;|&nbsp;&nbsp; <span style="color:#f7525f">SL: {format_price(live_sl)}</span></div>
                <p style="margin:10px 0 0 0; color:{live_color}; font-size:1.1rem; font-weight:600;">🤖 KURUMSAL ML GÜVENİ: %{live_prob:.1f}</p></div>'''
        else:
            stat_html = f'''<div class="metric-card" style="border-color:#434651; background:rgba(67, 70, 81, 0.1); text-align:center; padding: 15px;">
                <h3 style="color:#787b86; margin:0; font-weight:800; font-size:1.3rem;">⏳ İZLEMEDE (Eşik Altı) - Şu an işleme GİRME.</h3>
                <p style="color:#787b86; font-size:1.0rem; margin-top:10px; margin-bottom:5px;">Beklenen Yön: <b style="color:{live_color}">{live_dir}</b> (Güven: %{live_prob:.1f})</p>
                <p style="color:#5d606b; font-size:0.9rem; margin:0;"><i>Potansiyel Giriş: {format_price(last_price)} &nbsp;|&nbsp; TP: {format_price(live_tp)} &nbsp;|&nbsp; SL: {format_price(live_sl)}</i></p></div>'''
        st.markdown(stat_html, unsafe_allow_html=True)
    
        fig = build_realtime_chart(df.tail(300).copy(), ai_threshold, tp_m, sl_m)
        st.plotly_chart(fig, use_container_width=True, height=850, config={'displayModeBar': False, 'scrollZoom': True})
    
    render_classic_terminal()

if __name__ == "__main__":
    main()
