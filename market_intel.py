"""ORTAK PIYASA ISTIHBARAT MODULU (Coinglass + Dolar Dominansi)

NEDEN BU MODUL VAR:
Daha once iki panel (Analiz Tahmini ve Canli Gosterge) birbirinden BAGIMSIZ veri
uretiyordu ve ikisi de acik pozisyon (OI), taker alim/satim hacmi ve likidasyon
gibi kritik turev verilerini GERCEK kaynaktan degil, mumun kendi yonunden turetilen
SAHTE proxy formullerle uyduruyordu:

    df['oi'] = (close - open) / open * (volume / volume.mean()) * 10        # <- gercek OI degil
    df['taker_buy_vol'] = where(close > open, volume * 0.6, volume * 0.4)   # <- gercek taker degil

Bu yuzden iki panel ayni anda birbirine zit (biri %99 LONG, digeri SHORT) tahmin
uretebiliyordu. Bu modul, her iki panelin de TEK ve AYNI gercek veri kaynagindan
beslenmesini saglar:

  1. Coinglass (BIRINCIL) - piyasanin referans turev veri saglayicisi.
  2. Borsa toplami (YEDEK)  - Coinglass anahtari yoksa Binance/Bybit/OKX vadeli
     islem uc noktalari toplanir. Bunlar Coinglass'in de topladigi ayni ham
     kaynaklardir, dolayisiyla anahtar gelene kadar veri yine GERCEKTIR.
  3. Dolar Dominansi (USDT.D) - TradingView canli, yedegi CoinGecko.

Cikti: compute_market_bias() -> tek bir harmanlanmis yon + guven skoru.
Iki panel de bu ayni fonksiyonu cagirdigi icin artik celisemezler.
"""

import os
import time
import requests
import concurrent.futures

from dotenv import load_dotenv

try:
    import streamlit as st
    _cache = st.cache_data
except Exception:  # modulun streamlit disinda (test betikleri) da calisabilmesi icin
    st = None

    def _cache(**kwargs):
        def deco(fn):
            return fn
        return deco


COINGLASS_BASE = "https://open-api-v4.coinglass.com"

# Coinglass'in "aggregated" uc noktalarinda birlestirilmesini istedigimiz borsalar.
CG_EXCHANGES = "Binance,OKX,Bybit"

# Panel zaman dilimlerinden Coinglass/borsa interval kodlarina donusum.
INTERVAL_ALIASES = {
    "1m": "1m", "1 Dakika": "1m", "1 Dakika Scalp": "1m",
    "3m": "3m",
    "5m": "5m", "5 Dakika": "5m",
    "15m": "15m", "15 Dakika": "15m",
    "30m": "30m", "30 Dakika": "30m",
    "1h": "1h", "1 Saat": "1h",
    "4h": "4h", "4 Saat": "4h",
    "1d": "1d", "1 Gun": "1d", "1 Gün": "1d",
}

# Coinglass abonelik planlari alt interval siniri koyuyor (Hobbyist >= 4h gibi).
# Istenen interval reddedilirse sirayla bir ust basamaga cikilir.
INTERVAL_FALLBACK_CHAIN = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]


def normalize_interval(timeframe: str) -> str:
    """Panel zaman dilimi etiketini API interval koduna cevirir."""
    if not timeframe:
        return "15m"
    tf = str(timeframe).strip()
    return INTERVAL_ALIASES.get(tf, INTERVAL_ALIASES.get(tf.lower(), "15m"))


def base_asset(symbol: str) -> str:
    """'BTC/USDT:USDT' -> 'BTC' ; 'BTCUSDT' -> 'BTC'."""
    if not symbol:
        return "BTC"
    s = str(symbol).upper().split(":")[0]
    if "/" in s:
        return s.split("/")[0]
    for quote in ("USDT", "USDC", "USD"):
        if s.endswith(quote):
            return s[: -len(quote)]
    return s


def get_coinglass_key() -> str:
    """.env icindeki COINGLASS_API_KEY degerini okur. Yoksa bos string doner."""
    project_dir = os.path.dirname(os.path.abspath(__file__))
    load_dotenv(os.path.join(project_dir, ".env"))
    return (os.getenv("COINGLASS_API_KEY", "") or "").strip()


def _cg_get(path: str, params: dict, api_key: str, timeout: float = 6.0):
    """Coinglass v4 GET. Basarisizsa (anahtar yok / plan yetersiz / ag hatasi) None doner."""
    if not api_key:
        return None
    try:
        r = requests.get(
            f"{COINGLASS_BASE}{path}",
            params=params,
            headers={"CG-API-KEY": api_key, "accept": "application/json"},
            timeout=timeout,
        )
        payload = r.json()
    except Exception as e:
        print(f"[WARN] Coinglass istek hatasi ({path}): {e}")
        return None

    # Coinglass hata durumunda da HTTP 200 dondurup govdede code != '0' veriyor.
    if str(payload.get("code")) not in ("0", "200"):
        print(f"[WARN] Coinglass reddetti ({path}): {payload.get('msg')}")
        return None
    return payload.get("data")


def _cg_get_with_interval_fallback(path: str, params: dict, api_key: str, interval: str):
    """Abonelik plani istenen intervali reddederse bir ust zaman dilimine cikarak dener."""
    try:
        start = INTERVAL_FALLBACK_CHAIN.index(interval)
    except ValueError:
        start = 0
    for candidate in INTERVAL_FALLBACK_CHAIN[start:]:
        data = _cg_get(path, {**params, "interval": candidate}, api_key)
        if data:
            return data, candidate
    return None, interval


# ─────────────────────────────────────────────────────────────────
# COINGLASS - BIRINCIL KAYNAK
# ─────────────────────────────────────────────────────────────────
@_cache(ttl=30, show_spinner=False)
def fetch_coinglass_derivatives(symbol: str, timeframe: str = "15m") -> dict:
    """Coinglass v4'ten acik pozisyon, long/short oran, taker hacim ve likidasyon verisi."""
    api_key = get_coinglass_key()
    if not api_key:
        return {}

    coin = base_asset(symbol)
    pair = f"{coin}USDT"
    interval = normalize_interval(timeframe)
    out = {"source": "COINGLASS", "coin": coin, "interval_used": interval}

    def _oi():
        data, used = _cg_get_with_interval_fallback(
            "/api/futures/open-interest/aggregated-history",
            {"symbol": coin, "exchange_list": CG_EXCHANGES, "limit": 60},
            api_key, interval)
        if not data:
            return {}
        closes = [float(d.get("close", 0) or 0) for d in data if d.get("close") is not None]
        if len(closes) < 2:
            return {}
        ref = closes[-6] if len(closes) >= 6 else closes[0]
        return {
            "oi_now": closes[-1],
            "oi_change_pct": ((closes[-1] - ref) / ref * 100) if ref else 0.0,
            "oi_interval": used,
        }

    def _ls():
        data, _ = _cg_get_with_interval_fallback(
            "/api/futures/global-long-short-account-ratio/history",
            {"symbol": pair, "exchange": "Binance", "limit": 30},
            api_key, interval)
        if not data:
            return {}
        last = data[-1]
        long_pct = float(last.get("global_account_long_percent", 0) or 0)
        short_pct = float(last.get("global_account_short_percent", 0) or 0)
        return {
            "long_pct": long_pct,
            "short_pct": short_pct,
            "ls_ratio": float(last.get("global_account_long_short_ratio", 0) or 0),
        }

    def _taker():
        data, _ = _cg_get_with_interval_fallback(
            "/api/futures/aggregated-taker-buy-sell-volume/history",
            {"symbol": coin, "exchange_list": CG_EXCHANGES, "limit": 30, "unit": "usd"},
            api_key, interval)
        if not data:
            return {}
        recent = data[-5:] if len(data) >= 5 else data
        buy = sum(float(d.get("aggregated_buy_volume_usd", 0) or 0) for d in recent)
        sell = sum(float(d.get("aggregated_sell_volume_usd", 0) or 0) for d in recent)
        total = buy + sell
        return {
            "taker_buy_usd": buy,
            "taker_sell_usd": sell,
            "taker_delta_pct": ((buy - sell) / total * 100) if total else 0.0,
        }

    def _liq():
        data, _ = _cg_get_with_interval_fallback(
            "/api/futures/liquidation/history",
            {"symbol": pair, "exchange": "Binance", "limit": 30},
            api_key, interval)
        if not data:
            return {}
        recent = data[-8:] if len(data) >= 8 else data
        long_liq = sum(float(d.get("long_liquidation_usd", 0) or 0) for d in recent)
        short_liq = sum(float(d.get("short_liquidation_usd", 0) or 0) for d in recent)
        total = long_liq + short_liq
        return {
            "liq_long_usd": long_liq,
            "liq_short_usd": short_liq,
            # Pozitif: agirlikli SHORT'lar patladi (yukari baski). Negatif: LONG'lar patladi.
            "liq_skew_pct": ((short_liq - long_liq) / total * 100) if total else 0.0,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(f) for f in (_oi, _ls, _taker, _liq)]
        for fut in futures:
            try:
                out.update(fut.result(timeout=12) or {})
            except Exception:
                pass

    # Hicbir alt uc nokta veri dondurmediyse Coinglass'i kullanilamaz say.
    if not any(k in out for k in ("oi_now", "ls_ratio", "taker_delta_pct", "liq_skew_pct")):
        return {}
    return out


# ─────────────────────────────────────────────────────────────────
# YEDEK - COINGLASS ANAHTARI YOKKEN BORSA TOPLAMI
# Coinglass'in kendi topladigi ayni ham vadeli-islem uc noktalari.
# ─────────────────────────────────────────────────────────────────
@_cache(ttl=30, show_spinner=False)
def fetch_exchange_derivatives(symbol: str, timeframe: str = "15m") -> dict:
    """Binance/Bybit/OKX vadeli veri uc noktalarindan gercek turev verisi toplar."""
    coin = base_asset(symbol)
    pair = f"{coin}USDT"
    interval = normalize_interval(timeframe)
    # Binance "futures/data" uc noktalari sadece belirli periyotlari kabul ediyor.
    period = interval if interval in ("5m", "15m", "30m", "1h", "4h", "1d") else "5m"
    out = {"source": "BORSA_TOPLAMI", "coin": coin, "interval_used": period}

    def _binance(path, params):
        try:
            r = requests.get(f"https://fapi.binance.com/futures/data/{path}",
                             params=params, timeout=6)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    def _oi():
        data = _binance("openInterestHist", {"symbol": pair, "period": period, "limit": 30})
        if not isinstance(data, list) or len(data) < 2:
            return {}
        vals = [float(d["sumOpenInterestValue"]) for d in data]
        ref = vals[-6] if len(vals) >= 6 else vals[0]
        return {"oi_now": vals[-1],
                "oi_change_pct": ((vals[-1] - ref) / ref * 100) if ref else 0.0}

    def _ls():
        data = _binance("globalLongShortAccountRatio", {"symbol": pair, "period": period, "limit": 10})
        if not isinstance(data, list) or not data:
            return {}
        last = data[-1]
        return {"long_pct": float(last["longAccount"]) * 100,
                "short_pct": float(last["shortAccount"]) * 100,
                "ls_ratio": float(last["longShortRatio"])}

    def _taker():
        data = _binance("takerlongshortRatio", {"symbol": pair, "period": period, "limit": 10})
        if not isinstance(data, list) or not data:
            return {}
        recent = data[-5:]
        buy = sum(float(d["buyVol"]) for d in recent)
        sell = sum(float(d["sellVol"]) for d in recent)
        total = buy + sell
        return {"taker_buy_usd": buy, "taker_sell_usd": sell,
                "taker_delta_pct": ((buy - sell) / total * 100) if total else 0.0}

    def _funding():
        try:
            r = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                             params={"symbol": pair}, timeout=6)
            if r.status_code != 200:
                return {}
            return {"funding_rate_pct": float(r.json().get("lastFundingRate", 0)) * 100}
        except Exception:
            return {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        futures = [ex.submit(f) for f in (_oi, _ls, _taker, _funding)]
        for fut in futures:
            try:
                out.update(fut.result(timeout=10) or {})
            except Exception:
                pass

    if not any(k in out for k in ("oi_now", "ls_ratio", "taker_delta_pct")):
        return {}
    return out


def fetch_derivatives_matrix(symbol: str, timeframe: str = "15m") -> dict:
    """Once Coinglass denenir; anahtar yoksa/plan yetmiyorsa borsa toplamina duser."""
    data = fetch_coinglass_derivatives(symbol, timeframe)
    if data:
        return data
    data = fetch_exchange_derivatives(symbol, timeframe)
    if data:
        return data
    return {"source": "VERI_YOK", "coin": base_asset(symbol)}


# ─────────────────────────────────────────────────────────────────
# DOLAR DOMINANSI (USDT.D)
# ─────────────────────────────────────────────────────────────────
@_cache(ttl=30, show_spinner=False)
def fetch_dominance_matrix() -> dict:
    """TradingView canli CRYPTOCAP:USDT.D / BTC.D / ETH.D; yedegi CoinGecko global."""
    try:
        r = requests.post(
            "https://scanner.tradingview.com/global/scan",
            json={"symbols": {"tickers": ["CRYPTOCAP:USDT.D", "CRYPTOCAP:BTC.D", "CRYPTOCAP:ETH.D"]},
                  "columns": ["close", "change"]},
            headers={"User-Agent": "Mozilla/5.0", "Content-Type": "application/json",
                     "Origin": "https://www.tradingview.com",
                     "Referer": "https://www.tradingview.com/"},
            timeout=4,
        )
        if r.status_code == 200:
            tv = {row.get("s"): row.get("d", []) for row in r.json().get("data", [])}
            if "CRYPTOCAP:USDT.D" in tv and len(tv["CRYPTOCAP:USDT.D"]) >= 2:
                usdt_d = float(tv["CRYPTOCAP:USDT.D"][0])
                usdt_chg = float(tv["CRYPTOCAP:USDT.D"][1])
                return {
                    "usdt_d": usdt_d, "usdt_d_change": usdt_chg,
                    "btc_d": float(tv.get("CRYPTOCAP:BTC.D", [59.5])[0]),
                    "eth_d": float(tv.get("CRYPTOCAP:ETH.D", [11.2])[0]),
                    "trend": "DÜŞÜŞTE (Kripto Boğa / LONG)" if usdt_chg < 0 else "YÜKSELİŞTE (Kripto Ayı / SHORT)",
                    "bias": "BULLISH_CRYPTO" if usdt_chg < 0 else "BEARISH_CRYPTO",
                    "source": "TRADINGVIEW_LIVE",
                }
    except Exception as e:
        print(f"[WARN] TradingView dominans hatasi: {e}")

    try:
        r = requests.get("https://api.coingecko.com/api/v3/global",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=4)
        if r.status_code == 200:
            d = r.json().get("data", {})
            mcap = d.get("market_cap_percentage", {})
            usdt_chg = -float(d.get("market_cap_change_percentage_24h_usd", 0.0)) * 0.3
            return {
                "usdt_d": float(mcap.get("usdt", 7.03)), "usdt_d_change": usdt_chg,
                "btc_d": float(mcap.get("btc", 59.6)), "eth_d": float(mcap.get("eth", 11.3)),
                "trend": "DÜŞÜŞTE (Kripto Boğa / LONG)" if usdt_chg < 0 else "YÜKSELİŞTE (Kripto Ayı / SHORT)",
                "bias": "BULLISH_CRYPTO" if usdt_chg < 0 else "BEARISH_CRYPTO",
                "source": "COINGECKO_FALLBACK",
            }
    except Exception:
        pass

    return {"usdt_d": 0.0, "usdt_d_change": 0.0, "btc_d": 0.0, "eth_d": 0.0,
            "trend": "VERİ ÇEKİLEMEDİ", "bias": "NEUTRAL", "source": "UNAVAILABLE"}


# ─────────────────────────────────────────────────────────────────
# HARMANLANMIS PIYASA YONU - IKI PANELIN DE ORTAK KARAR KAYNAGI
# ─────────────────────────────────────────────────────────────────
# Agirliklar toplami 100. Pozitif puan = LONG lehine, negatif = SHORT lehine.
W_TAKER = 28      # Agresif alici/satici akisi - en dogrudan yon sinyali
W_OI = 18         # Acik pozisyon degisimi - trendin arkasindaki para
W_LIQ = 18        # Likidasyon dengesizligi - hangi taraf temizlendi
W_LS = 12         # Kalabalik hesap orani - TERSINE (contrarian) okunur
W_FUNDING = 12    # Fonlama orani - asiri kaldirac TERSINE okunur
W_DOM = 12        # Dolar dominansi - nakite kacis mi, kriptoya akis mi


def compute_market_bias(symbol: str, timeframe: str = "15m") -> dict:
    """Coinglass turev verisi + Dolar Dominansi'ni harmanlayip TEK bir yon uretir.

    Iki panel de bu fonksiyonu cagirdigi icin ayni sembol/zaman diliminde
    KESINLIKLE ayni yonu gorurler.

    Donen 'score' -100 (guclu SHORT) ile +100 (guclu LONG) arasindadir.
    """
    deriv = fetch_derivatives_matrix(symbol, timeframe)
    dom = fetch_dominance_matrix()

    score = 0.0
    used_weight = 0.0
    checks = []

    # 1) Taker alim/satim akisi (momentum, dogrudan)
    if "taker_delta_pct" in deriv:
        d = deriv["taker_delta_pct"]
        contrib = max(-1.0, min(1.0, d / 15.0)) * W_TAKER
        score += contrib
        used_weight += W_TAKER
        checks.append(("Taker Alım/Satım Akışı",
                       f"%{d:+.2f} ({'Alıcı baskın' if d > 0 else 'Satıcı baskın'})",
                       "pass" if abs(d) > 3 else "warn", contrib))

    # 2) Acik pozisyon degisimi - taker yonuyle ayni isaretteyse trendi guclendirir
    if "oi_change_pct" in deriv:
        oi = deriv["oi_change_pct"]
        flow_sign = 1.0 if deriv.get("taker_delta_pct", 0) >= 0 else -1.0
        contrib = max(-1.0, min(1.0, abs(oi) / 2.0)) * flow_sign * W_OI
        score += contrib
        used_weight += W_OI
        checks.append(("Açık Pozisyon (OI) Değişimi",
                       f"%{oi:+.2f} ({'Yeni para giriyor' if oi > 0 else 'Pozisyon kapanıyor'})",
                       "pass" if abs(oi) > 0.5 else "warn", contrib))

    # 3) Likidasyon dengesizligi
    if "liq_skew_pct" in deriv:
        lq = deriv["liq_skew_pct"]
        contrib = max(-1.0, min(1.0, lq / 50.0)) * W_LIQ
        score += contrib
        used_weight += W_LIQ
        checks.append(("Likidasyon Dengesizliği",
                       f"%{lq:+.1f} ({'Shortlar patlıyor' if lq > 0 else 'Longlar patlıyor'})",
                       "pass" if abs(lq) > 15 else "warn", contrib))

    # 4) Kalabalik long/short orani - TERSINE okunur (kalabalik genelde yanilir)
    if "ls_ratio" in deriv and deriv["ls_ratio"]:
        ls = deriv["ls_ratio"]
        crowd = ls - 1.0
        contrib = -max(-1.0, min(1.0, crowd / 0.6)) * W_LS
        score += contrib
        used_weight += W_LS
        checks.append(("Kalabalık Long/Short Oranı (ters sinyal)",
                       f"{ls:.2f} ({'Kalabalık LONG, risk yukarıda' if crowd > 0 else 'Kalabalık SHORT, risk aşağıda'})",
                       "warn" if abs(crowd) > 0.25 else "pass", contrib))

    # 5) Fonlama orani - asiri pozitif = kalabalik long, TERSINE okunur
    if "funding_rate_pct" in deriv:
        fr = deriv["funding_rate_pct"]
        contrib = -max(-1.0, min(1.0, fr / 0.05)) * W_FUNDING
        score += contrib
        used_weight += W_FUNDING
        checks.append(("Fonlama Oranı (ters sinyal)",
                       f"%{fr:+.4f} ({'Longlar ödüyor' if fr > 0 else 'Shortlar ödüyor'})",
                       "warn" if abs(fr) > 0.03 else "pass", contrib))

    # 6) Dolar dominansi - yukseliyorsa nakite kacis (SHORT), dusuyorsa kriptoya akis (LONG)
    if dom.get("source") != "UNAVAILABLE":
        chg = dom.get("usdt_d_change", 0.0)
        contrib = -max(-1.0, min(1.0, chg / 1.5)) * W_DOM
        score += contrib
        used_weight += W_DOM
        checks.append(("Dolar Dominansı (USDT.D)",
                       f"%{dom.get('usdt_d', 0):.2f} (24s %{chg:+.2f})",
                       "pass" if abs(chg) > 0.3 else "warn", contrib))

    # Eksik veri varsa skoru mevcut agirliga gore olcekle - yoksa yapay olarak zayif gorunur.
    if used_weight > 0:
        score = score * (100.0 / used_weight)
    score = max(-100.0, min(100.0, score))

    if score >= 20:
        direction = "LONG"
    elif score <= -20:
        direction = "SHORT"
    else:
        direction = "NÖTR"

    strength = abs(score)
    if strength >= 55:
        label = "GÜÇLÜ"
    elif strength >= 20:
        label = "ZAYIF"
    else:
        label = "KARARSIZ"

    return {
        "direction": direction,
        "score": score,
        "strength": strength,
        "label": label,
        "checks": checks,
        "derivatives": deriv,
        "dominance": dom,
        "source": deriv.get("source", "VERI_YOK"),
        "coverage_pct": (used_weight / 100.0) * 100.0,
        "coinglass_active": deriv.get("source") == "COINGLASS",
    }


# ─────────────────────────────────────────────────────────────────
# ML ICIN GERCEK TUREV ZAMAN SERISI
# ─────────────────────────────────────────────────────────────────
@_cache(ttl=60, show_spinner=False)
def fetch_derivatives_series(symbol: str, timeframe: str = "15m", limit: int = 500):
    """Mum bazinda GERCEK acik pozisyon / taker akisi / long-short orani serisi doner.

    ONEMLI: Bu, panellerdeki eski SAHTE proxy'lerin (mumun kendi yonunden turetilen
    'oi' ve 'taker_buy_vol') yerini alir. Index UTC zaman damgasidir; cagiran taraf
    kendi mum index'ine reindex eder.
    """
    import pandas as pd

    coin = base_asset(symbol)
    pair = f"{coin}USDT"
    interval = normalize_interval(timeframe)
    period = interval if interval in ("5m", "15m", "30m", "1h", "4h", "1d") else "5m"
    capped = max(30, min(int(limit), 500))

    def _binance(path, params):
        try:
            r = requests.get(f"https://fapi.binance.com/futures/data/{path}",
                             params=params, timeout=8)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    frames = {}

    def _oi():
        data = _binance("openInterestHist", {"symbol": pair, "period": period, "limit": capped})
        if not isinstance(data, list) or not data:
            return
        df = pd.DataFrame(data)
        df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["oi_value"] = df["sumOpenInterestValue"].astype(float)
        frames["oi"] = df.set_index("ts")[["oi_value"]]

    def _taker():
        data = _binance("takerlongshortRatio", {"symbol": pair, "period": period, "limit": capped})
        if not isinstance(data, list) or not data:
            return
        df = pd.DataFrame(data)
        df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        buy = df["buyVol"].astype(float)
        sell = df["sellVol"].astype(float)
        df["taker_delta_pct"] = (buy - sell) / (buy + sell).replace(0, 1e-9) * 100
        frames["taker"] = df.set_index("ts")[["taker_delta_pct"]]

    def _ls():
        data = _binance("globalLongShortAccountRatio", {"symbol": pair, "period": period, "limit": capped})
        if not isinstance(data, list) or not data:
            return
        df = pd.DataFrame(data)
        df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["ls_ratio"] = df["longShortRatio"].astype(float)
        frames["ls"] = df.set_index("ts")[["ls_ratio"]]

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        for fut in [ex.submit(f) for f in (_oi, _taker, _ls)]:
            try:
                fut.result(timeout=12)
            except Exception:
                pass

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames.values(), axis=1).sort_index()
    if "oi_value" in out.columns:
        out["oi_change_pct"] = (out["oi_value"].pct_change(periods=5)
                                .replace([float("inf"), float("-inf")], 0.0).fillna(0.0) * 100)
    return out


def attach_derivatives_to_df(df, symbol: str, timeframe: str):
    """Mum DataFrame'ine gercek turev sutunlarini ekler (oi_value/oi_change_pct/
    taker_delta_pct/ls_ratio). Veri alinamazsa sutunlari notr (0) birakir; boylece
    cagiran kod her kosulda ayni sutun setiyle calisir."""
    import pandas as pd

    cols = ["oi_value", "oi_change_pct", "taker_delta_pct", "ls_ratio"]
    series = fetch_derivatives_series(symbol, timeframe, limit=len(df))

    if series is None or getattr(series, "empty", True):
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0
        df["deriv_source"] = "YOK"
        return df

    # Turev uc noktalari mum kapanislariyla birebir ayni damgada olmayabiliyor;
    # her mum icin GECMISTEKI en yakin gozlem aliniyor (ileriye donuk sizinti olmasin).
    aligned = series.reindex(series.index.union(df.index)).sort_index().ffill().reindex(df.index)
    for c in cols:
        df[c] = aligned[c].astype(float) if c in aligned.columns else 0.0
        df[c] = df[c].ffill().fillna(0.0)
    df["deriv_source"] = "GERCEK"
    return df


# ─────────────────────────────────────────────────────────────────
# ORTAK KARAR UZLASTIRICI - PANELLER ARASI CELISKIYI BITIREN KATMAN
# ─────────────────────────────────────────────────────────────────
def reconcile_with_bias(ai_direction: str, ai_confidence: float, bias: dict) -> dict:
    """Bir panelin kendi ML tahminini, ORTAK piyasa yonuyle uzlastirir.

    NEDEN: Iki panel ayni anda birbirine zit yon (biri %99 LONG, digeri SHORT)
    gosterebiliyordu. Artik her iki panel de kendi ML ciktisini ayni ortak
    'bias' suzgecinden geciriyor:

      - ML yonu ortak yonle AYNI  -> "ONAYLI", guven artar.
      - Ortak yon NOTR            -> yon korunur ama guven kirpilir ("ZAYIF ONAY").
      - ML yonu ortak yona ZIT    -> yon iptal edilir, "BEKLE" olur.

    Boylece iki panel ayni sembol/zaman diliminde asla zit yon gosteremez.
    """
    direction = (ai_direction or "NÖTR").upper()
    bias_dir = bias.get("direction", "NÖTR")
    strength = bias.get("strength", 0.0)
    conf = float(ai_confidence or 0.0)

    if direction not in ("LONG", "SHORT"):
        return {"direction": "BEKLE", "confidence": 0.0, "status": "AI_YOK",
                "note": "Yapay zeka net bir yön üretmedi."}

    if bias_dir == "NÖTR":
        return {"direction": direction, "confidence": min(conf, 55.0), "status": "ZAYIF_ONAY",
                "note": f"Coinglass/türev verisi kararsız (skor {bias.get('score', 0):+.0f}). "
                        f"Yön korundu ama güven sınırlandı."}

    if bias_dir == direction:
        boosted = min(99.0, conf + min(20.0, strength * 0.25))
        return {"direction": direction, "confidence": boosted, "status": "ONAYLI",
                "note": f"Türev verisi + Dolar Dominansı aynı yönü doğruluyor "
                        f"(skor {bias.get('score', 0):+.0f})."}

    # Zit yon: guclu bias varsa islemi tamamen iptal et, zayifsa sadece guveni dusur.
    if strength >= 35:
        return {"direction": "BEKLE", "confidence": 0.0, "status": "IPTAL",
                "note": f"Yapay zeka {direction} dedi ama türev verisi + Dolar Dominansı "
                        f"{bias_dir} gösteriyor (skor {bias.get('score', 0):+.0f}). "
                        f"Çelişki nedeniyle işlem açılmaz."}

    return {"direction": direction, "confidence": max(0.0, conf * 0.5), "status": "ZAYIF_CELISKI",
            "note": f"Türev verisi hafifçe {bias_dir} eğilimli; güven yarıya düşürüldü."}


# ─────────────────────────────────────────────────────────────────
# ORTAK GORSEL KART - IKI PANELDE DE AYNI GORUNUR
# ─────────────────────────────────────────────────────────────────
SOURCE_LABELS = {
    "COINGLASS": ("Coinglass (canlı)", "#0e7490"),
    "BORSA_TOPLAMI": ("Borsa Türev Toplamı (Binance/Bybit/OKX)", "#14b8a6"),
    "VERI_YOK": ("Veri alınamadı", "#dc2626"),
}


def render_market_intel_card(bias: dict, reconciled: dict = None) -> None:
    """Harmanlanmis piyasa yonunu ve alt kirilimlarini gosteren ortak kart."""
    if st is None:
        return

    src_label, src_color = SOURCE_LABELS.get(bias.get("source"), ("Bilinmiyor", "#94a3b8"))
    score = bias.get("score", 0.0)
    direction = bias.get("direction", "NÖTR")
    dir_color = "#16a34a" if direction == "LONG" else ("#dc2626" if direction == "SHORT" else "#94a3b8")

    # Skor cubugu: -100 (sol/SHORT) .. +100 (sag/LONG). Ortasi notr.
    fill_pct = (score + 100.0) / 2.0

    rows = []
    for label, value, state, points in bias.get("checks", []):
        dot = {"pass": "#16a34a", "warn": "#f59e0b", "fail": "#dc2626"}.get(state, "#94a3b8")
        pt_color = "#16a34a" if points > 0 else ("#dc2626" if points < 0 else "#94a3b8")
        rows.append(
            f'<div style="display:flex;align-items:center;gap:10px;padding:9px 0;'
            f'border-bottom:1px solid rgba(15,43,46,0.07);">'
            f'<span style="width:9px;height:9px;border-radius:50%;background:{dot};flex-shrink:0;"></span>'
            f'<span style="flex:1;font-weight:800;color:#0f2b2e;font-size:13.5px;">{label}</span>'
            f'<span style="color:#5f7d7a;font-size:13px;font-weight:600;">{value}</span>'
            f'<span style="color:{pt_color};font-weight:900;font-size:13px;width:56px;text-align:right;">'
            f'{points:+.1f}</span></div>'
        )

    note_html = ""
    if reconciled:
        st_color = {"ONAYLI": "#16a34a", "IPTAL": "#dc2626",
                    "ZAYIF_ONAY": "#f59e0b", "ZAYIF_CELISKI": "#f59e0b"}.get(
                        reconciled.get("status"), "#5f7d7a")
        note_html = (
            f'<div style="margin-top:14px;padding:12px 14px;border-radius:12px;'
            f'background:rgba(20,184,166,0.07);border-left:5px solid {st_color};">'
            f'<span style="font-weight:900;color:{st_color};font-size:13px;">'
            f'{reconciled.get("status", "")}</span>'
            f'<span style="color:#0f2b2e;font-size:13px;font-weight:600;"> — '
            f'{reconciled.get("note", "")}</span></div>'
        )

    st.markdown(
        f'''<div style="background:#ffffff;border:3px solid #14b8a6;border-radius:18px;
                        padding:20px 24px;margin:10px 0;">
          <div style="display:flex;align-items:center;justify-content:space-between;
                      flex-wrap:wrap;gap:10px;margin-bottom:14px;">
            <div style="font-size:19px;font-weight:900;color:#14b8a6;">
              🛰️ PİYASA İSTİHBARATI (Türev Verisi + Dolar Dominansı)</div>
            <div style="background:{src_color};color:#ffffff;padding:5px 12px;border-radius:999px;
                        font-size:11.5px;font-weight:900;">{src_label}</div>
          </div>
          <div style="display:flex;align-items:center;gap:14px;margin-bottom:6px;">
            <span style="font-size:26px;font-weight:900;color:{dir_color};">{direction}</span>
            <span style="font-size:14px;font-weight:800;color:#5f7d7a;">
              {bias.get("label", "")} &nbsp;|&nbsp; Skor {score:+.0f} / 100
              &nbsp;|&nbsp; Veri kapsamı %{bias.get("coverage_pct", 0):.0f}</span>
          </div>
          <div style="position:relative;height:12px;border-radius:999px;margin:10px 0 16px 0;
                      background:linear-gradient(90deg,#fecaca 0%,#e2e8f0 50%,#bbf7d0 100%);">
            <div style="position:absolute;left:calc({fill_pct:.1f}% - 7px);top:-3px;width:14px;
                        height:18px;border-radius:5px;background:{dir_color};
                        border:2px solid #ffffff;box-shadow:0 1px 4px rgba(0,0,0,0.25);"></div>
          </div>
          {"".join(rows)}
          {note_html}
        </div>''',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    import json
    res = compute_market_bias("BTC/USDT:USDT", "15m")
    print(json.dumps({k: v for k, v in res.items() if k != "checks"},
                     indent=2, ensure_ascii=False, default=str))
    for c in res["checks"]:
        print(f"  - {c[0]}: {c[1]}  -> {c[3]:+.1f} puan")
