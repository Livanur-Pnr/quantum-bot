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

    # Coinglass'e ozel olmayan, her zaman ucretsiz calisan ek veri setleri
    # (basis, spot/vadeli hacim orani, Coinbase primi, buyuk oyuncu orani vb.)
    # Coinglass sonucunun UZERINE yazilir (Coinglass zaten OI/LS/taker/liq'i verdi).
    _period = interval if interval in ("5m", "15m", "30m", "1h", "4h", "1d") else "5m"
    extras = _fetch_free_derivative_extras(coin, _period)
    for k, v in extras.items():
        out.setdefault(k, v)
    return out


# ─────────────────────────────────────────────────────────────────
# COINGLASS VERI SETLERININ UCRETSIZ KARSILIKLARI
# Coinglass verinin kendisini uretmez; borsalarin genel (public) vadeli
# uc noktalarindan toplar. Asagidaki fonksiyonlar ayni ham kaynaklara
# dogrudan baglanir - dolayisiyla veri Coinglass'inkiyle ayni koktendir.
# ─────────────────────────────────────────────────────────────────
@_cache(ttl=3600, show_spinner=False)
def _okx_contract_size(inst_id: str) -> float:
    """OKX vadeli sozlesmelerinde 'sz' alani KONTRAT adedidir, coin miktari degil.
    USD degeri hesaplanirken kontrat carpani (ctVal) sarttir; coin basina degisir
    (BTC 0.01, ETH 0.1, SOL 1, DOGE 1000...). Bu atlanirsa likidasyon tutarlari
    yuzlerce kat sisik cikar."""
    try:
        r = requests.get("https://www.okx.com/api/v5/public/instruments",
                         params={"instType": "SWAP", "instId": inst_id}, timeout=6)
        data = r.json().get("data", [])
        if data:
            return float(data[0].get("ctVal", 1) or 1)
    except Exception:
        pass
    return 1.0


@_cache(ttl=30, show_spinner=False)
def fetch_liquidations(coin: str, window_minutes: int = 60) -> dict:
    """GERCEK likidasyon verisi (OKX genel uc noktasi).

    Coinglass'in 'Liquidation' panelinin ucretsiz karsiligi. Hangi tarafin
    temizlendigini gosterir: long'lar patliyorsa asagi baski, short'lar
    patliyorsa yukari baski (short squeeze) vardir.
    """
    inst_id = f"{coin}-USDT-SWAP"
    try:
        r = requests.get("https://www.okx.com/api/v5/public/liquidation-orders",
                         params={"instType": "SWAP", "uly": f"{coin}-USDT",
                                 "state": "filled", "limit": "100"}, timeout=8)
        payload = r.json()
        if str(payload.get("code")) != "0":
            return {}
        rows = payload.get("data", [])
        if not rows:
            return {}
    except Exception as e:
        print(f"[WARN] OKX likidasyon hatasi: {e}")
        return {}

    ct_val = _okx_contract_size(inst_id)
    cutoff_ms = (time.time() - window_minutes * 60) * 1000

    long_liq = 0.0
    short_liq = 0.0
    count = 0
    for row in rows:
        for d in row.get("details", []):
            try:
                ts = float(d.get("ts", 0) or 0)
                if ts < cutoff_ms:
                    continue
                usd = float(d.get("sz", 0) or 0) * ct_val * float(d.get("bkPx", 0) or 0)
                if d.get("posSide") == "long":
                    long_liq += usd
                elif d.get("posSide") == "short":
                    short_liq += usd
                count += 1
            except Exception:
                continue

    total = long_liq + short_liq
    if total <= 0:
        return {"liq_long_usd": 0.0, "liq_short_usd": 0.0,
                "liq_skew_pct": 0.0, "liq_count": 0, "liq_window_min": window_minutes}
    return {
        "liq_long_usd": long_liq,
        "liq_short_usd": short_liq,
        # Pozitif: agirlikli SHORT'lar patladi (yukari baski / short squeeze).
        "liq_skew_pct": (short_liq - long_liq) / total * 100,
        "liq_count": count,
        "liq_window_min": window_minutes,
    }


@_cache(ttl=30, show_spinner=False)
def fetch_aggregated_open_interest(coin: str) -> dict:
    """Coinglass'in 'Open Interest - Exchange List' karsiligi: acik pozisyonun
    USD cinsinden borsa borsa dagilimi ve toplami.

    Coinglass, OI/fonlama gibi metrikleri TEK bir borsadan degil, piyasadaki
    COGUNLUGU olusturan borsalarin agirlikli ortalamasindan/toplamindan uretir.
    Bu yuzden burada da ayni mantikla MUMKUN OLDUGUNCA COK borsa toplanir:
    Binance, Bybit, OKX, Bitget, Gate.io, HTX (Huobi), KuCoin, Hyperliquid.
    Herhangi biri erisilemezse (sembol yok, ag hatasi) sessizce atlanir - kalan
    borsalarla toplam hesaplanmaya devam eder."""
    pair = f"{coin}USDT"
    per_exchange = {}

    def _binance():
        try:
            oi_coin = float(requests.get("https://fapi.binance.com/fapi/v1/openInterest",
                                         params={"symbol": pair}, timeout=6).json()["openInterest"])
            mark = float(requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                                      params={"symbol": pair}, timeout=6).json()["markPrice"])
            per_exchange["Binance"] = oi_coin * mark
        except Exception:
            pass

    def _bybit():
        try:
            row = requests.get("https://api.bybit.com/v5/market/tickers",
                               params={"category": "linear", "symbol": pair},
                               timeout=6).json()["result"]["list"][0]
            per_exchange["Bybit"] = float(row["openInterest"]) * float(row["markPrice"])
        except Exception:
            pass

    def _okx():
        try:
            d = requests.get("https://www.okx.com/api/v5/public/open-interest",
                             params={"instType": "SWAP", "instId": f"{coin}-USDT-SWAP"},
                             timeout=6).json()["data"][0]
            per_exchange["OKX"] = float(d["oiUsd"])
        except Exception:
            pass

    def _bitget():
        try:
            oi_coin = float(requests.get(
                "https://api.bitget.com/api/v2/mix/market/open-interest",
                params={"symbol": pair, "productType": "usdt-futures"},
                timeout=6).json()["data"]["openInterestList"][0]["size"])
            mark = float(requests.get(
                "https://api.bitget.com/api/v2/mix/market/ticker",
                params={"symbol": pair, "productType": "usdt-futures"},
                timeout=6).json()["data"][0]["lastPr"])
            per_exchange["Bitget"] = oi_coin * mark
        except Exception:
            pass

    def _gate():
        try:
            d = requests.get(f"https://api.gateio.ws/api/v4/futures/usdt/contracts/{coin}_USDT",
                             timeout=6).json()
            # ONEMLI: 'position_size' KONTRAT adedidir, coin miktari degil - 'quanto_multiplier'
            # ile carpilmadan kullanilirsa OI onbinlerce kat sisik cikar (OKX'teki ctVal ile
            # ayni tuzak).
            per_exchange["Gate"] = (float(d["position_size"]) * float(d["quanto_multiplier"])
                                    * float(d["mark_price"]))
        except Exception:
            pass

    def _htx():
        try:
            d = requests.get("https://api.hbdm.com/linear-swap-api/v1/swap_open_interest",
                             params={"contract_code": f"{coin}-USDT"}, timeout=6).json()["data"][0]
            per_exchange["HTX"] = float(d["value"])
        except Exception:
            pass

    def _kucoin():
        try:
            # KuCoin BTC icin "XBT" ISO-4217-stili kod kullanir, "BTC" degil.
            symbol = f"{'XBT' if coin == 'BTC' else coin}USDTM"
            d = requests.get(f"https://api-futures.kucoin.com/api/v1/contracts/{symbol}",
                             timeout=6).json()["data"]
            # ONEMLI: 'openInterest' KONTRAT (lot) adedidir; coin karsiligi icin 'multiplier'
            # ile carpilmasi sart (OKX ctVal / Gate quanto_multiplier ile ayni tuzak turu).
            oi_coin = float(d["openInterest"]) * float(d["multiplier"])
            per_exchange["KuCoin"] = oi_coin * float(d["markPrice"])
        except Exception:
            pass

    def _hyperliquid():
        try:
            r = requests.post("https://api.hyperliquid.xyz/info",
                              json={"type": "metaAndAssetCtxs"}, timeout=8).json()
            universe, ctxs = r[0]["universe"], r[1]
            idx = next((i for i, u in enumerate(universe) if u["name"] == coin), None)
            if idx is not None:
                ctx = ctxs[idx]
                per_exchange["Hyperliquid"] = float(ctx["openInterest"]) * float(ctx["markPx"])
        except Exception:
            pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as ex:
        for fut in [ex.submit(f) for f in
                    (_binance, _bybit, _okx, _bitget, _gate, _htx, _kucoin, _hyperliquid)]:
            try:
                fut.result(timeout=10)
            except Exception:
                pass

    if not per_exchange:
        return {}
    return {"oi_total_usd": sum(per_exchange.values()), "oi_by_exchange": per_exchange}


@_cache(ttl=30, show_spinner=False)
def fetch_aggregated_funding(coin: str) -> dict:
    """Coinglass'in 'Funding Rate' tablosunun karsiligi: COK borsanin fonlama
    oranlari ve ortalamasi (Binance, Bybit, OKX, Bitget, Gate, HTX, KuCoin,
    Hyperliquid). Tek borsaya bakmak yaniltici olabiliyor - bazi borsalar
    kisa sureli asiri fonlamayla kalabaligi yaniltabilir, genis borsa
    ortalamasi bunu yumusatir."""
    pair = f"{coin}USDT"
    rates = {}

    def _binance():
        try:
            r = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                             params={"symbol": pair}, timeout=6).json()
            rates["Binance"] = float(r["lastFundingRate"]) * 100
        except Exception:
            pass

    def _bybit():
        try:
            r = requests.get("https://api.bybit.com/v5/market/tickers",
                             params={"category": "linear", "symbol": pair}, timeout=6).json()
            rates["Bybit"] = float(r["result"]["list"][0]["fundingRate"]) * 100
        except Exception:
            pass

    def _okx():
        try:
            r = requests.get("https://www.okx.com/api/v5/public/funding-rate",
                             params={"instId": f"{coin}-USDT-SWAP"}, timeout=6).json()
            rates["OKX"] = float(r["data"][0]["fundingRate"]) * 100
        except Exception:
            pass

    def _bitget():
        try:
            r = requests.get("https://api.bitget.com/api/v2/mix/market/current-fund-rate",
                             params={"symbol": pair, "productType": "usdt-futures"}, timeout=6).json()
            rates["Bitget"] = float(r["data"][0]["fundingRate"]) * 100
        except Exception:
            pass

    def _gate():
        try:
            d = requests.get(f"https://api.gateio.ws/api/v4/futures/usdt/contracts/{coin}_USDT",
                             timeout=6).json()
            rates["Gate"] = float(d["funding_rate"]) * 100
        except Exception:
            pass

    def _htx():
        try:
            r = requests.get("https://api.hbdm.com/linear-swap-api/v1/swap_funding_rate",
                             params={"contract_code": f"{coin}-USDT"}, timeout=6).json()
            rates["HTX"] = float(r["data"]["funding_rate"]) * 100
        except Exception:
            pass

    def _kucoin():
        try:
            # KuCoin BTC icin "XBT" ISO-4217-stili kod kullanir, "BTC" degil.
            symbol = f"{'XBT' if coin == 'BTC' else coin}USDTM"
            d = requests.get(f"https://api-futures.kucoin.com/api/v1/contracts/{symbol}",
                             timeout=6).json()["data"]
            rates["KuCoin"] = float(d["fundingFeeRate"]) * 100
        except Exception:
            pass

    def _hyperliquid():
        try:
            r = requests.post("https://api.hyperliquid.xyz/info",
                              json={"type": "metaAndAssetCtxs"}, timeout=8).json()
            universe, ctxs = r[0]["universe"], r[1]
            idx = next((i for i, u in enumerate(universe) if u["name"] == coin), None)
            if idx is not None:
                # Hyperliquid fonlamasi SAATLIK'tir; digerleri 8 saatlikle karsilastirilabilir
                # olsun diye x8 ile normalize edilir.
                rates["Hyperliquid"] = float(ctxs[idx]["funding"]) * 100 * 8
        except Exception:
            pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as ex:
        for fut in [ex.submit(f) for f in
                    (_binance, _bybit, _okx, _bitget, _gate, _htx, _kucoin, _hyperliquid)]:
            try:
                fut.result(timeout=10)
            except Exception:
                pass

    if not rates:
        return {}
    return {"funding_rate_pct": sum(rates.values()) / len(rates),
            "funding_by_exchange": rates}


@_cache(ttl=60, show_spinner=False)
def fetch_top_trader_ratios(coin: str, period: str = "15m") -> dict:
    """Coinglass'in 'Top Trader Long/Short Ratio' metriginin karsiligi.

    Kalabalik (global) hesap oranindan FARKLIDIR: bu, borsanin en buyuk
    bakiyeli hesaplarinin pozisyonudur - yani 'akilli para'. Kalabaligin
    tersi okunurken, buyuk oyuncunun yonu TAKIP edilir.
    """
    pair = f"{coin}USDT"
    out = {}

    def _get(path, key):
        try:
            r = requests.get(f"https://fapi.binance.com/futures/data/{path}",
                             params={"symbol": pair, "period": period, "limit": 3}, timeout=6)
            data = r.json()
            if isinstance(data, list) and data:
                out[key] = float(data[-1]["longShortRatio"])
                out[key + "_long_pct"] = float(data[-1]["longAccount"]) * 100
        except Exception:
            pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        futs = [ex.submit(_get, "topLongShortPositionRatio", "top_position_ratio"),
                ex.submit(_get, "topLongShortAccountRatio", "top_account_ratio")]
        for fut in futs:
            try:
                fut.result(timeout=9)
            except Exception:
                pass
    return out


@_cache(ttl=30, show_spinner=False)
def fetch_coinbase_premium(coin: str) -> dict:
    """Coinglass'in 'Coinbase Premium Index' karsiligi.

    Coinbase agirlikli olarak ABD kurumsal/bireysel talebini yansitir. Coinbase
    fiyati Binance'in uzerindeyse ABD tarafinda alim baskisi var demektir.
    """
    try:
        cb = requests.get(f"https://api.exchange.coinbase.com/products/{coin}-USD/ticker",
                          timeout=6).json()
        cb_price = float(cb["price"])
    except Exception:
        return {}
    try:
        bn = requests.get("https://api.binance.com/api/v3/ticker/price",
                          params={"symbol": f"{coin}USDT"}, timeout=6).json()
        bn_price = float(bn["price"])
    except Exception:
        return {}
    if bn_price <= 0:
        return {}
    return {"coinbase_premium_pct": (cb_price - bn_price) / bn_price * 100,
            "coinbase_price": cb_price, "binance_price": bn_price}


@_cache(ttl=15, show_spinner=False)
def fetch_futures_basis(coin: str) -> dict:
    """Coinglass'in 'Futures Basis' metriginin karsiligi: vadeli fiyatin spot
    fiyata gore priminin/iskontosunun yuzdesi.

    Pozitif basis (contango) -> piyasa vadeli tarafta LONG icin prim odemeye
    razi, yapisal olarak boga egilimli. Negatif basis (backwardation) -> panik
    satisi/short baskisi, ayi egilimli. Fonlama oranindan FARKLIDIR: fonlama
    8 saatlik donemsel odemedir, basis ise ANLIK fiyat farkidir.
    """
    pair = f"{coin}USDT"
    try:
        spot = float(requests.get("https://api.binance.com/api/v3/ticker/price",
                                  params={"symbol": pair}, timeout=6).json()["price"])
        fut = float(requests.get("https://fapi.binance.com/fapi/v1/premiumIndex",
                                 params={"symbol": pair}, timeout=6).json()["markPrice"])
    except Exception:
        return {}
    if spot <= 0:
        return {}
    return {"basis_pct": (fut - spot) / spot * 100, "spot_price": spot, "futures_price": fut}


@_cache(ttl=30, show_spinner=False)
def fetch_spot_futures_volume_ratio(coin: str) -> dict:
    """Spot ve vadeli 24s hacmi karsilastirir - piyasa katiliminin gercek
    yatirimdan mi (spot) yoksa kaldiracli spekulasyondan mi (vadeli) agirlikli
    oldugunu gosterir. Coinglass'in katilim/hacim panellerinin ucretsiz karsiligi.
    """
    pair = f"{coin}USDT"
    try:
        spot_v = float(requests.get("https://api.binance.com/api/v3/ticker/24hr",
                                    params={"symbol": pair}, timeout=6).json()["quoteVolume"])
        fut_v = float(requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr",
                                   params={"symbol": pair}, timeout=6).json()["quoteVolume"])
    except Exception:
        return {}
    total = spot_v + fut_v
    if total <= 0:
        return {}
    return {"spot_volume_usd": spot_v, "futures_volume_usd": fut_v,
            "futures_volume_share_pct": fut_v / total * 100}


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

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        for fut in [ex.submit(f) for f in (_oi, _ls, _taker)]:
            try:
                out.update(fut.result(timeout=10) or {})
            except Exception:
                pass

    if not any(k in out for k in ("oi_now", "ls_ratio", "taker_delta_pct")):
        return {}

    # Coinglass'in sundugu diger veri setlerinin ucretsiz karsiliklari - Coinglass
    # anahtari olsun olmasin HER ZAMAN eklenir (bunlar borsalarin kendi genel
    # uc noktalaridir, Coinglass'e ozel degildir).
    out.update(_fetch_free_derivative_extras(coin, period))
    return out


def _fetch_free_derivative_extras(coin: str, period: str) -> dict:
    """Coinglass anahtari olsun olmasin her zaman calisan, borsalarin genel
    uc noktalarindan gelen ek veri setleri (likidasyon, cok borsali OI/fonlama,
    buyuk oyuncu orani, Coinbase primi, basis, spot/vadeli hacim orani)."""
    extras = {}
    fetchers = {
        "funding": lambda: fetch_aggregated_funding(coin),
        "oi_agg": lambda: fetch_aggregated_open_interest(coin),
        "liq": lambda: fetch_liquidations(coin),
        "top": lambda: fetch_top_trader_ratios(coin, period),
        "premium": lambda: fetch_coinbase_premium(coin),
        "basis": lambda: fetch_futures_basis(coin),
        "volshare": lambda: fetch_spot_futures_volume_ratio(coin),
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(fetchers)) as ex:
        futs = {name: ex.submit(fn) for name, fn in fetchers.items()}
        for name, fut in futs.items():
            try:
                extras.update(fut.result(timeout=12) or {})
            except Exception:
                pass
    return extras


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
# Agirliklar toplami TAM 100 olmali (coverage_pct bunun uzerinden hesaplanir).
W_TAKER = 18      # Agresif alici/satici akisi - en dogrudan yon sinyali
W_OI = 10         # Acik pozisyon degisimi - trendin arkasindaki para
W_LIQ = 14        # Likidasyon dengesizligi - hangi taraf temizlendi (OKX gercek veri)
W_TOP = 14        # Buyuk oyuncu (top trader) pozisyonu - akilli para, YONU TAKIP EDILIR
W_LS = 8          # Kalabalik hesap orani - TERSINE (contrarian) okunur
W_FUNDING = 8     # Fonlama orani (3 borsa ort.) - asiri kaldirac TERSINE okunur
W_PREMIUM = 6     # Coinbase primi - ABD kurumsal talebi
W_BASIS = 10      # Vadeli-spot fiyat farki (contango/backwardation)
W_DOM = 12        # Dolar dominansi - nakite kacis mi, kriptoya akis mi
assert W_TAKER + W_OI + W_LIQ + W_TOP + W_LS + W_FUNDING + W_PREMIUM + W_BASIS + W_DOM == 100


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

    # 3) Likidasyon dengesizligi (OKX gercek likidasyon emirleri)
    if "liq_skew_pct" in deriv and deriv.get("liq_count", 0) > 0:
        lq = deriv["liq_skew_pct"]
        contrib = max(-1.0, min(1.0, lq / 50.0)) * W_LIQ
        score += contrib
        used_weight += W_LIQ
        _ll = deriv.get("liq_long_usd", 0.0)
        _sl = deriv.get("liq_short_usd", 0.0)
        checks.append(("Likidasyon Dengesizliği (son 1 saat)",
                       f"Long ${_ll:,.0f} / Short ${_sl:,.0f} "
                       f"({'Shortlar patlıyor' if lq > 0 else 'Longlar patlıyor'})",
                       "pass" if abs(lq) > 15 else "warn", contrib))

    # 3b) Buyuk oyuncu (top trader) pozisyon orani - kalabaligin AKSINE, bu yon TAKIP edilir
    if "top_position_ratio" in deriv:
        tp = deriv["top_position_ratio"]
        edge = tp - 1.0
        contrib = max(-1.0, min(1.0, edge / 0.8)) * W_TOP
        score += contrib
        used_weight += W_TOP
        checks.append(("Büyük Oyuncu Pozisyonu (akıllı para)",
                       f"{tp:.2f} ({'Büyükler LONG tarafta' if edge > 0 else 'Büyükler SHORT tarafta'})",
                       "pass" if abs(edge) > 0.2 else "warn", contrib))

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

    # 5b) Coinbase Primi - ABD kurumsal/bireysel talebi (BTC/ETH disinda genelde veri yok)
    if "coinbase_premium_pct" in deriv:
        cbp = deriv["coinbase_premium_pct"]
        contrib = max(-1.0, min(1.0, cbp / 0.15)) * W_PREMIUM
        score += contrib
        used_weight += W_PREMIUM
        checks.append(("Coinbase Primi (ABD talebi)",
                       f"%{cbp:+.3f} ({'ABD alım baskısı' if cbp > 0 else 'ABD satış baskısı'})",
                       "pass" if abs(cbp) > 0.05 else "warn", contrib))

    # 5c) Vadeli-Spot Basis - pozitif (contango) yapisal boga, negatif (backwardation) panik/ayi
    if "basis_pct" in deriv:
        bp = deriv["basis_pct"]
        contrib = max(-1.0, min(1.0, bp / 0.05)) * W_BASIS
        score += contrib
        used_weight += W_BASIS
        checks.append(("Vadeli-Spot Basis",
                       f"%{bp:+.4f} ({'Contango (yapısal boğa)' if bp > 0 else 'Backwardation (panik/ayı)'})",
                       "pass" if abs(bp) > 0.02 else "warn", contrib))

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
    "BORSA_TOPLAMI": ("8 Borsa Toplamı (Binance/Bybit/OKX/Bitget/Gate/HTX/KuCoin/Hyperliquid)", "#14b8a6"),
    "VERI_YOK": ("Veri alınamadı", "#dc2626"),
}


def render_market_intel_card(bias: dict, reconciled: dict = None) -> None:
    """Harmanlanmis piyasa yonunu ve alt kirilimlarini gosteren ortak kart."""
    if st is None:
        return

    n_ex = len(bias.get("derivatives", {}).get("oi_by_exchange", {}))
    src_label, src_color = SOURCE_LABELS.get(bias.get("source"), ("Bilinmiyor", "#94a3b8"))
    if bias.get("source") == "BORSA_TOPLAMI" and n_ex:
        src_label = f"{n_ex} Borsa Toplamı (OI/Fonlama)"
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
