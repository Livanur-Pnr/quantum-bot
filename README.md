# Pro-Quantum Trading Terminal (MEXC AI Quant Engine)

Yapay Zeka destekli kripto vadeli islem (futures) analiz ve sinyal botu.

## Kurulum

```bash
pip install -r requirements_gainzalgo.txt
```

## API Anahtari Ayarlama

1. `.env.example` dosyasini `.env` olarak kopyalayin:
   ```bash
   cp .env.example .env
   ```
2. `.env` dosyasina MEXC API anahtarlarinizi girin.

## Calistirma

```bash
streamlit run main.py
```

## Yapı

```
kripto/
├── main.py                  # Streamlit giris noktasi
├── pages/
│   ├── 1_Canlı_Gösterge.py  # AI Quant Engine (ML Ensemble)
│   └── 2_Analiz_Tahmini.py  # Klasik Indikatör Botu
├── .env                     # API anahtarlari (gizli, Git'e eklenmez)
├── .env.example             # API anahtar sablonu
├── .gitignore               # Git gizlilik kurallari
└── requirements_gainzalgo.txt
```

## Onemli Notlar

- **Islem Modu:** Otomatik giris motoru kaldirilmistir. Tum LONG/SHORT/Kapat emirleri "Islem Execution" sekmesinden manuel olarak, kullanicinin onayiyla gonderilir.
- **Slippage Korumasi:** Emir gonderilmeden once anlik fiyat kontrol edilir; piyasa fiyati %0.5'ten fazla kaymissa emir iptal edilir.
- **TP/SL:** Giristen sonra MEXC'in gercek tetikleyici (plan) emirleriyle kurulur; pozisyon TP veya SL ile kapaninca karsi taraftaki bekleyen emir otomatik temizlenir.
- **Strateji Modu (Korumali/Normal/Agresif):** Su an yalnizca onerilen kaldirac/guven esigi degerlerini gosteren bilgilendirme amaclidir; islem acilisini otomatik tetiklemez.
