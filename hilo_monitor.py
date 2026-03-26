# -*- coding: utf-8 -*-
# hilo_monitor.py — Varredura HiLo(20) diário em ações da B3
# Alertas via Telegram e exportação CSV diária

from __future__ import annotations
import os
import csv
import json
import logging
from datetime import datetime

# ==========================
# PARÂMETROS GERAIS
# ==========================
HILO_PERIOD   = 20          # ← Alterado de 10 para 20
LOOKBACK_DAYS = 300         # Aumentado para garantir warmup adequado ao período 20
EXPORT_CSV    = True
MIN_CANDLES   = HILO_PERIOD + 5   # Mínimo de candles para calcular com segurança

# ==========================
# LOGGING
# ==========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ==========================
# TICKERS B3
# ==========================
TICKERS_BR: list[str] = sorted(set([
    "PETR4","LREN3","VALE3","BOVA11","ITUB4","BBDC4","B3SA3","BBAS3","ABEV3",
    "WEGE3","SUZB3","PRIO3","GGBR4","CSNA3","ITSA4","BRKM3",
    "SANB11","EQTL3","TIMS3","VIVT3","HAPV3","BPAC11","RENT3","KLBN11",
    "RAIZ4","BBDC3","CMIG4","SBSP3","RADL3","RAIL3",
    "BEEF3","BRAV3","EMBJ3","SAPR11","MOVI3","RDOR3","USIM5","YDUQ3",
    # Substituições: ELET3/ELET6 → CPFE3 | BRFS3 → MRFG3 | CPLE6 → TAEE11
    "CPFE3","MRFG3","TAEE11","EGIE3",
]))

# ==========================
# DEPENDÊNCIAS
# ==========================
try:
    import pandas as pd
except ImportError as e:
    raise SystemExit("❌ Faltou 'pandas'. Instale com: pip install pandas") from e

try:
    import yfinance as yf
except ImportError as e:
    raise SystemExit("❌ Faltou 'yfinance'. Instale com: pip install yfinance") from e

# ==========================
# TELEGRAM
# ==========================
import urllib.request

TELEGRAM_BOT_TOKEN = os.getenv("TG_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TG_CHAT_ID", "")

def send_telegram(message: str) -> bool:
    """Envia texto ao Telegram. Retorna True se enviou, False caso contrário."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        url  = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": message}).encode("utf-8")
        req  = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
        return True
    except Exception as exc:
        log.warning("Telegram falhou: %s", exc)
        return False

# ==========================
# INDICADOR: HiLo Activator
# ==========================
def hilo_activator(df: pd.DataFrame, period: int = HILO_PERIOD) -> pd.Series:
    """
    HiLo Activator clássico.

    Regra:
    - Tendência ALTA  → linha = mínima das últimas `period` barras (suporte)
    - Tendência BAIXA → linha = máxima das últimas `period` barras (resistência)

    Virada para ALTA  : close > rolling_high[i]
    Virada para BAIXA : close < rolling_low[i]

    Parâmetros
    ----------
    df     : DataFrame com colunas 'high', 'low', 'close'
    period : janela do HiLo (padrão = HILO_PERIOD global)

    Retorna
    -------
    pd.Series com o valor da linha HiLo por barra (NaN nas primeiras `period` barras)
    """
    required = {"high", "low", "close"}
    if not required <= set(df.columns):
        raise ValueError(f"DataFrame precisa conter: {required}")
    if len(df) < period + 1:
        raise ValueError(f"Candles insuficientes ({len(df)}) para período {period}.")

    rolling_high = df["high"].rolling(period).max()
    rolling_low  = df["low"].rolling(period).min()

    hilo  = pd.Series(index=df.index, dtype=float)
    trend: int | None = None  # +1 = alta, -1 = baixa

    for i in range(period, len(df)):
        close = df["close"].iat[i]
        rh    = rolling_high.iat[i]
        rl    = rolling_low.iat[i]

        # Inicializa tendência na primeira barra completa
        if trend is None:
            trend = 1 if close >= rh else -1

        # ── Virada para ALTA ──────────────────────────────────────────────
        if trend == -1 and close > rh:
            trend = 1
        # ── Virada para BAIXA ─────────────────────────────────────────────
        elif trend == 1 and close < rl:
            trend = -1

        hilo.iat[i] = rl if trend == 1 else rh

    return hilo

# ==========================
# COLETA DE DADOS
# ==========================
_REQUIRED_COLS = {"open", "high", "low", "close", "volume"}

def _normalize_df(raw: pd.DataFrame) -> pd.DataFrame | None:
    """Padroniza colunas para lowercase e valida campos obrigatórios."""
    df = raw.copy()
    # Achata MultiIndex de coluna caso exista
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(c[0]).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    df = df.dropna(subset=list(_REQUIRED_COLS & set(df.columns)))
    return df if _REQUIRED_COLS <= set(df.columns) else None


def fetch_single(ticker_br: str, lookback_days: int) -> pd.DataFrame | None:
    """Download individual com fallback — usado quando o batch falha."""
    yf_t = f"{ticker_br}.SA"
    try:
        raw = yf.download(
            yf_t, period=f"{lookback_days}d", interval="1d",
            auto_adjust=True, progress=False, multi_level_index=False,
        )
        if raw is None or raw.empty:
            return None
        return _normalize_df(raw)
    except Exception as exc:
        log.debug("fetch_single(%s) falhou: %s", ticker_br, exc)
        return None


def fetch_ohlcv_daily(
    tickers_br: list[str],
    lookback_days: int = LOOKBACK_DAYS,
) -> dict[str, pd.DataFrame]:
    """
    Baixa OHLCV diário para a lista de tickers.

    Estratégia:
    1. Tenta batch download (mais rápido)
    2. Para tickers sem dados no batch, tenta download individual
    """
    out: dict[str, pd.DataFrame] = {}
    yf_tickers = [f"{t}.SA" for t in tickers_br]

    log.info("Baixando %d tickers via batch…", len(tickers_br))
    try:
        block = yf.download(
            yf_tickers,
            period=f"{lookback_days}d",
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
        )

        if isinstance(block.columns, pd.MultiIndex):
            # Estrutura (campo, ticker) — padrão do yfinance >= 0.2
            for t in tickers_br:
                y = f"{t}.SA"
                try:
                    sub = block.xs(y, axis=1, level=1) if y in block.columns.get_level_values(1) \
                          else block[y]
                    df  = _normalize_df(sub)
                    if df is not None and len(df) >= MIN_CANDLES:
                        out[t] = df
                except (KeyError, Exception):
                    pass
    except Exception as exc:
        log.warning("Batch download falhou (%s) — usando downloads individuais.", exc)

    # Fallback individual para tickers ausentes
    missing = [t for t in tickers_br if t not in out]
    if missing:
        log.info("Fallback individual para %d tickers…", len(missing))
    for t in missing:
        df = fetch_single(t, lookback_days)
        if df is not None and len(df) >= MIN_CANDLES:
            out[t] = df
        else:
            log.warning("Sem dados suficientes para %s.SA — ignorado.", t)

    log.info("Dados obtidos: %d/%d tickers.", len(out), len(tickers_br))
    return out

# ==========================
# DETECÇÃO DE VIRADA
# ==========================
def detect_turn(df: pd.DataFrame, hilo: pd.Series) -> dict | None:
    """
    Retorna dict com 'when' (today/yesterday) e 'direction' (alta/baixa)
    se houve virada do HiLo nas últimas 2 barras. Caso contrário, None.
    """
    df2 = df.copy()
    df2["hilo"] = hilo
    df2 = df2.dropna(subset=["hilo", "close"])

    if len(df2) < 3:
        return None

    def side(row) -> str:
        return "alta" if row["close"] > row["hilo"] else "baixa"

    last_side = side(df2.iloc[-1])
    prev_side = side(df2.iloc[-2])
    ante_side = side(df2.iloc[-3])

    if last_side != prev_side:
        return {"when": "today", "direction": last_side}
    if prev_side != ante_side:
        return {"when": "yesterday", "direction": prev_side}
    return None

# ==========================
# FORMATAÇÃO DO SINAL
# ==========================
def build_signal_text(ticker: str, info: dict, period: int) -> str:
    when_pt   = "HOJE" if info["when"] == "today" else "ONTEM"
    direction = info["direction"].upper()
    rec = "📈 COMPRAR CALL (ITM/ATM)" if info["direction"] == "alta" else "📉 COMPRAR PUT (ITM/ATM)"
    return f"[{ticker}] HiLo({period}) → {direction} ({when_pt})  {rec}"

# ==========================
# LÓGICA PRINCIPAL
# ==========================
def run():
    log.info("=== HiLo Monitor — período %d | %s ===", HILO_PERIOD, datetime.now().strftime("%Y-%m-%d"))

    data = fetch_ohlcv_daily(TICKERS_BR, lookback_days=LOOKBACK_DAYS)
    signals: list[dict] = []
    errors:  list[str]  = []

    for t, df in data.items():
        try:
            hilo = hilo_activator(df, HILO_PERIOD)
            info = detect_turn(df, hilo)
            if info:
                text = build_signal_text(t, info, HILO_PERIOD)
                signals.append({
                    "ticker":    t,
                    "when":      info["when"],
                    "direction": info["direction"],
                    "message":   text,
                })
        except Exception as exc:
            msg = f"[{t}] ERRO ao processar: {exc}"
            errors.append(msg)
            log.debug(msg)

    # ── Exibição no console ──────────────────────────────────────────────
    today_sigs     = [s for s in signals if s["when"] == "today"]
    yesterday_sigs = [s for s in signals if s["when"] == "yesterday"]

    print("\n" + "═" * 54)
    print(f"  HiLo({HILO_PERIOD}) — SINAIS DE VIRADA — {datetime.now():%Y-%m-%d}")
    print("═" * 54)

    if today_sigs:
        print(f"\n▶ HOJE ({len(today_sigs)} sinal{'is' if len(today_sigs)>1 else ''}):")
        for s in today_sigs:
            print("  •", s["message"])
    if yesterday_sigs:
        print(f"\n▶ ONTEM ({len(yesterday_sigs)} sinal{'is' if len(yesterday_sigs)>1 else ''}):")
        for s in yesterday_sigs:
            print("  •", s["message"])
    if not signals:
        print("\n  ✔ Nenhuma virada detectada nas últimas 2 barras.")
    if errors:
        print(f"\n⚠ Erros em {len(errors)} tickers (use --debug para detalhes).")

    print("═" * 54 + "\n")

    # ── Telegram ────────────────────────────────────────────────────────
    telegram_ok = 0
    if today_sigs:
        for s in today_sigs:
            if send_telegram(s["message"]):
                telegram_ok += 1
        log.info("Telegram: %d mensagem(ns) enviada(s).", telegram_ok)
    else:
        msg_vazio = f"📊 HiLo({HILO_PERIOD}) — {datetime.now():%d/%m/%Y}\n✔ Nenhuma virada detectada hoje."
        if send_telegram(msg_vazio):
            log.info("Telegram: aviso 'sem sinais' enviado.")

    # ── CSV ─────────────────────────────────────────────────────────────
    if EXPORT_CSV:
        today  = datetime.now().strftime("%Y-%m-%d")
        fname  = f"signals_{today}.csv"
        rows   = signals + [{"ticker": "ERROR", "when": "-", "direction": "-", "message": e} for e in errors]
        fields = ["ticker", "when", "direction", "message"]
        with open(fname, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        log.info("CSV exportado: %s (%d linha(s)).", fname, len(rows))


# ==========================
# ENTRYPOINT
# ==========================
if __name__ == "__main__":
    run()
