from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

# Ympäristömuuttujat ja oletusarvot
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "smollm2:135m-instruct-q2_K")
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ECG_DIR = BASE_DIR / "apple_health_export" / "electrocardiograms"
DEFAULT_WATCH_DIR = BASE_DIR / "watch_export"
DEFAULT_FS = 512
MIN_AMPLITUDE_UV = 100


class LlmError(RuntimeError):
    """Ollama-yhteyden tai -vastauksen virhe."""


def log(message: str) -> None:
    print(message, file=sys.stderr)


def ollama_generate(prompt: str, model: str, host: str) -> str:
    """Pyytää tekstivastauksen paikalliselta Ollama-mallilta."""
    try:
        response = requests.post(
            f"{host.rstrip('/')}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "0",
                "options": {"temperature": 0.2, "num_predict": 40},
            },
            timeout=60,
        )
        response.raise_for_status()
    except requests.ConnectionError as exc:
        raise LlmError(f"Ollamaan ei saatu yhteyttä ({host}). Onko se käynnissä?") from exc
    except requests.HTTPError as exc:
        raise LlmError(
            f"Ollama palautti virheen {response.status_code}: {response.text[:300]}. "
            f"Lataa malli komennolla: ollama pull {model}"
        ) from exc
    return response.json().get("response", "").strip()


# ---------------------------------------------------------------------------
# EKG-analyysi
# ---------------------------------------------------------------------------

ECG_META_KEYS = {
    "pvm": ("Tallennuspäivä", "Recorded Date"),
    "luokitus": ("Luokitus", "Classification"),
    "laite": ("Laite", "Device"),
}


def as_float(text: str) -> float | None:
    try:
        return float(text.strip().strip('"').replace("−", "-").replace(",", "."))
    except ValueError:
        return None


def load_ecg(path: Path) -> tuple[np.ndarray, dict[str, str]]:
    """Lukee Apple Healthin ecg_*.csv-tiedoston ilman henkilötietoja."""
    meta = {"pvm": "Ei tiedossa", "luokitus": "Ei luokiteltu", "laite": "Apple Watch"}
    samples: list[float] = []
    
    with path.open(encoding="utf-8-sig", errors="replace") as file:
        for raw in file:
            line = raw.strip()
            if not line:
                continue
            value = as_float(line)
            if value is not None:
                samples.append(value)
                continue
            if samples:
                continue
            row = next(csv.reader([line]), [])
            if len(row) >= 2:
                for field, labels in ECG_META_KEYS.items():
                    if row[0].strip() in labels:
                        meta[field] = row[1].strip()
                        
    return np.asarray(samples, dtype=float), meta


@dataclass
class EcgMeasurement:
    label: str
    signal: np.ndarray
    fs: int
    meta: dict[str, str]
    rpeaks: np.ndarray
    heart_rate: np.ndarray
    features: dict[str, Any]

    @property
    def time(self) -> np.ndarray:
        return np.arange(len(self.signal)) / self.fs


def analyze_ecg(path: Path, fs: int) -> EcgMeasurement:
    signal, meta = load_ecg(path)
    if len(signal) < fs * 5:
        raise ValueError(f"Signaali liian lyhyt ({len(signal)} näytettä)")
    try:
        from biosppy.signals import ecg as bp_ecg
    except ImportError as exc:
        raise RuntimeError("EKG-analyysi vaatii biosppy-paketin (pip install biosppy).") from exc

    peaks = bp_ecg.ecg(signal=signal, sampling_rate=fs, show=False)["rpeaks"]
    if len(peaks) < 3:
        raise ValueError(f"R-piikkejä löytyi liian vähän ({len(peaks)} kpl)")
        
    rr = np.diff(peaks) / fs
    rr = rr[rr > 0]  # Varmistetaan, ettei nollajakoa tapahdu
    if len(rr) == 0:
        raise ValueError("Virheellinen RR-intervalldata.")
        
    rr_diff = np.diff(rr)
    mean_rr = float(np.mean(rr))
    
    features: dict[str, Any] = {
        "avg_hr": round(60 / mean_rr, 1) if mean_rr > 0 else 0.0,
        "hr_range": f"{60 / float(np.max(rr)):.1f} - {60 / float(np.min(rr)):.1f}",
        "rmssd": round(float(np.sqrt(np.mean(rr_diff ** 2)) * 1000), 1) if len(rr_diff) > 0 else 0.0,
        "sdnn": round(float(np.std(rr, ddof=1) * 1000), 1),
        "pnn50": round(float(np.mean(np.abs(rr_diff) > 0.05) * 100), 1) if len(rr_diff) > 0 else 0.0,
        "amplitude_uv": round(float(np.ptp(signal)), 1),
        "duration": round(len(signal) / fs, 1),
        "beats": int(len(peaks)),
    }
    features["quality"] = "Korkea" if features["amplitude_uv"] > MIN_AMPLITUDE_UV else "Heikko"
    return EcgMeasurement(path.stem.removeprefix("ecg_"), signal, fs, meta, peaks, 60 / rr, features)


def ecg_table(measurements: list[EcgMeasurement]) -> pd.DataFrame:
    return pd.DataFrame([{
        "mittaus": item.label, 
        "laite_luokitus": item.meta["luokitus"],
        "keskisyke_bpm": item.features["avg_hr"], 
        "sykealue_bpm": item.features["hr_range"],
        "rmssd_ms": item.features["rmssd"], 
        "sdnn_ms": item.features["sdnn"],
        "pnn50_pct": item.features["pnn50"], 
        "kesto_s": item.features["duration"],
        "r_piikkeja": item.features["beats"], 
        "signaalin_laatu": item.features["quality"],
    } for item in measurements])


def plot_ecg(measurements: list[EcgMeasurement]) -> None:
    """Näyttää EKG-kuvaajat visualisointia varten."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log("Varoitus: matplotlib ei ole asennettu. Kuvaajia ei näytetä.")
        return

    for item in measurements:
        figure, axes = plt.subplots(2, 1, figsize=(16, 9))
        axes[0].plot(item.time, item.signal, color="#d62728", linewidth=0.8)
        axes[0].plot(item.time[item.rpeaks], item.signal[item.rpeaks], "o", color="black", markersize=4)
        axes[0].set(title=f"Apple Watch EKG ja R-piikit ({item.label})", xlabel="Aika (s)", ylabel="EKG (µV)")
        axes[1].plot(item.heart_rate, "o-", color="#d62728", markersize=3)
        axes[1].set(title="Hetkellinen syke", xlabel="Lyönti", ylabel="BPM")
        for axis in axes:
            axis.grid(alpha=0.3)
        figure.tight_layout()

    '''if len(measurements) > 1:
        figure, axis = plt.subplots(figsize=(16, 6))
        for item in measurements:
            axis.plot(item.time, item.signal, linewidth=0.7, alpha=0.7, label=item.label)
        axis.set(title="Kaikki Apple Watch -EKG:t", xlabel="Aika (s)", ylabel="EKG (µV)")
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()'''

    if len(measurements) > 1:
        figure, axis = plt.subplots(figsize=(16, 6))
        for item in measurements:
            # Paksunnettu viiva: linewidth nostettu 0.7 -> 1.8
            axis.plot(item.time, item.signal, linewidth=1.8, alpha=0.8, label=item.label)
        axis.set(title="Kaikki Apple Watch -EKG:t", xlabel="Aika (s)", ylabel="EKG (µV)")
        # TAUSTARUUDUKKO POISTETTU:
        # axis.grid(alpha=0.3)  <- Poistettu käytöstä
        axis.legend()
        figure.tight_layout()

    plt.show()
    plt.close("all")


def ecg_prompt(table: pd.DataFrame) -> str:
    return f"""Olet varovainen sydänterveysdatan avustaja. Vastaa suomeksi tiiviisti.
Seuraavat ovat Apple Watchin yhden kytkennän EKG-mittausten anonyymejä,
teknisesti laskettuja tunnuslukuja:\n{table.to_csv(index=False)}

Kerro laitteen luokitukset sellaisinaan, kuvaa syke- ja RMSSD-vaihtelu sekä
tekniset rajoitteet (30 s, yksi kytkentä, automaattinen R-piikkitunnistus).
Älä tee diagnoosia, vahvista laitteen luokitusta tai keksi tietoja. Oireiden
tai huolen yhteydessä ohjaa terveydenhuollon ammattilaisen arvioon."""


def run_ecg(args: argparse.Namespace) -> int:
    if args.file:
        files = [Path(args.file).expanduser()]
        if not files[0].exists() and not files[0].is_absolute():
            files = [Path(args.dir).expanduser() / files[0]]
    else:
        files = sorted(Path(args.dir).expanduser().glob("ecg_*.csv"))
        if args.latest:
            files = files[-args.latest:]

    if not files or not all(path.exists() for path in files):
        log("Virhe: EKG-tiedostoa ei löytynyt. Tarkista hakemisto tai tiedostopolku.")
        return 1

    measurements = []
    for path in files:
        try:
            measurements.append(analyze_ecg(path, args.fs))
        except Exception as exc:
            log(f"Ohitetaan {path.name}: {exc}")

    if not measurements:
        log("Virhe: Yhtään kelvollista EKG-mittausta ei pystytty analysoimaan.")
        return 1

    table = ecg_table(measurements)
    print("\nEKG-TUNNUSLUVUT\n" + table.to_string(index=False))

    if not args.no_plot:
        plot_ecg(measurements)

    if args.no_llm:
        return 0

    try:
        report = ollama_generate(ecg_prompt(table), args.model, args.host)
    except LlmError as exc:
        log(f"Virhe: {exc}")
        return 1

    print("\nAGENTIN YHTEENVETO\n" + "-" * 22 + "\n" + report)
    return 0


# ---------------------------------------------------------------------------
# Muut terveysmittarit
# ---------------------------------------------------------------------------

LINE_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})\s+klo\s+(\d{1,2})\.(\d{2}),\s*([\-\d\.]+)\s*$")


@dataclass(frozen=True)
class MetricConfig:
    key: str
    label: str
    unit: str
    direction: str
    abs_threshold: float | None
    manual_baseline: float | None

# ASETA BASELINE MANUAALISESTI
METRICS = {
    "ls":  MetricConfig("ls",  "Leposyke", "/min", "high", 6.0, 65.0),
    "hrv": MetricConfig("hrv", "Sykevälivaihtelu (HRV)", "ms", "low", None, 45.0),
    "rl":  MetricConfig("rl",  "Rannelämpö", "°C", "high", 0.3, 36.0),
    "ht":  MetricConfig("ht",  "Hengitystiheys", "/min", "high", 2.0, 15.0),
    "spo": MetricConfig("spo", "Veren happitaso", "%", "low", 2.0, 97.0),
}

REL_THRESHOLDS = {"hrv": 0.20}


def load_metric(path: Path) -> pd.Series:
    rows = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            match = LINE_RE.match(line.strip())
            if match:
                day, month, year, hour, minute, value = match.groups()
                rows.append((datetime(int(year), int(month), int(day), int(hour), int(minute)), float(value)))

    if not rows:
        raise ValueError(f"Ei jäsenneltävää dataa tiedostossa: {path}")

    frame = pd.DataFrame(rows, columns=["timestamp", "value"])
    return frame.groupby(frame.timestamp.dt.date).value.mean().rename(path.stem)


def load_metrics(data_dir: Path) -> dict[str, pd.Series]:
    data: dict[str, pd.Series] = {}
    for key in METRICS:
        path = data_dir / f"{key}.txt"
        if path.exists():
            data[key] = load_metric(path)

    if not data:
        raise FileNotFoundError(f"Ei löytynyt mittaritiedostoja kohteesta '{data_dir}'.")
    return data


def evaluate(key: str, series: pd.Series, window: int) -> dict[str, Any]:
    cfg = METRICS[key]
    dates = pd.to_datetime(series.index)
    dated = pd.Series(series.to_numpy(), index=dates).sort_index().dropna()
    cutoff = dated.index.max() - timedelta(days=window - 1)
    recent = dated[dated.index >= cutoff]
    
    baseline = cfg.manual_baseline
    source = "käsin annettu"
    if baseline is None:
        pool = dated[dated.index < cutoff]
        baseline = float((pool if len(pool) else dated).mean())
        source = f"datasta ({len(pool)} pv)"

    current = float(recent.mean()) if len(recent) > 0 else baseline
    delta = current - baseline
    threshold = cfg.abs_threshold

    flagged = (
        (cfg.direction == "high" and threshold is not None and delta >= threshold)
        or (cfg.direction == "low" and threshold is not None and delta <= -threshold)
    )
    relative = REL_THRESHOLDS.get(key)
    if relative and baseline:
        flagged = flagged or (cfg.direction == "low" and (delta / baseline) <= -relative)

    return {
        "key": key,
        "label": cfg.label,
        "unit": cfg.unit,
        "baseline": round(baseline, 2),
        "baseline_source": source,
        "recent": round(current, 2),
        "delta": round(delta, 2),
        "flagged": bool(flagged),
        "recent_days": len(recent),
    }


def deterministic_report(results: list[dict[str, Any]]) -> str:
    lines = ["TERVEYSDATAN POIKKEAMA-ANALYYSI", "=" * 36]
    for item in results:
        marker = " ⚠ POIKKEAMA" if item["flagged"] else ""
        lines.append(
            f"{item['label']}: baseline {item['baseline']} {item['unit']} ({item['baseline_source']}), "
            f"viimeisin jakso {item['recent']} {item['unit']}, muutos {item['delta']:+}{item['unit']}{marker}"
        )
    count = sum(item["flagged"] for item in results)
    lines.append("🔴 Yhdistetty hälytys" if count >= 2 else "🟡 Yksi poikkeama" if count else "🟢 Ei poikkeamia")
    return "\n".join(lines)


def cardio_prompt(results: list[dict[str, Any]], question: str) -> str:
    return f"""Vastaa suomeksi enintään kahdella lyhyellä virkkeellä.
Käytä vain näitä laskettuja tuloksia: {json.dumps(results, ensure_ascii=False)}
Kerro poikkeamien määrä ja yksi yleinen seuraava askel. Älä diagnosoi.
Pyyntö: {question}"""


def run_cardio(args: argparse.Namespace) -> int:
    try:
        data = load_metrics(Path(args.dir).expanduser())
    except (FileNotFoundError, ValueError) as exc:
        log(f"Virhe: {exc}")
        return 1

    results = [evaluate(key, series, args.window) for key, series in data.items()]
    print(deterministic_report(results))

    if args.no_llm:
        return 0

    question = args.question or f"Tee yhteenveto viimeisten {args.window} päivän datasta ja kerro seuraava askel."
    try:
        print("\nAGENTIN YHTEENVETO\n" + "-" * 22)
        print(ollama_generate(cardio_prompt(results, question), args.model, args.host))
    except LlmError as exc:
        log(f"Virhe: {exc}")
        return 1

    if args.chat:
        print("\nKeskustelutila; lopeta komennolla 'exit' tai 'lopeta'.")
        while True:
            try:
                question = input("Sinä: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if question.lower() in {"exit", "quit", "q", "lopeta"}:
                break
            if question:
                try:
                    print("Agentti:", ollama_generate(cardio_prompt(results, question), args.model, args.host))
                except LlmError as exc:
                    print(f"[virhe] {exc}")
    return 0


# ---------------------------------------------------------------------------
# CLI-määritelmät
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Apple Watch -terveysagentti")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama-mallin nimi")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Ollama-palvelimen osoite")

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # EKG CLI -asetukset
    ecg_parser = subparsers.add_parser("ekg", help="Analysoi EKG-tiedostoja")
    ecg_parser.add_argument("--dir", default=DEFAULT_ECG_DIR, help="EKG-tiedostojen hakemisto")
    ecg_parser.add_argument("--file", help="Yksittäisen EKG-tiedoston polku")
    ecg_parser.add_argument("--latest", type=int, help="Analysoi vain N viimeisintä tiedostoa")
    ecg_parser.add_argument("--fs", type=int, default=DEFAULT_FS, help="Näytteenottotaajuus (Hz)")
    ecg_parser.add_argument("--no-plot", action="store_true", help="Älä näytä kuvaajia")
    ecg_parser.add_argument("--no-llm", action="store_true", help="Jätä kielimalliraportti luomatta")

    # Cardio CLI -asetukset
    cardio_parser = subparsers.add_parser("cardio", help="Analysoi yleisiä terveysmittareita")
    cardio_parser.add_argument("--dir", default=DEFAULT_WATCH_DIR, help="Terveysmittareiden hakemisto")
    cardio_parser.add_argument("--window", type=int, default=3, help="Analysoitavien päivien määrä")
    cardio_parser.add_argument("--question", help="Kysymys kielimallille")
    cardio_parser.add_argument("--chat", action="store_true", help="Käynnistä interaktiivinen keskustelu")
    cardio_parser.add_argument("--no-llm", action="store_true", help="Jätä kielimalliraportti luomatta")

    args = parser.parse_args()

    if args.subcommand == "ekg":
        return run_ecg(args)
    elif args.subcommand == "cardio":
        return run_cardio(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
