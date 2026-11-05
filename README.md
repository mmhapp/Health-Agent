# Health Agent

Use Apple Fitness and Health data together with a local LLM-based AI agent to support your own health. The agent works with cardiovascular data (heart rate variability, respiratory rate, resting heart rate, wrist temperature, blood oxygen saturation) and ECG signals.

All data is processed locally, and inference is run with the local language model `smollm2:135m-instruct-q2_K`. Privacy is built in: your personal data never leaves your machine, all computation happens locally, and no data is sent to external servers.

The language model compares values against baselines that are defined manually in the code. To get started, download the code, export your own health data using the iPhone Health and Shortcuts apps, and save the files in the project folders.

---

## Instructions

Install the dependencies:

* `ollama pull smollm2:135m-instruct-q2_K`
* `pip install numpy pandas requests biosppy matplotlib`

Place `health.py` in the `Health Agent` folder.

Open a terminal in that folder and run either:

* `python health.py ekg`
* `python health.py cardio`

The ECG plots will open on screen, after which the AI report is printed in the terminal.

## Project

```
Health Agent/
├── health.py
├── README.md
├── apple_health_export/
│   └── electrocardiograms/
│       └── ecg_*.csv
└── watch_export/
    ├── hrv.txt
    ├── ht.txt
    ├── ls.txt
    ├── rl.txt
    └── spo.txt
```

```
[ Data (Apple Watch) ] ➔ [ ETL (Python) ] ➔ [ Agent (Python) ] ➔ [ LLM (Local with Ollama) ]
                                                                                           │
[ Outputs & Report ] ◄─────────────────────────────────────────────────────────────────────┘
```

## License

MIT — see [LICENSE](LICENSE) for details.
