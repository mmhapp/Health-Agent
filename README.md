# Health Agent

Hyödynnä Applen Kuntoilu- ja Terveys-dataa ja paikallinen LLM-pohjainen tekoälyagentti oman terveyden edistämiseen. Käytössä kardiovaskulaarinen data (sykevälivaihtelu, hengitystaajuus, leposyke, rannelämpö, happisaturaatio) ja EKG-data

Data käsitellään paikallisesti ja kielimallikutsut tehdään paikallisella kielimallilla `smollm2:135m-instruct-q2_K`. Tietosuoja on huomioitu – yksityiset tiedot eivät poistu koneelta. Laskenta tapahtuu paikallisesti, eikä dataa lähetetä ulkopuolisille palvelimille.
Kielimalli vertaa arvoja perusarvoihin (baseline), jotka on määritetty koodissa käsin. Tämä ei ole lääkinnällinen laite eikä tee diagnooseja.

Käyttöönotto: lataa koodi, vie oma terveysdatasi Applen Terveys- ja Pikakomennot-sovelluksilla ja tallenna tiedostot projektin kansioihin.

---

## Instructions


Asenna riippuvuudet
* (ollama pull smollm2:135m-instruct-q2_K)
* (pip install numpy pandas requests biosppy matplotlib)

Laita health.py Health Agent -kansioon.

Avaa pääte tässä kansiossa ja aja:

* python health.py ekg
* python health.py cardio

EKG-kuvat avautuvat näyttöön, jonka jälkeen tekoälyraportti tulostuu päätteeseen.

---

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
[ Tulosteet & Raportti ] ◄─────────────────────────────────────────────────────────────────┘
```
---

## License

MIT — see [LICENSE](LICENSE) for details.
