# Health Agent

Hyödynnä *Applen Kuntoilu- ja Terveys-dataa* ja *paikallinen LLM-pohjainen tekoälyagentti* oman terveyden edistämiseen
* Kardiovaskulaarinen data (sykevälivaihtelu, hengitystaajuus, leposyke, rannelämpö, happisaturaatio)
* EKG-data

Data käsitellään paikallisesti ja kielimallikutsut tehdään paikallisella kielimallilla 'smollm2:135m-instruct-q2_K.
Tietosuoja on huomioitu – yksityiset tiedot eivät poistu koneelta.
Laskenta tapahtuu paikallisesti, eikä dataa lähetetä ulkopuolisille palvelimille.
Kielimalli vertaa arvoja perusarvoihin (baseline), jotka on määritetty koodissa käsin.
Tämä ei ole lääkinnällinen laite eikä tee diagnooseja.

Agentissa käytössä oleva data on satunnaisgeneroitua, eikä se sisällä henkilötietoja – kuitenkin muodoltaan samaa kuin Applen data.
Mikäli halutaan käyttää omaa dataa, niin Applen Terveys-sovelluksen data voidaan tuoda Pikakomennot-sovelluksen avulla.

---

## Instructions


0. Asenna riippuvuudet

    (ollama pull smollm2:135m-instruct-q2_K)
    (pip install numpy pandas requests biosppy matplotlib)

1. Laita health.py Health Agent -kansioon.

2. Avaa pääte tässä kansiossa ja aja:

   python health.py ekg
   python health.py cardio

3. EKG-kuvat avautuvat näyttöön, jonka jälkeen tekoälyraportti tulostuu päätteeseen.

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