# HPC Maturity Assessment Agent - GUI

Graficki korisnicki interfejs za procjenu zrelosti organizacija za koristenje HPC infrastrukture.

## Sadrzaj

- [Pokretanje](#pokretanje)
- [Zahtjevi](#zahtjevi)
- [Funkcionalnosti](#funkcionalnosti)
- [Struktura projekta](#struktura-projekta)
- [Koristenje](#koristenje)

## Pokretanje

```bash
# Navigirajte u folder sa aplikacijom
cd hpc_assessment

# Pokrenite GUI
python run_gui.py

# Ili direktno
python gui.py
```

## Zahtjevi

- **Python 3.8+**
- **tkinter** (obicno dolazi sa Python-om)
  - Ubuntu/Debian: `sudo apt-get install python3-tk`
  - Fedora: `sudo dnf install python3-tkinter`
  - Windows: Dolazi sa Python instalacijom
  - macOS: Dolazi sa Python instalacijom

Nema dodatnih pip paketa - sve je implementirano koristenjem standardne biblioteke!

## Funkcionalnosti

### 1. Analiza Infrastrukture
- Dodavanje racunarskih cvorova (CPU, memorija, GPU)
- Definisanje workload-a i njihovih karakteristika
- Konfiguracija job scheduler-a, filesystem-a i interconnect-a

### 2. Evaluacija Tima
- Unos clanova tima i njihovih vjestina
- Procjena nivoa vjestina za: MPI, OpenMP, CUDA, SLURM, itd.
- Pracenje HPC projekata i iskustva

### 3. Analiza Softvera
- Softverski profil (jezik, tehnologije)
- Opcioni unos uzorka koda za analizu anti-pattern-a
- Procjena paralelizacije i skalabilnosti

### 4. Simulacija Target Klastera
- Konfiguracija HPC klastera za simulaciju
- Procjena speedup-a i efikasnosti
- Analiza troskova

### 5. Rezultati i Izvjestaji
- Maturity Score (0-100)
- Nivo zrelosti (1-5)
- Detaljni izvjestaji
- Eksport u JSON i TXT format

## Struktura Projekta

```
hpc_assessment/
├── run_gui.py              # Launcher skripta
├── gui.py                  # GUI aplikacija (Tkinter)
├── cli.py                  # CLI verzija (originalna)
├── agent.py                # Glavni assessment agent
├── config.py               # Konfiguracija i konstante
├── infrastructure_analyzer.py  # Analiza infrastrukture
├── team_evaluator.py       # Evaluacija tima
├── software_analyzer.py    # Analiza softvera
├── simulator.py            # HPC simulator
├── recommendation_engine.py    # Engine za preporuke
├── data_interoperability_analyzer.py  # Analiza podataka
└── README.md               # Ovaj fajl
```

## Koristenje

### Brzi start sa Demo podacima

1. Pokrenite aplikaciju: `python run_gui.py`
2. Kliknite **Demo** dugme u gornjem desnom uglu
3. Pogledajte rezultate u **Rezultati** tabu

### Unos vlastitih podataka

1. **Organizacija** - Unesite naziv organizacije
2. **Infrastruktura** - Dodajte cvorove i workload-e
3. **Tim** - Dodajte clanove tima i njihove vjestine
4. **Softver** - Opisite softverski profil
5. **Target Klaster** - Konfigurisite ciljni HPC klaster
6. Kliknite **Pokreni Procjenu**

### Eksport rezultata

- **Eksportuj** - Sacuvaj rezultate kao JSON
- **Sacuvaj izvjestaj** - Sacuvaj kompletan izvjestaj kao TXT

## Nivoi Zrelosti

| Nivo | Naziv | Opis |
|------|-------|------|
| 1 | Pocetni | Minimalna HPC spremnost, potrebna znacajna ulaganja |
| 2 | Razvijajuci | Osnovni kapaciteti postoje, potrebna obuka |
| 3 | Definisani | Strukturirani procesi, djelimicna paralelizacija |
| 4 | Upravljani | Optimizovani procesi, efikasno koristenje resursa |
| 5 | Optimizovani | Potpuna HPC spremnost, kontinuirano unapredjenje |

## Rjesavanje problema

### "No module named 'tkinter'"
```bash
# Ubuntu/Debian
sudo apt-get install python3-tk

# Fedora
sudo dnf install python3-tkinter
```

### GUI se ne prikazuje ispravno
- Provjerite da imate Python 3.8+
- Na Linux-u mozda treba instalirati dodatne X11 biblioteke

## Licenca

Ovaj projekat je razvijen za internu upotrebu.

---

Razvijeno za HPC zajednicu
