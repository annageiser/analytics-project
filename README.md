
# Analytics Project

A model that predicts a) whether a product will be bought in a session and (b) if bought, in which quantity (sales forecast) for a dynamically priced mail-order pharmacy

## Quick Start

1. Create and activate a virtual environment:
   - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
2. Install dependencies:
   - `pip install --upgrade pip && pip install -r requirements.txt`
3. Place input data in the `data/` folder:
   - `data/train.csv`
   - `data/items.csv`
4. Run notebooks in this order:
   - `exploratory_data:analysis.ipynb` (EDA and data quality checks)
   - `modeling_example.ipynb` (initial preprocessing sample)
   - `analytics_project_jupyter_draft.ipynb` (feature engineering draft)



# Order Prediction Pipeline – Architecture Diagram

## Goal

Vorhersage, ob ein Produkt gekauft wird (Order = 1 oder 0), basierend auf Preis, Produktdaten, Zeitverhalten und historischen Mustern.


```
┌──────────────────────────────┐      ┌──────────────────────────────┐
│ train.csv                    │      │ items.csv                    │     1. Raw data (Input)
│ - Klicks / Orders            │      │ - Produktinfos               │
│ - Preis / Verhalten          │      │ - Kategorie / Hersteller     │
│ - Zeit (day)                 │      │ - Produktattribute           │
└──────────────┬───────────────┘      └──────────────┬───────────────┘
               │                                     │
               └──────────────────┬──────────────────┘
                                  ▼
                  ┌──────────────────────────────┐
                  │ JOIN über pid (Produkt-ID)   │                         2. Datenverknüpfung (Merge)
                  │ -> verbindet Verhalten +     │                         Ergebnis: eine komplette Tabelle mit Produkt + Verhalten
                  │    Produktinformationen      │
                  └───────────────┬──────────────┘
                                  ▼
                  ┌──────────────────────────────┐                         3. Zeitliche Sortierung
                  │ SORTIERUNG                   │                         Warum wichtig:
                  │ - nach day (Zeit)            │                         - Modell darf keine Zukunft sehen
                  │ - nach lineID (Reihenfolge)  │                         - Daten müssen chronologisch verarbeitet werden
                  └───────────────┬──────────────┘
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │ FEATURE ENGINEERING                                    │  4. Feature Engineering (neue Signale bauen) 
      │                                                        │  **Ziel:** komplexe Muster sichtbar machen (z.B. Rabatt + Werbung)
      │ Preis Features:                                        │
      │ - priceRatio (Preis vs UVP)                            │
      │ - priceVsCompetitor                                    │
      │ - priceDiscount                                        │
      │                                                        │
      │ Zeit Features:                                         │
      │ - Wochentag als sin/cos (Zyklus)                       │
      │                                                        │
      │ Interaktionen:                                         │
      │ - Werbung × Preis                                      │
      │ - Verfügbarkeit × Preis                                │
      └───────────────────────────┬────────────────────────────┘
                                  ▼
                  ┌──────────────────────────────┐    5. Zeitlicher Split (Train/Test)
                  │ TRAIN (erste 80%)            │    **Wichtig:**
                  │ TEST (letzte 20%)            │    * kein zufälliges Mischen
                  └──────────────┬───────────────┘    * echte Zukunft wird getestet
                                 ▼


```
┌──────────────────────────────────────────────────────────────┐     6. Rolling Window Cross Validation (Kernlogik)
│ TRAINING IN ZEITFENSTERN                                     │     **Warum:** simuliert reale Produktion (Zeit vergeht)
│                                                              │
│ Beispiel:                                                    │
│ Fenster 1: Tage 1–7 → train                                  │
│            Tag 8 → validieren                                │
│ Fenster 2: Tage 2–8 → train                                  │
│            Tag 9 → validieren                                │
│ usw.                                                         │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐     7. Feature Verarbeitung pro Fold (sehr wichtig)
│ PRO FOLD: DATENAUFBEREITUNG                                  │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
```

### 7.1 Historische Features (nur Vergangenheit!)

```
- Produkt-Kaufrate (pid_order_rate)
- Klickrate / Basketrate
- Preis-Volatilität
- Produkt bereits gesehen?
```

### 7.2 Target Encoding

```
- Hersteller → Kaufwahrscheinlichkeit
- Kategorie → Kaufwahrscheinlichkeit
- Produktgruppe → Kaufwahrscheinlichkeit
```

### 7.3 Cleaning

```
- Entfernen von Leakage (z.B. revenue)
- Entfernen von IDs
- Entfernen von Rohpreisvariablen
```

### 7.4 Feature Filter

```
- Entfernen stark korrelierter Features (>0.95)
```

### 7.5 Vorbereitung fürs Modell

```
- Missing Values → Median / "Missing"
- Skalierung (RobustScaler)
- OneHot Encoding (Kategorien)
```

---

## 8. Modelle

```
┌──────────────────────────────┐
│ MODELLE                    │
│                            │
│ - Decision Tree           │
│ - Random Forest           │
│ - XGBoost (stärkstes)     │
└──────────────┬─────────────┘
               ▼
```

---

## 9. Hyperparameter Tuning

```
┌──────────────────────────────┐
│ RANDOM SEARCH + CV         │
│                            │
│ - verschiedene Parameter   │
│ - TimeSeriesSplit         │
│ - ROC AUC Optimierung     │
└──────────────┬─────────────┘
               ▼
```

---

## 10. Bewertung pro Fold

```
┌──────────────────────────────┐
│ METRIKEN                  │
│                            │
│ - ROC AUC                │
│ - F1 Score               │
└──────────────┬─────────────┘
               ▼
```

---

## 11. Finales Modell (wichtigster Teil)

```
┌──────────────────────────────────────────────────────────────┐
│ EXPANDING WINDOW FEATURES                                   │
│ (immer mehr historische Daten über Zeit)                   │
└──────────────────────────────┬─────────────────────────────┘
                               ▼
```

### Idee:

* Modell lernt aus kompletter Vergangenheit
* kein Sampling mehr
* realistischste Simulation

---

## 12. Final Training

```
┌──────────────────────────────┐
│ XGBOOST FINAL MODEL         │
│ - optimierte Parameter      │
│ - TimeSeries CV tuning     │
└──────────────┬─────────────┘
               ▼
```

---

## 13. Final Test Evaluation

```
┌──────────────────────────────┐
│ TEST SET RESULT            │
│                            │
│ - ROC AUC                │
│ - F1 Score               │
│ - Classification Report   │
└──────────────────────────────┘
```

---

## Gesamtlogik (einfach erklärt)

```
Rohdaten
   ↓
Merge + Sortierung
   ↓
Feature Engineering
   ↓
Zeitbasierter Split
   ↓
Rolling Window Training
   ↓
Feature Cleaning + Encoding
   ↓
Modelle testen + vergleichen
   ↓
Bestes Modell wählen
   ↓
Finales Training (alles Daten)
   ↓
Finale Evaluation
```

---

## Kernidee der gesamten Pipeline

* Alles ist zeitlich korrekt aufgebaut (kein Leakage)
* Features basieren nur auf Vergangenheit
* Modelle werden realistisch getestet (Rolling Windows)
* XGBoost ist das finale starke Modell