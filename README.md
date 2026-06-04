# Tesina 5: Machine Learning per la Predizione del Rischio Cardiovascolare

Questo progetto implementa un sistema di predizione del rischio cardiovascolare a 10 anni basato sui dati clinici del **Framingham Heart Study**, confrontando diversi modelli di Machine Learning e analizzando le decisioni del modello ottimale tramite tecniche di **Explainable AI (XAI)**.

## Informazioni dello Studente
* **Studente**: Daniele Careri
* **Insegnamento**: I.A. per la Medicina
* **Traccia**: Tesina 5 - Machine Learning interpretabile per predizione del rischio cardiovascolare
* **Dataset**: https://www.kaggle.com/datasets/noeyislearning/framingham-heart-study

---

## Struttura della Pipeline (Notebook Sequenziali)

I file Jupyter Notebook si trovano nella cartella `Code/` e devono essere eseguiti nell'ordine numerato indicato:

### 1. [1_preelaborazione_dati.ipynb]
* **Obiettivo**: Caricamento, ispezione e pulizia iniziale dei dati clinici di Framingham.
* **Interventi**:
  - Individuazione e correzione di incongruenze cliniche (es. fumatori registrati con 0 sigarette al giorno, pressioni sistoliche inferiori alle diastoliche, farmaci per l'ipertensione assunti da pazienti considerati sani).
  - Gestione dei dati mancanti: esportazione del dataset parziale pre-imputazione per evitare target leakage e imputazione globale con KNN (stratificato per genere) per la visualizzazione generale.

### 2. [2_visualizzazione_dati.ipynb]
* **Obiettivo**: Analisi esplorativa dei dati e statistiche descrittive delle variabili cliniche.
* **Grafici**:
  - Matrice di correlazione lineare di Pearson.
  - Pairplot multivariato delle feature cliniche stratificato per la classe di rischio.
  - Diagramma di dispersione 3D per confrontare la distribuzione spaziale dei dati originali rispetto a quelli imputati con il KNN.

### 3. [3_separazione_dati.ipynb]
* **Obiettivo**: Suddivisione dei dati in **Training (60%)**, **Validation (20%)** e **Test Set (20%)** in modo stratificato.
* **Prevenzione del Data Leakage**:
  - Imputazione KNN, standardizzazione RobustScaler e riduzione di dimensionalità PCA calcolano i parametri (`fit`) solo sulla porzione di Train e applicano le trasformazioni (`transform`) su Val e Test.
  - Generazione di **8 combinazioni di dataset** basate su modalità di scaling (`Clean` vs `Normalized`) e bilanciamento/riduzione delle classi (`Normal` sbilanciato, `Aug` con SMOTE, `Pca` sbilanciato, `Aug+Pca` bilanciato con SMOTE e ridotto).

### 4. [4_addestramento_modelli.ipynb]
* **Obiettivo**: Addestramento sistematico dei modelli su tutte le 8 combinazioni (40 configurazioni totali).
* **Modelli**:
  - *Tuning completo* (via `RandomizedSearchCV` e Stratified 5-Fold CV): Logistic Regression, Random Forest e XGBoost.
  - *Addestramento diretto leggero* (senza hypertuning per massimizzare la velocità): SVM e LightGBM.
  - Salvataggio dei modelli in formato `.joblib`.

### 5. [5_confronto_modelli.ipynb]
* **Obiettivo**: Confronto delle metriche (Recall, F1-Score, ROC-AUC) sul Validation set e test finale sul Test set isolato.
* **Metrica Clinica**: Classifica basata su un punteggio medico ponderato (70% Recall + 30% F1-Score) per massimizzare la sensibilità diagnostica.
* **Risultati sul Test Set**: Valutazione del miglior modello (**Regressione Logistica** su `Clean + Aug`) con stampa del report di classificazione e heatmap Seaborn della Matrice di Confusione.
* **Confronto con il Baseline senza Preprocessing**: Addestramento del modello sul dataset grezzo gestito con sola rimozione dei nulli (`dropna`), evidenziando l'incremento di Recall del **+23.38%** grazie alla pipeline di preprocessing.

### 6. [6_interpretabilita_xai.ipynb]
* **Obiettivo**: Interpretazione post-hoc del modello raccomandato.
* **Metodi**:
  - *Permutation Feature Importance (PFI)* sul Test Set valutando la perdita di Recall.
  - *Partial Dependence Plots (PDP)* per esaminare gli effetti marginali non lineari di età, pressione, colesterolo e glucosio.
  - *SHAP (SHapley Additive exPlanations)*: Importanza globale delle feature e waterfall plot locali per spiegare le predizioni cliniche dei pazienti ad indice 2 e 36.
  - *Feature Selection*: Selezione automatica delle Top-5 variabili SHAP ed addestramento/valutazione di un modello clinico ridotto con matrici di confusione affiancate.

### 7. [7_metodi_ensemble.ipynb]
* **Obiettivo**: Implementazione e valutazione di metodi di Ensemble Learning sul Test Set isolato per verificare il potenziale miglioramento rispetto alla Regressione Logistica singola.
* **Metodi**:
  - *Voting Classifier (Soft Voting)*: Combina le probabilità predette da tutti i 5 modelli di base (Regressione Logistica, Random Forest, XGBoost, LightGBM, SVM).
  - *Stacking Classifier*: Utilizza i 5 modelli base come predittori di primo livello e una Regressione Logistica a pesi bilanciati come meta-classificatore finale.

---

## Risultati Chiave (Test Set)

| Metrica | Modello Completo (15 Feature) | Modello Ridotto (Top 5 Feature) | Voting Classifier (Soft) | Stacking Classifier | Modello Senza Preprocessing (dropna) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Recall (Sensibilità)** | **89.15%** | 62.02% | 27.13% | 15.50% | 65.77% |
| **Accuracy** | 47.64% | 64.98% | 73.94% | **79.25%** | 67.90% |
| **ROC-AUC** | 68.75% | **69.04%** | 65.96% | 61.53% | 72.37% |

* **Considerazione Clinica**: Il modello completo con preprocessing (Regressione Logistica su `Clean + Aug`) offre la massima affidabilità per lo screening (Recall all'89.15%), minimizzando i falsi negativi (soggetti a rischio cardiopatico non diagnosticati). Sebbene i metodi di ensemble (in particolare lo Stacking) migliorino drasticamente l'accuratezza globale (fino al 79.25%), essi causano un grave crollo della sensibilità clinica (Recall fino al 15.50%), rendendoli non idonei in questo specifico scenario di medicina preventiva senza un'attenta taratura delle soglie decisionali.

---

## Requisiti e Installazione

Il progetto utilizza l'ambiente virtuale pre-configurato `.venv`. Per riprodurlo manualmente, installare le dipendenze da `requirements.txt`:

```bash
pip install -r requirements.txt
```
