========================================================================
TESINA DI MACHINE LEAVNING PER LA SALUTE / MEDICAL IA
========================================================================

§ TITOLO DELLA TESINA:
  Machine Learning per predizione del rischio cardiovascolare

§ NOME E COGNOME DELLO STUDENTE:
  Daniele Careri

§ TRACCIA SCELTA:
  Tesina 5 - Predizione del rischio cardiovascolare a 10 anni (Framingham Heart Study)
  e interpretazione clinica tramite tecniche di Explainable AI (XAI).

§ LINK AL DATASET:
  Framingham Heart Study Dataset (Kaggle):
  https://www.kaggle.com/datasets/noeyislearning/framingham-heart-study

§ ISTRUZIONI PER RIESEGUIRE I NOTEBOOK:
  I file notebook si trovano all'interno della cartella 'Code/' e devono essere
  eseguiti in ordine sequenziale (da 1 a 7) poiché ciascun
  notebook genera i dati, i modelli o le valutazioni necessarie per il successivo:

  1. 1_preelaborazione_dati.ipynb
     Caricamento dei dati originari, pulizia delle incoerenze fisiologiche ed esportazione
     dei dataset parziale e completo.
     
  2. 2_visualizzazione_dati.ipynb
     Analisi esplorativa grafica (EDA), heatmaps di correlazione e grafico a dispersione 3D.
     
  3. 3_separazione_dati.ipynb
     Creazione dello split stratificato disgiunto 60% Train / 20% Validation / 20% Test
     e trasformazione dei dataset per le 8 combinazioni per evitare target leakage.
     
  4. 4_addestramento_modelli.ipynb
     Ottimizzazione con RandomizedSearchCV per Logistic Regression, Random Forest e XGBoost,
     ed addestramento diretto leggero per SVM e LightGBM.
     
  5. 5_confronto_modelli.ipynb
     Valutazione comparativa sul Validation set, test finale del miglior modello (Logistic
     Regression Clean+Aug) sul Test set, confusione heatmap e confronto con il modello
     senza preprocessing (solo dropna).
     
  6. 6_interpretabilita_xai.ipynb
     Interpretazione del modello tramite PFI, PDP, SHAP (grafici beeswarm globale e spiegazioni
     locali con waterfall plot per i pazienti 2 e 36) e Feature Selection basata su SHAP.

  7. 7_metodi_ensemble.ipynb
     Valutazione di Voting Classifier e Stacking Classifier sul Test set per verificare il
     potenziale miglioramento prestazionale globale rispetto alla singola Regressione Logistica.

========================================================================
