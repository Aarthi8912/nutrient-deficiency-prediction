# Conference Paper & Final Year Submission Checklist

Use this to turn your 14-input AGAE-MTL project into a **final-year worthy** submission and a **conference-ready paper**.

---

## 1. Experiments to Run (Reproducibility)

Run in order and keep outputs:

| # | Script | Purpose |
|---|--------|--------|
| 1 | `prepare_enhanced_dataset.py` | 14-feature dataset, real BMI |
| 2 | `train_enhanced.py` | Proposed model (fix seed: 42) |
| 3 | `evaluate_enhanced.py` | Test metrics + figures |
| 4 | `run_baseline_enhanced.py` | LR, RF baselines |
| 5 | `compare_enhanced.py` | Bar chart: baselines vs proposed |
| 6 | `ablation_study.py` | Table: 6-in vs 14-in vs LR vs RF |

**For paper:** Report metrics from the same random seed (42) and 80/20 stratified split. Mention “single split” or add 5-fold CV (see below) for stronger claims.

---

## 2. Paper Structure (Suggested Sections)

- **Title:** Attention-Guided Denoising Autoencoder with Multi-Task Learning for Nutrient Deficiency Prediction Among Women and Adolescent Girls (15–49 Years)
- **Abstract:** Problem (nutrition risk in women 15–49), data (NFHS-5), method (AGAE-MTL, 14 inputs), main result (e.g. F1/AUC vs baselines), conclusion.
- **Introduction:** Motivation, gap (explainable prediction with diet/physiological factors), contribution (14-input model + attention + MTL).
- **Related Work:** Nutrition prediction, NFHS/DHS studies, multi-task learning, explainable AI in health.
- **Data:** NFHS-5 IR, population (women 15–49), 14 features (list them), risk label definition (BMI < 18.5 or anemia 1/2), train/test split, class balance.
- **Method:** Architecture (encoder → latent → attention → classifier + regressor + decoder), loss, training (optimizer, epochs, early stopping).
- **Experiments:** Baselines (LR, RF), metrics (Accuracy, Precision, Recall, F1, AUC-ROC), main result table (from `ablation_results.csv`), figures (ROC, confusion matrix, attention/feature importance).
- **Discussion:** Interpretation of attention (which factors matter), limitations (proxy label, single survey, no external validation).
- **Conclusion:** Summary and future work (e.g. biomarkers, other surveys).
- **References:** NFHS-5 report, DHS methodology, PyTorch/sklearn if required by venue.
- **Appendix (optional):** Hyperparameters, extra plots, ethical/clinical disclaimer.

---

## 3. Tables and Figures for the Paper

- **Table 1 – Dataset:** Sample size, number of features, risk prevalence (%), train/test sizes.
- **Table 2 – Comparison:** From `ablation_results.csv`: Model | Accuracy | Recall | Precision | F1 | AUC. Include: Logistic Regression, Random Forest, AGAE-MTL (6 inputs), AGAE-MTL (14 inputs).
- **Figure 1 – Pipeline:** Block diagram: Input (14) → Encoder → Latent → Attention → Classifier / Regressor / Decoder.
- **Figure 2 – ROC:** Use `roc_curve_enhanced.png` (AGAE-MTL 14-in). Optionally overlay LR and RF ROC.
- **Figure 3 – Confusion matrix:** Use `confusion_matrix_enhanced.png`.
- **Figure 4 – Feature / latent importance:** Use `attention_enhanced.png` and briefly interpret (e.g. BMI, anemia, diet).

---

## 4. Strengthening for Conference Review

- **Cross-validation:** Add 5-fold CV; report mean ± std for F1 and AUC. Use same preprocessing and seed per fold.
- **Statistical testing:** Compare proposed vs best baseline (e.g. McNemar or bootstrap) and report p-value.
- **Ablation:** If you add “w/o attention” or “w/o MTL” variants (by changing the model/training), add one row each to Table 2 and one short paragraph in Experiments.
- **Limitations:** Clearly state: proxy label (no gold-standard deficiency), NFHS-5 subsample (e.g. diet module), no external cohort validation.
- **Ethics / disclaimer:** “For research and decision support only; not a medical device; not a substitute for clinical diagnosis.”

---

## 5. Final Year Project Specifics

- **Report/thesis:** Use the same sections as above; add “System design” (data flow, app interface) and “Implementation” (PyTorch, Streamlit, sklearn).
- **Demo:** Run `streamlit run app_enhanced.py` and show: input form (14 fields) → prediction → risk level → top factors → PDF download.
- **Code submission:** Zip repo or share Git; include README with run order and `requirements.txt`.
- **Reproducibility:** Document Python version (e.g. 3.8+), `pip install -r requirements.txt`, and “Run steps 1–6 in README.”

---

## 6. Quick Reference – 14 Inputs

| # | Feature | Source (NFHS-5) |
|---|---------|------------------|
| 1 | Age | v012 |
| 2 | BMI | v445 (÷100) |
| 3 | Anemia | v457 |
| 4 | Education | v106 |
| 5 | Wealth | v190 |
| 6 | Residence | v025 |
| 7 | Pregnant | v213 |
| 8 | Breastfeeding | v404 |
| 9 | Insurance | v481 |
| 10 | Milk | v414n |
| 11 | Eggs | v414e |
| 12 | Fruit | v414p |
| 13 | Pulses | v414s |
| 14 | Curd | v414v |

---

## 7. Suggested Venues (Health / ML)

- Health informatics: IEEE BHI, MEDINFO, AMIA.
- ML for health: ML4H workshop (NeurIPS/ICML), CHIL.
- Regional: ICBB, INDIACom, or university conference in public health / CS.

Adapt abstract and emphasis (clinical vs methodological) to each venue’s scope.
