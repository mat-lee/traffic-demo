<div align="center">
  
# 🚦 Traffic Accident Severity Predictor

**Interactive Streamlit app** that estimates traffic accident **severity (1–4)** from tunable inputs: location, weather, road context, time, and a short description.

[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-ff4b4b?logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![scikit‑learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-1f6feb)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](#license)

<img src="docs/screenshot.png" alt="App screenshot" width="800" />
<br/>
<a href="#quickstart">Quickstart</a> • <a href="#features">Features</a> • <a href="#model-artifacts">Model Artifacts</a>

</div>

---

## Features
- 🗺️ **Map click** to set **latitude/longitude**
- 🌤️ Sliders/toggles for **weather, road features, and time**
- 📝 Optional **free‑text description** (vectorized under the hood)
- ⚡ Instant prediction with a clear label: <kbd>1</kbd> Minor · <kbd>2</kbd> Moderate · <kbd>3</kbd> Severe · <kbd>4</kbd> Very Severe

---

## Quickstart

```bash
# Clone and enter the project
git clone <your-repo-url>
cd <your-repo-folder>

# (Optional) create & activate a virtual environment
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

> Model pickle files (`*.pkl`) should live next to `app.py`. See **Model Artifacts**.

---

## Requirements
Minimal `requirements.txt`:
```
streamlit
pandas
numpy
scikit-learn
xgboost
joblib
folium
streamlit-folium
```

---

## Model Artifacts
| File | Purpose |
|---|---|
| `clf.pkl` | Classifier (e.g., XGBoost) |
| `tfidf.pkl` | `TfidfVectorizer` for text description |
| `pca.pkl` | `TruncatedSVD` (LSA), if used |
| `sc.pkl` | `StandardScaler` for numeric features |
| `le.pkl` | `LabelEncoder` mapping `{1,2,3,4}` |

> If you change features or SVD dimensions during training, update the app accordingly.

---

## Structure
```
.
├── app.py
├── requirements.txt
├── clf.pkl
├── tfidf.pkl
├── pca.pkl
├── sc.pkl
└── le.pkl
```

---

## Data
Built on the **US Accidents (2016–2023)** dataset. See Kaggle: https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents

---

## License
MIT — see [LICENSE](LICENSE).
