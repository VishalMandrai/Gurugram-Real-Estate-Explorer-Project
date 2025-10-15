# Gurugram Real Estate Explorer

> Interactive data exploration and price-prediction toolkit for residential properties in Gurugram (Gurgaon), India — built with Python (Pandas, NumPy), geospatial tools, and ML models.

---

## Project overview

This repository contains the code, data processing pipelines, exploratory analysis, visualization dashboards, and machine learning models developed for the **Gurugram Real Estate Explorer** project. The goal is to create an end-to-end reproducible workflow that collects, cleans, analyzes and models property listings to help stakeholders explore pricing patterns, locality effects, and to predict property prices at a per-unit level.

### Data collection

The project begins with a large-scale **data collection** phase:

* **Web scraping** was done using **Selenium** and **Beautiful Soup** from publicly available real-estate portals such as **Housing.com** and **99acres.com**.
* Collected a dataset of **7100+ residential property listings** across various sectors and localities of **Gurugram**.
* Apart from this, **OpenStreetMap GeoData** for Gurugram was used to extract **sector boundaries, centroids**, and to identify **nearby amenities** (schools, hospitals, banks, restaurants, etc.) for each locality — forming a key part of the geospatial analysis.

Following data collection, the dataset was cleaned and transformed using **Pandas** and **NumPy**, making it suitable for exploration, visualization, and modeling.

### Data cleaning & feature extraction

The raw scraped dataset contained numerous property-level attributes, including:

* **Flat Type**, **Address**, **Price Density (₹/Sq.Ft)**, **Property Price**, **Built-up Area**, **Number of Bedrooms**, **Bathrooms**, **Balconies**, **Parking availability**, **Furnishing Status**, **Floor details**, **EMI estimate**, **Brokerage Cost**, **Society/Project name**, **Societal Amenities**, **Seller/Developer name**, and **Property Description**.

Each of these features was cleaned and standardized using **Pandas** and **NumPy** — handling missing values, normalizing numeric units, unifying formats (e.g., price units, area values), and parsing textual attributes (like floor or furnishing status) into structured data.

### Feature engineering

From the cleaned data, several **derived features** were engineered to capture richer information:

* **Sector/Locality** extraction from property addresses.
* **Broader Locality** categorization (e.g., Golf Course Extension, Sohna Road, Dwarka Expressway, etc.).
* **Floor Number** and **Building Height** inferred from textual floor descriptions.
* **Parking Type and Count** split into **Open** and **Covered** parkings.
* Additional computed variables like **Price per Sq.Ft**, **EMI**, and **Amenity Counts/Scores**.

These engineered features provided stronger contextual understanding and improved model interpretability and performance.

### Exploratory Data Analysis (EDA)

EDA forms the **core analytical section** of this project — a deep and comprehensive exploration of Gurugram’s residential property market.

* Conducted using **Pandas** and **NumPy** for data manipulation and a range of visualization libraries — **Matplotlib**, **Seaborn**, **Plotly (Graph Objects & Express)**, and **Folium** for map-based insights.
* The EDA explored **every aspect of the Gurugram Real Estate Market** — from **Flat Types** to **Sectors** to **Broader Localities**.
* **Price distribution analyses** were carried out across **Flats, Sectors, Localities, and Property Specifications**.
* Detailed examination of **property specifications** including:

  * **Furnishing Status** and its relationship with pricing trends.
  * **Floor and Building Height features** and their effect on price premiums.
  * **Parking Types** and distribution across different flat types (and correlation with price).
  * **Balconies, Bedrooms, Bathrooms counts** — distributions and influence on pricing.
* Created **dedicated sections** for every major property aspect — **pricing, location, specifications, premium-ness, and affordability**.
* Introduced a **4-tier sector segmentation** based on **Price Density**, classifying Gurugram sectors into affordability types.
* Performed **statistical analyses**, including **Pearson’s Correlation Tests**, to quantify relationships between variables wherever relevant.
* The EDA notebook is designed almost like a **research report**, filled with observations, insights, and actionable implications for **buyers, investors, builders, and policymakers**.

The visual analysis employs **nearly every classical and modern visualization type**, including:

* **Pie charts, Line plots, Bar charts, Count plots, Treemaps, Sunbursts, Box plots, Violin plots, Histograms, KDE plots**, and **Interactive map visualizations** using **Folium**.

This section transforms raw numbers into powerful visual narratives — explaining not only what the data says but also **what it means** for the Gurugram housing market.

Key deliverables:

* A cleaned, analysis-ready dataset derived from scraped real-estate listings.
* EDA notebooks highlighting supply, pricing density, floor-rise premium, and sector-level effects.
* An interactive Streamlit app for exploring properties, maps and prediction results.
* One or more trained regression models (with saved pipelines) to estimate price / price-per-sqft.
* Documentation (this README) and reproducible instructions to run the project locally.

---

## Features

* Automated web scraping pipeline using **Selenium** and **Beautiful Soup**.
* Data cleaning and robust preprocessing pipeline (handles missing values, inconsistent formats, currencies and units).
* Geospatial visualizations (plots on OpenStreetMap, choropleth/heatmaps of price density and amenities).
* Sector- and locality-level aggregation and insights (median price, amenity scores, parking, furnishing effects).
* Feature engineering specific to real-estate: floor-rise premium, building age, area-normalized price density, amenity counts/scores, EMI estimate, etc.
* Model training & evaluation notebooks using scikit-learn / XGBoost with cross-validation and hyperparameter tuning.
* A Streamlit-based interactive explorer for recruiters/employers to view analyses and test the predictor.

---

## Repository structure

```
Gurugram-Real-Estate-Explorer/
├── data/                      # Raw and processed datasets (do not store sensitive data here)
├── notebooks/                 # Jupyter notebooks: EDA, feature engineering, modeling
├── src/                       # Reusable modules (data cleaning, features, modeling utils)
├── models/                    # Saved model artifacts and preprocessing pipelines (.pkl/.joblib)
├── app/                       # Streamlit app and supporting scripts
├── reports/                   # Plots, figures and summary tables
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── LICENSE
```

---

## Quick demo

1. Browse the `notebooks/` folder for exploratory analysis and model training steps.
2. Run the Streamlit app in `app/` to interactively explore the maps and try price predictions.

---

## Getting started (run locally)

> These commands assume you have Python 3.9+ installed.

1. Clone the repository

```bash
git clone https://github.com/<your-username>/Gurugram-Real-Estate-Explorer.git
cd Gurugram-Real-Estate-Explorer
```

2. Create a virtual environment and install dependencies

```bash
python -m venv venv
# Unix / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

3. (Optional) If you have the raw data files, place them in `data/raw/` and run the cleaning notebook or `src` scripts to regenerate processed data.

4. Run the Streamlit app (from project root)

```bash
streamlit run app/app.py
```

5. Open `notebooks/` in JupyterLab / Jupyter Notebook to reproduce the EDA and model training steps.

---

## Data

* The dataset was collected from publicly available listing sources and subsequently anonymised and cleaned before analysis. See `notebooks/00-data-cleaning.ipynb` for details.
* **Important:** This repository intentionally omits or masks any personally identifiable information and any content that would violate the source websites' terms of service. If you need to reproduce the data collection step, consult the `notebooks/` for the scraping notes and obey the source website's robots.txt and terms.

---

## Modeling summary

* Models used: baseline linear regression, tree-based regressors (RandomForest / XGBoost), and stacked pipelines.
* Cross-validation strategy and metrics: K-Fold CV with MAE, RMSE and R² reported. See `notebooks/03-modeling.ipynb` for full results.
* The final model pipeline (preprocessor + model) is saved under `models/` as `best_pipeline.joblib`.

Sample model performance (example):

* MAE: ~₹X,XXX (on hold-out test set)
* RMSE: ~₹X,XXX
* R²: ~0.XX

(Replace the above with your final model metrics before publishing.)

---

## Notes on feature engineering & leakage

* Carefully engineered features such as locality median price density and amenities counts contributed strongly to model performance.
* **Caution:** Some aggregated locality-level features can introduce leakage if computed using target data that overlaps the train/test split. Notebooks show the correct way to compute aggregation features within cross-validation to avoid leakage.

---

## How recruiters / employers can evaluate the project

1. Open the notebooks in `notebooks/` in the following order:

   * `00-data-collection.ipynb` — web scraping and dataset creation
   * `01-data-cleaning.ipynb` — data cleaning & preprocessing
   * `02-exploratory-analysis.ipynb` — EDA and insights
   * `03-feature-engineering.ipynb` — feature derivations and justifications
   * `04-modeling.ipynb` — models, validation and evaluation
2. Run the Streamlit app in `app/` and try sample localities or upload a hypothetical property to see predicted prices.
3. Inspect `src/` to review code quality, modularization, and unit-testable functions.
4. Review `reports/` for the visual narrative and key takeaways.

---

## Improvements & future work

* Expand dataset (more listings, additional sectors and micro-localities) and re-train models.
* Incorporate external data sources: proximity to transit, planned infrastructure, zoning, and crime/safety indices.
* Build an automated ML pipeline (CI) to retrain models when new data arrives.
* Add unit tests and CI checks for data contracts.

---

## Licensing

This project is released under the MIT License. See `LICENSE` for details.

---

## Acknowledgements / References

* Built using open-source Python libraries: Pandas, NumPy, scikit-learn, XGBoost, GeoPandas, Folium/Leaflet, Streamlit, Matplotlib/Plotly.
* The exploratory analyses and modelling approaches draw on common real-estate analytics practices.

---

## Contact

If you'd like to discuss the project, invite me to an interview, or review any part of the code, contact:

**Vishal Mandrai** — GitHub: `https://github.com/<your-username>` — Email: `your.email@example.com`

---

### Quick checklist before publishing

* [ ] Replace placeholder metrics with final numbers from `03-modeling.ipynb`.
* [ ] Ensure `requirements.txt` is up-to-date.
* [ ] Remove any raw files with PII from `data/` and add them to `.gitignore`.
* [ ] Add screenshots/gif of the Streamlit app to this README (optional).

*Happy coding — feel free to ask me to tailor this README for job applications or to produce a shorter one-page project summary.*
