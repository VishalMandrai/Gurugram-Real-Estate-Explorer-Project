<img width="1917" height="178" alt="image" src="https://github.com/user-attachments/assets/94f9d469-d72e-4ee7-ac9b-8251fc5abdb8" />

> Interactive data exploration and price-prediction toolkit for residential properties in Gurugram (Gurgaon), India — built with Python (Pandas, NumPy, Visualization libraries), geospatial tools, and ML models.

---

## **PROJECT OVERVIEW**

This repository contains the code, data processing pipelines, exploratory analysis, visualization dashboards, machine learning models, and recommendation engine developed for the **Gurugram Real Estate Explorer** project. The goal is to create an **end-to-end reproducible workflow** that collects, cleans, analyzes, and models property listings to help stakeholders explore pricing patterns, locality effects, and predict property prices at a per-unit level.


<br>


## **`I.` Data Collection**

The project begins with a large-scale **data collection** phase:

* **Web scraping** was done using **Selenium** and **Beautiful Soup** from publicly available real-estate portals such as **Housing.com** and **99acres.com**.
* Collected a dataset of **7100+ residential property listings** across various sectors and localities of **Gurugram**.
* Apart from this, **OpenStreetMap GeoData** for Gurugram was used to extract **sector boundaries, centroids**, and to identify **nearby amenities** (schools, hospitals, banks, restaurants, etc.) for each locality — forming a key part of the geospatial analysis.

Following data collection, the dataset was cleaned and transformed using **Pandas** and **NumPy**, making it suitable for exploration, visualization, and modeling.


<br>


## **`II.` Data Cleaning & Feature Extraction**

The raw scraped dataset contained numerous property-level attributes, including:

* **Flat Type**, **Address**, **Price Density (₹/Sq.Ft)**, **Property Price**, **Built-up Area**, **Number of Bedrooms**, **Bathrooms**, **Balconies**, **Parking availability**, **Furnishing Status**, **Floor details**, **EMI estimate**, **Brokerage Cost**, **Society/Project name**, **Societal Amenities**, **Seller/Developer name**, and **Property Description**.

Each of these features was cleaned and standardized using **Pandas** and **NumPy** — handling missing values, normalizing numeric units, unifying formats (e.g., price units, area values), and parsing textual attributes (like floor or furnishing status) into structured data.


<br>


## **`III.` Feature Engineering**

From the cleaned data, several **derived features** were engineered to capture richer information:

* **Sector/Locality** extraction from property addresses.
* **Broader Locality** categorization (e.g., Golf Course Extension, Sohna Road, Dwarka Expressway, etc.).
* **Floor Number** and **Building Height** inferred from textual floor descriptions.
* **Parking Type and Count** split into **Open** and **Covered** parkings.
* Additional computed variables like **Price per Sq.Ft**, **EMI**, and **Amenity Counts/Scores**.

These engineered features provided stronger contextual understanding and improved model interpretability and performance.


<br>


## **`IV.` Exploratory Data Analysis (EDA)**

EDA forms the **core analytical section** of this project — a **deep and comprehensive exploration** of Gurugram’s residential property market.

* Conducted using **Pandas** and **NumPy** for data manipulation and a range of visualization libraries — **Matplotlib**, **Seaborn**, **Plotly (Graph Objects & Express)**, and **Folium** for map-based insights.
* The EDA explored **every aspect of the Gurugram Real Estate Market** — from **Flat Types** to **Sectors** to **Broader Localities**.
* **Price distribution analyses** were carried out across **Flats, Sectors, Localities, and Property Specifications**.
* Detailed examination of **property specifications** including:
  * **Furnishing Status** and its relationship with pricing trends.
  * **Floor and Building Height features** and their effect on price premiums.
  * **Parking Types** and distribution across different flat types (and correlation with price).
  * **Balconies, Bedrooms, Bathrooms counts** — distributions and influence on pricing.
* Created **dedicated sections** for every major property aspect — **pricing, location, specifications, premium-ness, and affordability**.
* Introduced a **4-tier Sector Segmentation** based on **Sector Aggregated Price Density**, classifying Gurugram sectors into affordability types.
* Performed **Statistical Analyses**, including **Pearson’s Correlation Tests**, to quantify relationships between variables wherever relevant.
* The EDA notebook is designed almost like a **Research Report**, filled with observations, insights, and actionable implications for **buyers, investors, builders, and policymakers**. ***` PLEASE CONSIDER READING! `***

The visual analysis employs **nearly every classical and modern visualization type**, including:

* Dynamic and Static Charts - **Pie charts, Line plots, Bar charts, Count plots, Treemaps, Sunbursts, Box plots, Violin plots, Histograms, KDE plots**, and **Interactive map visualizations** using **Folium**.

This section **transforms raw numbers into powerful visual narratives** — explaining not only what the data says but also **what it means** for the Gurugram housing market.


<br>


## **`V.` Final Preprocessing**

After the exploratory phase, a **final preprocessing pipeline** was developed to refine the dataset for modeling. The main objective was to retain only those features that contribute meaningfully to the **Price Prediction Model** — ensuring high accuracy and reliability with minimal redundancy.

Key preprocessing steps included:

* **Feature selection:** Removed all features that could lead to **data leakage** during model training and evaluation — such as *Price Density*, *EMI*, and *Brokerage*.
* **Data type correction:** Converted categorical and numerical columns into consistent formats compatible with scikit-learn pipelines.
* **Outlier removal:** Applied rule-based and quantile-based trimming of extreme values across price, area, and per-sqft metrics.
* **Imputation:** Handled missing values using **pattern-based and insight-driven imputation**, inspired by findings from EDA. Instead of using constant or mean-based imputations, values were filled logically according to observed trends and correlations between property features (e.g., filling missing parking counts based on flat type or furnishing level).

The outcome of this preprocessing step was a **high-quality dataset of over 7000 cleaned and complete property listings**, ready for feature encoding, scaling, and model selection.


<br>


## **`VI.` Model Selection & Outcomes**

The **Model Selection** phase **focused on developing** a robust, generalizable, and efficient **regression model for predicting property prices** across Gurugram’s real estate market.

#### **Target Transformation**

* The target variable, **Property Price**, exhibited strong **right skewness** due to a concentration of high-end luxury properties. To stabilize variance and improve model interpretability, a **log transformation** was applied.
* Similarly, the **Built-up Area** feature was also log-transformed to better align with linear model assumptions and reduce the influence of outliers.

#### **Modeling Pipeline**

* All modeling steps were handled through **scikit-learn Pipelines**, ensuring clean integration of preprocessing, transformation, and model training steps.
* The pipeline included **feature scaling, categorical encoding, and model fitting**.
* **Initial sanity checks were done with simple models** to validate pipeline functioning before advancing to full-scale model comparisons.

#### **Cross-validated Model Training**

* Performed multiple rounds of **cross-validation** using different model architectures and feature encoding strategies.
* Explored three encoding setups:
  1. **Label Encoding only**
  2. **Label + One-Hot Encoding (OHE)**
  3. **Label + OHE + Target Encoding** — applied especially for high-cardinality categorical features like **Sector/Locality**.

#### **Models Evaluated**

A broad range of regression models was tested at default and tuned configurations:
* **Linear Regression** (baseline)
* **Lasso and Ridge Regression**
* **Support Vector Regression (SVR)**
* **Bagging-based models:** Random Forest Regressor, Extra Trees Regressor
* **Boosting-based models:** Gradient Boosting Regressor, XGBoost Regressor

#### **Custom Feature Engineering within Pipeline**

**To enrich model inputs** without causing **Data Leakage**, a **Custom Transformation Class** was created to inject **historical Sector/Locality-level Median Price Densities** into the training data. **This transformer:**
* **Computes and stores** median price densities per sector/locality **during training**.
* **Imputes** the pre-calculated density records **to the test set** — ensuring that no future information leaks into model evaluation.

#### **Model Tuning & Optimization**

* Based on multiple rounds of experimentation, four top-performing models were shortlisted:
  * Random Forest Regressor
  * Extra Trees Regressor
  * Gradient Boosting Regressor
  * XGBoost Regressor
* Applied **Bayesian Optimization** using **`Optuna`** for **hyperparameter tuning** on these contenders, with their best-performing feature encodings.

#### **Final Model Selection**

* **XGBoost Regressor** emerged as the **final selected model** — providing a balance of lightness, robustness, and reliability while maintaining strong generalization performance.
* **Compared to Random Forest** (which was computationally heavier due to its large ensemble size), **XGBoost offered better runtime efficiency** with nearly equivalent accuracy.

#### **Final Model Results**

| Metric         | Value (approx.) | Notes                          |
| -------------- | --------------- | ------------------------------ |
| **MAE (Test)** | ₹30 Lacs        | On hold-out test set           |
| **R² (Test)**  | 0.92            | Excellent generalization       |
| **R² (Train)** | 0.95            | Slight but acceptable variance |

#### Deployment

The complete model pipeline — including preprocessing, custom transformers, and the trained **XGBoost model** — was serialized and saved as a `.pkl` file for use in the **Streamlit deployment app**.


---


## Society Recommendation Engine

To complement the predictive model, we built a **Society Recommendation Engine**, following a **content-based filtering** approach. The recommender suggests similar housing societies based on three primary factors:

1. **Property Pricing**
2. **Nearby Location Proximity**
3. **Society Amenities**

Three individual recommendation modules were developed — each focusing on one of the above aspects. These modules were then **combined using weighted priorities** to create a unified and balanced recommendation system.

The engine accepts a **Society Name** as input and returns the **Top 10 most similar societies**. All components were tested manually for accuracy, interpretability, and consistency before being integrated and deployed within the Streamlit App.


---


## Streamlit App – Gurugram Real Estate Explorer

The **Gurugram Real Estate Explorer App** is an interactive **Streamlit-based web application** designed for **home buyers, investors, developers, and policymakers**. It delivers a data-driven understanding of Gurugram’s real estate landscape through multiple modules:

* **Analytics & Insights Module:** Provides deep exploratory visualizations and insights into pricing, specifications, and sector-level patterns.
* **Sector Explorer Tab:** Enables users to explore any sector of Gurugram via maps, graphs, and key property metrics (price, built-up area, etc.).
* **Price Prediction Tool:** Predicts potential property price ranges based on input specifications using the trained ML model.
* **Society Recommendation Engine:** Suggests similar societies across Gurugram, along with available property listings and external links.

**Live App Link:** [Gurugram Real Estate Explorer](https://gurugram-real-estate-explorer.streamlit.app/)


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

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------------------------------------------

## Model Selection & Outcomes

In the **Model Selection Step**, the dataset was split into input and target features. The target feature, *Property Price*, exhibited a strong right skew due to the presence of luxury properties, and thus a **log transformation** was applied — similar treatment was done for the *Built-up Area* feature.

Model development was managed through **Pipelines**, enabling integrated feature scaling, transformation, and model training. Various encoding strategies were tested — **Label Encoding**, **Label + One-Hot Encoding**, and **Label + OHE + Target Encoding** (for high-cardinality features like Sector/Locality). Multiple regression algorithms were explored, including **Linear Regression, SVM, Random Forest, Extra Trees, Gradient Boosting, and XGBoost**.

A key innovation was introducing a **custom transformation class** to compute and attach *Historical Sector/Locality Median Price Density* features — ensuring no data leakage by restricting the transformation to training data and reusing precomputed statistics for the test set.

After several cross-validated rounds and performance comparisons, four top-performing models were shortlisted — Random Forest, Extra Trees, Gradient Boosting, and XGBoost. **Bayesian Optimization** was used to fine-tune hyperparameters for these models. Ultimately, **XGBoost Regressor** emerged as the most efficient and robust model, achieving:

* **MAE:** ~ ₹30 Lakh (on hold-out test set)
* **Test R²:** ~ 0.92
* **Train R²:** ~ 0.95

While Random Forest also performed well, XGBoost was selected for deployment due to its lighter structure and superior generalization. The final pipeline was serialized and saved as a Pickle file for deployment.

---

## Society Recommendation Engine

A **Content-based Society Recommendation Engine** was developed to assist users in discovering similar societies within Gurugram. The engine evaluates similarity across three key dimensions:

1. **Property Pricing Patterns**
2. **Nearby Geographic Locations**
3. **Societal Amenities**

Three separate similarity engines were designed for each factor, later integrated into a unified system with priority-based weighting. The engine takes a society name as input and returns the **Top 10 most similar societies**. It was manually tested for reliability and successfully deployed in the Streamlit App.
















