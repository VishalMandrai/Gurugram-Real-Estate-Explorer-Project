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


### **`Target Transformation`**

* The target variable, **Property Price**, exhibited strong **right skewness** due to a concentration of high-end luxury properties. To stabilize variance and improve model interpretability, a **log transformation** was applied.
* Similarly, the **Built-up Area** feature was also log-transformed to better align with linear model assumptions and reduce the influence of outliers.


### **`Modeling Pipeline`**

* All modeling steps were handled through **scikit-learn Pipelines**, ensuring clean integration of preprocessing, transformation, and model training steps.
* The pipeline included **feature scaling, categorical encoding, and model fitting**.
* **Initial sanity checks were done with simple models** to validate pipeline functioning before advancing to full-scale model comparisons.


### **`Cross-validated Model Training`**

* Performed multiple rounds of **cross-validation** using different model architectures and feature encoding strategies.
* Explored three encoding setups:
  1. **Label Encoding only**
  2. **Label + One-Hot Encoding (OHE)**
  3. **Label + OHE + Target Encoding** — applied especially for high-cardinality categorical features like **Sector/Locality**.


### **`Models Evaluated`**

- A broad range of regression models was tested at default and tuned configurations:
 * **Linear Regression** (baseline)
 * **Lasso and Ridge Regression**
 * **Support Vector Regression (SVR)**
 * **Bagging-based models:** Random Forest Regressor, Extra Trees Regressor
 * **Boosting-based models:** Gradient Boosting Regressor, XGBoost Regressor


### **`Custom Feature Engineering within Pipeline`**

* **To enrich model inputs** without causing **Data Leakage**, a **Custom Transformation Class** was created to inject **historical Sector/Locality-level Median Price Densities** into the training data. **This transformer:**
 * **Computes and stores** median price densities per sector/locality **during training**.
 * **Imputes** the pre-calculated density records **to the test set** — ensuring that no future information leaks into model evaluation.


### **`Model Tuning & Optimization`**

* Based on multiple rounds of experimentation, four top-performing models were shortlisted:
  * Random Forest Regressor
  * Extra Trees Regressor
  * Gradient Boosting Regressor
  * XGBoost Regressor
* Applied **Bayesian Optimization** using **`Optuna`** for **hyperparameter tuning** on these contenders, with their best-performing feature encodings.


### **`Final Model Selection`**

* **XGBoost Regressor** emerged as the **final selected model** — providing a balance of lightness, robustness, and reliability while maintaining strong generalization performance.
* **Compared to Random Forest** (which was computationally heavier due to its large ensemble size), **XGBoost offered better runtime efficiency** with nearly equivalent accuracy.

### **`Final Model Results`**
| Metric         | Value (approx.) | Notes                          |
| -------------- | --------------- | ------------------------------ |
| **MAE (Test)** | ~ ₹30 Lacs      | On hold-out test set           |
| **R² (Test)**  | 0.92            | Excellent generalization       |
| **R² (Train)** | 0.95            | Slight but acceptable variance |


### **`Deployment`**

* The complete model pipeline — including preprocessing, custom transformers, and the trained **XGBoost model** — was serialized and saved as a **`XGBoost_model_pipeline.pkl`** file for use in the **Streamlit deployment app**.


<br>


## **`VII.` Society Recommendation Engine**

**To complement the predictive model**, we developed a **Society Recommendation Engine**, based on **Content-based Filtering** Approach. The recommender suggests similar housing societies based on three primary factors:

1. **Property Pricing**
2. **Nearby Location Proximity**
3. **Society Amenities**

**Three individual recommendation modules were developed** — each focusing on one of the above aspects. These modules were then **combined using weighted priorities** to create a **unified and balanced recommendation system**.

The **engine accepts** a **Society Name** as input and **returns** the **Top 10 most similar societies**. All components were tested manually for accuracy, interpretability, and consistency before being integrated and deployed within the Streamlit App. 


<br>


## **`VIII.` Streamlit App – Gurugram Real Estate Explorer** [🔗](https://gurugram-real-estate-explorer.streamlit.app/)


The **Gurugram Real Estate Explorer App** is an interactive **Streamlit-based web application** designed for **home buyers, investors, developers, and policymakers**. It delivers a data-driven understanding of Gurugram’s real estate landscape through multiple modules:

* **Analytics & Insights Module:** Provides deep exploratory visualizations and insights into pricing, specifications, and sector-level patterns.
* **Sector Explorer Tab:** Enables users to explore any sector of Gurugram via maps, graphs, and key property metrics (price, built-up area, etc.).
* **Price Prediction Tool:** Predicts potential property price ranges based on input specifications using the trained ML model.
* **Society Recommendation Engine:** Suggests similar societies across Gurugram, along with available property listings and external links.

**Live App Link:** [Gurugram Real Estate Explorer](https://gurugram-real-estate-explorer.streamlit.app/)

<img width="1905" height="781" alt="image" src="https://github.com/user-attachments/assets/78fbea8e-248a-4711-bb45-75181ac34256" />


---
<br>

## **KEY FEATURES**
* Web scraping pipeline using **Selenium** and **Beautiful Soup**.
* **Data cleaning** and **robust preprocessing pipeline** (handles missing values, inconsistent formats, currencies, and units).
* **Geospatial visualizations** (dynamic plots using OpenStreetMap and folium of price density and amenities).
* **Sector- and locality-level aggregation and insights** (median price, amenity scores, parking, furnishing effects).
* **Feature engineering specific to real estate:** floor-rise premium, building age, area-normalized price density, amenity counts/scores, EMI estimate, etc.
* **Model training & evaluation** notebooks using scikit-learn / XGBoost with cross-validation and hyperparameter tuning.
* **Society Recommendation Engine** using **content-based filtering** to **recommend similar societies** based on Pricing, Location, and Amenities.
* A **Streamlit-based interactive application** for recruiters/employers to view analyses and test the predictor.


---
<br>


## **Repository Structure**

```
Gurugram-Real-Estate-Explorer/
├── 01. Data Gathering/                      # Contains scripts and modules for structured data gathering
    └── 01. Data Gathering.ipynb                 # Contains all scripts
    └── file_1(Links).csv                        # Contains Links to property listings
├── 02. 02. Data Cleaning/                   
    └── 02. Data Cleaning.ipynb                  # Contains all code scripts for cleaning and pre-processing the data
├── 03. Feature Engineering/                 # Feature Engineering notebook and all Gurugram Geo data extracted from OpenStreetMap
    └── All_Amenities_with_counts.txt
    └── sector_geo_centroid_data.csv
    └── sectors_geo_boundary_data.csv
├── 04. Exploratory Data Analysis (EDA)/     # EDA Notebooks and support files
    └── 04. Exploratory Data Analysis - Part 1.ipynb
    └── 04. Exploratory Data Analysis - Part 2.ipynb
    └── Facing_Directions.jpeg
├── 05. Final Preprocessing/                 # Notebooks for Final Preprocessing
    └── Final Preprocessing.ipynb
├── 06. Analytic Module & Deployment/        # Separate Repo for Streamlit App (Link in saved file)
├── 07. Model Selection & Training/          # Model Selection & Recommendation Engine Notebooks with saved model pipeline and support files
    └── Model Selection - Price Model.ipynb
    └── Recommendation_Engine_Building.ipynb
    └── XGBoost_model_pipeline.pkl
    └── REC_1_Pricing_similarity_matrix.csv
    └── REC_2_Facility_similarity_matrix.csv
    └── REC_3_Locations_similarity_matrix.csv
└── README.md                                # This file
```

---
<br>

## **Data** 

* The dataset was collected from publicly available listing sources and subsequently anonymized and cleaned before analysis. See `02. Data Cleaning/02.  Data Cleaning.ipynb` for details.
* **Important:** This repository intentionally omits or masks any personally identifiable information and any content that would violate the source websites' terms of service. If you need to reproduce the data collection step, consult the `02. Data Gathering/02. Data Gathering.ipynb` for the scraping notes and obey the source website's robots.txt and terms.

---
<br>


## **How Recruiters / Employers can Evaluate the Project**

1. Open the notebooks saved in separate numbered folders with intuitive names in the following order:
   * `01. Data Gathering/01. Data Gathering.ipynb` — web scraping and dataset creation
   * `02. Data Cleaning/02. Data Cleaning.ipynb` — data cleaning & preprocessing
   * `03. Feature Engineering/03. Feature Engineering.ipynb` — feature derivations and unpacking
   * `04. Exploratory Data Analysis (EDA)/04. Exploratory Data Analysis (EDA).ipynb` — EDA and insights
   * `07. Model Selection and Training/07. Model Selection and Training.ipynb` — models, validation, and evaluation
2. Go to the Separate repo for the Streamlit app (link attached in `06. Analytic Module & Deployment` folder) and try analytic modules or upload a hypothetical property specs to see predicted prices and try the society recommendation engine.
3. Review code quality, modularization, and unit-testable functions for Streamlit app by visiting its dedicated repo (link attached in `06. Analytic Module & Deployment` folder).
4. Review `04. Exploratory Data Analysis (EDA)/04. Exploratory Data Analysis (EDA).ipynb` and Analytic and Insights modules of the Streamlit app for the visual narrative and key takeaways.

---

## **Improvements & Future Work**

* Expand dataset (more listings, additional sectors and micro-localities) and re-train models.
* Incorporate external data sources: proximity to transit, planned infrastructure, zoning, and crime/safety indices.
* Build an automated ML pipeline (CI) to retrain models when new data arrives.
* Add unit tests and CI checks for data contracts.


---
<br>


## **Acknowledgments / References**

* **Built using open-source Python libraries**: Pandas, NumPy, Selenium, scikit-learn, XGBoost, GeoPandas, Folium/Leaflet, Streamlit, Matplotlib/Plotly.
* The exploratory analyses and modeling approaches draw on common real-estate analytics practices.


---
<br>


## **Contact Me**

If you'd like to discuss the project, invite me to an interview, or review any part of the code, contact:

**Vishal Mandrai**
* GitHub: `https://github.com/VishalMandrai`
* Email: `vishalm.nitt@gmail.com` [📧](mailto:vishalm.nitt@gmail.com)
* LinkedIn: [LinkedIn](https://www.linkedin.com/in/vishal-mandrai999/)
 

---
---
