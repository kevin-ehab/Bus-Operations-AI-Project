# 🚌 Public Bus Operations ML

A machine learning project for predicting **bus delay levels** and **trip duration** using operational, route, traffic, weather, and driver-related data.

The project uses two supervised learning tasks:

* **Classification:** Predict whether a trip is *On Time*, experiencing a *Moderate Delay*, or a *Severe Delay*.
* **Regression:** Predict the expected **trip duration in minutes**.

## 📌 Project Overview

The goal of this project is to explore how machine learning can be applied to public bus operations to support scheduling, delay prediction, and trip-duration estimation.

Two different models were selected after comparing multiple algorithms using **5-fold GridSearchCV**:

| Task                     | Selected Model      | Best Parameters                               |
| ------------------------ | ------------------- | --------------------------------------------- |
| Delay Classification     | SVC with RBF kernel | `C=1`, `gamma=0.1`, `class_weight="balanced"` |
| Trip Duration Regression | KNN Regressor       | `n_neighbors=9`, `weights="distance"`, `p=2`  |

---

## 📊 Dataset

The project uses:

```text
Public_Bus_Operations_Multi_Target.csv
```

The dataset contains information about bus trips, including:

* `traffic_level`
* `weather_condition`
* `bus_type`
* `route_distance_km`
* `stops_count`
* `passenger_count`
* `driver_experience_years`
* `scheduled_start_hour`
* `delay_level`
* `trip_duration_minutes`

### Data Quality

The initial dataset contained:

* Missing values in several features
* **150 duplicated rows**
* Categorical target values that needed to be converted into numerical labels

Duplicate rows were removed before modeling.

---

## 🔧 Preprocessing

The features were divided into three groups.

### Ordinal Feature

```python
traffic_level
```

Traffic levels were encoded according to their natural order:

```text
Low → Moderate → High → Severe
```

Missing values were filled using the most frequent value.

### Nominal Features

```python
weather_condition
bus_type
```

These were processed using:

* Most-frequent imputation
* One-hot encoding `drop="first", handle_unknown="ignore"`

### Numerical Features

```python
route_distance_km
stops_count
passenger_count
driver_experience_years
scheduled_start_hour
```

Missing values were replaced using the median and the features were standardized using `StandardScaler`.

All preprocessing was implemented inside a scikit-learn `ColumnTransformer` and `Pipeline`.

---

## 🧠 Classification

### Objective

Predict the delay category of a bus trip:

```text
0 → On Time
1 → Moderate Delay
2 → Severe Delay
```

### Models Compared

* Logistic Regression
* Support Vector Classifier (SVC)

`GridSearchCV` with **5-fold cross-validation** was used to select the best hyperparameters. The search was optimized using **macro F1-score**.

### Cross-Validation Results

| Model               | Mean CV F1-macro |
| ------------------- | ---------------: |
| Logistic Regression |           0.6718 |
| **SVC**             |       **0.7007** |

The SVC achieved the highest cross-validation F1 score and was therefore selected as the final classification model.

### Final SVC

```text
Kernel: RBF
C: 1
Gamma: 0.1
Class weight: balanced
```

### Test Results

The model was evaluated on **1,970 test trips**.

| Metric          |      Score |
| --------------- | ---------: |
| Accuracy        | **71.83%** |
| Macro Precision | **71.59%** |
| Macro Recall    | **71.82%** |
| Macro F1        | **71.68%** |

Classification performance by class:

| Delay Level    | Precision | Recall |   F1 |
| -------------- | --------: | -----: | ---: |
| On Time        |      0.76 |   0.80 | 0.78 |
| Moderate Delay |      0.59 |   0.56 | 0.58 |
| Severe Delay   |      0.80 |   0.79 | 0.79 |

The model performed best on **Severe Delay** and **On Time** trips, while **Moderate Delay** was more difficult to distinguish from the other classes.

---

## ⏱️ Regression

### Objective

Predict:

```text
trip_duration_minutes
```

### Models Compared

* Linear Regression
* KNN Regressor

The models were compared using **5-fold GridSearchCV**, optimized for `R²`.

### Cross-Validation Results

| Model             | Mean CV R² |
| ----------------- | ---------: |
| Linear Regression |     0.9237 |
| **KNN Regressor** | **0.9266** |

KNN achieved the highest cross-validation R² and was selected as the final regression model.

### Final KNN

```text
n_neighbors: 9
weights: distance
p: 2
```

### Test Results

| Metric |             Score |
| ------ | ----------------: |
| MAE    | **6.671 minutes** |
| RMSE   | **8.552 minutes** |
| R²     |         **0.936** |

The model's typical prediction error is approximately **6.67 minutes**, which is relatively small compared with typical trip durations. The RMSE of **8.55 minutes** indicates that some predictions have larger errors.

The `R²` score of **0.936** indicates that the model explains a large proportion of the variation in trip duration on the test set.

The actual-vs-predicted plot also shows that predictions generally follow the actual trip durations, although errors become more noticeable for some longer trips.

---

## 🔎 Exploratory Data Analysis

Several relationships were investigated before modeling.

### Weather and Trip Duration

Rainy and foggy conditions tended to have greater average trip durations than windy and clear conditions.

### Route Distance and Trip Duration

A positive relationship was observed between route distance and trip duration: as route distance increases, trip duration generally increases as well.

### Trip Duration Distribution

Most trip durations were between approximately **25 and 50 minutes**.

The dataset also contained some longer trips, with a maximum recorded duration of **180 minutes**.

---

## 📁 Project Structure

```text
.
├── Public_Bus_Operations_Multi_Target.csv
├── New_Bus_Trips.csv
├── EYOUTH-30811012400538_bus_operations_project.ipynb
├── EYOUTH-30811012400538_best_bus_delay_classifier.joblib
├── EYOUTH-30811012400538_best_bus_duration_regressor.joblib
└── README.md
```

### Important Files

* `Public_Bus_Operations_Multi_Target.csv` — original bus operations dataset.
* `New_Bus_Trips.csv` — new trips used to demonstrate predictions.
* `EYOUTH-30811012400538_bus_operations_project.ipynb` — complete EDA, preprocessing, model training, tuning, evaluation, and prediction workflow.
* `EYOUTH-30811012400538_best_bus_delay_classifier.joblib` — trained SVC classification pipeline.
* `EYOUTH-30811012400538_best_bus_duration_regressor.joblib` — trained KNN regression pipeline.

---

## 📦 Requirements

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Joblib**
* **Jupyter Notebook / Google Colab**
  
Install the required Python libraries with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

---

## 🔮 Making Predictions

The trained pipelines can be loaded using `joblib`:

```python
import joblib

classification_model = joblib.load(
    "EYOUTH-30811012400538_best_bus_delay_classifier.joblib"
)

regression_model = joblib.load(
    "EYOUTH-30811012400538_best_bus_duration_regressor.joblib"
)
```

The models include the preprocessing steps, so new input data can be passed directly to the pipelines using the same transformations used during training.

Example:

```python
delay_prediction = classification_model.predict(new_data)
duration_prediction = regression_model.predict(new_data)
```

---

## 📋 Example Predictions

The notebook also tests the trained models on 10 new bus trips.

The predictions include:

```text
trip_id
predicted_delay_level
predicted_trip_duration_minutes
```

Example output:

| Trip   | Predicted Delay | Predicted Duration |
| ------ | --------------- | -----------------: |
| NEW001 | On Time         |          22.08 min |
| NEW002 | Severe Delay    |          96.60 min |
| NEW003 | Severe Delay    |         170.73 min |
| NEW004 | On Time         |          45.11 min |
| NEW005 | Severe Delay    |          65.41 min |
| NEW006 | Severe Delay    |         159.34 min |
| NEW007 | On Time         |          21.94 min |
| NEW008 | Moderate Delay  |          78.57 min |
| NEW009 | Severe Delay    |          61.45 min |
| NEW010 | Severe Delay    |         161.29 min |

---

## ⚠️ Limitations

* The classification model has noticeably weaker performance on **Moderate Delay** trips.
* The models were evaluated using a single held-out test split.
* Model performance may differ on data from different routes, cities, seasons, or transportation systems.
* Predictions should be treated as estimates rather than guaranteed outcomes.
* Real-world deployment would require additional validation and monitoring.

---

## 🔮 Future Improvements

Potential improvements include:

* Testing additional machine learning algorithms such as Random Forest, Gradient Boosting, and XGBoost.
* Performing more extensive hyperparameter tuning.
* Using repeated cross-validation for more robust evaluation.
* Investigating feature importance and model interpretability.
* Improving classification of the Moderate Delay class.
* Testing the models on an independent dataset.
* Adding real-time traffic and weather data.
* Building a dashboard or web application for live predictions.

---

## 📄 Disclaimer

This project is an educational machine learning project demonstrating how supervised learning can be applied to public transportation data. The predictions should not be considered guaranteed real-world outcomes.

---

## 👨‍💻 Author

**Kevin Ehab**

Built as a machine learning project focused on applying data preprocessing, exploratory data analysis, model comparison, hyperparameter tuning, and evaluation to a real-world-style transportation problem.
