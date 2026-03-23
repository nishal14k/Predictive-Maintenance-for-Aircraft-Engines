# Predictive-Maintenance-for-Aircraft-Engines

This project implements a predictive maintenance system using NASA's Turbofan Engine Degradation Simulation dataset. The system includes unsupervised clustering, supervised classification and regression, and a final risk scoring system for proactive maintenance alerts.

Two versions of the system were developed using different machine learning strategies:

- nasa1.py: Uses KMeans clustering and a Random Forest classifier.  
- nasa2.py: Uses Agglomerative clustering and an SVM classifier with SMOTE.

---

Features

Data Preprocessing
- Load and clean data using predefined column names.
- Normalize operational settings and sensor readings.

Phase 1: Clustering
- nasa1.py: KMeans clustering for engine health stage grouping.  
- nasa2.py: Agglomerative clustering for hierarchical grouping.  
- Visualized with PCA-reduced scatter plots and cluster-based sensor trends.

Phase 2: Classification
- nasa1.py: Random Forest classifier to predict engine degradation stages.  
- nasa2.py: SVM classifier trained with SMOTE for class balancing.  
- Performance evaluated via accuracy, classification report, and confusion matrix.

Phase 3: Regression Model (Time-to-Next-Failure Prediction)
- Predict the time (in cycles) remaining until the next degradation stage (e.g., Stage 1 → 2, Stage 3 → 4).
- Models used:
  - nasa1.py: Ridge Regression  
  - nasa2.py: Random Forest Regression  
- Other potential models: Support Vector Regression (SVR)
- Evaluation Metrics: MAE, RMSE, R² Score

Phase 4: Risk Score & Alerts
- Combines stage classification and RUL estimation.  
- Calculates normalized risk scores.  
- Triggers maintenance alerts if the risk score exceeds a threshold (0.7).  
- Visualizes risk scores over time per unit.

---

Outputs
- Visualizations for clusters, stage trends, risk scores.  
- time_to_next_stage_output.csv: Contains time-based predictions per unit.  
- Printed alerts and model evaluation metrics.

