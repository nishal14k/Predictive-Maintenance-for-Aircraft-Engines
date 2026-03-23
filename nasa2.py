import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, r2_score, mean_absolute_error
from imblearn.over_sampling import SMOTE
from sklearn.cluster import AgglomerativeClustering
from sklearn.svm import SVC

def loaddata(filename):
    column_names = ['unit', 'time'] + [f'op_setting_{i+1}' for i in range(3)] + [f'sensor_{i+1}' for i in range(21)]
    df = pd.read_csv(filename, sep=' ', header=None)
    df.dropna(axis=1, how='all', inplace=True)
    df.columns = column_names
    return df

def preprocess(df):
    df = df.copy()
    op_settings = df[[f'op_setting_{i+1}' for i in range(3)]]
    sensor_data = df[[f'sensor_{i+1}' for i in range(21)]]
    features = pd.concat([op_settings, sensor_data], axis=1)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)
    return scaled_data, features.columns.tolist()

def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def agglomerative(data, k):
    model = AgglomerativeClustering(n_clusters=k, linkage='ward')
    clusters = model.fit_predict(data)
    return clusters

def visualize(data, clusters, k):
    pca = PCA(n_components=2, random_state=30)
    reduced_data = pca.fit_transform(data)
    plt.figure(figsize=(8, 6))
    for i in range(k):
        cluster_points = reduced_data[clusters == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, label=f'Cluster {i}')
    plt.title('Engine Degradation Clusters (PCA Reduced)')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot(df, clusters, sensor_names):
    df = df.copy()
    df['cluster'] = clusters
    for sensor in sensor_names:
        plt.figure(figsize=(6, 4))
        for cluster in np.unique(clusters):
            cluster_data = df[df['cluster'] == cluster]
            mean_series = cluster_data.groupby('time')[sensor].mean()
            plt.plot(mean_series, label=f'Cluster {cluster}')
        plt.title(f'Trend of {sensor} by Cluster')
        plt.xlabel('Time')
        plt.ylabel(sensor)
        plt.legend()
        plt.tight_layout()
        plt.show()

def trainmodel(X, y):
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    print("Before SMOTE:")
    print(pd.Series(ytrain).value_counts().to_string())
    smote = SMOTE(k_neighbors=3, sampling_strategy={0: 3767, 1: 2642, 2: 4296, 3: 1000, 4: 2000})
    Xtrainresampled, ytrainresampled = smote.fit_resample(Xtrain, ytrain)
    print("\nAfter SMOTE:")
    print(pd.Series(ytrainresampled).value_counts().to_string())
    model = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        decision_function_shape='ovr',
        probability=True,
    )
    model.fit(Xtrainresampled, ytrainresampled)
    ytrainpred = model.predict(Xtrainresampled)
    trainacc = accuracy_score(ytrainresampled, ytrainpred)
    print(f"\nTraining Accuracy: {trainacc:.4f}")
    ytestpred = model.predict(Xtest)
    testacc = accuracy_score(ytest, ytestpred)
    print(f"Testing Accuracy: {testacc:.4f}\n")
    print("Classification Report (SVM):")
    print(classification_report(ytest, ytestpred))
    cm = confusion_matrix(ytest, ytestpred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title('Confusion Matrix (SVM)')
    plt.xlabel('Predicted Stage')
    plt.ylabel('True Stage')
    plt.tight_layout()
    plt.show()

    return model

def regressionmodel(df, top_sensors):
    df = df[df['time_to_next_stage'] > 0].copy()
    df['stage'] = df['stage'].astype(int)
    X = df[top_sensors + ['stage']]
    y = df['time_to_next_stage']
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        min_samples_split=4,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    mse = mean_squared_error(ytest, ypred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(ytest, ypred)
    r2 = r2_score(ytest, ypred)
    print("\nRUL Regression Report(RF):")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.2f}")

    return model

def timetonext(df, stage_labels):
    df = df.copy()
    df['stage'] = stage_labels
    df['time_to_next_stage'] = -1
    for unit in df['unit'].unique():
        unit_df = df[df['unit'] == unit]
        times = unit_df['time'].values
        stages = unit_df['stage'].values
        for idx in range(len(unit_df)):
            current_stage = stages[idx]
            for j in range(idx + 1, len(stages)):
               if stages[j] > current_stage:
                    df.loc[unit_df.index[idx], 'time_to_next_stage'] = times[j] - times[idx]
                    break
    return df

if __name__ == "__main__":
    filename = "/content/train_FD001[2].txt"
    df = loaddata(filename)
    data, feature_names = preprocess(df)
    # phase 1, clustering and plotting
    k = 5
    clusters = agglomerative(data, k)
    visualize(data, clusters, k)
    selected_sensors = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7']
    plot(df, clusters, selected_sensors)
    # assign stages
    cluster_to_stage = {2: 1, 0: 4, 4: 0, 1: 2, 3: 3}
    stages = np.array([cluster_to_stage[c] for c in clusters])
    # phase 2, SVC model
    SVCmodel = trainmodel(data, stages)
    # phase 3, time to next
    df['cycle'] = df['time']
    labeled_df = timetonext(df, stages)
    df['stage'] = labeled_df['stage'].values
    df['time_to_next_stage'] = labeled_df['time_to_next_stage'].values
    rul = labeled_df['time_to_next_stage'].values
    top_sensors = ['sensor_11', 'sensor_12', 'sensor_14', 'sensor_9', 'sensor_4', 'cycle']
    regression_model = regressionmodel(labeled_df, top_sensors)
    output_filename = "time_to_next_stage_output.csv"
    labeled_df[['unit', 'time', 'stage', 'time_to_next_stage']].to_csv(output_filename, index=False)
    print(f"\nAll time_to_next_stage values saved to '{output_filename}'")
    # Phase 4, risk score decision logic
    print("\nPhase 4: Risk Score Computation and Alerts")
# Step 1: Predict time_to_next_stage for all rows
Xtop = df[top_sensors + ['stage']]
df['predicted_time_to_next'] = regression_model.predict(Xtop)

# Step 2: Initialize estimated RUL column
df['estimated_rul'] = 0.0

# Step 3: Compute cumulative future times per unit
for unit in df['unit'].unique():
    unit_df = df[df['unit'] == unit].copy()
    unit_df = unit_df.sort_values('time')
    unit_df = unit_df.reset_index()
    estimated_ruls = []

    for idx in range(len(unit_df)):
        current_stage = unit_df.loc[idx, 'stage']
        if current_stage == 4:
            estimated_ruls.append(0.0)
            continue
        cum_rul = 0.0
        temp_stage = current_stage
        for j in range(idx, len(unit_df)):
            stage_j = unit_df.loc[j, 'stage']
            if stage_j == temp_stage:
                cum_rul += unit_df.loc[j, 'predicted_time_to_next']
                temp_stage += 1
            if temp_stage > 4:
                break
        estimated_ruls.append(cum_rul)

    df.loc[unit_df['index'], 'estimated_rul'] = estimated_ruls
 # Step 4: Compute risk score
failureprobs = SVCmodel.predict_proba(data)[:, 4]
raw_risk_scores = failureprobs * df['estimated_rul']

# Step 5: Normalize
min_score = np.min(raw_risk_scores)
max_score = np.max(raw_risk_scores)
df['risk_score'] = (raw_risk_scores - min_score) / (max_score - min_score + 1e-6)

# Step 6: Maintenance alerts
df['maintenance_alert'] = df['risk_score'] > 0.7
alert_units = df[df['maintenance_alert']]['unit'].unique()
if len(alert_units) > 0:
    print(f"Maintenance Alert! Units at risk: {alert_units}")
else:
    print("No critical maintenance alerts triggered.")

# Step 7: Plot
plt.figure(figsize=(15, 10))
for unit in df['unit'].unique():
    unit_data = df[df['unit'] == unit]
    plt.plot(unit_data['time'], unit_data['risk_score'], label=f'Unit {unit}')
plt.axhline(0.7, color='red', linestyle='--', label='Alert Threshold')
plt.title('Risk Score Over Time by Engine')
plt.xlabel('Time')
plt.ylabel('Normalized Risk Score')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.show()
