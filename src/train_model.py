import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import matplotlib.pyplot as plt

tx_merge = pd.read_csv('../data/tx_merge_clustered.csv')
print("Loaded columns:", tx_merge.columns.tolist())

X = tx_merge[['Recency', 'Revenue', 'Revenue_per_Transaction']]
y = tx_merge['LTVCluster']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=56)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Original y_train distribution:")
print(y_train.value_counts())

original_counts = y_train.value_counts()
smote = SMOTE(
    sampling_strategy={0: max(2000, original_counts[0]), 1: 1200, 2: 300},
    random_state=42,
    k_neighbors=3
)
X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
print("Post-SMOTE y_train distribution:")
print(pd.Series(y_train).value_counts())

xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.03,
    scale_pos_weight=4,
    gamma=2,
    min_child_weight=5,
    random_state=42,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train_scaled, y_train)

y_pred_proba = xgb_model.predict_proba(X_test_scaled)
threshold = 0.9 
y_pred_test = np.argmax(y_pred_proba, axis=1)
for i in range(len(y_pred_test)):
    if y_pred_proba[i, 2] > threshold:  
        y_pred_test[i] = 2

print("\nXGBoost Results:")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_test):.2f}")
print("\nClass Metrics (Precision | Recall | F1):")
report = classification_report(y_test, y_pred_test, target_names=['Low', 'Mid', 'High'], output_dict=True)
for cls in ['Low', 'Mid', 'High']:
    print(f"{cls}: {report[cls]['precision']:.2f} | {report[cls]['recall']:.2f} | {report[cls]['f1-score']:.2f}")

plt.figure(figsize=(8, 4))
plt.scatter(range(20), y_test.values[:20], color='blue', label='Actual', marker='o')
plt.scatter(range(20), y_pred_test[:20], color='red', label='Predicted', marker='x')
plt.yticks([0, 1, 2], ['Low', 'Mid', 'High'])
plt.xlabel('Sample')
plt.ylabel('Cluster')
plt.title('Actual vs. Predicted (First 20)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()
