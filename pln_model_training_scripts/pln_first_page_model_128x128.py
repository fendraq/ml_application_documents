# Script for training ensemble model on dataset converted to 128x128
# Imports
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

print("Loading data")
X = np.load('../data/processed/pln_X_features_raw_128x128.npy')
y = np.load('../data/processed/pln_y_labels.npy')

print("Keeping top 40 pixel rows")
X_reshape = X.reshape(-1, 128, 128)
X_top40 = X_reshape[:, :40, :]
X_top40_flat = X_top40.reshape(X_top40.shape[0], -1)

print("Splitting data in train and test")
X_train, X_test, y_train, y_test = train_test_split(
    X_top40_flat, y, test_size=0.20, 
    random_state=42,
    stratify=y
)

print("Scaling data")
scaler = StandardScaler()
X_trained_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Initializing models")
knc = KNeighborsClassifier(metric='minkowski', n_neighbors=2, weights='distance')
rfc = RandomForestClassifier(max_depth= None, min_samples_split=5, n_estimators=150)

print("Training KNC")
knc.fit(X_trained_scaled, y_train) 

print("Training RFC")
rfc.fit(X_trained_scaled, y_train)

print("Setting threshold to 0.25")
threshold = 0.25
rfc_y_proba = rfc.predict_proba(X_test_scaled)[:, 1] 
rfc_y_pred = (rfc_y_proba >= threshold).astype(int)

knc_y_proba = knc.predict_proba(X_test_scaled)[:, 1] 
knc_y_pred = (knc_y_proba >= threshold).astype(int)

print("Testing prediction")
ensemble_pred = np.maximum(rfc_y_pred, knc_y_pred)
cm = confusion_matrix(y_test, ensemble_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

print(cm)
print(classification_report(y_test, ensemble_pred))

print("Saving models")
joblib.dump(rfc, "../models/pln_first_page_model_128x128/rfc_model.pkl")
print("RFC saved")
joblib.dump(knc, "../models/pln_first_page_model_128x128/knc_model.pkl")
print("KNC saved")
joblib.dump(scaler, "../models/pln_first_page_model_128x128/scaler_model.pkl")
print("Scaler saved")
