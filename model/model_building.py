import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load Data
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['cultivar'] = data.target

# 2. Select 6 Features (Alcohol, Flavanoids, Color Intensity, Hue, Proline, Magnesium)
selected_features = [
    'alcohol', 'flavanoids', 'color_intensity', 
    'hue', 'proline', 'magnesium'
]

X = df[selected_features]
y = df['cultivar']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Create Pipeline (Scaler + Model)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Train
pipeline.fit(X_train, y_train)

# 6. Evaluate
print("Accuracy:", accuracy_score(y_test, pipeline.predict(X_test)))

# 7. Save Model
joblib.dump(pipeline, 'wine_cultivar_model.pkl')
print("Model saved to wine_cultivar_model.pkl")