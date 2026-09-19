import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

csv_file = "hand_gestures.csv"

if not os.path.exists(csv_file):
    print(f"Error: {csv_file} not found! Run collect_data.py first.")
    exit()

df = pd.read_csv(csv_file)
X = df.drop('label', axis=1)
y = df['label']

print(f"Dataset loaded: {len(df)} samples across classes: {df['label'].unique()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model Training Complete! Accuracy: {accuracy * 100:.2f}%")

# Save model into the server folder
os.makedirs('server', exist_ok=True)
with open('server/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Saved trained model to server/model.pkl")