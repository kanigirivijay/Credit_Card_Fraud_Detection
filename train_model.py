import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Step 1: Load the data
print("📥 Loading data...")
df = pd.read_csv("fraud_dataset.csv")
print("✅ Data loaded")

# Step 2: Drop unnecessary columns
print("🧹 Dropping name columns...")
df = df.drop(["nameOrig", "nameDest"], axis=1)

# Step 3: Encode the transaction type
print("🔤 Encoding transaction type...")
le = LabelEncoder()
df["type"] = le.fit_transform(df["type"])

# Step 4: Split the data
print("📊 Splitting data...")
X = df.drop(["isFraud", "isFlaggedFraud"], axis=1)
y = df["isFraud"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
print("🧠 Training model...")
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 6: Save the model with error check
try:
    print("💾 Saving model...")
    joblib.dump(model, "fraud_detection_model.pkl")
    print("✅ Model training and saving complete!")
except Exception as e:
    print(f"❌ Error saving model: {e}")
