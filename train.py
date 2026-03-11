import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from utils import fetch_weather_data, process_data

print("Loading data...")

raw_data = fetch_weather_data(38.7223, -9.1333, "2023-01-01", "2024-12-31")
df = process_data(raw_data)

X = df.drop(columns=["target", "precipitation_sum", "rain_sum"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "LogisticRegression": LogisticRegression(),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

best_acc = 0
best_model = None
best_name = ""

print("Training models...")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"Model: {name}, Accuracy: {acc:.2f}")
    
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name


print(f"Best model: {best_name}")
joblib.dump(best_model, "weather_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")
joblib.dump(best_acc, "best_acc.pkl")

df.to_csv("weather_daily.csv")