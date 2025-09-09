import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ==============================
# 1. Cargar los datos
# ==============================
df = pd.read_csv("./data/prestamos_knn_1000_limpio.csv")

# Variables predictoras y variable objetivo
X = df[['edad','ingreso_mensual_cop','historial_credito','monto_prestamo_cop']]
y = df['aprobado']

# ==============================
# 2. Dividir en train/test
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 3. Escalado
# ==============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# 4. Entrenar modelo KNN
# ==============================
modelo_knn = KNeighborsClassifier(n_neighbors=10, weights="distance")
modelo_knn.fit(X_train, y_train)

# ==============================
# 5. Evaluación del modelo
# ==============================
y_pred = modelo_knn.predict(X_test)

print("🔎 Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))

print("\n📊 Reporte de clasificación:")
print(classification_report(y_test, y_pred))

# ==============================
# 6. Predicción para nuevo cliente
# ==============================
New_Cliente = [[50,207532600.0,0.0,27937197.0]]
New_Cliente_Scaler = scaler.transform(New_Cliente)
Prediccion_Nueva = modelo_knn.predict(New_Cliente_Scaler)

print(f"\nPredicción para el nuevo cliente: {Prediccion_Nueva[0]}")
