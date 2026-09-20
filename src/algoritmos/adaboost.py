import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

inicio = time.time()

df = pd.read_csv("../../data/creditcard.csv")

# ==========================================
# FEATURES E TARGET
# ==========================================
X = df.drop("Class", axis=1)
y = df["Class"]

# =========================================
# DIVIDIR TREINO E TESTE
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# =========================================
# BALANCEAR TREINO
# =========================================
smote = SMOTE(random_state=42) # balancendo apenas o treino para evitar vazamento de dados

X_train_res, y_train_res = smote.fit_resample(
    X_train,
    y_train
)

# =========================================
# MODELO
# =========================================

ada = AdaBoostClassifier(
    n_estimators=100,
    random_state=42
)

# =========================================
# TREINAR
# =========================================

print("Treinando AdaBoost...")

ada.fit(X_train_res, y_train_res)

print("Treinamento concluído!")

# =========================================
# PREVISÕES
# =========================================

y_pred = ada.predict(X_test)

# =========================================
# RESULTADOS
# ========================================

print("\n=== ADABOOST ===")

print("\nMatriz de confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório:")
print(classification_report(
    y_test,
    y_pred,
    digits=4
))

fim = time.time()
print(f"\nTempo de execução: {fim - inicio:.2f} segundos")
