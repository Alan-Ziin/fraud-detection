import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

from catboost import CatBoostClassifier

inicio = time.time()

# ==========================================
# CARREGAR DATASET
# ==========================================

df = pd.read_csv("creditcard.csv")


# ==========================================
# FEATURES E TARGET
# ==========================================

X = df.drop("Class", axis=1)
y = df["Class"]


# ==========================================
# DIVIDIR TREINO E TESTE
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)


# ==========================================
# BALANCEAR TREINO
# ==========================================

smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(
    X_train,
    y_train
)


# ==========================================
# MODELO
# ==========================================

cat = CatBoostClassifier(
    iterations=100, # número de árvores
    learning_rate=0.1, # taxa de aprendizado
    depth=6, # profundidade da árvore
    random_seed=42, # semente para reprodutibilidade
    verbose=False # desativa a saída de log durante o treinamento
)


# ==========================================
# TREINAR
# ==========================================

print("Treinando CatBoost...")

cat.fit(X_train_res, y_train_res)

print("Treinamento concluído!")


# ==========================================
# PREVISÕES
# ==========================================

y_pred = cat.predict(X_test)


# ==========================================
# RESULTADOS
# ==========================================

print("\n=== CATBOOST ===")

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