import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

inicio = time.time()

df = pd.read_csv("../../data/creditcard.csv")

X = df.drop("Class", axis=1)
y = df["Class"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

smote = SMOTE(random_state=42) # balancendo apenas o treino para evitar vazamento de dados

X_train_res, y_train_res = smote.fit_resample(
    X_train,
    y_train
)

ada = AdaBoostClassifier(
    n_estimators=100,
    random_state=42
)

print("Treinando AdaBoost...")

ada.fit(X_train_res, y_train_res)

print("Treinamento concluído!")

y_pred = ada.predict(X_test) # fazendo previsões no conjunto de teste

#resultado
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
