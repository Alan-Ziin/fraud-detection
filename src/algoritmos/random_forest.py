import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

df = pd.read_csv("../../data/creditcard.csv")

# seperando dados em features e target
X = df.drop("Class", axis=1)
y = df["Class"]

# dividindo os dados em treino e teste, mantendo a proporção das classes
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3, # tamanho do conjunto de teste (30% dos dados)
    random_state=42, # semente para reprodutibilidade
    stratify=y # mantém a proporção das classes
)

# balanceamento das classes usando SMOTE, apenas no conjunto de treino para evitar vazamento de dados
smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(
    X_train,
    y_train
)

# O modelo
rf = RandomForestClassifier(
    n_estimators=100, # número de árvores na floresta
    random_state=42,
    class_weight="balanced",
    n_jobs=-1 # utiliza todos os núcleos da CPU para treinamento
)

print("Treinando Random Forest...")
rf.fit(X_train_res, y_train_res)
print("Treinamento concluído!")


y_pred = rf.predict(X_test) # Fazendo previsões no conjunto de teste


print("\n=== RANDOM FOREST ===") # o resultado

print("\nMatriz de confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório:")
print(classification_report(
    y_test,
    y_pred,
    digits=4
))