import pandas as pd

df = pd.read_csv("../data/creditcard.csv")

print("Primeiras Linhas:")
print(df.head()) #Pega as 5 primeiras linhas do dataset

print("\nInformações do Dataset:")
print(df.info()) #Informações sobre o dataset, como número de linhas, colunas e tipos de dados

print("\nEstatísticas:")
print(df.describe())

print("\nValores Nulos:")
print(df.isnull().sum()) 

print("\nDistribuição das Classes:")
print(df["Class"].value_counts()) # Mostra o total de cada classe no dataset, será usado para
# verificar o balanceamento das classes
