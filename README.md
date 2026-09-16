# Análise Comparativa de Algoritmos de Aprendizado de Máquina para Detecção de Fraudes em Transações Online

> **Repositório reupado e atualizado** referente ao Trabalho de Conclusão de Curso (TCC) do tecnólogo em **Análise e Desenvolvimento de Sistemas** pela **Universidade de Taubaté (UNITAU)**[cite: 1].

---

## 📌 Sobre o Projeto
Este projeto tem como objetivo avaliar e comparar o desempenho de diferentes algoritmos supervisionados de aprendizado de máquina aplicados à detecção automática de transações fraudulentas em cartões de crédito[cite: 1]. O estudo aborda o desafio crítico do **desbalanceamento de classes** em bases de dados financeiras reais[cite: 1].

### 🚀 Principais Tecnologias e Ferramentas
* **Linguagem:** Python[cite: 1]
* **Manipulação de Dados:** Pandas, NumPy[cite: 1]
* **Pré-processamento e Balanceamento:** Scikit-learn (`RobustScaler`), Imbalanced-learn (`SMOTE`)[cite: 1]
* **Modelos de Machine Learning (Ensemble):** 
  * Random Forest[cite: 1]
  * CatBoost[cite: 1]
  * LightGBM[cite: 1]
  * XGBoost[cite: 1]
  * AdaBoost[cite: 1]

---

## 📊 Metodologia e Resultados
1. **Pré-processamento:** Tratamento de variáveis originais (`Time`, `Amount`) e componentes transformados via PCA (`V1` a `V28`)[cite: 1].
2. **Balanceamento:** Aplicação da técnica **SMOTE** exclusivamente no conjunto de treino para evitar vazamento de dados e equilibrar a representatividade das fraudes[cite: 1].
3. **Avaliação:** Uso de métricas robustas para cenários desbalanceados (Precisão, *Recall*, *F1-Score* e Matriz de Confusão)[cite: 1].
4. **Destaque:** O **Random Forest** apresentou o melhor desempenho geral e equilíbrio (maior *F1-Score* e alta precisão), mostrando-se a alternativa mais viável para cenários de produção reais[cite: 1].

---

## 📂 Estrutura do Repositório
* `Reconstruindo`

---

## 👤 Autores
* **Alan Alves de Sales**[cite: 1]
* **Gabriel Peixoto Fialho Costa**[cite: 1]

*Orientador: Prof. Dr. Luis Fernando de Almeida*[cite: 1]
