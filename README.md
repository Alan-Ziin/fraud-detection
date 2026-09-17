# Análise Comparativa de Algoritmos de Aprendizado de Máquina para Detecção de Fraudes em Transações Online

> **Repositório reupado e atualizado** referente ao Trabalho de Conclusão de Curso (TCC) do tecnólogo em **Análise e Desenvolvimento de Sistemas** pela **Universidade de Taubaté (UNITAU)**.

---

## Sobre o Projeto
Este projeto tem como objetivo avaliar e comparar o desempenho de diferentes algoritmos supervisionados de aprendizado de máquina aplicados à detecção automática de transações fraudulentas em cartões de crédito. O estudo aborda o desafio crítico do **desbalanceamento de classes** em bases de dados financeiras reais.

###  Principais Tecnologias e Ferramentas
* **Linguagem:** Python
* **Manipulação de Dados:** Pandas, NumPy
* **Pré-processamento e Balanceamento:** Scikit-learn (`RobustScaler`), Imbalanced-learn (`SMOTE`)
* **Modelos de Machine Learning (Ensemble):** 
  * Random Forest
  * CatBoost
  * LightGBM
  * XGBoost
  * AdaBoost

---

## Metodologia e Resultados
1. **Pré-processamento:** Tratamento de variáveis originais (`Time`, `Amount`) e componentes transformados via PCA (`V1` a `V28`).
2. **Balanceamento:** Aplicação da técnica **SMOTE** exclusivamente no conjunto de treino para evitar vazamento de dados e equilibrar a representatividade das fraudes.
3. **Avaliação:** Uso de métricas robustas para cenários desbalanceados (Precisão, *Recall*, *F1-Score* e Matriz de Confusão).
4. **Destaque:** O **Random Forest** apresentou o melhor desempenho geral e equilíbrio (maior *F1-Score* e alta precisão), mostrando-se a alternativa mais viável para cenários de produção reais.

---

## Estrutura do Repositório
* `Reconstruindo`

---

## Autores do Projeto
* **Alan Alves de Sales**
* **Gabriel Peixoto Fialho Costa**

*Orientador: Prof. Dr. Luis Fernando de Almeida*
