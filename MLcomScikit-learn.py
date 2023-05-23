# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Carrega o dataset
df = pd.read_csv('\dataset.csv')
print(df.shape)
print(df.columns)
print(df.head())
print(df.info())
# Análise Exploratória - Resumo Estatístico
# Verifica se há valores ausentes
print(df.isnull().sum())

# Correlação
# print(df.corr())

# Resumo estatístico do dataset 
print(df.describe())

# Resumo estatístico da variável preditora
print(df["horas_estudo_mes"].describe())

# Histograma da variável preditora
sns.histplot(data = df, x = "horas_estudo_mes", kde = True)

## Preparação dos Dados

# Prepara a variável de entrada X
X = np.array(df['horas_estudo_mes'])

print(type(X))

# Ajusta o shape de X
X = X.reshape(-1, 1)
print(X)

# Prepara a variável alvo
y = df['salario']
# print(y)
# Gráfico de dispersão entre X e y
plt.scatter(X, y, color = "blue", label = "Dados Reais Históricos")
plt.xlabel("Horas de Estudo")
plt.ylabel("Salário")
plt.legend()
plt.show()

# Dividir dados em treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.2, random_state = 42)

print(X_treino.shape)

print(X_teste.shape)

print(y_treino.shape)

print(y_teste.shape)

# ## Modelagem Preditiva (Machine Learning)


# Cria o modelo de regressão linear simples
modelo = LinearRegression()

# Treina o modelo
modelo.fit(X_treino, y_treino)
# print(modelo.fit(X_treino, y_treino))

# # Visualiza a reta de regressão linear (previsões) e os dados reais usados no treinamento
plt.scatter(X, y, color = "blue", label = "Dados Reais Históricos")
plt.plot(X, modelo.predict(X), color = "red", label = "Reta de Regressão com as Previsões do Modelo")
plt.xlabel("Horas de Estudo")
plt.ylabel("Salário")
plt.legend()
plt.show()

# Avalia o modelo nos dados de teste
score = modelo.score(X_teste, y_teste)
print(f"Coeficiente R^2: {score:.2f}")

# Intercepto - parâmetro w0
modelo.intercept_
print(modelo.intercept_)

# Slope - parâmetro w1
modelo.coef_
print(modelo.coef_)
## Deploy do Modelo

# # Usaremos o modelo para prever o salário com base nas horas de estudo.

# Define um novo valor para horas de estudo
horas_estudo_novo = np.array([[48]]) 
# print(horas_estudo_novo)
# Faz previsão com o modelo treinado
salario_previsto = modelo.predict(horas_estudo_novo)

print(f"Se você estudar cerca de", horas_estudo_novo, "horas por mês seu salário pode ser igual a", salario_previsto)

# Mesmo resultado anterior usando os parâmetros (coeficientes) aprendidos pelo modelo
# y_novo = w0 + w1 * X
salario = modelo.intercept_ + (modelo.coef_ * horas_estudo_novo)
# print(salario)

# Define um novo valor para horas de estudo
horas_estudo_novo = np.array([[65]]) 

# Faz previsão com o modelo treinado
salario_previsto = modelo.predict(horas_estudo_novo)

print(f"Se você estudar cerca de", horas_estudo_novo, "horas por mês seu salário pode ser igual a", salario_previsto)

# Define um novo valor para horas de estudo
horas_estudo_novo = np.array([[73]]) 

# Faz previsão com o modelo treinado
salario_previsto = modelo.predict(horas_estudo_novo)

print(f"Se você estudar cerca de", horas_estudo_novo, "horas por mês seu salário pode ser igual a", salario_previsto)



