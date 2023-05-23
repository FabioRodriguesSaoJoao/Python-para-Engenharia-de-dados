# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Carrega o dataset
df = pd.read_csv('dataset.csv')
print(df.shape)
print(df.columns) 
print(df.head()) 
print(df.info())

# Verifica se há valores ausentes
print(df.isnull().sum())
# Resumo estatístico do dataset - ATENÇÃO
print(df.describe())
# Resumo estatístico da variável alvo
print(df["valor_aluguel"].describe())

# Histograma da variável alvo
sns.histplot(data = df, x = "valor_aluguel", kde = True)

# Correlação entre as variáveis
print(df.corr())

sns.scatterplot(data = df, x = "area_m2", y = "valor_aluguel")

# ## Construção do Modelo OLS (Ordinary Least Squares) com Statsmodels em Python

print(df.head())
# Definimos a variável dependente
y = df["valor_aluguel"]


# Definimos a variável independente
X = df["area_m2"]


# # O Statsmodels requer a adição de uma constante à variável independente
X = sm.add_constant(X)


# Criamos o modelo
modelo = sm.OLS(y, X)# OLS y vem antes de x

# Treinamento do modelo
resultado = modelo.fit()

print(resultado.summary())


plt.figure(figsize = (12, 8))
plt.xlabel("area_m2", size = 16)
plt.ylabel("valor_aluguel", size = 16)
plt.plot(X["area_m2"], y, "o", label = "Dados Reais")
plt.plot(X["area_m2"], resultado.fittedvalues, "r-", label = "Linha de Regressão (Previsões do Modelo)")
plt.legend(loc = "best")
plt.show()

# # ## ** ------------------------------------------------------------------------------------------------
# # ## Conclusão

# # # Claramente existe uma forte relação entre a área (em m2) dos imóveis e o valor do aluguel. Entretanto, apenas a área dos imóveis não é suficiente para explicar a variação no valor do aluguel, pois nosso modelo obteve um coeficiente de determinação (R²) de apenas 0.34.

# # # O ideal seria usar mais variáveis de entrada para construir o modelo a fim de compreender se outros fatores influenciam no valor do aluguel.

# # # É sempre importante deixar claro que correlação não implica causalidade e que não podemos afirmar que o valor do aluguel muda apenas devido à área dos imóveis. Para estudar causalidade devemos aplicar Análise Causal.
