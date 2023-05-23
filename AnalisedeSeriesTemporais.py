# # Análise de Séries Temporais em Python
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# Carrega o dataset
df = pd.read_csv("dataset.csv")

print(df.shape)

print(df.columns)

print(df.head())

print(df.tail())#pegando o DF de tras para frente

# Valor mínimo da coluna data
print(df['Data'].min())


# Valor máximo da coluna data
print(df['Data'].max())

print(df.info())

# Converte a coluna de data no tipo datetime
df['Data'] = pd.to_datetime(df['Data'])

print(df.head())

print(df.info())

# Converter o DataFrame em uma série temporal com a data como índice
serie_temporal = df.set_index('Data')['Total_Vendas']


# Fornece a frequência da série temporal (diária, neste caso)
serie_temporal = serie_temporal.asfreq('D')#asfreq é a frequencia ('d') se refere a frequencia diaria 

# ## Análise Exploratória


# # Cria o gráfico da série temporal (sem formatação)
plt.figure(figsize = (12, 6))
plt.plot(serie_temporal)
plt.xlabel('Data')
plt.ylabel('Vendas')
plt.title('Série Temporal de Vendas')
plt.grid(True)
plt.show()


# Cria o gráfico da série temporal (com formatação)

# Criar o gráfico da série temporal com layout de contraste
plt.figure(figsize = (12, 6))
plt.plot(serie_temporal, color = 'white', linewidth = 2)

# Configurar cores e estilo do gráfico
plt.gca().set_facecolor('#2e03a3')
plt.grid(color = 'yellow', linestyle = '--', linewidth = 0.5)

# Configurar rótulos dos eixos, título e legenda
plt.xlabel('Data', color = 'black', fontsize = 14)
plt.ylabel('Vendas', color ='black', fontsize = 14)
plt.title('Série Temporal de Vendas', color = 'black', fontsize = 18)

# Configurar as cores dos eixos e dos ticks (marcadores)
plt.tick_params(axis = 'x', colors  ='black')
plt.tick_params(axis = 'y', colors = 'black')

plt.show()


# Cria o modelo
modelo = SimpleExpSmoothing(serie_temporal)

# Treinamento (ajuste) do modelo
modelo_ajustado = modelo.fit(smoothing_level = 0.2) #fit ajustando o modelo


# # Extrai os valores previstos pelo modelo = fittedvalues
suavizacao_exponencial = modelo_ajustado.fittedvalues  #fittedvalues valores previstos


# O resultado final é uma nova série temporal chamada suavizacao_exponencial, que representa a versão suavizada da série original de vendas, com menos ruído e flutuações de curto prazo.
# Plot
plt.figure(figsize = (12, 6))
plt.plot(serie_temporal, label = 'Valores Reais')
plt.plot(suavizacao_exponencial, label = 'Valores Previstos', linestyle = '--')
plt.xlabel('Data')
plt.ylabel('Vendas')
plt.title('Modelo de Suavização Exponencial')
plt.legend()
plt.show()

## Deploy e Previsão com o Modelo Treinado
# Fazer previsões
num_previsoes = 1 # esta querendo 1 dia so
previsoes = modelo_ajustado.forecast(steps = num_previsoes) #forecast previsoes
print('Previsão do Total de Vendas Para Janeiro/2024:', round(previsoes[0], 4)) #round(previsoes[0], 4) primeiro valor com 4 casas decimais

