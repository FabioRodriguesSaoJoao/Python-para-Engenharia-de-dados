import pandas as pd

#cria um dicionarios
dados={"Estado":['Santa Catarina', 'Rio de Janeiro', 'Tocantis', 'Bahia', 'Minas Gerais'],'Ano':[2004,2005,2006,2007,2008], 'Taxa Desemprego': [1.5,1.7,1.6,2.4,2.7]}

from pandas import DataFrame
#converte o dicionario em um dataframe
df = DataFrame(dados)
#visualiza as 5 primeiras linhas

print(df.head())
print(type(df))

#Reorganizando as colunas
df = DataFrame(dados,columns=['Estado','Taxa Desemprego','Ano'])
print(df.head())

#Criando outro dataframe com os mesmo dados anteriores mas adicionando uma coluna
df2 = DataFrame(dados,columns=['Estado','Taxa Desemprego','Taxa Crescimento','Ano'],index=['estado1','estado2','estado3','estado4','estado5'])
print(df2.head())
print("Valores da tabela:\n",df2.values)
print("tipos:\n",df2.dtypes)
print("colunas:\n",df2.columns)
#imprimindo apenas uma coluna
print(df2['Estado'])
# imprimindo apenas duas coluna
print(df2[['Taxa Crescimento','Ano']])
print(df2.index)
# filtrando pelo index
print(df2.filter(items=['estado3'],axis=0))#me retornou somente a linha que eu quero, chamado pelo index


# MANIPULANDO PANDAS E NUMPY
# REUMO ESTATISTICO DO DF
print(df2.describe())#retorna somente estatisticas de colunas com valores numericos
print(df2.isna())#tem valor ausente?

import numpy as np
#Usando o Numpy para alimentar uma das colunas do df
df2['Taxa Crescimento'] = np.arange(5.)
print(df2)
print(df2['Taxa Crescimento'].isna())
print(df2.describe())

#slicing de DF do Pandas
#fatiamento do DF
print(df2['estado2':'estado4'])# pegando todos os valores entre essas duas linhas
print(df2[df2['Taxa Desemprego']<2])#filtrando a coluna taxa desemprego quando o valor for menor que 2
print(df2[['Estado','Taxa Crescimento']])#retornou toda as duas colunas

# Preenchendo valores ausentes em df do pandas
#Primeiro importamos um dataset
df1=pd.read_csv("dataset.csv")
print(df1.head(5))
print(df1.isna().sum())
#Extraímos a moda da coluna Quantity
moda = df1['Quantidade'].value_counts().index[0]# valor que mais se repete
print(moda)

#E por fim preenchemos os valores NA com a moda
df1['Quantidade'].fillna(value=moda,inplace=True)# chamando a coluna quantidade da tabela, usando a funçao fillna para preenchimento, passando o value(valores que irao ser usados para preenchimento), inplace para salvar no proprio df
print(df1)
print(df1.isna().sum())

#TODO QUERY(CONSULTA) de dados no df do pandas
df1=pd.read_csv("dataset.csv")

#checamos os valores minimos e maximos da coluna Valor_Venda
print(df1.Valor_Venda.describe())
#Geramos um novo df apenas com o intervalo de vendas entre 229 e 10000
df22= df1.query('229<Valor_Venda<10000')#query para verificar
#entao confirmamos os valore minimo e maximo
print(df22.Valor_Venda.describe())
#Geramos um novo df apenas com os valores de venda acima da media
df3 = df22.query('Valor_Venda>766')
print(df3.head())

#todo Verificando a Ocorrencia de Diversos Valores em uma coluna
print(df1.shape)# shape linha,coluna
#entao aplicamos o filtro
print(df1[df1['Quantidade'].isin([5,7,9,11])])# isin verifica se os valores estao na coluna

#unindo os dois comando anteriores
print(df1[df1['Quantidade'].isin([5,7,9,11])].shape)
print(df1[df1['Quantidade'].isin([5,7,9,11])][:10])#retornando somente as 10 primeiras linhas

#todo OPERADORES LOGICOS PARA MANUPULAÇÃO DE DADOS COM PANDAS
#Filtrando as vendas que ocorreram para o segmento de Home Ofice e nao regiao South
print(df1[(df1.Segmento == "Home Office") & (df1.Regiao == "South")].head())

#Filtrando as vendas que ocorreram para o segmento de Home office ou regiao South
print(df1[(df1.Segmento == "Home Office") | (df1.Regiao == "South")].tail())#tail é a parte final dos registros

#Filtrando as vendas que nao ocorreram para o segmento de Home Ofice e nem na regiao South
print(df1[(df1.Segmento != "Home Office") & (df1.Regiao != "South")].sample(5)) # sample amostra de 5 valores

#todo AGRUPAMENTO DE DADOS EM DF COM GROUP BY
print(df1[["Segmento","Regiao", "Valor_Venda"]].groupby(['Segmento','Regiao']).mean())# PRIMEIRO VOCE ESCOLHE QUAIS COLUNAS VC QUER QUE APAREÇA , DPS CHAMA O METODO GROUPBY PARA VER, NO CASO, QUAL SEGMENTO EXISTE, DENTRO DO SEGMENTO, QUAIS REGIOES EXISTEM, E A MEDIA DE VALOR POR CADA REAGIAO


#TODO AGREGAÇÃO MULTIPLA COM GROUP BY
# agg = AGREGAÇÃO
print(df1[["Segmento","Regiao", "Valor_Venda"]].groupby(['Segmento','Regiao']).agg(['mean','std','count']))# PRIMEIRO VOCE ESCOLHE QUAIS COLUNAS VC QUER QUE APAREÇA , DPS CHAMA O METODO GROUPBY PARA VER, NO CASO, QUAL SEGMENTO EXISTE, DENTRO DO SEGMENTO, QUAIS REGIOES EXISTEM, E A MEDIA DE VALOR POR CADA REAGIAO, agora tambem me da a media, o dp e a contagem

# todo Filtrando DF do pandas com base em STR
print(df1[df1.Segmento.str.startswith('Con')].head()) # pegando o df, coluna Segmento, somente as str iniciadas com 'Con'

# Filtrando o df pela coluna segmento com valores que terminam com as lentras 'mer'
print(df1[df1.Segmento.str.endswith('mer')].head())

#Todo split de str em DF com pandas
print(df1['ID_Pedido'].head())# trabalhando com somento uma coluna

# Split da coluna pelo caracter
print(df1['ID_Pedido'].str.split('-'))

# Extraindo somente o ano dentro das listas
print(df1['ID_Pedido'].str.split('-').str[1].head())

# Fazemos o split da coluna e extraímos o item na posiçaõ 2 (indice1)
df1['Ano'] = df1['ID_Pedido'].str.split('-').str[1]

print(df1.head())

#todo parte 2
print(df1['Data_Pedido'].head())
# #Vamos remover os digitos 2 e 0 a esquerda do valor da varialvel "Data_pedido"
print(df1['Data_Pedido'].str.lstrip('20')) #lstrip L É DE ESQUERDA


#todo Replace de str em df do pandas
# Substituímos os caracteres CG por AX na coluna 'ID_Cliente'
df1['ID_Cliente'] = df1['ID_Cliente'].str.replace('CG', 'AX')
print(df1.head())

#todo combinação de str em df do pandas
# Concatenando strings
df1['Pedido_Segmento'] = df1['ID_Pedido'].str.cat(df1['Segmento'], sep = '-') # criando nova coluna , juntando duas outras colunas, ela vai para o final da tabela
print(df1.head())

#todo construção de Graficos a partir de DF do pandas 1/2
import sklearn
# Vamos começar importando o dataset iris do Scikit-learn
from sklearn.datasets import load_iris
data = load_iris()# tabela inicial
# E então carregamos o dataset iris como dataframe do Pandas
# import pandas as pd
df = pd.DataFrame(data['data'], columns = data['feature_names'])
df['species'] = data['target']
# print(df.head())
# #para criar um grafico de linhas com todas as variaveis do df, basta fazer isso:
print(df.plot())

# Que tal um scatter plot com duas variáveis? 
df.plot.scatter(x = 'sepal length (cm)', y = 'sepal width (cm)') # scatter mostra a relação entre duas variaveis, x e y no caso

# # E mesmo gráficos mais complexos, como um gráfico de área, pode ser criado:
columns = ['sepal length (cm)', 'petal length (cm)', 'petal width (cm)', 'sepal width (cm)']
df[columns].plot.area()
# print(columns)
# todo parte 2 
# Calculamos a média das colunas agrupando pela coluna species e criamos um gráfico de barras com o resultado
df.groupby('species').mean().plot.bar()

# Ou então, fazemos a contagem de classes da coluna species e plotamos em um gráfico de pizza
print((df.groupby('species').count().plot.pie(y = 'sepal length (cm)'))) # grafico de pizza pie

# Gráfico KDE (Kernel Density Function) para cada variável do dataframe
df.plot.kde(subplots = True, figsize = (8,8))

# Boxplot de cada variável numérica
columns = ['sepal length (cm)', 'petal length (cm)', 'petal width (cm)', 'sepal width (cm)']
df[columns].plot.box(figsize = (8,8))
