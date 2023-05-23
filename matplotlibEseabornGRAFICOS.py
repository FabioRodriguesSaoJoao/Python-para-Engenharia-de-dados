## Matplotlib

import matplotlib as mpl

# O matplotlib.pyplot é uma coleção de funções e estilos do Matplotlib 


# Construindo Plots

import matplotlib.pyplot as plt

# O método plot() define os eixos do gráfico
plt.plot([1, 3, 5], [2, 4, 7])
print(plt.show())# show é mostrar


x = [2, 3, 5]
y = [3, 5, 7]

plt.plot(x, y)
plt.xlabel('Variável 1')
plt.ylabel('Variável 2')
plt.title('Teste Plot')
plt.show()


x2 = [1, 2, 3]
y2 = [11, 12, 15]

plt.plot(x2, y2, label = 'Gráfico com Matplotlib')
plt.legend()# com esse metodo, o label fica dentro da legenda
plt.show()

# ------------------------------------------------------------------------------------------------------------------

## Gráfico de Barras
x1 = [2,4,6,8,10]
y1 = [6,7,8,2,4]

plt.bar(x1, y1, label = 'Barras', color = 'green')# para criar um grafico de barras é so chamar a funçao bar
plt.legend()
plt.show()

x2 = [1,3,5,7,9]
y2 = [7,8,2,4,2]

plt.bar(x1, y1, label = 'Listas1', color = 'blue')
plt.bar(x2, y2, label = 'Listas2', color = 'red')# se chamamos duas vezes, nós criamos um grafico em cima do outro, na mesma area de plotagem
plt.legend()
plt.show()

idades = [22,65,45,55,21,22,34,42,41,4,99,101,120,122,130,111,115,80,75,54,44,64,13,18,48]

ids = [x for x in range(len(idades))]#criando  indices

# print(ids)

plt.bar(ids, idades)
plt.show()

bins = [0,10,20,30,40,50,60,70,80,90,100,110,120,130]

plt.hist(idades, bins, histtype = 'bar', rwidth = 0.8)# hist faz o histograma do tipo barra (bar), usou o bins para mudar os indices
plt.show()


plt.hist(idades, bins, histtype = 'stepfilled', rwidth = 0.8)#'stepfilled' nao deixamos espaços entre as barras
plt.show()

# ------------------------------------------------------------------------------------------------------------------

# Gráfico de Dispersão

x = [1,2,3,4,5,6,7,8]
y = [5,2,4,5,6,8,4,8]

plt.scatter(x, y, label = 'Pontos', color = 'r', marker = '*')# scatter graficos de pontos 
plt.legend()
plt.show()
plt.scatter(x, y, label = 'Pontos', color = 'black', marker = 'o')# scatter graficos de pontos 
plt.legend()
plt.show()

# ------------------------------------------------------------------------------------------------------------------

## Gráfico de Área Empilhada


dias = [1,2,3,4,5]
dormir = [7,8,6,77,7]
comer = [2,3,4,5,3]
trabalhar = [7,8,7,2,2]
passear = [8,5,7,8,13]

plt.stackplot(dias, dormir, comer, trabalhar, passear, colors = ['m','c','r','k','b'])# stackplot grafico de area empilhada
plt.show()

# ------------------------------------------------------------------------------------------------------------------

## Gráfico de Pizza


fatias = [7, 2, 2, 13] #valores
atividades = ['dormir', 'comer', 'passear', 'trabalhar']#legenda
cores = ['olive', 'lime', 'violet', 'royalblue']#cores respectivas


plt.pie(fatias, labels = atividades, colors = cores, startangle = 90, shadow = True, explode = (0,0.2,0,0))#pie para pizza, explode usa para destacar a fatia que queremos, cada valor é respectivo para cada fatia, entao devemos testar para ver qual queremos destacar
plt.show()

# ------------------------------------------------------------------------------------------------------------------

## Criando Gráficos Customizados com Pylab



# O Pylab combina funcionalidades do pyplot com funcionalidades do Numpy
from pylab import *


# ------------------------------------------------------------------------------------------------------------------

# Gráfico de linha

# Dados
x = linspace(0, 5, 10)
y = x ** 2

# Cria a figura
fig = plt.figure()

# Define a escala dos eixos
axes = fig.add_axes([0, 0, 0.8, 0.8])

# Cria o plot
axes.plot(x, y, 'r')

# Labels e título
axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('Gráfico de Linha');

# ------------------------------------------------------------------------------------------------------------------

# Gráficos de linha com 2 figuras

# Dados
x = linspace(0, 5, 10)
y = x ** 2

# Cria a figura
fig = plt.figure()

# Cria os eixos
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # eixos da figura principal
axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3]) # eixos da figura secundária

# Figura principal
axes1.plot(x, y, 'r')
axes1.set_xlabel('x')
axes1.set_ylabel('y')
axes1.set_title('Figura Principal')

# Figura secundária
axes2.plot(y, x, 'g')
axes2.set_xlabel('y')
axes2.set_ylabel('x')
axes2.set_title('Figura Secundária');
print(axes1,axes2)

# ------------------------------------------------------------------------------------------------------------------

# Gráficos de linha em Paralelo

# Dados
x = linspace(0, 5, 10)
y = x ** 2

# Divide a área de plotagem em dois subplots
fig, axes = plt.subplots(nrows = 1, ncols = 2)

# Loop pelos eixos para criar cada plot
for ax in axes:
    ax.plot(x, y, 'r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Título')
    
# Ajusta o layout
fig.tight_layout()

# ------------------------------------------------------------------------------------------------------------------

# Gráficos de linha com diferentes escalas

# Dados
x = linspace(0, 5, 10)
y = x ** 2

# Cria os subplots
fig, axes = plt.subplots(1, 2, figsize = (10,4))
      
# Cria o plot1
axes[0].plot(x, x**2, x, exp(x))
axes[0].set_title("Escala Padrão")

# Cria o plot2
axes[1].plot(x, x**2, x, exp(x))
axes[1].set_yscale("log")# set_yscale muda o padrao do grafico
axes[1].set_title("Escala Logaritmica (y)");


# Grid

# Dados
x = linspace(0, 5, 10)
y = x ** 2

# Cria os subplots
fig, axes = plt.subplots(1, 2, figsize = (10,3))

# Grid padrão
axes[0].plot(x, x**2, x, x**3, lw = 2)
axes[0].grid(True)#grid adiciona quadrantes 

# Grid customizado
axes[1].plot(x, x**2, x, x**3, lw = 2)
axes[1].grid(color = 'b', alpha = 0.7, linestyle = 'dashed', linewidth = 0.8)#grid quadrantes personalizados

# ------------------------------------------------------------------------------------------------------------------

# Diferentes estilos de Plots

# Dados
xx = np.linspace(-0.75, 1., 100)
n = np.array([0,1,2,3,4,5])

# Subplots
fig, axes = plt.subplots(1, 4, figsize = (12,3))# o numero 1 é o numero de linhas, e o 4 de colunas

# Plot 1
axes[0].scatter(xx, xx + 0.25 * randn(len(xx)), color = "black")
axes[0].set_title("scatter")

# Plot 2
axes[1].step(n, n ** 2, lw = 2, color = "blue")
axes[1].set_title("step")

# Plot 3
axes[2].bar(n, n ** 2, align = "center", width = 0.5, alpha = 0.5, color = "magenta")
axes[2].set_title("bar")

# Plot 4
axes[3].fill_between(x, x ** 2, x ** 3, alpha = 0.5, color = "green");
axes[3].set_title("fill_between");
axes.show()

## Histogramas

# ------------------------------------------------------------------------------------------------------------------

# Dados
n = np.random.randn(100000)

# Cria os subplots
fig, axes = plt.subplots(1, 2, figsize = (12,4))

# Plot 1
axes[0].hist(n)
axes[0].set_title("Histograma Padrão")
axes[0].set_xlim((min(n), max(n)))

# Plot 2
axes[1].hist(n, cumulative = True, bins = 50)
axes[1].set_title("Histograma Cumulativo")
axes[1].set_xlim((min(n), max(n)));

# ------------------------------------------------------------------------------------------------------------------

from mpl_toolkits.mplot3d.axes3d import Axes3D


# Dados
alpha = 0.7
phi_ext = 2 * np.pi * 0.5

# Função para um mapa de cores
def ColorMap(phi_m, phi_p):
    return ( + alpha - 2 * np.cos(phi_p)*cos(phi_m) - alpha * np.cos(phi_ext - 2*phi_p))

# Mais dados
phi_m = np.linspace(0, 2*np.pi, 100)
phi_p = np.linspace(0, 2*np.pi, 100)
X,Y = np.meshgrid(phi_p, phi_m)#juntas os dados meshgrid
Z = ColorMap(X, Y).T # ColorMap para gerar mais uma dimensao


# Cria a figura
fig = plt.figure(figsize = (14,6))

# Adiciona o subplot 1 com projeção 3d
ax = fig.add_subplot(1, 2, 1, projection = '3d')# projection tem que especificar as 3 dimenções
p = ax.plot_surface(X, Y, Z, rstride=4, cstride=4, linewidth=0)

# Adiciona o subplot 2 com projeção 3d
ax = fig.add_subplot(1, 2, 2, projection = '3d')
p = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Cria a barra de cores como legenda
cb = fig.colorbar(p, shrink=0.5) #colorbar para legenda

# ------------------------------------------------------------------------------------------------------------------

# # Wire frame

# Cria a figura
fig = plt.figure(figsize=(8,6))

# Subplot
ax = fig.add_subplot(1, 1, 1, projection = '3d')

# Wire frame
p = ax.plot_wireframe(X, Y, Z, rstride=4, cstride=4) # plot_wireframe para nao ter preenchimento

# ------------------------------------------------------------------------------------------------------------------

# # todo Visualização de Dados com Seaborn
# Imports
import random
import numpy as np
import pandas as pd
import matplotlib as mat
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")# para tirar as mensagens de wraning
import seaborn as sea

# Carregando um dos datasets que vem com o Seaborn
dados = sea.load_dataset("tips")
print(dados.head())

# O método joinplot cria plot de 2 variáveis com gráficos bivariados e univariados
sea.jointplot(data = dados, x = "total_bill", y = "tip", kind = 'reg')

# O método lmplot() cria plot com dados e modelos de regressão
sea.lmplot(data = dados, x = "total_bill", y = "tip", col = "smoker")

# Construindo um dataframe com Pandas
df = pd.DataFrame()

# Alimentando o Dataframe com valores aleatórios
df['idade'] = random.sample(range(20, 100), 30)
df['peso'] = random.sample(range(55, 150), 30)

print(df.shape)
print(df.head())

# lmplot
sea.lmplot(data = df, x = "idade", y = "peso", fit_reg = True)# fit_reg para criar um modelo de regreção, lmplot modelo linear

# kdeplot
sea.kdeplot(df.idade) # kdeplot grafico de densidade

# kdeplot
sea.kdeplot(df.peso)


# distplot
sea.distplot(df.peso)# une o histograma com o grafico de densidade

# Histograma
plt.hist(df.idade, alpha = .3)
sea.rugplot(df.idade) # adc tracinhos com rugplot

# Box Plot
sea.boxplot(df.idade, color = 'm')# grafico de caixa boxplot, para dividir os quartis

# Box Plot
sea.boxplot(df.peso, color = 'y')

# Violin Plot
sea.violinplot(df.idade, color = 'g')# violino violinplot

# Violin Plot
sea.violinplot(df.peso, color = 'cyan')

# Clustermap
sea.clustermap(df) # visualiza hierarquia completa clustermap

# Usando Matplotlib, Seaborn, NumPy e Pandas na Criação de Gráfico Estatístico

# Valores randômicos
np.random.seed(42)#seed para 
n = 1000
pct_smokers = 0.2#porcentagem

# Variáveis
flag_fumante = np.random.rand(n) < pct_smokers
# print(flag_fumante)
idade = np.random.normal(40, 10, n)
altura = np.random.normal(170, 10, n)
peso = np.random.normal(70, 10, n)
# print(idade)
# Dataframe
dados = pd.DataFrame({'altura': altura, 'peso': peso, 'flag_fumante': flag_fumante})# chave para coluna
# print(dados)
# Cria os dados para a variável flag_fumante
dados['flag_fumante'] = dados['flag_fumante'].map({True: 'Fumante', False: 'Não Fumante'})

print(dados.shape)
print(dados.head())

# Style
sea.set(style = "ticks")

# lmplot
sea.lmplot(x = 'altura', 
           y = 'peso', 
           data = dados, 
           hue = 'flag_fumante', #hue preenche os conteudos do grafico
           palette = ['tab:blue', 'tab:orange'], 
           height = 7)

# Labels e título
plt.xlabel('Altura (cm)')
plt.ylabel('Peso (kg)')
plt.title('Relação Entre Altura e Peso de Fumantes e Não Fumantes')

# Remove as bordas
sea.despine()

# Show
plt.show()

import matplotlib.pyplot as plt 
year = ['2010', '2002', '2004', '2006', '2008'] 
production = [25, 15, 35, 30, 10] 
plt.bar(year, production) 
plt.savefig("output.jpg") 
plt.savefig("output1", facecolor='y', bbox_inches="tight", 
            pad_inches=0.3, transparent=True)