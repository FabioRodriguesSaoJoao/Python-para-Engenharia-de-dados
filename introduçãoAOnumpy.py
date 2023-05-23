
#TODO NUMPY
import numpy as np
arr1 = np.array([10, 21, 32, 43, 48, 15, 76, 57, 89])
print(arr1)
print(type(arr1))
print(arr1.shape) # resultado :9, 9 elementos no array, tendo um eixo (x)
#indexação fatiação
print(arr1[4])
print(arr1[1:4])
print(arr1[1:4+1])#puxando o index 4 tbm
indices = [1,2,5,6]
print(arr1[indices])#puxando os valores do arr1 usando a lista "indices" como indeces para puxar os valores
mask = (arr1 % 2 == 0)
print(arr1[mask])#retornando numeros pares dentro do arr1
arr1[0] = 100
print(arr1)

#nao é possivel incluir elemento de outro tipo
try:
    arr1[0] = "Novo elemento"
except:
    print("operação nao permitida!")

#todo trabalhando comfunções numpy
arr2 = np.array([1,2,3,4,5])
print(arr2)
print(arr2.cumsum())#soma acumulada
print(arr2.cumprod())#produto acumulado
arr3 = np.arange(0,50,5)# criando uma lista, começando com zero, terminando com 50 e pulando de 5 em 5
print(arr3)
print(np.shape(arr3))#formato do array
print(arr3.dtype)
arr4 = np.zeros(10)# array feito de zeros de tamanho 10
arr5 = np.eye(3) # array diagonal preenchido com um
arr6 = np.diag(np.array([1,2,3,4])) #os valores passados cria uma matriz diagonal
arr7 = np.array([True,False])
arr8 = np.array(["Linguagem Python", "Linguagem R", 'Linguagem Julia'])
print(np.linspace(0,10)) #sequencia de numeros com valores conhecidos
print(np.linspace(0,10,15))
print(np.logspace(0,5,10))# agora com padrao logaritmicos


# todo MANIPULANDO MATRIZES
arr9 = np.array([[1,2,3],[4,5,6]]) #listas de listas
print(arr9)
print(arr9.shape)# matriz com duas linhas e 3 colunas
arr10 = np.ones((2,3)) #criando uma matriz 2x3 apenas com numeros 1
lista = [[13,81,22],[0,34,59,],[21,48,94]]
arr11 = np.matrix(lista)# a função matrix cria uma matriz a partir de uma lista de listas
print(arr11)
print(np.shape(arr11))
print((arr11.size))
print(arr11.dtype)

# indexação com duas dimenções
print(arr11[2,1])#linha 2 coluna 1 #indece começa por 0
print(arr11 [0:2,2])
print(arr11 [1,])# linha de indice 1 e todas as colunas
arr11[1,0] = 100
print(arr11)
x = np.array([1,2])
y = np.array([1.0,2.0])
z = np.array([1,2], dtype=np.float64)#forçando um tipo de dado em particular
print(x.dtype,y.dtype,z.dtype)
arr12 = np.array([[24,76,94,14],[47,35,89,2]],dtype=float)
print(arr12)
#verificando o tamanho do tamanho do array
print(arr12.itemsize)#em bytes
print(arr12.nbytes)#total de bytes do array
print(arr12.ndim)#quantidade de dimensoes

# todo manipulando objetos de 3 e 4 dimensoes com NumPy
arr_3d = np.array([
   [ 
    [1,2,3,9],
    [4,5,6,9],
    [7,8,9,9]
    ],
    [
    [10,11,12,9],
    [13,14,15,9],
    [16,17,18,9],
    ]
])
print(arr_3d)
print(arr_3d.ndim)#3d
print(arr_3d.shape)#2 listas, 3 linhas, 4 colunas
print(arr_3d[0,2,1])#primeira lista, ultima linha, coluna 2
arr_4d = np.array([[
    [
    [1,2,3,4,5],
    [6,7,8,9,10],
    [11,12,13,14,15],
    [16,17,18,19,20],
    ],
    [
    [12,22,23,24,25],
    [26,27,28,29,30],
    [31,32,33,34,35],
    [36,37,38,39,40],
    ],
    [
    [41,42,43,44,45],
    [46,47,48,49,50],
    [51,52,53,54,55],
    [56,57,58,59,60],
    ],
],
[
    [
    [61,62,63,64,65],
    [66,67,68,69,70],
    [71,72,73,74,75],
    [76,77,78,79,80],
    ],
    [
    [81,82,83,84,85],
    [86,87,88,89,90],
    [91,92,93,94,95],
    [96,97,98,99,100],
    ],
    [
    [101,102,103,104,105],
    [106,107,108,109,110],
    [111,112,113,114,115],
    [116,117,118,119,120],
    ]
]])
print(arr_4d)
print(arr_4d.ndim)
print(arr_4d.shape)
print(arr_4d[0,2,1])
print(arr_4d[0,2,1,4])

# todo Manipulando arquivos com Numpy
import os
filename = os.path.join("dataset.csv")#pegou o caminho
arr13 = np.loadtxt(filename,delimiter=',', usecols= (0,1,2,3), skiprows=1)# delimiter delimitador # usecols pegando somente as colunas que desejo. skiprows = 1 pulando a primeira linha
print(arr13)
# carregando apenas duas variaveis (colunas ) do arquivo
var1,var2 = np.loadtxt(filename, delimiter=',', usecols=(0,1), skiprows=1, unpack=True)
print(var1,var2)
#gerando um plot a partir de um arquivo usando numpy
# import matplotlib.pyplot as plt
(plt.show(plt.plot(var1, var2, 'o', markersize = 6, color = 'red')))

# todo Estatistica basica
arr14 = np.array([15,23,63,94,75])
#media
m = np.mean(arr14)
#desvio padrao
dp = np.std(arr14)# se o dp forem alto, entao os valores estao distante da media 

#variancia
vari = np.var(arr14)
print(m,dp,vari)

# todo operações matematicas
arr15 = np.arange(1,10)
#somando os elementos
s = np.sum(arr15)
#soma acumulada dos elementos
sa = np.cumsum(arr15)
print(s,sa)
arr16 = np.array([3,2,1])
arr17 = np.array([1,2,3])
#soma dos arrays
arr18 = np.add(arr16,arr17)
print(arr18)
arr19 =  np.array([[1,2],[3,4]])
arr20 =  np.array([[5,6],[0,7]])
#multiplicar as duas matrizes, NO 1°ARRAY A QUANTIDADE DE colunas TEM QUE SER IGUAL A QuaNTIDADE DE linhas DO 2° ARRAY
arr21 = np.dot(arr19,arr20)
print(arr21)
arr22 = arr19 @ arr20
print(arr22)
arr23 = np.tensordot(arr19,arr20,axes = ((1),(0)))
print(arr23)
# 
# TODO FATIAMENTO DE ARRAYS
arr22 = np.diag(np.arange(3))
print(arr22)
print(arr22[1,1])
print(arr22[1])
print(arr22[:,2])
arr23 = np.arange(10)
print(arr23)
print(arr23[2:9:3])# start:end:step
a=np.array([1,2,3,4])
b=np.array([4,2,2,4])
#comparação
print(a==b)# retorna boolean
print(np.array_equal(a,b))
print(arr23.min())
print(arr23.max())

#somando um valor a cada elemento do array
i = np.array([1,2,3])+1.5
print(i)
arr24 = np.array([1.2,1.5,1.6,2.5,3.5,4.5])
print(np.around(arr24))#arrebondando

arr25 = np.array([[1,2,3,4],[5,6,7,8]])#duas dimensões
arr27 = arr25.flatten() # achata para 1 dimensão
print(arr27)
arr28 = np.array([1,2,3])
print(np.repeat(arr28,3)) # repete os valores 3x
print(np.tile(arr28,3)) # repete os valores 3x em sequencia
print(np.copy(arr28))# copiando o array