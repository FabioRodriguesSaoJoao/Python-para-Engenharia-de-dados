# # Construir um modelo de Inteligência Artificial capaz de classificar imagens considerando 10 categorias: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']. Dada uma nova imagem de uma dessas categorias o modelo deve ser capaz de classificar e indicar o que é a imagem.


# Imports
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Carrega o dataset CIFAR-10
(imagens_treino, labels_treino), (imagens_teste, labels_teste) = datasets.cifar10.load_data()

# Clases das imagens
nomes_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Normaliza os valores dos pixels para que os dados fiquem na mesma escala
imagens_treino = imagens_treino / 255.0 #(255.0 é o maximo de posições de um pixel)
imagens_teste = imagens_teste / 255.0

# # Função para exibir as imagens
def visualiza_imagens(images, labels):
    plt.figure(figsize = (10,10))# figsize = (10,10) define o tamanho da figura
    for i in range(25):
        plt.subplot(5, 5, i+1)# (5, 5, i+1) imprime 5 imagem por linha, em 5 linhas
        plt.xticks([])# removendo os tiks de linhas
        plt.yticks([])
        plt.grid(False)# removendo o grid para so ter as imagens
        plt.imshow(images[i], cmap = plt.cm.binary) # imshow para mpstrar as imagens, pegando cada imagem do indice i images[i]
        plt.xlabel(nomes_classes[labels[i][0]])
    plt.show()

# # Executa a função
visualiza_imagens(imagens_treino, labels_treino)

# # Modelo

# Cria o objeto de sequência de camadas ( cada camada é uma equação matematica)
modelo = models.Sequential()

# # Adiciona o primeiro bloco de convolução e max pooling (camada de entrada)
modelo.add(layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (32, 32, 3))) # quantidade de neuronios, (3, 3) matriz de convolução , input_shape = (32, 32, 3) como é a matriz de entrada, devemos colocar o input_shape, 32,32 altura e largura, 3 por ser colorida
modelo.add(layers.MaxPooling2D((2, 2))) # uma matriz 2x2

# Adiciona o segundo bloco de convolução e max pooling (camada intermediária)
modelo.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
modelo.add(layers.MaxPooling2D((2, 2)))

# Adiciona o terceiro bloco de convolução e max pooling (camada intermediária)
modelo.add(layers.Conv2D(64, (3, 3), activation = 'relu'))
modelo.add(layers.MaxPooling2D((2, 2)))

# Adicionar camadas de classificação
modelo.add(layers.Flatten()) # achatando
modelo.add(layers.Dense(64, activation = 'relu'))
modelo.add(layers.Dense(10, activation = 'softmax')) # 10 classes

# Sumário do modelo
# modelo.summary()

# Compilação do modelo
modelo.compile(optimizer = 'adam', #paramentro do back propagation
                   loss = 'sparse_categorical_crossentropy', # loss = 'sparse_categorical_crossentropy',  função de erro
                   metrics = ['accuracy']) #metrics = ['accuracy'] metrica de sucesso


history = modelo.fit(imagens_treino, # fit para treinar o modelo
                         labels_treino, 
                         epochs = 10, #epochs = 10, treinar por 10 passadas
                         validation_data = (imagens_teste, labels_teste)) #validation_data = (imagens_teste, labels_teste) avaliação do treino

# # Avalia o modelo
erro_teste, acc_teste = modelo.evaluate(imagens_teste, labels_teste, verbose = 2)

print('\nAcurácia com Dados de Teste:', acc_teste)

# Formação Engenheiro de Machine Learning

# Carrega uma nova imagem
nova_imagem = Image.open("nova_imagem.jpg")

# Dimensões da imagem (em pixels)print
print(nova_imagem.size)

# Obtém largura e altura da imagem
largura = nova_imagem.width
altura = nova_imagem.height


print("A largura da imagem é: ", largura)
print("A altura da imagem é: ", altura)


# Redimensiona para 32x32 pixels
nova_imagem = nova_imagem.resize((32, 32))



# Exibir a imagem
plt.figure(figsize = (1,1))
plt.imshow(nova_imagem)
plt.xticks([])
plt.yticks([])
plt.show()

# Converte a imagem para um array NumPy e normaliza
nova_imagem_array = np.array(nova_imagem) / 255.0

# Expande a dimensão do array para que ele tenha o formato (1, 32, 32, 3)
nova_imagem_array = np.expand_dims(nova_imagem_array, axis = 0) 

# # Previsões
previsoes = modelo.predict(nova_imagem_array)

print(previsoes)

# Obtém a classe com maior probabilidade e o nome da classe
classe_prevista = np.argmax(previsoes) # argmax para pegar o maior valor da lista
nome_classe_prevista = nomes_classes[classe_prevista]

print("A nova imagem foi classificada como:", nome_classe_prevista)