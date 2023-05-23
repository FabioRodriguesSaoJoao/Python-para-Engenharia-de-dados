# Import
import random
#Função para limpar a tela
from os import system, name
def limpa_tela():
    if name =='int':
        _ = system('cls')

# Board (tabuleiro)
board = ['''

>>>>>>>>>>Hangman<<<<<<<<<<

+---+
|   |
    |
    |
    |
    |
=========''', '''

+---+
|   |
O   |
    |
    |
    |
=========''', '''

+---+
|   |
O   |
|   |
    |
    |
=========''', '''

 +---+
 |   |
 O   |
/|   |
     |
     |
=========''', '''

 +---+
 |   |
 O   |
/|\  |
     |
     |
=========''', '''

 +---+
 |   |
 O   |
/|\  |
/    |
     |
=========''', '''

 +---+
 |   |
 O   |
/|\  |
/ \  |
     |
=========''']


# Classe
class Hangman:

	# Método Construtor
     def __init__(self,palavra):
         self.palavra = palavra
         self.letras_erradas = []
         self.letras_escolhidas = []
	# Método para adivinhar a letra
     def guess(self,letra):
          if letra in self.palavra and letra not in self.letras_escolhidas:
             self.letras_escolhidas.append(letra)
          elif letra not in self.palavra and letra not in self.letras_erradas:
             self.letras_erradas.append(letra)
          else:
             return False
	# Método para verificar se o jogo terminou
     def hangman_over(self):
          return self.hangman_won() or (len(self.letras_erradas) == 6)
	# Método para verificar se o jogador venceu
     def hangman_won(self):
          if '_' not in self.hide_palavra():
              return True
          return False
     
	# Método para não mostrar a letra no board
     def hide_palavra(self):
          rtn = ''
          for letra in self.palavra:
               if letra not in self.letras_escolhidas:
                    rtn += '_'
               else:
                    rtn += letra	
          return rtn
	# Método para checar o status do game e imprimir o board na tela
     def print_game_status(self):
          print(board[len(self.letras_erradas)])
          print('\nPalavra: ' + self.hide_palavra())
          print('\nLetras erradas: ',)
          for letra in self.letras_erradas:
              print(letra,)
          print()
          print('letras corretas:',)
          for letra in self.letras_escolhidas:
              print(letra,)
          print()
#metodo para ler uma palavra 
def rand_palavra():
    #Lista de palavras para o jogo
    palavras = ['banana','abacate','uva','morango','laranja']
    #escolhe randomicamente uma palavra
    palavra = random.choice(palavras)
    return palavra
#metodo main - execução do programa
def main():
     limpa_tela()
    #cria o objeto e seleciona uma palavra randomicamente
     game = Hangman(rand_palavra())
    #enquanto o joto nao tiver terminado, print do status, solicita uma letra e faz a leitura do caracter
     while not game.hangman_over():
        #status do game
          game.print_game_status()
          #recebe input do terminal
          user_input = input('\n Digite uma letra: ')
          #verifica se a letra digitada faz parte da palavra
          game.guess(user_input)

     #verifica o status do jogo
     game.print_game_status()
     #de acordo com o status, imprime mensagem na tela para o usuario
     if game.hangman_won():
         print('\n parabens voce venceu!')
     else:
         print('\n Game over! Voce perdeu!')
         print("a palavra era " + game.palavra)
#executa o programa
if __name__ == "__main__":
     main()

