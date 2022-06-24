# Fábio Franz
# Matheus Pasold
# Minéia Maschio

import demoD1
import scipy.io as scipy
import numpy as np

if __name__ == '__main__':
    mat = scipy.loadmat('grupoDados3.mat')

    dadosTrain = mat['grupoTrain']
    rotuloTrain = mat['trainRots']
    dadosTeste = mat['grupoTest']
    testRots = mat['testRots']

    rotuloPrevisto = demoD1.meuKnn(dadosTrain, rotuloTrain, dadosTeste, 1)

    estaCorreto = rotuloPrevisto == testRots

    numCorreto = sum(estaCorreto)

    totalNum = len(testRots)

    acuracia = np.round((numCorreto / totalNum), 2)

    print('Q3.1: Aplique o kNN ao problema usando k = 1. Qual é a acurácia na classificação?')
    print('Acurácia do KNN com k = 1')
    print(acuracia)

    print('\n')
    demoD1.normalizacao(dadosTrain)
    demoD1.normalizacao(dadosTeste)
    demoD1.calcularAcuracias(dadosTrain, rotuloTrain, dadosTeste, testRots)

    print('\nQ3.2: A acurácia pode ser igual a 92% com o kNN. Descubra por que o resultado atual é muito menor. Ajuste o conjunto de dados ou k de tal forma que a acurácia se torne 92% e explique o que você fez e por quê.')
    print('Foi aplicado a normalização dos dados nos dados de treinamento e teste para que para evitar que as medidas de distância sejam dominadas por uma única característica.')
    print('Depois testamos com k do tamanho da base de testes (50) obtivemos alguns resultados com 92% de acurácia como k = 14 e também com 94% como k = 25')