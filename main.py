# Fábio Franz
# Matheus Pasold
# Minéia Maschio

import scipy.io as scipy
import math
import numpy as np
import statistics
import matplotlib.pyplot as plt
from operator import itemgetter, attrgetter


def getDadosRotulo(dados, rotulos, rotulo, indice):
    ret = []

    for idx in range(0, len(dados)):

        if (rotulos[idx] == rotulo):
            ret.append(dados[idx][indice])

    return ret


def visualizaPontos(dados, rotulos, d1, d2):
    fig, ax = plt.subplots()

    ax.scatter(getDadosRotulo(dados, rotulos, 1, d1), getDadosRotulo(dados, rotulos, 1, d2), c='red', marker='^')

    ax.scatter(getDadosRotulo(dados, rotulos, 2, d1), getDadosRotulo(dados, rotulos, 2, d2), c='blue', marker='+')

    ax.scatter(getDadosRotulo(dados, rotulos, 3, d1), getDadosRotulo(dados, rotulos, 3, d2), c='green', marker='.')

    plt.show()

def normalizacao(dados):
    min = np.amin(dados, axis=0)
    max = np.amax(dados, axis=0)
    for i in range(len(dados[0])):
        for x in range(len(dados)):
            divisor = dados[x][i] - min[i]
            dividendo = max[i] - min[i]
            dados[x][i] = divisor / dividendo

def dist(dadosTrain, dadosTeste, rotuloTrain):
    #Lista de distâncias
    listDist = []

    #Para cada dado de teste
    for i in range(len(dadosTrain)):
        soma = 0
        d = []

        #Para cada atributo do teste calcular distancia euclidiana
        for x in range(len(dadosTeste)):
            soma += pow((dadosTrain[i][x] - dadosTeste[x]),2)

        #Adiciona distância
        d.append(math.sqrt(soma))

        #Adiciona rotulo
        d.append(rotuloTrain[i][0])

        #Adiciona aos resultados
        listDist.append(d)

    return listDist

def meuKnn(dadosTrain, rotuloTrain, dadosTeste, k):
    #Lista de resultados
    listResult = []

    #Normalização
    #normalizacao(dadosTrain)
    #normalizacao(dadosTeste)

    #Para cada exemplo de teste
    for x in range(len(dadosTeste)):
        listRotulos = []
        listRotuloMode = []

        #Calcule a distância entre o exemplo de teste e os dados de treinamento
        listDist = dist(dadosTrain, dadosTeste[x], rotuloTrain)

        # Ordenar lista
        listSorted = sorted(listDist, key=itemgetter(0))

        # Cria uma nova lista com k sendo a quantidade de itens
        listK = listSorted[:k]

        # Pega apenas os rotulos
        for x in range(len(listK)):
            listRotulos.append(listK[x][1])

        #Calcula a moda
        listRotuloMode.append(statistics.mode(listRotulos))

        #Adiciona ao resultado
        listResult.append(listRotuloMode)

    return listResult

if __name__ == '__main__':
    import scipy.io as scipy

    mat = scipy.loadmat('grupoDados1.mat')

    dadosTrain = mat['grupoTrain']
    rotuloTrain = mat['trainRots']
    dadosTeste = mat['grupoTest']
    testRots = mat['testRots']
    k = 1

    list = []
    for x in range(len(testRots)):
        list.append(testRots[x][0])


    rotuloPrevisto = meuKnn(dadosTrain, rotuloTrain, dadosTeste, 1)

    estaCorreto = rotuloPrevisto == testRots

    numCorreto = sum(estaCorreto)

    totalNum = len(testRots)

    acuracia = np.round((numCorreto / totalNum), 2)

    print('Acurácia do KNN com k = 1')
    print(acuracia)

    rotuloPrevisto2 = meuKnn(dadosTrain, rotuloTrain, dadosTeste, 10)

    estaCorreto = rotuloPrevisto2 == testRots

    numCorreto = sum(estaCorreto)

    totalNum = len(testRots)

    acuracia = np.round((numCorreto / totalNum), 2)

    print('Acurácia do KNN com k = 10')
    print(acuracia)

    visualizaPontos(dadosTeste, rotuloPrevisto2, 1, 2)



