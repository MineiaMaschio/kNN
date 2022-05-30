# Fábio Franz
# Matheus Pasold
# Minéia Maschio

import scipy.io as scipy
import math
import numpy as np
import statistics
from operator import itemgetter, attrgetter

def normalizacao(dados):
    min = np.amin(dados, axis=0)
    max = np.amax(dados, axis=0)
    for i in range(len(dados[0])):
        for x in range(len(dados)):
            divisor = dados[x][i] - min[i]
            dividendo = max[i] - min[i]
            dados[x][i] = divisor / dividendo

def distanciaEuclidiana(dadosTrain, dadosTeste, rotuloTrain):
    listDist = []

    for i in range(len(dadosTrain)):
        soma = 0
        d = []
        for x in range(len(dadosTeste)):
            soma += pow((dadosTrain[i][x] - dadosTeste[x]),2)
        d.append(math.sqrt(soma))
        d.append(rotuloTrain[i])
        listDist.append(d)

    return listDist

def meuKnn(dadosTrain, rotuloTrain, dadosTeste, k):
    list = []
    listDist = []
    listK = []
    listSorted = []
    listResult = []

    #Normalização
    #normalizacao(dadosTrain)
    #normalizacao(dadosTeste)

    #Para cada exemplo de teste
    for x in range(len(dadosTeste)):
        #Calcule a distância entre o exemplo de teste e os dados de treinamento
        listDist = distanciaEuclidiana(dadosTrain, dadosTeste[x], rotuloTrain)

        # Ordenar lista
        listSorted = sorted(listDist, key=itemgetter(0))

        for i in range(len(listSorted)):
            list.append(listSorted[i][1])

        listK = list[:k]

        listResult.append(statistics.mode(listK))

    print(listResult)

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

    print(list)
    rotuloPrevisto = meuKnn(dadosTrain, rotuloTrain, dadosTeste, k)

    estaCorreto = rotuloPrevisto == testRots

    numCorreto = sum(estaCorreto)

    totalNum = len(testRots)

    acuracia = np.round((numCorreto / totalNum), 2)

    print(acuracia)



