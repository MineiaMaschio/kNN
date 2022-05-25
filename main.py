# Fábio Franz
# Matheus Pasold
# Minéia Maschio

import scipy.io as scipy

def meuKnn(dadosTrain, rotuloTrain, dadosTeste, k):

    #Para cada exemplo de teste
    #Calcule a distância entre o exemplo de teste e os dados de treinamento
    #Ordene as distâncias. A ordem iX de cada elemento ordenado é importante:
    #[distOrdenada ind] = sort(...);
    #O rótulo previsto corresponde ao rótulo do exemplo mais próximo (iX(1))


if __name__ == '__main__':
    # Fábio Franz
    # Matheus Pasold
    # Minéia Maschio

    import scipy.io as scipy

    mat = scipy.loadmat('grupoDados1.mat')

    dadosTrain = mat['grupoTrain']
    rotuloTrain = mat['trainRots']
    dadosTeste = mat['grupoTest']
    # rotuloTeste = mat['testRots']
    k = 5





