import numpy as np

class BasicPartner:
    def __init__(self, nb_partner = 10):
        self.nb_partner = nb_partner
    def __len__(self,):
        return self.nb_partner
    def __get_item__(self, idx):
        return idx%self.nb_partner


def powerlaw(n, power = 2):
    """
    Cette fonction veut renvoyer une répartition en puissance de somme nulle et de taille n
    """
    result = np.zeros(n)
    for i in range(n):
        result[i]  = power ** i
    return result/np.sum(result)
