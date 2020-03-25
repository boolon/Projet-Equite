from surprise import AlgoBase
from surprise import SVD

class duckSVD(SVD):
    def preprocess(self, test_set):
        pass

class NaiveAlgo(AlgoBase):
    def __init__(self):
        """
        Cette méthode combinatoire cherche la meilleure solution et n'est absolument pas applicable sur des trucs trop grands
        """
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        return self

    def preprocess(self, test_set):
        pass

    def estimate(self, u, i):

        return 1

class PerUserAlgo(AlgoBase):
    def __init__(self):
        """
        Cette méthode applique les proportions pour chaque utilisateur
        """
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        return self
    def preprocess(self, test_set):
        pass

    def estimate(self, u, i):

        return 1

class GlobalProportionAlgo(AlgoBase):
    def __init__(self):
        """
        Cette méthode consiste à recommander peu à peu des objets en prenant à chaque fois l'objet avec la meilleure similarité
        dans la catégorie des objets qui est le plus loin de sa valeur cible en proportion parmi les résultat déjà obtenus
        """
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        return self
    def preprocess(self, test_set):
        pass

    def estimate(self, u, i):

        return 1

class MeanScoreRelaxation(AlgoBase):
    def __init__(self):
        """
        Cette méthode utilise une relaxation convexe décrite dans le PDF.
        """
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        return self
    def preprocess(self, test_set):
        pass

    def estimate(self, u, i):

        return 1
