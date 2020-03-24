from surprise import AlgoBase

class NaiveAlgo(AlgoBase):
    def __init__(self):
        """
        Cette méthode combinatoire cherche la meilleure solution et n'est absolument pas applicable sur des trucs trop grands
        """
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        return self

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

    def estimate(self, u, i):

        return 1
