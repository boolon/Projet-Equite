from surprise import AlgoBase
from surprise import SVD
from collections import defaultdict
import numpy as np

class duckSVD(AlgoBase):
    def __init__(self):
        """
        Le SVD avec un top1 à la fin
        """
        AlgoBase.__init__(self)
        self.SVD = SVD()

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.SVD.fit(trainset)
        return self

    def preprocess(self, test_set):

        self.predicted = dict()
        possible_prediction = defaultdict(list)
        for u, i, _ in test_set:
            possible_prediction[u].append((i, self.SVD.estimate(u,i)))
        for u in possible_prediction:
            max_sim = -1
            for el in possible_prediction[u]:
                if float(el[1])>max_sim:
                    max_sim = el[1]
                    self.predicted[int(u)]=int(el[0])


    def estimate(self, u, i):
        if type(i)==str and i.startswith("UKN"):
            return -1
        if int(u) in self.predicted:
            return int(self.predicted[int(u)]==int(i))
        else:
            return -1

class NaiveAlgo(AlgoBase):
    def __init__(self):
        """
        Cette méthode combinatoire cherche la meilleure solution et n'est absolument pas applicable sur des trucs trop grands
        Dans le cas d'un top 1, c'est un blossom algorithm
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
    def __init__(self, cat_products, cat_target):
        """
        Cette fonction décide de manière random pour chaque utilisateur dans quelle catégorie tirer les résultats.
        """
        AlgoBase.__init__(self)

        # Le modèle qui nous donne les \hat{r}_ij.
        self.SVD = SVD()

        # Les informations pour la partnership
        self.cat_products = cat_products
        self.cat_target = cat_target

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.SVD.fit(trainset)
        return self

    def preprocess(self, test_set):
        self.predicted = dict()
        possible_prediction = defaultdict(list)
        for u, i, _ in test_set:
            possible_prediction[u].append((i, self.SVD.estimate(u,i),self.cat_products[i]))

        for u in possible_prediction:
            # print([[el for el in possible_prediction[u] if el[2]==selected_category]!=[] for selected_category in range(10)])
            while 1:
                selected_category = np.random.choice(np.arange(0, len(self.cat_target)), p=self.cat_target)
                selected_possible = [el for el in possible_prediction[u] if el[2]==selected_category]
                if selected_possible !=[]:
                    break
            max_sim = -1

            for el in selected_possible:
                if el[1]>max_sim:
                    max_sim = el[1]
                    self.predicted[int(u)] = int(el[0])
        # print([len([1 for el in self.predicted if self.predicted[el]%10==i]) for i in range(10)])


    def estimate(self, u, i):
        if type(i)==str and i.startswith("UKN"):
            return -1
        if int(u) in self.predicted:
            return int(self.predicted[int(u)]==int(i))
        else:
            return -1


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
