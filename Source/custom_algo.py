from surprise import AlgoBase
from surprise import SVD
from collections import defaultdict
import numpy as np
import sys
from itertools import product






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

    def __init__(self, cat_products, cat_target,A = 1000):
        """
        Cette méthode combinatoire cherche la meilleure solution et n'est absolument pas applicable sur des trucs trop grands
        Dans le cas d'un top 1, c'est un blossom algorithm

        """
        AlgoBase.__init__(self)

        # Le modèle qui nous donne les \hat{r}_ij.
        self.SVD = SVD()

        # Les informations pour la partnership
        self.cat_products = cat_products
        self.cat_target = cat_target
        self.A=A
        self.nb_categories = len(cat_products)


    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.SVD.fit(trainset)
        return self



    #find the mapping that maximizes the score function
    #best_mapping=max(self.possible_predictions(n_users, test_set), key=score)

    def preprocess(self,test_set):
        U=list(set([u for u,_,_ in test_set]))
        I=list(set([i for _,i,_ in test_set]))
        n_users=len(U)
        n_items=len(I)
        cats = set([self.cat_products[i] for i in I])
        r = {}
        categories = {}
        best_item = {}

        for u in U:
            categories[u] = []

            for i in I:
                cat_i = self.cat_products[i]
                estimation = self.SVD.estimate(u,i)

                if not(cat_i in categories[u]):
                    categories[u] += [cat_i]
                    best_item[(u, cat_i)] = [i, estimation]

                r[(u,i)] = estimation
                best_item_u_i = best_item[(u, cat_i)][1]

                if estimation > best_item_u_i:
                    best_item[(u, cat_i)] = [i, estimation]

        N = self.nb_categories ** n_users
        print("   nombre de cas à traiter : "+str(N))
        print("")
        #print("N = "+str(N))
        possibles_predictions = [{} for k in range(N)]
        possibles_predictions_bis = [zip(U, item) for item in product(cats, repeat=len(U))]
        #print(len(possibles_predictions_bis))
        possibles_predictions_scores = []
        count = 0
        gain_max = 0
        prediction_dic = {}
        D = 10
        Av = [k * int(N / D) for k in range(D + 1)]
        tiret = ""

        for pred in possibles_predictions_bis:
            #print(count)
            #a = (100 * count / N) / 10


            if count in Av:
                tiret = tiret + "-"
                a_printer = "      avancement : "+str(round(100 * count / N, 1))+"%"
                longueur = len(a_printer)
                diff = 30 - longueur

                for k in range(diff):
                    a_printer += " "

                print(a_printer + tiret)

            loc = []
            count_category = [0 for c in cats]
            loc_func = {}

            for el in pred:
                bil = best_item[el]
                possibles_predictions[count][el[0]] = bil[0]
                loc += [bil[1]]
                loc_func[el[0]] = bil[0]
                count_category[el[1]] += 1

            su = sum(loc)
            freq_category = np.array(count_category) / self.nb_categories
            gain_freq = self.A / (sum((freq_category - np.array(self.cat_target)) ** 2))
            gain_tot = su + gain_freq

            if gain_tot > gain_max:
                gain_max = gain_tot
                prediction_dic = loc_func

            count += 1

        self.predicted = prediction_dic


    def estimate(self, u, i):
        if type(i)==str and i.startswith("UKN"):
            return -1
        if int(u) in self.predicted:
            return int(self.predicted[int(u)]==int(i))
        else:
            return -1





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
