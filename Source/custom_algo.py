from surprise import AlgoBase
from surprise import SVD
from surprise import PredictionImpossible
from collections import defaultdict
import numpy as np
import sys
from itertools import product
import heapq
import autograd.numpy as agnp
from autograd import grad
from gradient_descent import gradient_descent





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
        return -1

class PerUserAlgo(AlgoBase):
    def __init__(self, cat_products, cat_target):
        """
        Cette fonction décide de manière aléatoire pondéré par la fréquence cible
        pour chaque utilisateur dans quelle catégorie tirer les résultats.
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
            custom_target = np.zeros(len(self.cat_target))
            for i in range(len(self.cat_target)):
                if [el for el in possible_prediction[u] if el[2]==i]!=[]:
                    custom_target[i] = self.cat_target[i]
            custom_target/=np.sum(custom_target)
            selected_category = np.random.choice(np.arange(0, len(self.cat_target)), p=custom_target)
            selected_possible = [el for el in possible_prediction[u] if el[2]==selected_category]

            max_sim = -1

            for el in selected_possible:
                if el[1]>max_sim:
                    max_sim = el[1]
                    self.predicted[int(u)] = int(el[0])



    def estimate(self, u, i):
        return -1






class GlobalProportionAlgo(AlgoBase):
    def __init__(self, cat_products, cat_target):
        """
        Cette méthode consiste à recommander peu à peu des objets en prenant à chaque fois l'objet avec la meilleure similarité
        dans la catégorie des objets qui est le plus loin de sa valeur cible en proportion parmi les résultat déjà obtenus
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
        C = len(self.cat_target)
        self.predicted = dict()
        heaps = [[] for _ in range(C)]
        n = 0
        current_prop = np.zeros(C)
        # We use C heaps to gather the similarities
        for u, i, _ in test_set:
            heapq.heappush(heaps[self.cat_products[i]],(self.SVD.estimate(u,i),u,i))
        while 1:
            if n == 0:
                selected_category = np.argmax(np.array([heap==[] for heap in heaps]))
            else:
                status = current_prop - self.cat_target*(n+1)
                status = np.abs(np.clip(status,a_min = None, a_max = 0))
                for c in range(C):
                    if heaps[c]==[]:
                        status[c]=-1
                selected_category = np.argmax(status)

            continu = True
            while heaps[selected_category]!=[] and continu:
                est, u, i = heapq.heappop(heaps[selected_category])
                if not (int(u) in self.predicted):
                    self.predicted[int(u)]=int(i)
                    current_prop[selected_category]+=1
                    n+=1
                    continu = False
            if heaps == [[] for _ in range(C)]:
                return


    def estimate(self, u, i):
        return -1



class MeanScoreRelaxation(AlgoBase):
    def __init__(self,cat_products, cat_target, lamb = 10000, latent_dimension = 10,
                 mu = 10, lr =0.0005, alpha = 0.99, nb_epochs = 1, nb_main_epochs = 50):
        """
        Cette méthode cherche à faire correspondre score moyen et performances
        Le résultat est très mauvais avec cette méthode
        """
        AlgoBase.__init__(self)
        self.latent_dimension = latent_dimension
        self.lamb = 10
        self.mu = 10

        # Les informations pour la partnership
        self.cat_products = cat_products
        self.cat_target = cat_target

        # The AlgoBase got its own functions and dictionnaries to remember the
        # know uid and iid, but we choose to use our own to have a better control
        # on it as we don't need the estimate function
        self.u_to_raw = dict()
        self.i_to_raw = dict()
        self.raw_to_u = dict()
        self.raw_to_i = dict()

        self.number_raw_u = 0
        self.number_raw_i = 0

        #Parameters of the gradient descent
        self.lr = lr
        self.alpha = alpha
        self.nb_epochs = nb_epochs
        self.nb_main_epochs = nb_main_epochs

    def add_to_known(self, u, i):

        if u in self.u_to_raw:
            ru = self.u_to_raw[u]
        else:
            self.u_to_raw[u]= self.number_raw_u
            self.number_raw_u +=1
            ru = self.u_to_raw[u]
            self.raw_to_u[ru] = u
        if i in self.i_to_raw:
            ri = self.i_to_raw[i]
        else:
            self.i_to_raw[i]= self.number_raw_i
            self.number_raw_i +=1
            ri = self.i_to_raw[i]
            self.raw_to_i[ri] = i
        return ru, ri

    def from_known_ri(self,ri):
        return self.raw_to_i[ri]

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        n = trainset.n_users
        m = trainset.n_items
        #print(n,m)
        self.K = agnp.zeros((n,m))
        self.R = agnp.zeros((n,m))
        for u, i, rating in trainset.all_ratings():
            ru, ri = self.add_to_known(u,i)
            self.K[ru,ri]=1
            self.R[ru,ri]=rating

        self.U = agnp.random.normal(size = (n, self.latent_dimension))
        self.M = agnp.random.normal(size = (self.latent_dimension, m))
        self.C = agnp.array([[self.cat_products[self.from_known_ri(ri)] == c for c in range(len(self.cat_target))] for ri in range(m)])



        self.fun_U = lambda U : (agnp.sum(self.K*(self.R - agnp.dot(U,self.M))**2)+ self.mu * (agnp.sum(U**2) + agnp.sum(self.M**2))
                                    +self.lamb*agnp.sum((1/n *  agnp.dot(agnp.dot(agnp.ones(n),agnp.dot(U, self.M)),self.C) -self.cat_target)**2))
        self.fun_M = lambda M : (agnp.sum(self.K*(self.R - agnp.dot(self.U,M))**2)+ self.mu * (agnp.sum(self.U**2) + agnp.sum(M**2))
                                    +self.lamb*agnp.sum((1/n *  agnp.dot(agnp.dot(agnp.ones(n),agnp.dot(self.U, M)),self.C) -self.cat_target)**2))

        self.grad_U = grad(self.fun_U)
        self.grad_M = grad(self.fun_M)


        for epoch in range(self.nb_main_epochs):
            self.M = gradient_descent(self.M, self.grad_M, N = 1, lr = self.lr, alpha = 1)
            self.U = gradient_descent(self.U, self.grad_U, N = 1, lr = self.lr, alpha = 1)
            self.lr*=self.alpha

        return self

    def preprocess(self, test_set):
        self.predicted = dict()
        possible_prediction = defaultdict(list)
        for u, i, _ in test_set:
            possible_prediction[u].append((i, self.estimate(u,i)))
        for u in possible_prediction:
            max_sim = -10000
            for el in possible_prediction[u]:
                if float(el[1])>max_sim:
                    max_sim = el[1]
                    self.predicted[int(u)]=int(el[0])


    def estimate(self, u, i):
        # This estimate function DOES NOT comply with the predict and test methods, but we could modify it so it would while compliying with our
        if (type(u)==str and u.startswith("UKN")) or (type(i) ==str and i.startswith("UKN")):
            return -1
        if int(i) in self.i_to_raw and int(u) in self.u_to_raw:
            ri, ru = self.i_to_raw[int(i)], self.u_to_raw[int(u)]
            return np.dot(self.U[ru],self.M[:,ri])
        else:
            return -1
