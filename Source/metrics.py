import numpy as np
from collections import defaultdict

def main_metric(predictions,cat_products, cat_target, K = 1, lamb = 1, verbose = False, model = None):
    assert model!=None
    assert K==1

    top_K = defaultdict(list)
    #for uid in model.predicted:
    #    top_K[uid].append((model.predicted[uid],None,[el[2] for el in predictions if int(el[0])==uid and int(el[1])==int(model.predicted[uid])][0]))
    for uid, iid, true_r, _, _ in predictions:
        if int(iid) == int(model.predicted[int(uid)]):
            top_K[int(uid)].append((int(iid),None,true_r))
    tot_similarity = 0
    proportions = np.zeros(len(cat_target))
    n = 0
    for uid, ratings in top_K.items():
        n+=1
        for rating in ratings:
            # On gagne en similarité en fonction du rating véritable
            tot_similarity+=rating[2]
            # On ajoute aux proportions en fonction de l'appartenance
            proportions[cat_products[rating[0]]]+=1
    regularization_term = np.sum((proportions/K/n-cat_target)**2)
    return tot_similarity + lamb * (1+regularization_term), tot_similarity, regularization_term
