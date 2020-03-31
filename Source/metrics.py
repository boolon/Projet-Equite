import numpy as np
from collections import defaultdict
from custom_algo import NaiveAlgo

def get_top_K(predictions, K=10):
    '''
    From surprise documentation
    Return the top-K recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        K(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''

    # First map the predictions to each user.
    top_K = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_K[uid].append((iid, est, true_r))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_K.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_K[uid] = user_ratings[:K]
    return top_K

def main_metric(predictions,cat_products, cat_target, K = 10, lamb = 1, verbose = False, model = None):
    if model!= None:
        top_K = defaultdict(list)
        for uid in model.predicted:
            top_K[uid].append((model.predicted[uid],None,[el[2] for el in predictions if int(el[0])==uid and int(el[1])==int(model.predicted[uid])][0]))
    else:
        top_K =get_top_K(predictions, K = K)
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
    regularization_term = lamb / (1+np.sum((proportions/K/n-cat_target)**2))
    return tot_similarity + regularization_term, tot_similarity, regularization_term
