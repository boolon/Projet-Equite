import sys, getopt
import surprise
from surprise import Dataset
from surprise.model_selection import cross_validate,train_test_split

from partnership import BasicPartner, powerlaw,geolaw

from metrics import main_metric

from custom_algo import *
from draw import draw_moustaches

import time

def build_dataset(dataset):
    # Chargement des jeux de données
    if dataset in ("ml-100k", "ml-1m", "jester",):
        data = Dataset.load_builtin(dataset, prompt = False)
        train_set, test_set = train_test_split(data, test_size=.50)
    else:
        print("Erreur dans le dataset")
        sys.exit(2)
    return data, train_set, test_set
    
def build_model(model,cat_products,cat_target):
    # Construction du Modèle
    if model == "SVD":
        model = duckSVD()
    elif model == "perUser":
        model = PerUserAlgo(cat_products,cat_target)
    elif model == "global":
        model = GlobalProportionAlgo(cat_products, cat_target)
    elif model == "relax":
        model = MeanScoreRelaxation(cat_products, cat_target)
    else:
        print("Erreur dans le modèle : {}".format(model))
        sys.exit(2)
    return model

def mesure_performance(model, data, train_set, test_set, cat_products, cat_target, K = 10, verbose = True):
    t0 = time.time()
    if verbose:print("Training...")
    model.fit(train_set)
    if verbose:print("Preprocessing...")
    model.preprocess(test_set)
    t = time.time()-t0
    # We use the tests function to gather the true_r for the predictions but the estimated r are not correctly estimated because we only care about one predicted element given by a parameter of the model
    predictions = model.test(test_set)
    if verbose:print("Computing performance...")
    performance = main_metric(predictions, cat_products, cat_target, K = K, model = model)
    return performance,t


def main(argv):
    # Traitement des arguments
    try:
        opts, args = getopt.getopt(argv,"ham:d:c:",["method=","dataset=","nb-categories=","all", "nb-experiment="])
    except getopt.GetoptError:
        sys.exit(2)

    model = "SVD"
    K = 1 # We can't support anything besides K = 1 for the top_K
    dataset = "ml-100k"
    nb_experiment = 1
    nb_categories = 10
    mode_all = False
    for opt, arg in opts:
        if opt=="-h":

            sys.exit(0)
        elif opt in ("-m","--method"):
            if arg in ("SVD", "perUser", "global", "relax"):
                model = arg
            else:
                print("Erreurs dans l'argument de method : {}".format(arg))
                sys.exit(2)
        elif opt in ("-d", "--dataset"):
            if arg in ("ml-100k", "ml-1m", "jester"):
                dataset = arg
            else:
                print("Erreurs dans l'argument de dataset: {}".format(arg))
                sys.exit(2)
        elif opt in ("--nb-experiment"):
            try:
                nb_experiment = int(arg)
            except ValueError:
                print("Erreur dans l'argument de nb-experiment : {}".format(arg))
                sys.exit(2)
        elif opt in ("-a", "all"):
            mode_all = True
        elif opt in ("-c","--nb-categories"):
            try:
                nb_categories=int(arg)
            except ValueError:
                print("Erreur dans l'argument de nb-categories : {}".format(arg))
                sys.exit(2)
        else:
            print("Erreur dans l'option : {}".format(opt))
            sys.exit(2)

    if mode_all:
        # Dans le mode_all contrôlé par le paramètre -a, des tests sont effectués sur toutes les méthodes et des graphiques sont créés en sortie
        seeds = [np.random.randint(1,10000000) for _ in range(nb_experiment)]
        models = ("SVD", "perUser", "global", "relax")
        nb_models = len(models)
        similarities =[[] for i in range(nb_models)]
        equities = [[] for i in range(nb_models)]
        times = [[] for i in range(nb_models)]
        for i in range(nb_models):
            for seed in seeds:
                np.random.seed(seed)
                data, train_set, test_set = build_dataset(dataset)
                cat_products = BasicPartner(nb_categories)
                cat_target = geolaw(nb_categories)
                model = build_model(models[i],cat_products,cat_target)

                (_, sim, eq),t = mesure_performance(model, data, train_set, test_set, cat_products, cat_target, K = 1, verbose = False)

                similarities[i].append(sim)
                equities[i].append(eq)
                times[i].append(t)
                print(seed, models[i], sim, eq, t)
        draw_moustaches(similarities, models, "la similarité", dataset)
        draw_moustaches([equities[1]], [models[1]], "l'équité", dataset)
        draw_moustaches([equities[2]], [models[2]], "l'équité", dataset)
        draw_moustaches([equities[0], equities[3]], [models[0], models[3]], "l'équité", dataset)
        draw_moustaches([times[-1]], [models[-1]], "le temps d'exécution", dataset)
        draw_moustaches(times[:-1], models[:-1], "le temps d'exécution", dataset)

    else:
        my_seed = 55
        np.random.seed(my_seed)

        data, train_set, test_set = build_dataset(dataset)

        # Construction des Groupes de Produits et des Cibles de Produits
        cat_products = BasicPartner(nb_categories)
        cat_target = geolaw(nb_categories)

        # Construction du Modèle
        model = build_model(model,cat_products,cat_target)

        # Mesure des performances et affichage des performances
        print(mesure_performance(model, data, train_set, test_set,cat_products, cat_target, K = K))



if __name__ == "__main__":
   main(sys.argv[1:])
   sys.exit(0)
