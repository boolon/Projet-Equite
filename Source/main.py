import sys, getopt
import surprise
from surprise import Dataset
from surprise.model_selection import cross_validate,train_test_split

from partnership import BasicPartner, powerlaw,geolaw

from metrics import main_metric

from custom_algo import *

def mesure_performance(model, data, train_set, test_set, cat_products, cat_target, K = 10):
    # Il faudra qu'on ait un truc à nous pour mesurer vraiment les performances pour pouvoir changer les arguments
    # cross_validate(model, data, measures=["RMSE","MAE"],cv = 5, verbose = True)
    # Pour l'instant on se contente de mesurer nos performances à la fin du script
    print("Training...")
    print("")
    model.fit(train_set)
    print("Preprocessing...")
    print("")
    model.preprocess(test_set)
    print("")
    print("Predicting...")
    print()
    predictions = model.test(test_set)
    print("Computing performance...")
    print("")
    performance = main_metric(predictions, cat_products, cat_target, K = K)
    print("Performance : "+str(performance))


def main(argv):
    # Traitement des arguments
    try:
        opts, args = getopt.getopt(argv,"ham:d:c:",["method=","dataset=","nb-categories="])
    except getopt.GetoptError:
        sys.exit(2)

    model = "SVD"
    K = 10
    dataset = "ml-100k"
    nb_categories = 5
    for opt, arg in opts:
        if opt=="-h":
            # Help pour l'utilisation de la fonction
            sys.exit(0)
        elif opt in ("-m","--method"):
            if arg in ("SVD", "perUser","naiveAlgo"):
                model = arg
            else:
                print("Erreurs dans l'argument de method : {}".format(arg))
                sys.exit(2)
        elif opt in ("-d", "--dataset"):
            if arg in ("ml-100k", "ml-1m", "jester","simonml"):
                dataset = arg
            else:
                print("Erreurs dans l'argument de dataset: {}".format(arg))
                sys.exit(2)
        elif opt in ("-c","nb-categories"):
            try:
                nb_categories=int(arg)
            except:
                print("Erreur dans l'argument de nb-categories")
                sys.exit(2)


    my_seed = 1995
    random.seed(my_seed)
    np.random.seed(my_seed)

    # Chargement des jeux de données
    if dataset in ("ml-100k", "ml-1m", "jester",):
        data = Dataset.load_builtin(dataset, prompt = False)
        train_set, test_set = train_test_split(data, test_size=.25)
    elif dataset == "simonml":
        data = Dataset.load_builtin("ml-100k", prompt = False)
        train_set, test_set = train_test_split(data, test_size=.25)
        test_set = [x for x in test_set if x[0] in ['1', '2', '3', '4', '5', '6']]
    else:
        print("Erreur dans le dataset")
        sys.exit(2)


    # Construction des Groupes de Produits et des Cibles de Produits
    cat_products = BasicPartner(nb_categories)
    cat_target = geolaw(nb_categories)

    # Construction du Modèle
    if model == "SVD":
        model = duckSVD()
    elif model == "perUser":
        model = PerUserAlgo(cat_products,cat_target)
        K = 1
    elif model == "naiveAlgo":
        model = NaiveAlgo(cat_products, cat_target)
        if dataset!="simonml":
            print("Warning : Computation has only few chances (none, tbh) to end if the dataset is to big")
    else:
        print("Erreur dans le modèle : {}".format(model))
        sys.exit(2)

    # Mesure des performances
    mesure_performance(model, data, train_set, test_set,cat_products, cat_target, K = K)

    sys.exit(0)

if __name__ == "__main__":
   main(sys.argv[1:])
