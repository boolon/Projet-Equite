import sys, getopt
import surprise
from surprise import Dataset
from surprise.model_selection import cross_validate,train_test_split

from partnership import BasicPartner, powerlaw

from metrics import main_metric

def mesure_performance(model, data, train_set, test_set, cat_products, cat_target):
    # Il faudra qu'on ait un truc à nous pour mesurer vraiment les performances pour pouvoir changer les arguments
    cross_validate(model, data, measures=["RMSE","MAE"],cv = 5, verbose = True)
    # Pour l'instant on se contente de mesurer nos performances à la fin du script
    predictions = model.test(test_set)
    performance = main_metric(predictions, cat_products, cat_target)
    print(performance)


def main(argv):
    # Traitement des arguments
    try:
        opts, args = getopt.getopt(argv,"ham:d:c:",["method=","dataset=","nb-categories="])
    except getopt.GetoptError:
        sys.exit(2)

    model = surprise.SVD()
    dataset = "ml-100k"
    nb_categories = 10
    for opt, arg in opts:
        if opt=="-h":
            # Help pour l'utilisation de la fonction
            sys.exit(0)
        elif opt in ("-m","--method"):
            if arg == "SVD":
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
        elif opt in ("-c","nb-categories"):
            try:
                nb_categories=int(arg)
            except:
                print("Erreur dans l'argument de nb-categories")
                sys.exit(2)

    # Chargement des jeux de données
    if type(dataset)==str:
        data = Dataset.load_builtin(dataset, prompt = False)
    else:
        print("Erreur dans le dataset")
        sys.exit(2)

    train_set, test_set = train_test_split(data, test_size=.25)

    # Construction des Groupes de Produits et des Cibles de Produits
    cat_products = BasicPartner(nb_categories)
    cat_target = powerlaw(nb_categories)

    # Construction du Modèle
    if model == "SVD":
        model = surprise.SVD()
    else:
        print("Erreur dans le modèle : {}".format(model))
        sys.exit(2)

    # Mesure des performances
    mesure_performance(model, data, train_set, test_set,cat_products, cat_target)

    sys.exit(0)

if __name__ == "__main__":
   main(sys.argv[1:])
