import sys, getopt
import surprise
from surprise import Dataset
from surprise.model_selection import cross_validate

def mesure_performance(model, data):
    cross_validate(model, data, measures=["RMSE","MAE"],cv = 5, verbose = True)

def main(argv):
    # Traitement des arguments
    try:
        opts, args = getopt.getopt(argv,"ham:d:",["method=","dataset="])
    except getopt.GetoptError:
        sys.exit(2)
    model = surprise.SVD()
    dataset = "ml-100k"
    for opt, arg in opts:
        if opt=="-h":
            # Help pour l'utilisation de la fonction
            sys.exit(0)
        elif opt in ("-m","--method"):
            if arg == "SVD":
                model = surprise.SVD()
            else:
                print("Erreurs dans l'argument de method : {}".format(arg))
                sys.exit(2)
        elif opt in ("-d", "--dataset"):
            if arg in ("ml-100k", "ml-1m"):
                dataset = arg
            else:
                print("Erreurs dans l'argument de dataset: {}".format(arg))
                sys.exit(2)

    # Chargement des jeux de données
    if type(dataset)==str:
        data = Dataset.load_builtin(dataset, prompt = False)

    # Mesure des performances
    mesure_performance(model, data)

    sys.exit(0)

if __name__ == "__main__":
   main(sys.argv[1:])
