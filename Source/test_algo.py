import sys, getopt
import surprise
from surprise import Dataset
from surprise.model_selection import cross_validate,train_test_split

from partnership import BasicPartner, powerlaw, geolaw

from metrics import main_metric

from custom_algo import *

from main import mesure_performance

dataset = "ml-100k"
data = Dataset.load_builtin(dataset, prompt = False)
train_set, test_set = train_test_split(data, test_size=.25)

#Je ne prends que les 6 premiers bonhommes
test_set_bis = [x for x in test_set if x[0] in ['1', '2', '3', '4', '5', '6']]

#Je construis les catégories
nb_categories = 10
cat_products = BasicPartner(nb_categories)
cat_target = geolaw(nb_categories)
total_cat = set([cat_products[i] for _,i,_ in test_set_bis])
print("Catégories : "+str(total_cat))
print("")

#Je construis le modèle

model = NaiveAlgo(cat_products,cat_target)

#Je lance le test
mesure_performance(model, data, train_set, test_set_bis, cat_products, cat_target, K = 1)
