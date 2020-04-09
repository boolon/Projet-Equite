import matplotlib.pyplot as plt
import numpy as np
import os

def draw_moustaches(X, models, mesure, dataset):
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    bp = ax.boxplot([np.array(x) for x in X], patch_artist=True)

    ax.set_xticklabels(models)

    n = len(X[0])
    fig.suptitle("Diagramme à moustache pour {} experiences sur différents modèles\n utilisant le dataset {} comparant {}".format(n, dataset, mesure))
    models_str = ""
    for model in models:
        models_str+=model

    os.makedirs(os.path.join("..","output"), exist_ok = True)
    fig.savefig(os.path.join("..","output",'{}{}{}{}.png'.format(n,dataset,mesure[4:6],models_str)), bbox_inches='tight')
