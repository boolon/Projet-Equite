# Projet-Equite
Projet Équité dans le Filtrage Collaboratif avec Romain Warlop et Cyrine Hlioui de 55.

# Use
`python -m method -d dataset -c number_of_categories`
Pour afficher les performances pour cette métrique sous la forme (combination similarité+équité,similarité, inéquité),temps d'exécution
`python -a -d dataset -c number_of_categories --nb-experiment n`
Pour tester un certain nombre de fois chaque méthode et enregistrer des graphiques des performances de ces méthodes

Argument possible pour method : SVD, perUser, global, relax
Argument possible pour dataset : ml-100k, ml-1m, jester
