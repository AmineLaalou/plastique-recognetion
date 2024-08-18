import torch
dtype = torch.float
device = torch.device("cpu") # Tous les calculs seront exécutés sur le processeur
# device = torch.device("cuda:0") # Tous les calculs seront exécutés sur la carte graphique

# Création d'un tenseur rempli avec des nombres aléatoires
a = torch.randn(2, 3, device=device, dtype=dtype)
print(a) # Affichage du tenseur a
# Output: tensor([[-1.1884,  0.8498, -1.7129],
#                  [-0.8816,  0.1944,  0.5847]])

# Création d'un tenseur rempli avec des nombres aléatoires
b = torch.randn(2, 3, device=device, dtype=dtype)
print(b) # Affichage du tenseur b
# Output: tensor([[ 0.7178, -0.8453, -1.3403],
#                  [ 1.3262,  1.1512, -1.7070]])

print(a*b) # Affichage du produit (terme à terme) des deux tenseurs
# Output: tensor([[-0.8530, -0.7183,  2.58],
#                  [-1.1692,  0.2238, -0.9981]])

print(a.sum()) # Affichage de la somme de tous les éléments du tenseur a
# Output: tensor(-2.1540)

print(a[1,2]) # Affichage de l'élément de la 2ème rangée et de la 3ème colonne de a
# Output: tensor(0.5847)

print(a.min()) # Affichage de la valeur minimale du tenseur a
# Output: tensor(-1.7129)