import os
import numpy as np
import matplotlib.pyplot as plt

dossier = r"C:\Users\noah\OneDrive - Université Laval\A26\Travaux pratiques avancés\géophysique\Data"
#changer dans votre répertoire

espacement = 0.5    # distance entregéophones
n_sigma = 5         # seuil de détection : n_sigma fois le bruit (à changer au besoin)

# (premier scope, dernier scope, distance source - géophone)
# on ignore scope 45
groupes = [
    (0, 2, 0.1),
    (3, 5, 0.2),
    (6, 8, 0.3),
    (9, 11, 0.4),
    (12, 14, 0.5),
    (15, 17, 0.6),
    (18, 20, 0.7),
    (21, 23, 0.8),
    (24, 26, 0.9),
    (27, 29, 1.0),
    (30, 32, 1.5),
    (33, 35, 2.0),
    (36, 38, 2.5),
    (39, 41, 3.0),
    (42, 44, 5.0),
    (46, 48, 6.0),
    (49, 51, 3.5),
    (52, 54, 3.8),
    (55, 57, 4.2),
    (58, 60, 3.25),
    (61, 63, 3.1),
    (64, 66, 2.8),
]


def temps_arrivee(fichier):
    data = np.loadtxt(fichier, delimiter=",", skiprows=2)
    t = data[:, 0] * 1000          #en ms
    trig = data[:, 4]              # canal 4 déclencheur de la masse

    #le déclencheur tombe sous la moitié de sa valeur de départ
    t0 = t[np.argmax(trig < np.mean(trig[:50]) / 2)]

    temps = []
    for canal in [1, 2, 3]:
        v = data[:, canal]
        bruit = v[t < t0]
        moy, sigma = np.mean(bruit), np.std(bruit)

        # 1) premier point après le coup qui sort du bruit
        candidats = (t > t0) & (np.abs(v - moy) > n_sigma * sigma)
        if not candidats.any():
            temps.append(np.nan)
            continue
        i = np.argmax(candidats)

        #début de l'onde
        while i > 0 and np.abs(v[i - 1] - moy) > sigma:
            i -= 1
        temps.append(t[i] - t0)
    return temps


# boucle pour les scopes
print(f"{'fichier':<14}{'géophone':>9}{'distance (m)':>14}{'temps (ms)':>12}")

x, t = [], []

for debut, fin, d in groupes:
    for n in range(debut, fin + 1):
        nom = f"scope_{n}.csv"
        chemin = os.path.join(dossier, nom)

        if not os.path.exists(chemin):
            print(f"{nom:<14}  fichier introuvable")
            continue

        temps = temps_arrivee(chemin)
        for k in range(3):
            print(f"{nom:<14}{k + 1:>9}{d + k * espacement:>14.2f}{temps[k]:>12.2f}")
            x.append(d + k * espacement)
            t.append(temps[k])
            
plt.plot(x, t, "o", color="k", ms=4)
plt.xlabel("Distance (m)")
plt.ylabel("Temps (ms)")
plt.xlim(0)
plt.ylim(0)
plt.grid(alpha=0.5)
plt.show()
            