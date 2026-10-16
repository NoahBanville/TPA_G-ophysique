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
    (45, 46, 6.0),
    (48, 48, 3.5),
    (50, 51, 3.5),
    (52, 54, 3.8),
    (55, 57, 4.2),
    (58, 60, 3.25),
    (61, 63, 3.1),
    (64, 66, 2.8),
]
# géophones à ignorer (scope, géophone) : la première onde ne sort pas du bruit
exclure = [(42, 2), (53, 2), (56, 2), (45, 1), (46, 1), (45, 2), (46, 2), (45, 3)]

def temps_arrivee(fichier):
    data = np.loadtxt(fichier, delimiter=",", skiprows=2)
    t = data[:, 0] * 1000          #en ms
    trig = data[:, 4]              # canal 4 déclencheur de la masse

    #le déclencheur tombe sous la moitié de sa valeur de départ
    t0 = t[np.argmax(trig < np.mean(trig[:50]) / 2)]

    temps = []
    for canal in [1, 2, 3]:
        v = data[:, canal]
        v = np.convolve(v - np.mean(v[t < t0]), np.ones(5) / 5, mode="same")
        bruit = v[t < t0]
        moy, sigma = np.mean(bruit), np.std(bruit)

        # 1) premier point après le coup qui sort du bruit
        dehors = np.abs(v - moy) > n_sigma * sigma
        # le signal doit rester hors du bruit 3 points de suite (0,3 ms)
        dehors = dehors[:-2] & dehors[1:-1] & dehors[2:]
        candidats = (t[:-2] > t0) & dehors
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
            if (n, k + 1) in exclure:
                continue
            x.append(d + k * espacement)
            t.append(temps[k])
            
# ---------------- Moyenne à chaque distance ----------------
x = np.round(np.array(x), 2)     
t = np.array(t)
ok = ~np.isnan(t)                # on enlève les géophones sans détection
x, t = x[ok], t[ok]

xs = np.unique(x)                                             # distances différentes
t_moy = np.array([np.mean(t[x == d]) for d in xs])            # moyenne
n = np.array([np.sum(x == d) for d in xs])                    # nombre de mesures
t_std = np.array([np.std(t[x == d], ddof=1) if np.sum(x == d) > 1 else 0 for d in xs])
t_err = t_std / np.sqrt(n)                                    # incertitude sur la moyenne

print()
print(f"{'distance (m)':>12}{'n':>4}{'moyenne (ms)':>14}{'incertitude (ms)':>18}")
for k in range(len(xs)):
    print(f"{xs[k]:>12.2f}{n[k]:>4}{t_moy[k]:>14.2f}{t_err[k]:>18.2f}")

# ---------------- Vitesses des deux couches ----------------
c1 = x <= 2                      # couche 1 : points de 0 à 2 m
c2 = (x >= 3) & (x <= 7)         # couche 2 : points de 3 à 7 m
(a1, b1), cov1 = np.polyfit(x[c1], t[c1], 1, cov=True)   # t = a1 x + b1
(a2, b2), cov2 = np.polyfit(x[c2], t[c2], 1, cov=True)   # t = a2 x + b2

# pente en ms/m  ->  vitesse en m/s
V1 = 1000 / a1
V2 = 1000 / a2
sV1 = V1 * np.sqrt(cov1[0, 0]) / a1
sV2 = V2 * np.sqrt(cov2[0, 0]) / a2

print(f"V1 = {V1:.0f} ± {sV1:.0f} m/s")
print(f"V2 = {V2:.0f} ± {sV2:.0f} m/s")


xc = (b2 - b1) / (a1 - a2)
tc = a1 * xc + b1

plt.plot([0, xc], [b1, tc], "--", label=f"V1 = {V1:.0f} m/s")
plt.plot([xc, 7], [tc, a2 * 7 + b2], "--", label=f"V2 = {V2:.0f} m/s")
plt.axvline(xc, color="black", ls="-.")
plt.legend()
print(f"Point de rencontre : x = {xc:.2f} m, t = {tc:.1f} ms")

plt.errorbar(xs, t_moy, yerr=t_err, fmt="o", color="k", ms=4, capsize=3)
plt.xlabel("Distance (m)")
plt.ylabel("Temps moyen (ms)")
plt.xlim(0)
plt.ylim(0)
plt.grid(alpha=0.5)
plt.show()
            