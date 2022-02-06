# -*- coding: utf-8 -*-

# Vorlage Tolerierung korreliert


""" Bibliotheken importieren """

from scipy.io import loadmat    
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm, t, chi2, f
from scipy.stats import skew
import statsmodels.api as sm
from statsmodels.formula.api import ols
import sympy as syms    

""" Plots in 'HD' ausgeben :) """

plt.rcParams["figure.dpi"] = 300



""" Definition der gegebenen Parameter """

# Normalverteilung: Teilen durch das gegebene Sigma der Toleranz
# Gleichverteilung: Teilen durch np.sqrt(12)
# Dreieckverteilung: Teilen durch np.sqrt(24)

Uref_0 = 5
Uref_tol = 0.05
Uref_min = Uref_0 - Uref_tol / 2
Uref_max = Uref_0 + Uref_tol / 2
Uref_sig = Uref_tol / np.sqrt(12)

R1_0 = 100
R1_sig = 1
R1_tol = R1_sig * 6

R2_0 = 100
R2_sig = 1
R2_tol = R2_sig * 6

U_0 = R2_0 / (R1_0 + R2_0) * Uref_0

rho = 0.95



""" Linearisierung im Arbeitspunkt - Symbolische Differentiation """

# Variablen und Funktionen symbolisch definieren
Uref_sym, R1_sym, R2_sym, U_sym = syms.symbols('Uref_sym, R1_sym, R2_sym, U_sym')
U_sym = R2_sym / (R1_sym + R2_sym) * Uref_sym

# Empfindlichkeiten symbolisch berechnen
EUref_sym = U_sym.diff(Uref_sym)
ER1_sym = U_sym.diff(R1_sym)
ER2_sym = U_sym.diff(R2_sym)

# Werte definioeren
values = {Uref_sym:Uref_0, R1_sym:R1_0, R2_sym: R2_0}

#Empfindlichkeiten numerisch berechnen
EUref = float(EUref_sym.evalf(subs = values))
ER1 = float(ER1_sym.evalf(subs = values))
ER2 = float(ER2_sym.evalf(subs = values))



""" Linearisierung im Arbeitspunkt - Simulation und Regression """

# Generierung von korrelierten Zufallsgrößen
N = 10000
z1 = np.random.normal(0, 1, N)
z2 = np.random.normal(0, 1, N)
R1_sim = R1_sig * z1
R2_sim = rho * R2_sig * z1 + np.sqrt(1 - rho**2) * R2_sig * z2
R1_sim = R1_0 + R1_sim
R2_sim = R2_0 + R2_sim
Uref_sim = np.random.uniform(Uref_min, Uref_max, N)

# Korrelation mit numpy berechnen
Rkor = np.corrcoef(R1_sim, R2_sim)
print()
print('Korrelationskoeffizient mit numpy: rR1R2 =', round(Rkor[0, 1], 4))

# Korrelation mit Regression berechnen
regress = pd.DataFrame({'R1' : R1_sim,
                        'R2' : R2_sim})
poly = ols('R2 ~ R1', regress)
model = poly.fit()
print('Korrelationskoeffizient mit Regression: rR1R2 =', round(model.params[1], 4))



""" Arithmetische Tolerierung im Arbeitspunkt """

U_tol_ari = np.abs(EUref * Uref_tol) + np.abs(ER1 * R1_tol) + np.abs(ER2 * R2_tol)
print()
print("Toleranz bei arithmetischer Tolerierung deltaU =", round(U_tol_ari, 4))



""" Statistische Tolerierung - Grenzwertmethode """

U_tol_unk = 6 * np.sqrt((EUref * Uref_sig)**2 + (ER1 * R1_sig)**2 + (ER2 * R2_sig)**2)
U_tol_kor = 6 * np.sqrt((EUref * Uref_sig)**2 + (ER1 * R1_sig)**2 + (ER2 * R2_sig)**2
                        + 2 * rho * ER1 * ER2 * R1_sig * R2_sig)
print()
print('Toleranzbereich bei Grenzwertmethode ohne Korrelation: deltaU =', round(U_tol_unk, 4))
print('Toleranzbereich bei Grenzwertmethode mit Korrelation: deltaU =', round(U_tol_kor, 4))



""" Statistische Tolerierung - Simulation """

# In Vorlage bereits oben definiert

# Generierung von korrelierten Zufallsgrößen
#N = 10000
#z1 = np.random.normal(0, 1, N)
#z2 = np.random.normal(0, 1, N)
#R1_sim = R1_sig * z1
#R2_sim = rho * R2_sig * z1 + np.sqrt(1 - rho**2) * R2_sig * z2
#R1_sim = R1_0 + R1_sim
#R2_sim = R2_0 + R2_sim
#Uref_sim = np.random.uniform(Uref_min, Uref_max, N)

# Berechnung der Zielgröße und der statistischen Kennwerte
U_sim = R2_sim / (R1_sim + R2_sim) * Uref_sim
U_mean = np.mean(U_sim)
U_std = np.std(U_sim, ddof = 1)
Uplot = np.arange(2.35, 2.65, 0.001)
f_sim = norm.pdf(Uplot, U_mean, U_std)
F_sim = norm.cdf(Uplot, U_mean, U_std)

# Auswertung Toleranz als Prognoseintervall
gamma = 0.9973
c1 = t.ppf((1 - gamma) / 2, N - 1)
c2 = t.ppf((1 + gamma) / 2, N - 1)
U_tol_sim_t = U_std * np.sqrt(1 + 1 / N) * (c2 - c1)
print()
print('Toleranzbereich bei statistischer Simulation mit Prognoseintervall: deltaU =', round(U_tol_sim_t, 4))

# Auswertung Toleranz numerisch
U_sort = np.append(np.append(2.35, np.sort(U_sim)), np.max(U_sim))
U_cdf = np.append(np.append(0, np.arange(1, N + 1) / N), 1)
indexmin = np.min(np.where(U_cdf >= (1 - gamma) / 2))
indexmax = np.min(np.where(U_cdf >= (1 + gamma) / 2))
U_min_sim = U_sort[indexmin]
U_max_sim = U_sort[indexmax]
U_tol_sim_num = U_max_sim - U_min_sim
print('Toleranzbereich bei statistischer Simulation mit numerischer Auswertung: deltaU =', round(U_tol_sim_num, 4))
