# -*- coding: utf-8 -*-

# Vorlage Univariate Auswertung

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

# Einheit definieren
Einheit0 = 'Länge'
Einheit_sign = 'l'
Einheit_unit = r'µm'

#Daten laden und vorbereiten
# data0 = loadmat(Dateiname)['data']
data0 = np.random.normal(0, 1, 10000)
data = np.array(data0).flatten()

# Datenumfang und Spannweite
N = np.size(data)

# Lagekennwerte
data_mean = np.mean(data)

# Streuungskennwerte
data_var = np.var(data, ddof = 1)
data_std = np.std(data, ddof = 1)


#Daten laden und vorbereiten
#data0 = loadmat(Dateiname)['data']
data0 = np.random.normal(0, 1, 10000)
data2 = np.array(data0).flatten()

# Datenumfang und Spannweite
M = np.size(data2)

# Lagekennwerte
data_mean2 = np.mean(data2)

# Streuungskennwerte
data_var2 = np.var(data2, ddof = 1)
data_std2 = np.std(data2, ddof = 1)



""" Konfidenzbereiche """

# Mittelwert - bekannte Varianz - Standardnormalverteilung z
gamma = 0.9973
sigma = 1
c1 = norm.ppf((1 - gamma) / 2)
c2 = norm.ppf((1 + gamma) / 2)
mu_min = data_mean - c2 * sigma / np.sqrt(N)
mu_max = data_mean - c1 * sigma / np.sqrt(N)
print('Mittelwert:', round(mu_min, 4), Einheit_unit, '< µ <=', round(mu_max, 4), Einheit_unit)

# Mittelwert - unbekannte Varianz - t-Verteilung mit N - 1 FG
gamma = 0.9973
c1 = t.ppf((1 - gamma) / 2, N - 1)
c2 = t.ppf((1 + gamma) / 2, N - 1)
mu_min = data_mean - c2 * data_std / np.sqrt(N)
mu_max = data_mean - c1 * data_std / np.sqrt(N)
print('Mittelwert:', round(mu_min, 4), Einheit_unit, '< µ <=', round(mu_max, 4), Einheit_unit)

# Standardabweichung - Chi2-Verteilung mit N - 1 FG
gamma = 0.9973
c1 = chi2.ppf((1 - gamma) / 2, N - 1)
c2 = chi2.ppf((1 + gamma) / 2, N - 1)
sigma_min = np.sqrt((N - 1) / c2) * data_std
sigma_max = np.sqrt((N - 1) / c1) * data_std
print('Standardabweichung:', round(sigma_min, 4), Einheit_unit, '< sigma <=', round(sigma_max, 4), Einheit_unit)


# Differenz zweier Mittelwerte (dmu = mu1 - mu2) - bekannte Varianz - Standardnormalverteilung z
gamma = 0.9973
sigma = 1 # evtl Summe aus beiden Varianzen
c1 = norm.ppf((1 - gamma) / 2)
c2 = norm.ppf((1 + gamma) / 2)
dmu_min = (data_mean - data_mean2) - c2 * np.sqrt(sigma**2 * (1 / N + 1 / M))
dmu_max = (data_mean - data_mean2) - c1 * np.sqrt(sigma**2 * (1 / N + 1 / M))
print('Differenz Mittelwert:', round(dmu_min, 4), Einheit_unit, '< (µ1 - µ2) <=', round(dmu_max, 4), Einheit_unit)

# Differenz zweier Mittelwerte (dmu = mu1 - mu2) - unbekannte Varianz - t-Verteilung mit N + M -2
gamma = 0.9973
c1 = t.ppf((1 - gamma) / 2, N + M - 2)
c2 = t.ppf((1 + gamma) / 2, N + M - 2)
s = np.sqrt(((N - 1) * data_var + (M - 1) * data_var2) / (N + M -2))
dmu_min = (data_mean - data_mean2) - c2 * np.sqrt(1 / N + 1 / M) * s
dmu_max = (data_mean - data_mean2) - c1 * np.sqrt(1 / N + 1 / M) * s
print('Differenz Mittelwert:', round(dmu_min, 4), Einheit_unit, '< (µ1 - µ2) <=', round(dmu_max, 4), Einheit_unit)

# Verhältnis zweier Standardabweichungen (sigma2 / sigma1) - f-Verteilung mit (N - 1, M - 1) FG
gamma = 0.9973
c1 = f.ppf((1 - gamma) / 2, N - 1, M - 1)
c2 = f.ppf((1 + gamma) / 2, N - 1, M - 1)
rsigma_min = np.sqrt(c1 * data_var / data_var2)
rsigma_max = np.sqrt(c2 * data_var / data_var2)
print('Verhältnis der Standardabweichungen:', round(rsigma_min, 4), Einheit_unit, '< (sigma2 / sigma1) <=', round(rsigma_max, 4), Einheit_unit)


""" Prognosebereich """

# Bekannter Mittelwert, Bekannte Varianz - Normalverteilung z
gamma = 0.9973
mu = 0
sigma = 1
c1 = norm.ppf((1 - gamma) / 2)
c2 = norm.ppf((1 + gamma) / 2)
x_prog_min = mu + c1 * sigma
x_prog_max = mu + c2 * sigma
print('Prognosewert:', round(x_prog_min, 4), Einheit_unit, '< x <=', round(x_prog_max, 4), Einheit_unit)

# Unbekannter Mittelwert, Bekannte Varianz - Normalverteilung z
gamma = 0.9973
sigma = 1
c1 = norm.ppf((1 - gamma) / 2)
c2 = norm.ppf((1 + gamma) / 2)
x_prog_min = data_mean + c1 * sigma * np.sqrt(1 + 1 / N)
x_prog_max = data_mean + c2 * sigma * np.sqrt(1 + 1 / N)
print('Prognosewert:', round(x_prog_min, 4), Einheit_unit, '< x <=', round(x_prog_max, 4), Einheit_unit)

# Bekannter Mittelwert, Unbekannte Varianz - t-Verteilung mit N - 1 FG
gamma = 0.9973
mu = 0
c1 = t.ppf((1 - gamma) / 2, N - 1)
c2 = t.ppf((1 + gamma) / 2, N - 1)
x_prog_min = mu + c1 * data_std
x_prog_max = mu + c2 * data_std
print('Prognosewert:', round(x_prog_min, 4), Einheit_unit, '< x <=', round(x_prog_max, 4), Einheit_unit)

# Unbekannter Mittelwert, Unbekannte Varianz - t-Verteilung mit N - 1 FG
gamma = 0.9973
c1 = t.ppf((1 - gamma) / 2, N - 1)
c2 = t.ppf((1 + gamma) / 2, N - 1)
x_prog_min = data_mean + c1 * data_std * np.sqrt(1 + 1 / N)
x_prog_max = data_mean + c2 * data_std * np.sqrt(1 + 1 / N)
print('Prognosewert:', round(x_prog_min, 4), Einheit_unit, '< x <=', round(x_prog_max, 4), Einheit_unit)



""" Hypothesentest und Güte """

print()
# zb zur Erkennung von Systematischen Messfehlern
# Annahme für Grundgesamtheit aufstellen, Annahmegrenzen mit Stichprobenmittelwert bewerten
# Formeln aus Konfidenzbereich ableiten und je nachdem sigma einsetzen (normal)

# Vorgehen:
    
# Aufgabenstellung verbal beschreiben
# Ein systematischer Messfehler kann z. B. durch einen Test auf den Mittelwert mu_0 erkannt werden'
# Die Varianz ist unbekannt -> t-Verteilung mit N - 1 Freiheitsgraden'
# Mithilfe der Nullhypothese werden die Annahmegrenzen bestimmt'
# Die nominale Temperatur beträgt mu_0 = 0°C'

# Definition von Nullhypothese und Alternativhyypothese
print('Test auf Mittelwert bei unbekannter Varianz')
print('Nullhypothese H0: Keine signifikante Abweichung: mu = mu_0')
print('Alternativhypothese H1: Signifikante Abweichung: mu != mu_0')
mu_0 = 0

# Rückführen auf Verteilung
print('Die Stichprobe ist t-verteilt')

# Signifikanzniveau definieren
gamma = 0.95
alpha = 1 - gamma
print('alpha =', round(alpha, 4))

# Verwerfungsbereich definieren
print('Der Verwerfungsbereich ist beidseitig')

# Definition der Bereichsgrenzen
print('Mithilfe der konstanten c1 und c2 kann der Annahmebereich definiert werden:')
c1 = t.ppf((1 - gamma) / 2, N - 1)
print('c1 =', round(c1, 4))
c2 = t.ppf((1 + gamma) / 2, N - 1)
print('c2 =', round(c2, 4))
print('Der Annahmebereich bestimmt sich damit zu:')
data_mean_min = mu_0 - ((c2 * data_std) / np.sqrt(N))
data_mean_max = mu_0 - ((c1 * data_std) / np.sqrt(N))
print(round(data_mean_min, 4), Einheit_unit, '< x_quer <=', round(data_mean_max, 4), Einheit_unit)
if (data_mean > data_mean_min) & (data_mean <= data_mean_max):
    print('Der Mittelwert der Stichprobe x_quer =', round(data_mean, 4), Einheit_unit, 'liegt im Annahmebereichs. Die Nullhypothese wird nicht verworfen. Es gilt keine signifikante Abweichung')
else:
    print('Der Mittelwert der Stichprobe x_quer =', round(data_mean, 4), Einheit_unit, 'liegt nicht im Annahmebereichs. Die Nullhypothese wird verworfen. Es gilt eine signifikante Abweichung')


# Hypothesentest auf Mittelwert bei bekannter Varianz - z
mu_0 = 0
sigma = 1
gamma = 0.95

c1 = norm.ppf((1 - gamma) / 2)
c2 = norm.ppf((1 + gamma) / 2)
data_mean_min = mu_0 + (c1 * sigma / np.sqrt(N))
data_mean_max = mu_0 + (c2 * sigma / np.sqrt(N))
# Test ob data_mean in Annahmebereich
# x_plot-Vektor definieren und Güte berechnen
dmu = np.linspace(-0.1, 0.1, 10000)
Guete = 1 + norm.cdf((data_mean_min - dmu) / (sigma / np.sqrt(N))) - norm.cdf((data_mean_max - dmu) / (sigma / np.sqrt(N)))
# Abweichung um x wird mit y% erkannt


# Hypothesentest auf Mittelwert bei unbekannter Varianz - t mit N - 1 FG
mu_0 = 0
gamma = 0.95
c1 = t.ppf((1 - gamma) / 2, N - 1)
c2 = t.ppf((1 + gamma) / 2, N - 1)
data_mean_min = mu_0 + (c1 * data_std / np.sqrt(N))
data_mean_max = mu_0 + (c2 * data_std / np.sqrt(N))
# Test ob data_mean in Annahmebereich
# x_plot-Vektor definieren und Güte berechnen
dmu = np.linspace(-0.1, 0.1, 10000)
Guete = 1 + t.cdf((data_mean_min - dmu) / (data_std / np.sqrt(N)), N - 1) - t.cdf((data_mean_max - dmu) / (data_std / np.sqrt(N)), N - 1)
# Abweichung um x wird mit y% erkannt


# Hypothesentest Varianz - chi2 mit N - 1 FG
sigma_0 = 1
gamma = 0.95
c1 = chi2.ppf((1 - gamma) / 2, N - 1)
c2 = chi2.ppf((1 + gamma) / 2, N - 1)
sigma_min = np.sqrt(c1 / (N - 1) * sigma_0**2)
sigma_max = np.sqrt(c2 / (N - 1) * sigma_0**2)
# Test ob data_sigma in Annahmebereich
# x_plot-Vektor definieren und Güte berechnen
dsigma = np.linspace(0.95, 1.05, 10000)
Guete = 1 + chi2.cdf(sigma_min**2 / (dsigma**2) * (N - 1), N - 1) - chi2.cdf(sigma_max**2 / (dsigma**2) * (N - 1), N - 1)
# Abweichung um x wird mit y% erkannt


# Hypothesentest auf gleiche Mittelwerte bei bekannter Varianz
dmu_0 = 0
sigma = 1
gamma = 0.95
c1 = norm.ppf((1 - gamma) / 2)
c2 = norm.ppf((1 + gamma) / 2)
data_dmean_min = dmu_0 + c1 * np.sqrt(sigma**2*(1 / N + 1 / M))
data_dmean_max = dmu_0 + c2 * np.sqrt(sigma**2*(1 / N + 1 / M))
# Test ob data_dmean in Annahmebereich
# x_plot-Vektor definieren und Güte berechnen
dmu = np.linspace(-0.1, 0.1, 10000)
Guete = 1 + norm.cdf((data_dmean_min - dmu) / np.sqrt(sigma**2 * (1 / N + 1 / M))) - norm.cdf((data_dmean_max - dmu) / np.sqrt(sigma**2 * (1 / N + 1 / M)))
# Abweichung um x wird mit y% erkannt


# Hypothesentest auf gleiche Mittelwerte bei unbekannter Varianz - t mit N + M - 2 FG
# Nullhypothese sagt aus das die Differenz beider Mittelwerte 0 sein sollte..
dmu_0 = 0
gamma = 0.95
c1 = t.ppf((1 - gamma) / 2, N + M - 2)
c2 = t.ppf((1 + gamma) / 2, N + M - 2)
data_s = ((N - 1) * data_std**2 + (M - 1) * data_std2**2) / (N + M - 2)
data_dmean_min = dmu_0 + c1 * np.sqrt(1 / N + 1 / M) * data_s
data_dmean_max = dmu_0 + c2 * np.sqrt(1 / N + 1 / M) * data_s
# Test ob data_dmean in Annahmebereich
# x_plot-Vektor definieren und Güte berechnen
dmu = np.linspace(-0.1, 0.1, 10000)
Guete = 1 + t.cdf((data_dmean_min - dmu) / (np.sqrt(1 / N + 1 / M) * data_s), N + M - 2) - norm.cdf((data_dmean_max - dmu) / (np.sqrt(1 / N + 1 / M) * data_s), N + M - 2)
# Abweichung um x wird mit y% erkannt


# Hypothesentest auf gleiche Varianz - f mit (N - 1, M - 1) FG
rsigma_0 = 1
gamma = 0.95
c1 = f.ppf((1 - gamma) / 2, N - 1, M - 1)
c2 = f.ppf((1 + gamma) / 2, N - 1, M - 1)
data_rsigma_min = np.sqrt(c1 * rsigma_0)
data_rsigma_max = np.sqrt(c2 * rsigma_0)
# Test ob data_rsigma in Annahmebereich
# x_plot-Vektor definieren und Güte berechnen
rsigma = np.linspace(0.75, 1.25, 10000)
Guete = 1 + f.cdf(data_rsigma_min* 1 / rsigma, N - 1, M - 1) - f.cdf(data_rsigma_max* 1 / rsigma, N - 1, M - 1)
# Abweichung um x wird mit y% erkannt



""" Gütefunktion """

print()
# Vorher muss Hypothesentest stattgefunden haben
# Untersucht wird der Drift des Mittelwerts vom wahren Wert
# Mit steigender Abweichung erhöht sich die Wahrscheinlichkeit der Korrekten Einstufung


# Standardnormalverteilung

# x_plot-Vektor definieren und Güte berechnen
dmu = np.linspace(-0.1, 0.1, 10000)
Guete = 1 + norm.cdf((data_mean_min - dmu) / (data_std / np.sqrt(N))) - norm.cdf((data_mean_max - dmu) / (data_std / np.sqrt(N)))

# Index berechnen und zugehörige Wahrscheinlichkeit ermitteln
index = np.min(np.where(dmu >= -0.02))
x_index = dmu[index]
P_index = Guete[index]

# Wahrscheinlichkeit ausgeben
print('Wahrscheinlichkeit der Erkennung einer Abweichung um +- 0.02: ', round(P_index, 4))



# t-Verteilung

# x_plot-Vektor definieren und Güte berechnen
dmu = np.linspace(-0.1, 0.1, 10000)
Guete = 1 + t.cdf((data_mean_min - dmu) / (data_std / np.sqrt(N)), N - 1) - t.cdf((data_mean_max - dmu) / (data_std / np.sqrt(N)), N - 1)

# Index berechnen und zugehörige Wahrscheinlichkeit ermitteln
index = np.min(np.where(dmu >= -0.02))
x_index = dmu[index]
P_index = Guete[index]

# Wahrscheinlichkeit ausgeben
print('Wahrscheinlichkeit der Erkennung einer Abweichung um +- 0.02: ', round(P_index, 4))



# Gütefunktion visualisieren
fig1 = plt.figure()
ax1 = fig1.subplots()

ax1.plot(dmu, Guete)

ax1.grid()
ax1.set_xticks([-0.1, -x_index, 0, x_index, 0.1])
ax1.set_yticks([0, P_index, 1])
