# -*- coding: utf-8 -*-

# Vorlage Tolerierung Unkorreliert


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

Q_0 = 0.5

Uadc_0 = 5
Uadc_tol = 0.5
Uadc_sig = Uadc_tol / 6

Uoff_0 = 0.01
Uoff_min = 0
Uoff_max = 0.02
Uoff_tol = 0.02
Uoff_sig = Uoff_tol / np.sqrt(12)

NA_0 = 0
NA_min = -1 / 2048
NA_max = 1 / 2048
NA_tol = 2 / 2048
NA_sig = NA_tol / np.sqrt(12)

k_0 = 0.2
m_0 = 0.04



""" Sollverhalten darstellen """

print()
print('a)')
print('Die Gleichung wird durch einsetzen der Gleichungen ineinander erreicht.')
print('Sollverhalten: Siehe Diagramm')

# Sollbereich definieren
Q_soll = np.arange(0, 1 + 0.01, 0.01)
N_soll = N = m_0 * Q_soll**2 / k_0**2 * ( 1 + Uoff_0 / Uadc_0) + NA_0

# Sollverhalten plotten
fig1 = plt.figure()
fig1.suptitle('Sollverhalten')
ax1 = fig1.add_subplot(111)
ax1.plot(Q_soll, N_soll, label = 'Sollverhalten')
ax1.set_xlabel('Durchfluss Q in m³/s')
ax1.set_ylabel('Normierter ADC-Wert $N$')
ax1.grid(True)
ax1.legend()



""" Linearisierung im Arbeitspunkt - Symbolische Differentiation """

# Variablen und Funktion symbolisch definieren
Q_sym, Uadc_sym, Uoff_sym, NA_sym, k_sym, m_sym = syms.symbols('Q_sym, Uadc_sym, Uoff_sym, NA_sym, k_sym, m_sym')
N_sym = m_sym * Q_sym**2 / k_sym**2 * (1 + Uoff_sym / Uadc_sym) + NA_sym

# Symbolische Berechnung der Empfindlichkeiten
EUadc_sym = N_sym.diff(Uadc_sym)
EUoff_sym = N_sym.diff(Uoff_sym)
ENA_sym = N_sym.diff(NA_sym)
EQ_sym = N_sym.diff(Q_sym) # Für spätere Umrechnung der Toleranz auf Durchfluss

# Werte definieren im Arbeitspunkt
values = {Q_sym:Q_0, Uadc_sym:Uadc_0, Uoff_sym:Uoff_0, NA_sym:NA_0, k_sym:k_0, m_sym:m_0}

# Empfindlichkeiten Numerisch berechnen
EUadc = float(EUadc_sym.evalf(subs = values))
EUoff = float(EUoff_sym.evalf(subs = values))
ENA = float(ENA_sym.evalf(subs = values))
EQ = float(EQ_sym.evalf(subs = values))

print()
print("b)")
print("Empfindlichkeit UADC =", round(EUadc, 4))
print("Empfindlichkeit UOFF =", round(EUoff, 4))
print("Empfindlichkeit NA =", round(ENA, 4))



""" Linearisierung im Arbeitspunkt - Simulation und Regression """

# Anzahl der Samples festlegen
N = 10000
Uadc_sim = np.random.normal(Uadc_0, Uadc_sig, N)
Uoff_sim = np.random.uniform(Uoff_min, Uoff_max, N)
NA_sim = np.random.uniform(NA_min, NA_max, N)
N_sim = m_0 * Q_0**2 / k_0**2 * (1 + Uoff_sim / Uadc_sim) + NA_sim

# Regressionsfunktion aufstellen
y_regress = pd.DataFrame({'Uadc': Uadc_sim.reshape(-1),
                          'Uoff': Uoff_sim.reshape(-1),
                          'NA': NA_sim.reshape(-1),
                          'N': N_sim.reshape(-1)})
poly = ols("N ~ Uadc + Uoff + NA", y_regress)
model = poly.fit()
print()
print("c)")
print("Empfindlichkeit UADC =", round(model.params.Uadc, 4))
print("Empfindlichkeit UOFF =", round(model.params.Uoff, 4))
print("Empfindlichkeit NA =", round(model.params.NA, 4))


""" Arithmetische Tolerierung im Arbeitspunkt """

N_tol_ari = np.abs(EUadc * Uadc_tol) + np.abs(EUoff * Uoff_tol) + np.abs(ENA * NA_tol)
print()
print('d)')
# Ziel ist die Toleranzangabe von deltaQ, daher teilen durch EQ
print("Toleranz bei arithmetischer Tolerierung deltaQ =", round(N_tol_ari / EQ, 4))



""" Statistische Tolerierung - Grenzwertmethode """

gamma = 0.9973 # 6 sigma

c1 = stats.norm.ppf((1 - gamma) / 2)
c2 = stats.norm.ppf((1 + gamma) / 2)
# Einzelne Varianzen addieren sich zur Gesamtvarianz
N_sig_clt = np.sqrt((EUadc * Uadc_sig)**2
                        + (EUoff * Uoff_sig)**2
                        + (ENA * NA_sig)**2)
# T = 6 * sigma
N_tol_clt = (c2 - c1) * N_sig_clt
print()
print('e)')
print("Toleranz bei Grenzwertmethode deltaQ =", round(N_tol_clt / EQ, 4))



""" Statistische Tolerierung - Simulation """

# Simulation ist in Vorlage bereits oben definiert
# Anzahl der Samples festlegen
#N = 10000
#Uadc_sim = np.random.normal(Uadc_0, Uadc_sig, N)
#Uoff_sim = np.random.uniform(Uoff_min, Uoff_max, N)
#NA_sim = np.random.uniform(NA_min, NA_max, N)
#N_sim = m_0 * Q_0**2 / k_0**2 * (1 + Uoff_sim / Uadc_sim) + NA_sim

# Toleranzrechnung basierend auf Simulation und Normalverteilung!
gamma = 0.9973
c1 = norm.ppf((1 - gamma) / 2)
c2 = norm.ppf((1 + gamma) / 2)
N_mean = np.mean(N_sim)
N_std = np.std(N_sim, ddof = 1)
N_tol_sim_prog = (c2 - c1) * N_std * np.sqrt(1 + 1 / N)
print()
print('f)')
print('Toleranz bei statistischer Simulation und Auswertung mit Prognosebereich:')
print('deltaQ =', round(N_tol_sim_prog / EQ, 4))

# Toleranzrechnung basierend auf Simulation und numerischer Auswertung
N_sim_sort = np.sort(N_sim)
N_sim_cdf = np.arange(1, N + 1, 1) / N
index_min = np.min(np.where(N_sim_cdf >= (1 - gamma) / 2))
index_max = np.min(np.where(N_sim_cdf >= (1 + gamma) / 2))
N_tol_sim_num = N_sim_sort[index_max] - N_sim_sort[index_min]
print('Toleranz bei statistischer Simulation und numerischer Auswertung:')
print('deltaQ =', round(N_tol_sim_num / EQ, 4))

# Mögliche Interpretation der Ergebnisse
print()
print("Durch die Umrechnung der Rechteckverteilung bei Offset-Spannung und ")
print("Auflösung auf eine äquivalente Standardabweichung wird der ")
print("Toleranzbereich mit gamma = 99.73 % vergrößert. Da nur wenige Maße")
print("überlagert werden, ist der Gewinn der statistischen Tolerierung ")
print("gering. Damit ist die Toleranz bei statistischer Tolerierung größer ")
print("als bei arithmetischer Tolerierung. Außerdem ist die Näherung, dass")
print("die Aufangsgröße eine Normalverteilung aufweist, wegen der geringen")
print("Anzahl von Eingangsgrößen nicht erfüllt.")



""" Statistische Tolerierung - Faltung """

# n... entspricht jeweils den "x-Werten"

# Einheitliche Schrittweite definieren
conv_min = -0.002
conv_max = 0.002
N_RES = 1e-6
gamma = 0.9973

# Erzeugen von Wahrscheinlichkeitsdichte 1
n_uadc = np.arange(conv_min, conv_max + N_RES, N_RES)
pdf_uadc = norm.pdf(n_uadc, 0, np.abs(EUadc * Uadc_sig))

# Erzeugen von Wahrscheinlichkeitsdichte 2
n_uoff = np.arange(conv_min, conv_max + N_RES, N_RES)
pdf_uoff = stats.uniform.pdf(n_uoff, EUoff * Uoff_min, EUoff * Uoff_tol)

# Faltung der ersten beiden Wahrscheinlichkeitsdichten
#n_uadc_uoff = np.arange(2 * conv_min, 2 * conv_max + N_RES, N_RES)
pdf_uadc_uoff = np.convolve(pdf_uadc, pdf_uoff) * N_RES

# Erzeugen von Wahrscheinlichkeitsdichte 3
n_na = np.arange(conv_min, conv_max + N_RES, N_RES)
pdf_na = stats.uniform.pdf(n_na, ENA * NA_min, ENA * NA_tol)

# Faltung der ersten beiden Wahrscheinlichkeitsdichten mit der dritten
n_uadc_uoff_na = np.arange(3 * conv_min, 3 * conv_max + N_RES, N_RES)
pdf_uadc_uoff_na = np.convolve(pdf_uadc_uoff, pdf_na) * N_RES

# Evtl. Fehlerkorrektur der Länge, auch anders herum möglich
pdf_uadc_uoff_na = pdf_uadc_uoff_na[0:np.size(n_uadc_uoff_na)]

# Berechne Verteilungsfunktion
cdf_uadc_uoff_na = np.cumsum(pdf_uadc_uoff_na) * N_RES
cdf_uadc_uoff_na = cdf_uadc_uoff_na / np.max(cdf_uadc_uoff_na) # Normierung um exakt 1 zu erreichen

# Suche Grenzen entsprechend der Signifikanzzahl
index_min = np.min(np.where(cdf_uadc_uoff_na >= (1 - gamma) / 2))
index_max = np.min(np.where(cdf_uadc_uoff_na >= (1 + gamma) / 2))

# Berechne Toleranz
N_tol_con = n_uadc_uoff_na[index_max] - n_uadc_uoff_na[index_min]
print ()
print("Toleranz bei Faltung =", round(N_tol_con / EQ, 4))

# Visualisierung
fig2 = plt.figure(2, figsize = (6, 4))
fig2.suptitle('')
ax1 = fig2.subplots(1, 1)

ax1.plot(n_uadc_uoff_na - 0.0005, pdf_uadc_uoff_na, 'b', label = "Faltung")
ax1.plot(n_uadc_uoff_na, norm.pdf(n_uadc_uoff_na, 0, N_sig_clt), 'r', label = "Grenzwertmethode")

ax1.set_xlabel('ADC-Einheiten $n$')
ax1.set_ylabel('Wahrscheinlichkeitsdichte $f(n)$')
ax1.axis([-0.0015, 0.0015, -50, 1050])
ax1.grid()
ax1.legend()



""" Visualisierung der Toleranzbeiträge """

fig3 = plt.figure()
ax1 = fig3.subplots()
x_plot = np.array(['Uoff', 'NA', 'Uadc'])
y_plot = np.array([np.abs((c2 - c1) * EUoff * Uoff_sig), np.abs((c2 - c1) * ENA * NA_sig), np.abs((c2 - c1) * EUadc * Uadc_sig)])
ax1.bar(x_plot, y_plot)
ax1.set_ylabel(r'Toleranzbeitrag in $10^{-3} \cdot \frac{m^3}{h} $')