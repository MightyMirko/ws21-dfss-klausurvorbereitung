# -*- coding: utf-8 -*-

# Vorlage Multivariate Auswertung


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



""" Randhäufigkeiten, Scatter Matrix, Kovarianzmatrix """

# Laden der Daten und Initialisiseren der Variablen
values = loadmat('spritzguss')
d = values['d']
T = values['T']
p = values['p']/1000
S = values['s']
X = np.append(np.append(np.append(d,T,axis=0),p,axis=0),S,axis=0)

# Kumulative Randhäufigkeiten berechnen
dsort = np.append(np.append(0.0,np.sort(d)),10)
Tsort = np.append(np.append(20,np.sort(T)),130)
psort = np.append(np.append(0.4,np.sort(p)), 1.1)
Ssort = np.append(np.append(0.5,np.sort(S)), 3)
H =np.append([0,0],np.cumsum(1/len(T)*np.ones(np.shape(T))))

# Kumulative Randhäufigkeiten darstellen
f1 = plt.figure(1, figsize=(12, 9))
axes1 = f1.subplots(2,2,gridspec_kw=dict(hspace=0.3))

axes1[0,0].step(dsort,H, color='b')
axes1[0,0].grid(True, ls='--')
axes1[0,0].set_xlabel('Wanddicke d / mm')
axes1[0,0].set_ylabel('Kumulative Randhäufigkeit H$_d$(d)')
axes1[0,0].set_title('Wanddicke')

axes1[0,1].step(Tsort, H,color='b')
axes1[0,1].grid(True, ls='--')
axes1[0,1].set_xlabel('Temperatur T / °C')
axes1[0,1].set_ylabel('Kumulative Randhäufigkeit H$_T$(T)')
axes1[0,1].set_title('Temperatur')

axes1[1,0].step(psort, H,color='b')
axes1[1,0].grid(True, ls='--')
axes1[1,0].set_xlabel('Nachdruck p / kbar')
axes1[1,0].set_ylabel('Kumulative Randhäufigkeit H$_p$(p)')
axes1[1,0].set_title('Nachdruck')

axes1[1,1].step(Ssort, H,color='b')
axes1[1,1].grid(True, ls='--')
axes1[1,1].set_xlabel('Schwindung S / %')
axes1[1,1].set_ylabel('Kumulative Randhäufigkeit H$_S$(S)')
axes1[1,1].set_title('Schwindung')

# Streudiagramm als Matrix
df = pd.DataFrame(np.transpose(X), columns=['d / mm', 'T / °C', 'p / kbar', 'S / %'])
f2 = plt.figure(1, figsize = (12, 8))
ax2 = f2.subplots()
pd.plotting.scatter_matrix(df, ax = ax2, alpha = 1, figsize = (12, 8), Color = 'b', hist_kwds = dict(Color = 'b'))

# Kennwerte berechnen in Numpy
mX = np.mean(X, axis=1)
cX = np.cov(X)

# Kennwerte berechnen in Pandas """
mX2 = df.mean() # Mittelwerte
cX2 = df.cov() # Kovarianzmatrix
rX2 = df.corr() # Korrelationsmatrix


""" Varianzanalyse ANOVA und plausibilisierung mit Boxplot """

# Data Frame Variable mit Daten erstellen
df = pd.DataFrame({'Schicht': np.tile(np.repeat([1, 2, 3], 3), 3),
                   'Linie': np.repeat(['A', 'B', 'C'], 9),
                   'Spannung': [16.1736, 16.0336, 16.0971,
                                16.1243, 15.9743, 16.0653,
                                15.9059, 15.8825, 15.8979,
                                16.4598, 16.5174, 16.4884,
                                16.7064, 16.5755, 16.4482,
                                16.7010, 16.7071, 16.7317,
                                16.4500, 16.5278, 16.3452,
                                16.5261, 16.4987, 16.4420,
                                16.5136, 16.2742, 16.1590]})
print('Datensatz')
print()
print(df)

# ANOVA durchführen, dazu Modell aufbauen
# C(...) sind kategorische Variablen
# C(...):C(...) ist das Produkt zweier kategorischer Variablen
# type=2 ist wieder ein dataframe    
poly = ols('Spannung ~ C(Linie) + C(Schicht) + C(Linie):C(Schicht)', df)
model = poly.fit()
anova2 = sm.stats.anova_lm(model, typ = 2)

print()
print()
print('ANOVA-Tabelle')
print()
print(anova2)
print('F gibt den Wert der F-verteilten Variable wieder. Ist dieser kleiner als der Grenzwert c, \
dann ist der Varianz zwischen den Gruppen größer als innerhalb der Gruppe. Dies wird durch den p-value plausiblisiert, indem dieser größer als das signifikanzniveau ist.')


# Boxplot zur plausibilisierung
fig1 = plt.figure(2, figsize=(12, 4))

ax1, ax2 = fig1.subplots(1,2)
ax1 = df.boxplot('Spannung',by = 'Linie',ax = ax1)
ax1.set_xlabel('Linie');
ax1.set_ylabel('Spannung U / V');  
ax1.set_title('');  
ax1.grid(True)

ax2 = df.boxplot('Spannung',by = 'Schicht',ax = ax2)
ax2.set_xlabel('Schicht');
ax2.set_ylabel('Spannung U / V');
ax2.set_title('');  
ax2.grid(True)

fig1.suptitle('')
print('Die Plausibilisierung erfolgt mit hilfe eines Boxplot: Die IQR überlappen sich, damit liegt kein signifikanter Einfluss des Waschmittels vor')
#print('Die Boxplots der Temperatur überlappen sich nicht, der p-value ist kleiner als 0.05. Der Einfluss der Temperatur ist somit signifikant.')
#print()
#print('Die Boxplots des Materials überlappen sich stark, der p-value ist größer als 0.05. Das Material hat somit keinen signifikanten Einfluss.')
print()
print('Der Einfluss des Wechelwirkungsterms ist laut dem p-value nicht signifikant und wird mithilfe des Liniendiagramms plausibilisiert da die Kurven annähernd parallel verlaufen.')

#fig3 = plt.figure()
#ax3 = fig3.subplots(1,1)
#ax3.plot([1,2,3], np.mean(data2[:,0:3], axis = 1),'b', label = '10 Grad')
#ax3.plot([1,2,3], np.mean(data2[:,4:7], axis = 1),'r', label = '18 Grad')
#ax3.grid(True)
#ax3.set_xlabel('Material')
#ax3.set_ylabel('Lebensdauer')
#ax3.legend()



""" Korrelationsanalyse """

# Wenn starke Abweichung siehe 09_2_Gussteile oder 09_3_Sportuntersuchung

# Einheit festlegen
Einheit = 'Ohm'

# Laden der Messdaten """
data = loadmat('Widerstandsdrift')['values']
df = pd.DataFrame({'Temperatur' :   np.tile([0, 20, 40, 80, 160], 10),
                   'Widerstand' :   data[1:,1:].flatten()})

# Dataframe verifizieren
print(' ')
print(df)

# Stichprobenanzahl definieren
N = df.shape[0]

# Streudiagramm der Messwerte
pd.plotting.scatter_matrix(df, alpha = 1, grid = True)

# Korrelationskoeffizient der Stichprobe
# Berechnung über Pandas Dataframe
r = df.corr(method = 'pearson').loc['Temperatur', 'Widerstand']
print()
print('b)')
print('Der Korrelationskoeffizient beträgt: r =', round(r, 4))

# Konfidenzbereich des Korrelationskoeffizienten der Grundgesamtheit
# Die Korrelation ist normalverteilt
gamma = 0.95
c1 = stats.norm.ppf((1 - gamma) / 2)
c2 = stats.norm.ppf((1 + gamma) / 2)
rho_min = np.tanh(np.arctanh(r) - c2 / (np.sqrt(N - 3)))
rho_max = np.tanh(np.arctanh(r) - c1 / (np.sqrt(N - 3)))
print()
print('c)')
print('Der 95%-Konfidenzbereich des Korrelationskoeffizient beträgt:')
print(round(rho_min, 4), '< rho <=', round(rho_max, 4))

# Hypothesentest des Korrelationskoeffizeinten der Grundgesamtheit mit variable t
print()
print('d)')
print('Nullhypothese       H0: rho = 0')
print('Alternativhypothese H1: rho != 0')
# t-Verteilung mit N - 2 Freiheitsgraden
# Annahmegrenzen für den Hypothesentest
# Bestimmung der Größe t aus dem Korrelationskoeffizienten der Stichprobe
# Liegt t0 im Annahmebereich wird die Hypothese nicht verworfen
alpha = 0.05
c1 = t.ppf(alpha / 2, N - 2)
c2 = t.ppf(1 - alpha / 2, N - 2)
print('Grenzen des Annahmebereichs:', round(c1, 4), 'bis', round(c2, 4))
t0 = r * np.sqrt((N - 2) / (1 - r**2))
print('t0 =', round(t0, 4))
if ((t0 > c1) & (t0 <= c2)): 
    print('Die Nullhypothese wird nicht verworfen.')
else: 
    print('Die Nullhypothese wird verworfen.')
    
# Hypothesentest des Korrelationskoeffizeinten der Grundgesamtheit mit variable z und p-value
print()
print('e)')
print('Nullhypothese       H0: rho = 0')
print('Alternativhypothese H1: rho != 0')
# Unter der Annahme rho = 0 vereinfacht sich der Term:
alpha = 0.05
c1 = stats.norm.ppf(alpha / 2)
c2 = stats.norm.ppf(1 - alpha / 2)
rc1_z = np.tanh(c1 / (np.sqrt(N - 3)))
rc2_z = np.tanh(c2 / (np.sqrt(N - 3)))
print('r =', round(r,3))
print('Annahmebereich:', round(rc1_z, 4), 'bis', round(rc2_z, 4))
if ((r > rc1_z) & (r <= rc2_z)): 
    print('Die Nullhypothese wird nicht verworfen.')
else: 
    print('Die Nullhypothese wird verworfen.')

# Vergleich der Annahmebereiche
# Einsetzen von rc1 und rc2 in t-Verteilung, Vergleich mit den Grenzen der t-Verteilung
print()
print('f)')
rc1_t = rc1_z * np.sqrt((N - 2) / (1 - rc1_z**2))
rc2_t = rc2_z * np.sqrt((N - 2) / (1 - rc2_z**2))
print('Grenzen t:', round(c1, 4), 'bis', round(c2, 4))
print('Grenzen z:', round(rc1_t, 4), 'bis', round(rc2_t, 4))
print('Der engere Annahmebereich zeigt, dass der Hypothesentest mit der standardnormalverteilten Zufallsvariable z strenger ist als der Test mit der t-verteilten Variable t.')