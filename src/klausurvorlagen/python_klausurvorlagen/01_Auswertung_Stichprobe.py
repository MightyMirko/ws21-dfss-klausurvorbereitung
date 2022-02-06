# -*- coding: utf-8 -*-

# Vorlage Auswertung einer Stichprobe


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
#data0 = loadmat(Dateiname)['data']
data0 = np.random.normal(0, 1, 100)
data = np.array(data0).flatten()

# Daten sortieren
data_sort = np.sort(data)

# Berechnung der statistischen Kennwerte

# Datenumfang und Spannweite
N = np.size(data)
data_min = np.amin(data)
data_max = np.amax(data)
data_span = data_max - data_min

# Lagekennwerte
data_mean = np.mean(data)
print()
print('Arithmetischer Mittelwert:', round(data_mean,4), Einheit_unit)
data_median = np.median(data)
print('Median:', round(data_median, 4), Einheit_unit)

# Streuungskennwerte
data_var = np.var(data, ddof = 1)
print()
print('Varianz:', round(data_var, 4), Einheit_unit + '²')
data_std = np.std(data, ddof = 1)
print('Standardabweichung:', round(data_std, 4), Einheit_unit)
data_q25 = np.quantile(data, 0.25)
data_q75 = np.quantile(data, 0.75)
data_iqr = data_q75 - data_q25
print('Inter-Quartil-Range:', round(data_iqr, 4), Einheit_unit)

# Schiefe
data_skew_mom = skew(data)
print()
print('Momentenkoeffizient der Schiefe:', round(data_skew_mom, 4))
data_skew_qua = (data_q75 - 2 * data_median + data_q25) / data_iqr
print('Quartilkoeffizient der Schiefe:', round(data_skew_qua, 4))
if data_skew_mom > 0.1:
    print('--> Die Messwerte sind linkssteil')
elif (data_skew_mom <= 0.1) & (data_skew_mom >= -0.1):
    print('--> Die Messwerte sind symmetrisch')
else:
    print('--> Die Messwerte sind rechtssteil')

# Häufigkeiten
data_abs_freq, Klassengrenzen = np.histogram(data, bins = np.arange(np.floor(data_min) - 0.5, np.ceil(data_max) + 1.5, 1))
#data_abs_freq, Klassengrenzen = np.histogram(data, bins = np.arange(np.floor(data_min) - 0.5, np.ceil(data_max) + 1.5, data_span / np.sqrt(N)))
data_rel_freq = data_abs_freq / N
data_abs_sum_freq = np.cumsum(data_abs_freq)
data_rel_sum_freq = data_abs_sum_freq / N
Klassenmitten = np.arange(np.floor(data_min), np.ceil(data_max) + 1)

data_freq = pd.DataFrame({'Gruppenwert':Klassenmitten,
                        'Absolute Häufigkeit H(x)': data_abs_freq,
                        'RelativeHaeufigkeit h(x)': data_rel_freq,
                        'Absolute Summenhäufigkeit H_sum(x)': data_abs_sum_freq,
                        'Relative Summenhäufigkeit h_sum(x)': data_rel_sum_freq})
print(' ')
print(data_freq) 


# Visualisierung

# Streugraph der Stichprobe
fig1 = plt.figure()
ax1 = fig1.subplots()

ax1.plot(np.arange(1, N + 1), data, 'ko', label = 'Stichprobe')

ax1.set_xlabel('Stichprobe')
ax1.set_ylabel(Einheit0 + ' ' + Einheit_sign + ' in ' + Einheit_unit)
ax1.grid()
ax1.legend()


# BoxPlot
fig2 = plt.figure()
ax1 = fig2.subplots()

ax1.boxplot(data)

ax1.set_xlabel('Messreihe')
ax1.set_ylabel(Einheit0 + ' ' + Einheit_sign + ' in ' + Einheit_unit)
ax1.grid()


# Absolute Häufigkeit
fig3 = plt.figure()
ax1 = fig3.subplots()

ax1.hist(data, Klassengrenzen, color = 'r', edgecolor = 'k')

ax1.set_xlabel(Einheit0 + ' ' + Einheit_sign + ' in ' + Einheit_unit)
ax1.set_ylabel(r'Absolute Häufigkeit H(' + Einheit_sign + ')')
ax1.grid()
ax1.set_axisbelow(True)


# Relative Häufigkeit
fig4 = plt.figure()
ax1 = fig4.subplots()

ax1.hist(data, Klassengrenzen, color = 'b', edgecolor = 'k', density = True)
#ax1.plot(data_sort, norm.pdf(data_sort, data_mean, data_std), 'r')

ax1.set_xlabel(Einheit0 + ' ' + Einheit_sign + ' in ' + Einheit_unit)
ax1.set_ylabel(r'Relative Häufigkeit h(' + Einheit_sign + ')')
ax1.grid()
ax1.set_axisbelow(True)


# Absolute Summenhäufigkeit
fig4 = plt.figure()
ax1 = fig4.subplots()

ax1.step(np.append(np.append(data_min, data_sort), data_max), np.append(np.append(0, np.arange(1, N + 1)), N), color = 'r', where = 'post')

ax1.set_xlabel(Einheit0 + ' ' + Einheit_sign + ' in ' + Einheit_unit)
ax1.set_ylabel(r'Absolute Summenhäufigkeit $H_{Sum}(' + Einheit_sign + ')$')
ax1.grid()


# Relative Summenhäufigkeit
fig5 = plt.figure()
ax1 = fig5.subplots()

ax1.step(np.append(np.append(data_min, data_sort), data_max), np.append(np.append(0, np.arange(1, N + 1) / N), 1), color = 'b', where = 'post')
#ax1.plot(data_q25, 0.25, 'bo')
#ax1.plot(data_median, 0.5, 'ro')
#ax1.plot(data_q75, 0.75, 'bo')

ax1.set_xlabel(Einheit0 + ' ' + Einheit_sign + ' in ' + Einheit_unit)
ax1.set_ylabel(r'Relative Summenhäufigkeit $h_{Sum}(' + Einheit_sign + ')$')
ax1.grid()