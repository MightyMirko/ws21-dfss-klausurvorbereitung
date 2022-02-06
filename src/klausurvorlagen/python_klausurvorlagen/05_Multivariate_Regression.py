# -*- coding: utf-8 -*-

# Vorlage Multivariate Regression


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


def conf_pred_band_ex(_regress_ex, _poly, _model, alpha=0.05):
    """ Function calculates the confidence and prediction interval for a
    given multivariate regression function poly according to lecture DFSS,
    regression parameters are already determined in an existing model,
    identical polynom is used for extrapolation
    
    Parameters
    ----------
    regress_ex : DataFrame
        Extended dataset for calculation.
    poly : OLS object of statsmodels.regression.linear_model modul
        definition of regression model.
    model : statsmodels.regression.linear_model.RegressionResultsWrapper
        Model parameters.
    alpha : float, optional
        Significance level. The default is 0.05.
    
    Returns
    -------
    lconf_ex : Series
        Distance of confidence limit to mean regression.
    lpred_ex : Series
        Distance of prediction limit to mean regression.
    """
    
    # ols is used to calculte the complets vector x_0 of input variables
    poly_ex = ols(_poly.formula, _regress_ex)
    x_0 = poly_ex.exog
    # Calculation according lecture book
    d = np.dot(x_0, np.dot(_poly.normalized_cov_params, x_0.T))
    c_1 = stats.t.isf(alpha/2, _model.df_resid)
    lconf_ex = c_1*np.sqrt(np.diag(d)*_model.mse_resid)
    lpred_ex = c_1*np.sqrt((1+np.diag(d))*_model.mse_resid)
    
    return lconf_ex, lpred_ex



""" Laden der Messdaten """
data0 = loadmat('ChemischeAusbeute')['data']
df = pd.DataFrame({'Temperatur': data0[:,0],
                   'Katalysatorkonzentration': data0[:,1],
                   'Ausbeute': data0[:,2]})

# Dataframe verifizieren
print()
print('Datensatz:')
print()
print(df)


# Stichprobenanzahl definieren
N = df.shape[0]


""" a) Regressionsfunktion und Darstellung """

poly1 = ols('Ausbeute ~ Temperatur + Katalysatorkonzentration + I(Temperatur*Katalysatorkonzentration)', df)
model1 = poly1.fit()
print(model1.summary())

b = model1._results.params

# Dataframe aufbauen und mit Regressionswerten befüllen
Meshzahl = 20

T_val = np.linspace(df['Temperatur'].min(), df['Temperatur'].max(), Meshzahl)
K_val = np.linspace(df['Katalysatorkonzentration'].min(), df['Katalysatorkonzentration'].max(), Meshzahl)

T_mesh, K_mesh = np.meshgrid(T_val, K_val)
regress_ex = pd.DataFrame({'Temperatur' : T_mesh.flatten(),
                           'Katalysatorkonzentration' : K_mesh.flatten()})
regress_ex['Ausbeute'] = model1.predict(regress_ex)  
A_mesh = regress_ex['Ausbeute'].to_numpy().reshape(T_mesh.shape)

fig1 = plt.figure(figsize = (10, 10))
ax1 = fig1.add_subplot(111, projection='3d')

ax1.plot(df['Temperatur'], df['Katalysatorkonzentration'], df['Ausbeute'], 'bo', label = 'Stichprobe')

ax1.plot_wireframe(T_mesh, K_mesh, A_mesh, color = 'green', label = 'Regressionsfunktion')

ax1.set_xlabel('Temperatur T in $\degree C$')
ax1.set_ylabel('Katalysatorkonzentration K in %')
ax1.set_zlabel('Ausbeute A in %')
ax1.set_title('Datensatz und Regression')
ax1.grid(True)
ax1.legend()


""" b) Bewertung der Residuen """

fig2 = plt.figure(figsize = (12, 6))
fig2.suptitle('Residuen')
ax1, ax2 = fig2.subplots(1, 2)

ax1.plot(df['Temperatur'], model1.resid, 'ro')
ax1.set_xlabel('Temperatur T in $\degree C$');
ax1.set_ylabel('Abweichung Ausbeute A in %');  
ax1.set_title('');  
ax1.grid(True)

ax2.plot(df['Katalysatorkonzentration'], model1.resid, 'ro')
ax2.set_xlabel('Katalysatorkonzentration K in %');
ax2.set_ylabel('Abweichung Ausbeute A in %');  
ax2.set_title('');  
ax2.grid(True)

print()
print('b)')
print('Es ist keine Struktur der Reststreuung zu erkennen, die auf einen systematischen Regressionsfehler schließen lässt.')


""" c) Regressionsgüte """

print()
print('c)')
print('Zur Evaluation der Regressionsgüte wird das adjungierte Bestimmtheitsmaß herangezogen.')
print('R_adj =', round(model1.rsquared_adj, 3))
print('Die Regression ist sehr gut.')
print()
print('Die Residuen lassen, wie in Aufgabenteil b) festgestellt, keinen strukturierten Fehler erkennen. Der t-Test zeigt, dass nicht alle Terme signifikant sind, deshalb wird in der folgende Teilaufgabe eine Reduktion der Regressionsterme vorgenommen.')


""" d) Reduktion auf signifikante Terme """

# Lineares Modell ist aufgestellt
# Linearer Term der Katalysatorkonzentration ist nicht signifikant und wird entfernt
# Adjungiertes Bestimmtheitsmaß ändert sich von 0.988 auf 0.988
# Konstanter Term ist nicht signifikant und wird entfernt
# Adjungiertes Bestimmtheitsmaß ändert sich von 0.988 auf 1.000
# Alle Terme sind signifikant

#poly = ols('Ausbeute ~ Temperatur + Katalysatorkonzentration + Temperatur:Katalysatorkonzentration', df)
#poly2 = ols('Ausbeute ~ 1 + Temperatur + Temperatur:Katalysatorkonzentration', df)
poly2 = ols('Ausbeute ~ -1 + Temperatur + Temperatur:Katalysatorkonzentration', df)
model2 = poly2.fit()
print(model2.summary())


""" e) Darstellung der neuen Regressionsfunktion und Konfidenzbereich """

gamma = 0.95
alpha = 1 - gamma

# Berechnung mit Herleitung aus der Vorlesung 
lconf_ex, lprog_ex = conf_pred_band_ex(regress_ex , poly2, model2, alpha = 1 - gamma)
regress_ex['L_conf'] = regress_ex['Ausbeute'] - lconf_ex
regress_ex['U_conf'] = regress_ex['Ausbeute'] + lconf_ex

# Konfidenzbereiche in Mesh umwandeln
Lconf = regress_ex['L_conf'].to_numpy().reshape(T_mesh.shape)
Uconf = regress_ex['U_conf'].to_numpy().reshape(T_mesh.shape)

# Darstellung der Daten
fig3 = plt.figure(figsize = (10, 10))
ax1 = fig3.add_subplot(111, projection='3d')

ax1.plot(df['Temperatur'], df['Katalysatorkonzentration'], df['Ausbeute'], 'bo', label = 'Stichprobe')
ax1.plot_wireframe(T_mesh, K_mesh, A_mesh, color = 'green', label = 'Regressionsfunktion')
ax1.plot_wireframe(T_mesh, K_mesh, Lconf, color = 'blue', label = 'Unterer Konfidenzbereich')
ax1.plot_wireframe(T_mesh, K_mesh, Uconf, color = 'red', label = 'Oberer Konfidenzbereich')

ax1.set_xlabel('Temperatur T in $\degree C$')
ax1.set_ylabel('Katalysatorkonzentration K in %')
ax1.set_zlabel('Ausbeute A in %')
ax1.set_title('Datensatz und Regression')
ax1.grid(True)
ax1.legend()


""" f) Berechnung Prognoseintervall für spezifische Werte """

gamma = 0.95

p0 = pd.DataFrame({'Temperatur' : np.array([125]),
                   'Katalysatorkonzentration' : np.array([0.4])})
p0['Ausbeute'] = model2.predict(p0)

lconf_ex, lprog_ex = conf_pred_band_ex(p0 , poly2, model2, alpha = 1 - gamma)

p0['L_pred'] = p0['Ausbeute'] - lprog_ex
p0['U_pred'] = p0['Ausbeute'] + lprog_ex

print()
print('f)')
print('Das Prognoseintervall für die Ausbeute beträgt:', round(p0['L_pred'][0], 3), '% bis', round(p0['U_pred'][0], 3), '%.')










""" Beispiel 2 - diskrete Berechnung """

""" Laden der Messdaten """
data0 = loadmat('LebensdauerMaschinenkomponenten')['values']
df = pd.DataFrame({'Betriebsspannung': data0[:,1],
                   'Drehzahl': data0[:,2],
                   'Lebensdauer': data0[:,0]})

# Dataframe verifizieren
print()
print('Datensatz:')
print()
print(df)


# Stichprobenanzahl definieren
N = df.shape[0]


""" a) Regressionsfunktion und Darstellung """

poly = ols('Lebensdauer ~ 1 + I(Betriebsspannung) + I(Drehzahl) + I(Betriebsspannung)*I(Drehzahl) + I(Betriebsspannung**2) + I(Drehzahl**2)', df)
model1 = poly.fit()
print(model1.summary())

b = model1.params

Meshzahl = 30
U_plot = np.linspace(df['Betriebsspannung'].min(), df['Betriebsspannung'].max(), Meshzahl)
n_plot = np.linspace(df['Drehzahl'].min(), df['Drehzahl'].max(), Meshzahl)

U_mesh, n_mesh = np.meshgrid(U_plot, n_plot)
L_mesh = b[0] + b[1] * U_mesh + b[2] * n_mesh + b[3] * U_mesh * n_mesh + b[4] * U_mesh**2 + b[5] * n_mesh**2;

fig1 = plt.figure(figsize = (10, 10))
ax1 = fig1.gca(projection='3d')
ax1.plot(df['Betriebsspannung'], df['Drehzahl'], df['Lebensdauer'], 'bo', label = 'Stichprobe')
ax1.plot_wireframe(U_mesh, n_mesh, L_mesh, color = 'green', label = 'Regressionsfunktion')
ax1.set_xlabel('Betriebsspannung U in V')
ax1.set_ylabel('Drehzahl n in 1/min')
ax1.set_zlabel('Lebensdauer L in min')
ax1.set_title('Datensatz und Regression')
ax1.grid(True)
ax1.legend()


""" b) Bewertung der Residuen """

fig2 = plt.figure(figsize = (12, 6))
fig2.suptitle('Residuen')
ax1, ax2 = fig2.subplots(1, 2)

ax1.plot(df['Betriebsspannung'], model1.resid, 'ro')
ax1.set_xlabel('Betriebsspannung U in V');
ax1.set_ylabel('Abweichung Lebensdauer L in min');  
ax1.set_title('');  
ax1.grid(True)

ax2.plot(df['Drehzahl'], model1.resid, 'ro')
ax2.set_xlabel('Drehzahl n in 1/min');
ax2.set_ylabel('Abweichung Lebensdauer L in min');  
ax2.set_title('');  
ax2.grid(True)

print()
print('b)')
print('Es ist keine Struktur der Reststreuung zu erkennen, die auf einen systematischen Regressionsfehler schließen lässt.')


""" c) Regressionsgüte """

print()
print('c)')
print('Zur Evaluation der Regressionsgüte wird das adjungierte Bestimmtheitsmaß herangezogen.')
print('R_adj =', round(model1.rsquared_adj, 3))
print('Die Regression ist mäßig.')
print()
print('Die Residuen lassen, wie in Aufgabenteil b) festgestellt, keinen strukturierten Fehler erkennen. Der t-Test zeigt, dass nicht alle Terme signifikant sind, deshalb wird in der folgende Teilaufgabe eine Reduktion der Regressionsterme vorgenommen.')


""" d) Reduktion auf signifikante Terme """

# Vollquadratisches Modell ist aufgestellt
# Quadratischer Term der Drehzahl ist nicht signifikant und wird entfernt
# Adjungiertes Bestimmtheitsmaß ändert sich von 0.882 auf 0.890
# Der Wechselwirkungsterm ist nicht signifikant und wird entfernt
# Adjungiertes Bestimmtheitsmaß ändert sich von 0.890 auf 0.904
# Alle Terme sind signifikant

#poly2 = ols('Lebensdauer ~ 1 + I(Betriebsspannung) + I(Drehzahl) + I(Betriebsspannung)*I(Drehzahl) + I(Betriebsspannung**2) + I(Drehzahl**2)', df)
#poly2 = ols('Lebensdauer ~ 1 + I(Betriebsspannung) + I(Drehzahl) + I(Betriebsspannung)*I(Drehzahl) + I(Betriebsspannung**2)', df)
poly2 = ols('Lebensdauer ~ 1 + I(Betriebsspannung) + I(Drehzahl) + I(Betriebsspannung**2)', df)

model2 = poly2.fit()
print(model2.summary())


""" e) Darstellung der neuen Regressionsfunktion und Konfidenzbereich """

b = model2._results.params
r = model2.resid

# Berechnung der neuen Regression
L_mesh = b[0] + b[1] * U_mesh + b[2] * n_mesh + b[3] * U_mesh**2;

# Berechnung des Konfidenzbereichs
alpha = 0.05

# Pseudoinverse
PSI = np.linalg.inv((np.array([np.ones(N), df['Betriebsspannung'], df['Drehzahl'], df['Betriebsspannung']**2])).dot(np.array([np.ones(N), df['Betriebsspannung'], df['Drehzahl'], df['Betriebsspannung']**2]).T))
# Freiheitsgrade
FG = (N - len(b))
# Summe der Residuen zum Quadrat
Sr = np.sqrt(1 / FG * r.dot(r.T))

Lmin = np.zeros((len(U_mesh), len(n_mesh)))
Lmax = np.zeros((len(U_mesh), len(n_mesh)))

for n in range(0, len(U_mesh)):
    for m in range(0, len(n_mesh)):
        x0 = np.array([1, U_mesh[n,m], n_mesh[n,m], U_mesh[n,m]**2])
        Lmin[n,m] = b.dot(x0) - stats.t.ppf(1 - alpha / 2, FG) * Sr * np.sqrt(x0.dot(PSI).dot(x0))
        Lmax[n,m] = b.dot(x0) - stats.t.ppf(alpha / 2, FG) * Sr * np.sqrt(x0.dot(PSI).dot(x0))



fig3 = plt.figure(figsize = (10, 10))
ax1 = fig3.add_subplot(111, projection='3d')
ax1.plot(df['Betriebsspannung'], df['Drehzahl'], df['Lebensdauer'], 'bo', label = 'Stichprobe')
ax1.plot_wireframe(U_mesh, n_mesh, Lmin, color = 'blue', label = 'Unterer Konfidenzbereich')
ax1.plot_wireframe(U_mesh, n_mesh, Lmax, color = 'red', label = 'Oberer Konfidenzbereich')
ax1.plot_wireframe(U_mesh, n_mesh, L_mesh, color = 'green', label = 'Regressionsfunktion')
ax1.set_xlabel('Betriebsspannung U in V')
ax1.set_ylabel('Drehzahl n in 1/min')
ax1.set_zlabel('Lebensdauer L in min')
ax1.set_title('Datensatz und Regression')
ax1.grid(True)
ax1.legend()


""" f) Berechnung Prognoseintervall """

b = model2.params
U0 = 120
n0 = 925
xp0 = np.array([1, U0, n0, U0**2])
LPmin = b.dot(xp0) - stats.t.ppf(1 - alpha / 2, FG) * Sr * np.sqrt(1 + (xp0.dot(PSI).dot(xp0)))
LPmax = b.dot(xp0) - stats.t.ppf(alpha / 2, FG) * Sr * np.sqrt(1 + (xp0.dot(PSI).dot(xp0)))
print()
print('f)')
print('Das Prognoseintervall für die Lebensdauer beträgt:', round(LPmin, 3), 'min bis', round(LPmax, 3), 'min.')
