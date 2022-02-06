# -*- coding: utf-8 -*-

# Vorlage Univariate Regression


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
data0 = loadmat('ScherfestigkeitSchweissen')['data']
df = pd.DataFrame({'Scherfestigkeit': data0[:,0],
                   'Durchmesser': data0[:,1]})

# Dataframe verifizieren
print()
print('Datensatz:')
print()
print(df)
print()


# Stichprobenanzahl definieren
N = df.shape[0]


""" Lineares Regressionsmodell definieren und berechnen """

#poly = ols("Scherfestigkeit ~ I(Durchmesser) + I(Durchmesser**2)" , df).fit()
poly = ols("Scherfestigkeit ~ I(Durchmesser)" , df)
model = poly.fit()
print(model.summary())


""" Darstellung der Regressionsfunktion zusammen mit Originaldaten """

fig = plt.figure(figsize = (10,5))
ax1, ax2 = fig.subplots(1, 2)

ax1.set_xlabel(r'Durchmesser / $\mu m$')
ax1.set_ylabel(r'Scherfestigkeit / $MPa$')
ax1.set_title('Datensatz');
ax1.grid(True)

# Dataframe aufbauen und mit Regressionswerten befüllen
regress_ex = pd.DataFrame({'Durchmesser': np.arange(0, 5000, 1)})
regress_ex['Scherfestigkeit'] = model.predict(regress_ex['Durchmesser'])

ax1.plot(df['Durchmesser'], df['Scherfestigkeit'], 'bo', label = 'Stichprobe')
ax1.plot(regress_ex['Durchmesser'], regress_ex['Scherfestigkeit'],'b', label = 'Regression')


""" Berechnung und Darstellung der Residuen """

ax2.stem(df['Durchmesser'], model.resid, 'r', use_line_collection = True, markerfmt = 'ro')
ax2.set_xlabel(r'Durchmesser / $\mu m$');
ax2.set_ylabel(r'Abweichung Scherfestigkeit / $MPa$');  
ax2.set_title('Residuen');  
ax2.grid(True)

fig.tight_layout()
print('Es ist keine Struktur der Reststreuung zu erkennen, die auf einen systematischen Regressionsfehler schließen lässt.')



""" Konfidenzbereich und Prognosebereich bei Regressionsfunktionen """

# Berechnung mit Herleitung aus der Vorlesung 
lconf_ex, lprog_ex = conf_pred_band_ex(regress_ex , poly, model, alpha = 0.05)
regress_ex['L_conf'] = model.predict(regress_ex['Durchmesser']) - lconf_ex
regress_ex['U_conf'] = model.predict(regress_ex['Durchmesser']) + lconf_ex
regress_ex['L_pred'] = model.predict(regress_ex['Durchmesser']) - lprog_ex
regress_ex['U_pred'] = model.predict(regress_ex['Durchmesser']) + lprog_ex

ax1.plot(regress_ex['Durchmesser'], regress_ex['L_conf'], 'r--', label = 'Konfidenzbereich')
ax1.plot(regress_ex['Durchmesser'], regress_ex['U_conf'], 'r--')
ax1.plot(regress_ex['Durchmesser'], regress_ex['L_pred'], 'g', label = 'Prognosebereich')
ax1.plot(regress_ex['Durchmesser'], regress_ex['U_pred'], 'g')

ax1.legend()





""" Beispiel 2 - inverse """

""" Laden der Messdaten """
data0 = loadmat('Messwerte')['Messwerte']
df = pd.DataFrame({'Massenstrom': data0[0,:],
                   'Spannung': data0[1,:]})

# Dataframe verifizieren
print()
print('Datensatz:')
print()
print(df)


# Stichprobenanzahl definieren
N = df.shape[0]


""" a) Lineares Regressionsmodell definieren und berechnen """

#model = ols("Spannung ~ I(Massenstrom) + I(Massenstrom**2) + I(Massenstrom**3)" , df).fit()
model = ols("Spannung ~ -1 + I(Massenstrom) + I(Massenstrom**2) + I(Massenstrom**3)" , df).fit()
print(model.summary())


""" b) Reduktion auf signifikante Terme """

# Kubisches Modell definiert
# Nur die konstante ist nicht signifikant, sie wird entfernt (Hinzufügen der -1)
# Adjungiertes Bestimmtheitsmaß erhöht sich von 0.989 auf 0.998
# Nun sind alle Terme signifikant, keine weitere Reduktion


""" c) Lineares Regressionsmodell der inversen definieren und berechnen """

#model_inv = ols("Massenstrom ~ I(Spannung) + I(Spannung**2) + I(Spannung**3)" , df).fit()
#model_inv = ols("Massenstrom ~ I(Spannung) + I(Spannung**2)" , df).fit()
#model_inv = ols("Massenstrom ~ I(Spannung**2)" , df).fit()
model_inv = ols("Massenstrom ~ -1 + I(Spannung**2)" , df).fit()
print(model_inv.summary())


""" d) Reduktion auf signifikante Terme """

# Kubisches Modell definiert
# Alle Terme außer quadratischer Term nicht signifikant
# Kubischer Term wird entfernt
# Adjungiertes Bestimmtheitsmaß verändert sich von 0.999 auf 0.999
# Quadratischer Term ist signifikant
# Linearer Term wird entfernt
# Adjungiertes Bestimmtheitsmaß verändert sich von 0.999 auf 0.999
# Konstanter Term ist nicht signifikant und wird entfernt
# Adjungiertes Bestimmtheitsmaß verändert sich von 0.999 auf 1.0
# Alle Terme sind signifikant

# Resultierende Gleichung lautet: m = 18.1703 * U²
# Umstellen liefert: U = sqrt(m) / sqrt(18.1703)


""" e) Darstellen der Stichprobe und der Regression """

regress_ex = pd.DataFrame({'Massenstrom': np.arange(0, 500, 1)})
regress_ex['Spannung'] = model.predict(regress_ex['Massenstrom'])

regress_ex2 = pd.DataFrame({'Spannung': np.arange(0, 5.5, 0.01)})
regress_ex2['Massenstrom'] = model_inv.predict(regress_ex2['Spannung'])

fig = plt.figure(figsize = (10,5))
ax1, ax2 = fig.subplots(1, 2)

ax1.plot(df['Massenstrom'], df['Spannung'], 'bo', label = 'Stichprobe')
ax1.plot(regress_ex['Massenstrom'], regress_ex['Spannung'], 'b', label = 'Regression aus b)')
ax1.plot(regress_ex2['Massenstrom'], regress_ex2['Spannung'], 'y', label = 'Regression aus d)')

ax1.set_xlabel(r'Massenstrom in $kg / h$')
ax1.set_ylabel(r'Spannung in $V$')
ax1.set_title('Regressionen');
ax1.grid(True)
ax1.legend()


""" f) Darstellen der Residuen """

ax2.stem(df['Massenstrom'], model.resid, 'b', use_line_collection = True, markerfmt = 'bo', label = 'Abweichung aus b)')
ax2.stem(df['Massenstrom'], model_inv.resid, 'y', use_line_collection = True, markerfmt = 'yo', label = 'Abweichung aus d)')
ax2.set_xlabel(r'Massenstrom in $kg / h$')
ax2.set_ylabel(r'Abweichung Spannung in $V$');  
ax2.set_title('Residuen');  
ax2.grid(True)
ax2.legend()
fig.tight_layout()
# Die kubische Regression aus Teilaufgabe b) ist deutlich akurater


""" g) Schlussfolgerung """
# Je größer die Anzahl der für die Regression verwendeten Terme, desto besser wird der Verlauf der 
# Stichprobe nachgebildet. Dies kann aber zu Overfitting führen
# Wenn ein funktionaler Zusammenhang zweier Größen bekannt ist, sich aber nicht durch ein Polynom darstellen lässt, 
# kann es zielführend sein, die Regressionsfunktion der Umkehrfunktion zu berechnen und dann nach der Zielgröße aufzulösen. 
# Dazu darf die Regressionsfunktion eine Ordnung von maximal 2 aufweisen. 
# Alternativ kann es sinnvoll sein, eine Achse zu transformieren, um ein Polynom als Regressionsfunktion zu bekommen.






