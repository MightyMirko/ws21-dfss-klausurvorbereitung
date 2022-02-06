# -*- coding: utf-8 -*-

# Vorlage Messsystemanalyse


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



""" Einheit definieren """
Einheit0 = 'Temperatur'
Einheit_sign = 'T'
Einheit_unit = r'$\degree C$'


""" a) Auflösung einschätzen """

Y_TOLERANCE = 1
Y_RESOLUTION = 0.0001
print()
print()
print('1. Bewertung der Auflösung')
print()
if Y_RESOLUTION / Y_TOLERANCE <= 0.05:
    print("Auflösung ausreichend")
else:
    print("Auflösung ist nicht ausreichend")
    

""" b) Systematische Messabweichung und Wiederholbarkeit """

# Daten laden
data_ref = np.array([18.3124, 18.3119, 18.3039, 18.3066, 18.2998, 18.2998, 18.2837, 18.2925, 18.2971, 18.2934, 18.2951, 18.3146, 18.3126, 18.3066, 18.2903, 18.2900, 18.3015, 18.2951, 18.3039, 18.3041, 18.3124, 18.3119, 18.3039, 18.3066, 18.2998, 18.2908, 18.2837, 18.2925, 18.2971, 18.2934, 18.2951, 18.3002, 18.2978, 18.2876, 18.2927, 18.2925, 18.2968, 18.3053, 18.2973, 18.2998, 18.3036, 18.2959, 18.3051, 18.2998, 18.3036, 18.2959, 18.3051, 18.3083, 18.3133, 18.3109])
y_repeat_test = data_ref.T
y_repeat_len = np.size(y_repeat_test)
y_repeat_reference = 18.3

# Visualisierung der systematischen Messabweichung
fig1 = plt.figure(1, figsize=(6, 4))
fig1.suptitle('')
ax1 = fig1.subplots(1, 1)

ax1.plot(np.arange(0, y_repeat_len) + 1, y_repeat_test, 'bo-')
ax1.plot(np.arange(0, y_repeat_len) + 1, y_repeat_reference*np.ones(y_repeat_len), 'r', label = 'Referenzwert')
ax1.plot(np.arange(0, y_repeat_len) + 1, (y_repeat_reference + 0.1 * Y_TOLERANCE) * np.ones(y_repeat_len), 'g--', label = 'Toleranzgrenze')
ax1.plot(np.arange(0, y_repeat_len) + 1, (y_repeat_reference - 0.1 * Y_TOLERANCE) * np.ones(y_repeat_len), 'g--')

ax1.set_xlabel('Messung')
ax1.set_ylabel(Einheit0 + ' ' +  Einheit_sign + ' in ' + Einheit_unit)
ax1.set_title('Visualisierung der systematischen Messabweichung')
ax1.grid()
ax1.legend()

# Berechnung des Fähigkeitsindex
y_deviation = np.mean(y_repeat_test) - y_repeat_reference
c_g = 0.1 * Y_TOLERANCE / 3 / np.std(y_repeat_test, ddof = 1)
print()
print()
print("2. Systematische Abweichung und Wiederholbarkeit")
print()
print("C_g = ", round(c_g, 4))
if c_g >= 1.33:
    print("Wiederholbarkeit ausreichend")
else:
    print("Wiederholbarkeit ist nicht ausreichend")
    
c_gk = (0.1 * Y_TOLERANCE - np.abs(y_deviation)) / 3 / np.std(y_repeat_test, ddof = 1)
print()
print("C_gk = ", round(c_gk, 4))
if c_gk >= 1.33:
    print("Wiederholbarkeit und sytematische Abweichung ausreichend")
elif c_g >= 1.33:
    print("Systematische Abweichung zu groß")
else:
    print("Auflösung und systematische Abweichung nicht ausreichend")

# Hypothesentest mit H0: y_repeat_test = y_repeat_reference
hypo_test = stats.ttest_1samp(y_repeat_test, y_repeat_reference)
print()
print("Hypothesentest auf Abweichung mit p-value = ", round(float(hypo_test[1]), 4))
if hypo_test[1] <= 0.05:
    print("Abweichung signifikant")
else:
    print("Abweichung nicht signifikant")

"""
# Konfidenzbereich für y_repeat_test
GAMMA = 0.9
c1 = stats.t.ppf((1 - GAMMA) / 2, y_repeat_len - 1)
c2 = stats.t.ppf((1 + GAMMA) / 2, y_repeat_len - 1)
y_repeat_min = np.mean(y_repeat_test) + c1 * np.std(y_repeat_test, ddof = 1) / np.sqrt(y_repeat_len)
y_repeat_max = np.mean(y_repeat_test) + c2 * np.std(y_repeat_test, ddof = 1) / np.sqrt(y_repeat_len)
print()
print("Referenzwert:", round(y_repeat_reference, 4))
print("Konfidenzbereich: Untere Grenze = ", round(y_repeat_min, 4))
print("Konfidenzbereich: Obere Grenze = ", round(y_repeat_max, 4))
if (y_repeat_reference >= y_repeat_min) & (y_repeat_reference <= y_repeat_max):
    print("Abweichung nicht signifikant")
else:
    print("Abweichung signifikant")
"""


""" c) Linearität """

# Daten laden
# Ziel-Datenframe besteht aus Referenzwerten und Abweichungen
data_lin_ref = np.tile(np.array([10.7000, 18.3000, 31.7000, 44.0000, 54.2000]), 10)
data_lin_mes = np.array([10.6529, 18.2937, 31.6723, 43.8981, 54.2330, 10.6675, 18.2961, 31.7330, 44.0024, 54.1845, 10.7184, 18.3155, 31.7306, 44.0291, 54.2087, 10.6966, 18.3131, 31.7209, 43.9757, 54.2112, 10.7039, 18.3058, 31.6917, 43.9903, 54.2209, 10.6602, 18.2913, 31.6723, 44.0049, 54.1650, 10.6917, 18.2913, 31.6723, 43.9879, 54.2063, 10.7451, 18.3010, 31.6772, 43.9830, 54.2573, 10.7136, 18.2961, 31.6723, 43.9927, 54.1820, 10.6335, 18.3034, 31.6748, 43.9879, 54.2209])
data_lin_dev = data_lin_mes - data_lin_ref
y_linearity = pd.DataFrame({'reference': data_lin_ref,
                            'deviation': np.reshape(data_lin_dev, -1)})

# Visualisierung
fig2 = plt.figure(2, figsize = (10, 4))
fig2.suptitle('')
ax1, ax2 = fig2.subplots(1, 2, gridspec_kw = dict(wspace = 0.3))

ax1.plot(y_linearity["reference"], y_linearity["deviation"], 'b+', alpha = 0.4, label = 'Stichprobe')

ax1.set_xlabel('Referenzwert '+  Einheit_sign + ' in ' + Einheit_unit)
ax1.set_ylabel(r'Abweichung $\Delta$' + Einheit_sign + ' in ' + Einheit_unit)
ax1.set_title('Bewertung des Konfidenzbereichs')
ax1.grid()

ax2.plot(y_linearity["reference"], y_linearity["deviation"], 'b+', alpha = 0.4, label = 'Stichprobe')

ax2.set_xlabel('Referenzwert ' + Einheit_sign + ' in ' + Einheit_unit)
ax2.set_ylabel(r'Abweichung $\Delta$' + Einheit_sign + ' in ' + Einheit_unit)
ax2.set_title('Mittelwerte zur Linearitätsbewertung')
ax2.grid()

# Regressionsfunktion mit Konfidenzbereich
poly = ols("deviation ~ reference", y_linearity)
model = poly.fit()
print(model.summary())

y_plot = np.linspace(y_linearity["reference"].min(), y_linearity["reference"].max(), 1000)
y_regress = pd.DataFrame({"reference": np.reshape(y_plot, -1)})
y_regress["deviation"] = model.predict(y_regress)

ax1.plot(y_linearity["reference"], np.zeros(np.size(y_linearity["reference"])), 'k', alpha = 0.7)
ax1.plot(y_regress["reference"], y_regress["deviation"], 'r', label = 'Regression')

gamma = 0.95
y_regress["confidence"], y_regress["prediction"] = conf_pred_band_ex(y_regress, poly, model, alpha = 1 - gamma)

ax1.plot(y_regress["reference"], y_regress["deviation"] + y_regress["confidence"], 'r:', label = str(gamma * 100) + '%-Konfidenzbereich')
ax1.plot(y_regress["reference"], y_regress["deviation"] - y_regress["confidence"], 'r:')

# Hypothesentest Gesamtlinearität
print()
print()
print("3. Linearität")
print()
print("Prüfung Regressionsgerade")
if (model.pvalues > 0.05).all(axis = None):
    print("Keine signifikante Abweichung zur Linearität")
else:
    print("Signifikante Abweichung zur Linearität")

# Test auf Abweichung jedes Mittelwerts zur Referenz
ax2.plot(y_linearity.groupby("reference").aggregate('mean'), 'ro', label = 'Mittelwerte')
ax2.plot(y_linearity["reference"], -np.ones(np.size(y_linearity["reference"])) * Y_TOLERANCE * 0.05, 'g-', label = 'Toleranzgrenze')
ax2.plot(y_linearity["reference"],  np.ones(np.size(y_linearity["reference"])) * Y_TOLERANCE * 0.05, 'g-')

ax1.legend()#loc = 9, ncol = 3)
ax2.legend()#loc = 9, ncol = 3)

print()
print("Prüfung individueller Abweichungen")
if (np.abs(y_linearity.groupby("reference").aggregate("mean"))["deviation"] <= 0.05 * Y_TOLERANCE).all(axis=None):
    print("Keine individuelle Abweichung zur Linearität")
else:
    print("Individuelle Abweichung zur Linearität")
    


""" d) Einschätzung der Prozessstreuung nach Verfahren 2 (Mit Einfluss des Prüfers) """

# Daten laden und aufbereiten
# Ziel ist ein Dataframe mit den Spalten: Part, Measurement, Appraiser und Value
data_var_prt = np.repeat(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 6)
data_var_mes = np.tile(np.array([1, 2]), 30)
data_var_apr = np.tile(np.repeat(np.array([1, 2, 3]), 2), 10)
data_var_val = np.array([19.4903, 19.4903, 19.4828, 19.4804, 19.4803, 19.4782, 19.4199, 19.4223, 19.4314, 19.4316, 19.4192, 19.4170, 19.3835, 19.3835, 19.3627, 19.3652, 19.3799, 19.3788, 19.4248, 19.4272, 19.4338, 19.4314, 19.4454, 19.4454, 19.4078, 19.4102, 19.4142, 19.4118, 19.4072, 19.4083, 19.4078, 19.4078, 19.4289, 19.4289, 19.4258, 19.4247, 19.4369, 19.4369, 19.4338, 19.4363, 19.4421, 19.4421, 19.4223, 19.4223, 19.4191, 19.4191, 19.4279, 19.4279, 19.4248, 19.4272, 19.4216, 19.4191, 19.4356, 19.4345, 19.4078, 19.4102, 19.4142, 19.4142, 19.4225, 19.4214])
Y_K = 10 # Anzahl Messobjekte
Y_J = 3 # Anzahl Prüfer
Y_N = 2 # Anzahl Messreihen
y_variation_2 = pd.DataFrame({'Part': data_var_prt,
                              'Measurement': data_var_mes,
                              'Appraiser': data_var_apr,
                              'Value': data_var_val})

# Berechnung der normalisierten Quadratsummen unter Nutzung der Anova-Tabelle
poly = ols('Value ~ C(Part) + C(Appraiser) + C(Part):C(Appraiser)', data = y_variation_2)
model = poly.fit()
anova2 = sm.stats.anova_lm(model, typ = 2)
anova2["M"] = anova2["sum_sq"] / anova2["df"]

# Varianzabschätzung und Berechnung von GRR und ndc
equipment_variation = np.sqrt(anova2.loc["Residual", "M"])
appraiser_variation = np.sqrt((anova2.loc["C(Appraiser)", "M"] - anova2.loc["C(Part):C(Appraiser)", "M"]) / Y_K / Y_N)
interaction_variation = np.sqrt((anova2.loc["C(Part):C(Appraiser)", "M"] - anova2.loc["Residual", "M"]) / Y_N)
part_variation = np.sqrt((anova2.loc["C(Part)", "M"] - anova2.loc["Residual", "M"]) / Y_J / Y_N)
grr = np.sqrt(appraiser_variation**2 + interaction_variation**2 + equipment_variation**2)
grr_relative = 6 * grr / Y_TOLERANCE
ndc = 1.41 * part_variation / grr
print()
print()
print("4. Streuverhalten: Verfahren 2")
print()
print("Relativer GRR-Wert %GRR = ", round(grr_relative * 100, 4), "%")
if (grr_relative <= 0.05):
    print("Messprozess ist entsprechend %GRR sehr gut")
elif (grr_relative <= 0.1):
    print("Messprozess ist entsprechend %GRR fähig")
elif (grr_relative <= 0.3):
    print("Messprozess ist entsprechend %GRR bedingt fähig")
else:
    print("Messprozess ist entsprechend %GRR nicht fähig")
print()
print("Number of Distict Categories ndc = ", round(ndc, 4))
if (ndc >= 5):
    print("System ist entsprechend ndc fähig")
else:
    print("System ist entsprechend ndc nicht fähig")
    
# Visualisierung jedes Prüfers mithilfe von Multi-Indexing
y_variation_2_multi = y_variation_2.set_index(['Appraiser', 'Measurement', 'Part'])
fig4 = plt.figure(4, figsize = (12, 4))
fig4.suptitle('')
ax1, ax2, ax3 = fig4.subplots(1, 3, gridspec_kw = dict(wspace = 0.3))

ax1.plot(np.arange(1, Y_K + 1, 1), y_variation_2_multi.loc[1, 1, :], 'b', label = 'Messreihe 1')
ax1.plot(np.arange(1, Y_K + 1, 1), y_variation_2_multi.loc[1, 2, :], 'r:', label = 'Messreihe 2')

ax1.set_xlabel('Stichprobe')
ax1.set_ylabel(Einheit0 + ' ' + Einheit_sign + ' in ' + Einheit_unit)
ax1.set_title('Messstation 1')
ax1.grid()
ax1.legend()

ax2.plot(np.arange(1, Y_K + 1, 1), y_variation_2_multi.loc[2, 1, :], 'b', label = 'Messreihe 1')
ax2.plot(np.arange(1, Y_K + 1, 1), y_variation_2_multi.loc[2, 2, :], 'r:', label = 'Messreihe 2')

ax2.set_xlabel('Stichprobe')
ax2.set_ylabel(Einheit0 + ' ' + Einheit_sign + ' in ' + Einheit_unit)
ax2.set_title('Messstation 2')
ax2.grid()
ax2.legend()

ax3.plot(np.arange(1, Y_K + 1, 1), y_variation_2_multi.loc[3, 1, :], 'b', label = 'Messreihe 1')
ax3.plot(np.arange(1, Y_K + 1, 1), y_variation_2_multi.loc[3, 2, :], 'r:', label = 'Messreihe 2')

ax3.set_xlabel('Stichprobe')
ax3.set_ylabel(Einheit0 + ' ' + Einheit_sign + ' in ' + Einheit_unit)
ax3.set_title('Messstation 3')
ax3.grid()
ax3.legend()

# Visualisierung des Mittelwerts jedes Prüfers mithilfe von Multi-Indexing
fig5 = plt.figure(5, figsize=(8, 6))
fig5.suptitle('')
ax1 = fig5.subplots(1, 1)

ax1.plot(np.arange(1, Y_K + 1, 1), y_variation_2_multi.loc[1, :, :].mean(level = ['Part']), 'b', label = 'Messstation 1')
ax1.plot(np.arange(1, Y_K + 1, 1), y_variation_2_multi.loc[2, :, :].mean(level = ['Part']), 'r:', label = 'Messstation 2')
ax1.plot(np.arange(1, Y_K + 1, 1), y_variation_2_multi.loc[3, :, :].mean(level = ['Part']), 'g--', label = 'Messstation 3')

ax1.set_xlabel('Stichprobe')
ax1.set_ylabel(Einheit0 + ' ' + Einheit_sign + ' in ' + Einheit_unit)
ax1.grid()
ax1.legend(loc = 9, ncol = 3)


""" d) Einschätzung der Prozessstreuung nach Verfahren 3 (Ohne Einfluss des Prüfers) """
"""
# Daten laden und formatieren
data_var_prt = np.repeat(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 3)
data_var_mes = np.tile(np.array([1, 2]), 15)
data_var_val = np.random.normal(18.3, 0.01, 30)
Y_K = 15 # Anzahl Messobjekte
Y_N = 2 # Anzahl Messreihen
y_variation_3 = pd.DataFrame({'Part': data_var_prt,
                              'Measurement': data_var_mes,
                              'Value': data_var_val})

# Berechnung der normalisierten Quadratsummen unter Nutzung der Anova-Tabelle
poly = ols('Value ~ C(Part)', data = y_variation_3)
model = poly.fit()
anova1 = sm.stats.anova_lm(model, typ = 2)
anova1["M"] = anova1["sum_sq"] / anova1["df"]

# Varianzabschätzung und Berechnung von GRR und ndc
equipment_variation = np.sqrt(anova1.loc["Residual", "M"])
part_variation = np.sqrt((anova1.loc["C(Part)", "M"] - anova1.loc["Residual", "M"]) / Y_N)
grr = equipment_variation
grr_relative = 6 * grr / Y_TOLERANCE
ndc = 1.41 * part_variation / grr
print()
print()
print("5. Streuverhalten: Verfahren 3")
print()
print("Relativer GRR-Wert %GRR = ", round(grr_relative * 100, 4), "%")
if (grr_relative <= 0.05):
    print("Messprozess ist entsprechend %GRR sehr gut")
elif (grr_relative <= 0.1):
    print("Messprozess ist entsprechend %GRR fähig")
elif (grr_relative <= 0.3):
    print("Messprozess ist entsprechend %GRR bedingt fähig")
else:
    print("Messprozess ist entsprechend %GRR nicht fähig")
print()
print("Number of Distict Categories ndc = ", round(ndc, 4))
if (ndc >= 5):
    print("System ist entsprechend ndc fähig")
else:
    print("System ist entsprechend ndc nicht fähig")

# Visualisierung
y_variation_3_multi = y_variation_3.set_index(['Measurement', 'Part'])
fig6 = plt.figure(6, figsize = (8, 6))
fig6.suptitle('')
ax1 = fig6.subplots(1, 1)

ax1.plot(np.arange(1, Y_K + 1, 1), y_variation_3_multi.loc[1, :], 'b', label = 'Messung 1')
ax1.plot(np.arange(1, Y_K + 1, 1), y_variation_3_multi.loc[2, :], 'r:', label = 'Messung 2')

ax1.set_xlabel('Stichprobe')
ax1.set_ylabel(Einheit0 + ' ' + Einheit_sign + ' in ' + Einheit_unit)
ax1.grid()
ax1.legend(loc = 9, ncol = 3)
"""


""" e) Langzeitstabilität """

# Daten laden und evaluieren
y_longterm1 = np.array([18.3080, 18.2978, 18.3038, 18.3052, 18.3021, 18.2924, 18.2894, 18.3052, 18.3074, 18.3019, 18.2945, 18.2948, 18.3095, 18.2871, 18.3006, 18.2924, 18.3095, 18.3054, 18.3069, 18.2945])
y_longterm2 = np.array([18.2867, 18.3034, 18.2903, 18.3026, 18.2948, 18.3000, 18.2996, 18.3026, 18.2998, 18.2844, 18.3082, 18.3069, 18.3000, 18.3008, 18.3066, 18.3043, 18.3000, 18.2989, 18.2988, 18.3081])
y_longterm3 = np.array([18.2953, 18.2866, 18.3005, 18.3087, 18.3080, 18.3048, 18.3028, 18.3087, 18.2994, 18.2801, 18.3046, 18.2825, 18.2814, 18.2960, 18.2898, 18.2969, 18.2814, 18.3066, 18.3069, 18.3000])
y_longterm = np.stack((y_longterm1, y_longterm2, y_longterm3)).T
Y_LONGTERM_MU = 18.3
Y_LONGTERM_SIG = 0.01
y_longterm_mean = np.mean(y_longterm, axis = 1)
y_longterm_std = np.std(y_longterm, ddof = 1, axis = 1)
y_longterm_len = y_longterm.shape[1]
GAMMA_WARN = 0.95
GAMMA_CORRECT = 0.99

# Hypothesentest für den Mittelwert
c1_warn = stats.norm.ppf((1 - GAMMA_WARN) / 2)
c2_warn = stats.norm.ppf((1 + GAMMA_WARN) / 2)
c1_correct = stats.norm.ppf((1 - GAMMA_CORRECT) / 2)
c2_correct = stats.norm.ppf((1 + GAMMA_CORRECT) / 2)
y_longterm_mean_warn_1 = Y_LONGTERM_MU + c1_warn * Y_LONGTERM_SIG / np.sqrt(y_longterm_len)
y_longterm_mean_warn_2 = Y_LONGTERM_MU + c2_warn * Y_LONGTERM_SIG / np.sqrt(y_longterm_len)
y_longterm_mean_correct_1 = + Y_LONGTERM_MU + c1_correct * Y_LONGTERM_SIG / np.sqrt(y_longterm_len)
y_longterm_mean_correct_2 = Y_LONGTERM_MU + c2_correct * Y_LONGTERM_SIG / np.sqrt(y_longterm_len)
print()
print()
print("4. Langzeitstabilität")
print()
print("Mittelwert")
mean_flag_warn = False
mean_flag_corr = False
if ((y_longterm_mean > y_longterm_mean_warn_1).all(axis = None) & (y_longterm_mean < y_longterm_mean_warn_2).all(axis = None)):
    print("Warngrenzen nicht überschritten")
else:
    print("Warngrenzen überschritten")
    mean_flag_warn = True
if ((y_longterm_mean > y_longterm_mean_correct_1).all(axis = None) & (y_longterm_mean < y_longterm_mean_correct_2).all(axis = None)):
    print("Eingriffsgrenzen nicht überschritten")    
else:
    print("Eingriffsgrenzen überschritten")
    mean_flag_corr = True

# Hypothesentest für die Standardabweichung
c1_warn = stats.chi2.ppf((1 - GAMMA_WARN) / 2, y_longterm_len - 1)
c2_warn = stats.chi2.ppf((1 + GAMMA_WARN) / 2, y_longterm_len - 1)
c1_correct = stats.chi2.ppf((1 - GAMMA_CORRECT) / 2, y_longterm_len - 1)
c2_correct = stats.chi2.ppf((1 + GAMMA_CORRECT) / 2, y_longterm_len - 1)
y_longterm_sig_warn_1 = np.sqrt(c1_warn / (y_longterm_len - 1)) * Y_LONGTERM_SIG
y_longterm_sig_warn_2 = np.sqrt(c2_warn / (y_longterm_len - 1)) * Y_LONGTERM_SIG
y_longterm_sig_correct_1 = np.sqrt(c1_correct/(y_longterm_len - 1)) * Y_LONGTERM_SIG
y_longterm_sig_correct_2 = np.sqrt(c2_correct/(y_longterm_len - 1)) * Y_LONGTERM_SIG
print()
print("Standardabweichung")
std_flag_warn = False
std_flag_corr = False
if ((y_longterm_std > y_longterm_sig_warn_1).all(axis = None) & (y_longterm_std < y_longterm_sig_warn_2).all(axis = None)):
    print("Warngrenzen nicht überschritten")
else:
    print("Warngrenzen überschritten")
    std_flag_warn = True
if ((y_longterm_std > y_longterm_sig_correct_1).all(axis = None) & (y_longterm_std < y_longterm_sig_correct_2).all(axis = None)):
    print("Eingriffsgrenzen nicht überschritten")
else:
    print("Eingriffsgrenzen überschritten")
    std_flag_corr = True
print()
if (not(mean_flag_warn & std_flag_warn)):
    print('Das Messsystem ist langzeitstabil')
elif (not(mean_flag_corr & std_flag_corr)):
    print('Das Messsystem ist bedingt langzeitstabil')
else:
    print('Das Messsystem ist nicht langzeitstabil')

# Visualisierung
fig3 = plt.figure(3, figsize = (12, 4))
fig3.suptitle('Shewhart Regelkarte')
ax1, ax2 = fig3.subplots(1, 2, gridspec_kw = dict(wspace = 0.3))

#ax1.plot(np.arange(0, y_longterm.shape[0]) + 1, Y_LONGTERM_MU * np.ones(y_longterm.shape[0]), 'k', alpha = 0.7)
ax1.plot(np.arange(0, y_longterm.shape[0]) + 1, y_longterm_mean, 'bo-', label = 'Mittelwert')
ax1.plot(np.arange(0, y_longterm.shape[0]) + 1, y_longterm_mean_correct_1 * np.ones(y_longterm.shape[0]), 'r:', label = 'EG')
ax1.plot(np.arange(0, y_longterm.shape[0]) + 1, y_longterm_mean_correct_2 * np.ones(y_longterm.shape[0]), 'r:')
ax1.plot(np.arange(0, y_longterm.shape[0]) + 1, y_longterm_mean_warn_1 * np.ones(y_longterm.shape[0]), 'g--', label = 'WG')
ax1.plot(np.arange(0, y_longterm.shape[0]) + 1, y_longterm_mean_warn_2 * np.ones(y_longterm.shape[0]), 'g--')

ax1.set_xlabel('Stichprobe')
ax1.set_ylabel(r'Mittelwert $\overline{' + Einheit_sign + '}$ in ' + Einheit_unit)
ax1.set_title('Kontrolle des Mittelwerts')
ax1.grid()
ax1.legend(loc = 9, ncol = 3)

ax2.plot(np.arange(0, y_longterm.shape[0]) + 1, y_longterm_std, 'bo-', label = 'Standardabweichung')
ax2.plot(np.arange(0, y_longterm.shape[0]) + 1, y_longterm_sig_correct_1 * np.ones(y_longterm.shape[0]), 'r:', label = 'EG')
ax2.plot(np.arange(0, y_longterm.shape[0]) + 1, y_longterm_sig_correct_2 * np.ones(y_longterm.shape[0]), 'r:')
ax2.plot(np.arange(0, y_longterm.shape[0]) + 1, y_longterm_sig_warn_1 * np.ones(y_longterm.shape[0]), 'g--', label = 'WG')
ax2.plot(np.arange(0, y_longterm.shape[0]) + 1, y_longterm_sig_warn_2 * np.ones(y_longterm.shape[0]), 'g--')

ax2.set_xlabel('Stichprobe')
ax2.set_ylabel('Standardabweichung s in ' + Einheit_unit)
ax2.set_title('Kontrolle der Standardabweichung')
ax2.grid()
ax2.legend(loc = 9, ncol = 3)














""" Beispiel 2 - Wenn Toleranz gesucht ist """

""" Einheit definieren """
Einheit0 = 'Masse'
Einheit_sign = 'm'
Einheit_unit = 'g'


""" a) Toleranz einschätzen """

Y_RESOLUTION = 0.01
print()
print()
print('a) Bewertung der Auflösung')
print()
y_tol_res = Y_RESOLUTION / 0.05
print("Die Toleranzgrenze der Auflösung beträgt: Delta " + Einheit_sign + ' =', round(y_tol_res, 4), Einheit_unit)

    
""" b) Systematische Messabweichung und Wiederholbarkeit """

# Daten laden
data_ref = np.array([146.99, 147.32, 147.66, 146.32, 147.66, 146.65, 146.99, 146.99, 147.32, 147.66, 146.65, 146.65, 146.99, 147.32, 147.66, 148.66, 147.99, 146.99, 147.32, 147.66, 147.99, 147.32, 147.99, 147.32, 146.99, 147.66, 147.66, 146.99, 146.32, 146.99, 147.32, 146.99, 147.66, 147.32, 146.99, 147.32, 146.99, 146.99, 147.99, 147.66, 147.66, 146.99, 146.99, 147.32, 148.32, 148.32, 147.99, 146.99, 147.66, 147.66])
y_repeat_test = data_ref.T
y_repeat_len = np.size(y_repeat_test)
y_repeat_reference = 147.35

# Visualisierung der systematischen Messabweichung
fig1 = plt.figure(1, figsize = (6, 4))
fig1.suptitle('')
ax1 = fig1.subplots(1, 1)

ax1.plot(np.arange(0, y_repeat_len) + 1, y_repeat_test, 'bo-')
ax1.plot(np.arange(0, y_repeat_len) + 1, y_repeat_reference * np.ones(y_repeat_len), 'r')

ax1.set_xlabel('Messung')
ax1.set_ylabel(Einheit0 + ' ' +  Einheit_sign + ' in ' + Einheit_unit)
ax1.set_title('Visualisierung der systematischen Messabweichung')
ax1.grid()

# Berechnung des Fähigkeitsindex
y_deviation = np.mean(y_repeat_test) - y_repeat_reference
y_tol_cg = 1.33 * 3 * np.std(y_repeat_test, ddof = 1) / 0.1
print()
print()
print("b) Systematische Abweichung und Wiederholbarkeit")
print()
print(r'Die Toleranzgrenze für einen cg-Wert von 1.33 beträgt: Delta ' + Einheit_sign + ' =', round(y_tol_cg, 4), Einheit_unit)
    
y_tol_cgk = ((1.33 * 3 * np.std(y_repeat_test, ddof = 1)) + np.abs(y_deviation)) / 0.1
print(r'Die Toleranzgrenze für einen cgk-Wert von 1.33 beträgt: Delta ' + Einheit_sign + ' =', round(y_tol_cgk, 4), Einheit_unit)

# Hypothesentest mit H0: y_repeat_test = y_repeat_reference
hypo_test = stats.ttest_1samp(y_repeat_test, y_repeat_reference)
print()
print("Hypothesentest auf Abweichung mit p-value = ", round(float(hypo_test[1]), 4))
if hypo_test[1] <= 0.05:
    print("Abweichung signifikant")
else:
    print("Abweichung nicht signifikant")
    

""" c) Linearität """

# Daten laden
# Ziel-Datenframe besteht aus Referenzwerten und Abweichungen
data_lin_ref = np.tile(np.array([98.50, 123.17, 147.35, 172.08, 196.82]), 10)
data_lin_mes = np.array([98.94, 123.30, 146.99, 173.35, 196.03, 98.94, 123.97, 147.32, 171.68, 197.03, 98.28, 124.30, 147.66, 172.34, 196.70, 98.61, 122.97, 146.32, 172.01, 197.03, 98.61, 122.97, 147.66, 171.68, 197.03, 99.28, 122.97, 146.65, 171.68, 197.03, 98.61, 122.97, 146.99, 171.68, 196.70, 98.61, 123.63, 146.99, 172.01, 197.70, 98.28, 123.30, 147.32, 171.68, 196.03, 98.28, 123.97, 147.66, 171.01, 196.70])
data_lin_dev = data_lin_mes - data_lin_ref
y_linearity = pd.DataFrame({'reference': data_lin_ref,
                            'deviation': np.reshape(data_lin_dev, -1)})

# Visualisierung
fig2 = plt.figure(2, figsize = (8, 5))
fig2.suptitle('')
ax1 = fig2.subplots(gridspec_kw = dict(wspace = 0.3))

ax1.plot(y_linearity["reference"], y_linearity["deviation"], 'b+', alpha = 0.4, label = 'Stichprobe')

ax1.set_xlabel('Referenzwert '+  Einheit_sign + ' in ' + Einheit_unit)
ax1.set_ylabel(r'Abweichung $\Delta$' + Einheit_sign + ' in ' + Einheit_unit)
ax1.set_title('Bewertung des Konfidenzbereichs')
ax1.grid(True)

# Regressionsfunktion mit Konfidenzbereich
poly = ols("deviation ~ reference", y_linearity)
model = poly.fit()
#print(model.summary())

y_plot = np.linspace(y_linearity["reference"].min(), y_linearity["reference"].max(), 1000)
y_regress = pd.DataFrame({"reference": np.reshape(y_plot, -1)})
y_regress["deviation"] = model.predict(y_regress)

ax1.plot(y_linearity["reference"], np.zeros(np.size(y_linearity["reference"])), 'k', alpha = 0.7)
ax1.plot(y_regress["reference"], y_regress["deviation"], 'r', label = 'Regression')

gamma = 0.95
y_regress["confidence"], y_regress["prediction"] = conf_pred_band_ex(y_regress, poly, model, alpha = 1 - gamma)

ax1.plot(y_regress["reference"], y_regress["deviation"] + y_regress["confidence"], 'r:', label = str(gamma * 100) + '%-Konfidenzbereich')
ax1.plot(y_regress["reference"], y_regress["deviation"] - y_regress["confidence"], 'r:')

ax1.legend()#loc = 9, ncol = 3)

# Hypothesentest Gesamtlinearität
print()
print()
print("c) Linearität")
print()
print("Prüfung Regressionsgerade")
if (model.pvalues > 0.05).all(axis = None):
    print("Keine signifikante Abweichung zur Linearität")
else:
    print("Signifikante Abweichung zur Linearität")

# Die Toleranzgrenze, die sich aus der Linearitätsbewertung ergibt, berechnet 
# sich aus der großten Abweichungdes Stichprobenmittelwertes von:
y_linearity_dev_max = np.abs(y_linearity.groupby("reference").aggregate('mean').max())
# Die Toleranzgrenze wird wie folgt berechnet:
y_linearity_tol = float(y_linearity_dev_max * 2 / 0.05)
print()
print('Die Toleranzgrenze der Linearität beträgt: Delta ' + Einheit_sign + ' =', round(y_linearity_tol, 4), Einheit_unit)


""" d) Einschätzung der Prozessstreuung nach Verfahren 3 (Ohne Einfluss des Prüfers) """

# Daten laden und formatieren
data_var_prt = np.repeat(np.arange(1, 26, 1), 2)
data_var_mes = np.tile(np.array([1, 2]), 25)
data_var_val = np.array([117.28, 116.82, 122.94, 122.16, 125.29, 125.29, 142.23, 141.64, 115.27, 115.08, 129.37, 129.82, 133.07, 131.82, 136.08, 135.09, 105.48, 104.92, 123.60, 123.24, 125.08, 123.28, 105.79, 106.01, 111.11, 108.83, 135.59, 135.73, 117.98, 119.06, 126.04, 127.25, 124.13, 122.08, 111.72, 111.00, 112.09, 111.57, 127.55, 126.65, 106.34, 107.00, 133.46, 132.85, 129.42, 128.62, 109.27, 108.59, 117.52, 116.58])
Y_K = 25 # Anzahl Messobjekte
Y_N = 2 # Anzahl Messreihen
y_variation_3 = pd.DataFrame({'Part': data_var_prt,
                              'Measurement': data_var_mes,
                              'Value': data_var_val})

# Berechnung der normalisierten Quadratsummen unter Nutzung der Anova-Tabelle
poly = ols('Value ~ C(Part)', data = y_variation_3)
model = poly.fit()
anova1 = sm.stats.anova_lm(model, typ = 2)
anova1["M"] = anova1["sum_sq"] / anova1["df"]

# Varianzabschätzung und Berechnung von GRR und ndc
equipment_variation = np.sqrt(anova1.loc["Residual", "M"])
part_variation = np.sqrt((anova1.loc["C(Part)", "M"] - anova1.loc["Residual", "M"]) / Y_N)
grr = equipment_variation
# Die Toleranz berechnet sich aus dem grr-Wert wie folgt:
grr_relative = 0.3
y_tol_grr = 6 * grr / grr_relative
ndc = 1.41 * part_variation / grr
print()
print()
print("d) Streuverhalten: Verfahren 3 (ohne Prüfer)")
print()
print('Die Toleranzgrenze des Streuverhaltens für einen Wert %GRR = 30% beträgt:')
print('Delta ' + Einheit_sign + ' =', round(y_tol_grr, 4), Einheit_unit)
print('Der ndc-Wert ist mit: ndc =', round(ndc, 4), 'ausreichend groß')

# Visualisierung
y_variation_3_multi = y_variation_3.set_index(['Measurement', 'Part'])
fig6 = plt.figure(6, figsize = (6, 4))
fig6.suptitle('')
ax1 = fig6.subplots(1, 1)

ax1.plot(np.arange(1, Y_K + 1, 1), y_variation_3_multi.loc[1, :], 'b', label = 'Messung 1')
ax1.plot(np.arange(1, Y_K + 1, 1), y_variation_3_multi.loc[2, :], 'r:', label = 'Messung 2')

ax1.set_xlabel('Stichprobe')
ax1.set_ylabel(Einheit0 + ' ' + Einheit_sign + ' in ' + Einheit_unit)
ax1.grid()
ax1.legend(loc = 9, ncol = 3)


""" e) Langzeitstabilität """

# Daten laden und evaluieren
y_longterm1=np.array([123.70, 122.49, 123.24, 123.41, 122.81, 123.84, 123.99, 124.43, 123.70, 122.92, 122.99, 123.63, 123.46, 123.50, 123.74, 123.48, 123.21, 123.50, 123.72, 123.25])
y_longterm2=np.array([123.09, 123.50, 123.79, 123.55, 123.09, 123.48, 122.95, 123.74, 122.66, 123.54, 123.33, 123.61, 123.49, 123.66, 123.78, 122.78, 123.12, 123.32, 122.81, 123.86])
y_longterm3=np.array([123.66, 123.07, 123.34, 123.85, 123.52, 123.45, 123.47, 123.02, 124.02, 123.78, 123.04, 123.21, 124.25, 123.14, 123.93, 123.32, 123.68, 123.34, 123.43, 123.30])
y_longterm4=np.array([122.93, 123.06, 123.80, 123.64, 122.96, 123.55, 123.63, 123.72, 122.71, 123.72, 122.68, 123.41, 122.43, 123.75, 123.35, 123.17, 123.52, 123.50, 123.29, 122.77])
y_longterm5=np.array([123.76, 123.44, 123.18, 123.26, 123.50, 123.89, 123.27, 123.67, 123.84, 123.40, 123.00, 123.57, 123.52, 123.45, 123.93, 122.66, 123.75, 123.87, 123.62, 123.96])
y_longterm = np.stack((y_longterm1, y_longterm2, y_longterm3, y_longterm4, y_longterm5)).T
Y_LONGTERM_MU = 123.38
Y_LONGTERM_SIG = 0.419
y_longterm_mean = np.mean(y_longterm, axis = 1)
y_longterm_std = np.std(y_longterm, ddof = 1, axis = 1)
y_longterm_len = y_longterm.shape[1]
GAMMA_WARN = 0.95
GAMMA_CORRECT = 0.99

# Hypothesentest für den Mittelwert
c1_warn = stats.norm.ppf((1 - GAMMA_WARN) / 2)
c2_warn = stats.norm.ppf((1 + GAMMA_WARN) / 2)
c1_correct = stats.norm.ppf((1 - GAMMA_CORRECT) / 2)
c2_correct = stats.norm.ppf((1 + GAMMA_CORRECT) / 2)
y_longterm_mean_warn_1 = Y_LONGTERM_MU + c1_warn * Y_LONGTERM_SIG / np.sqrt(y_longterm_len)
y_longterm_mean_warn_2 = Y_LONGTERM_MU + c2_warn * Y_LONGTERM_SIG / np.sqrt(y_longterm_len)
y_longterm_mean_correct_1 = + Y_LONGTERM_MU + c1_correct * Y_LONGTERM_SIG / np.sqrt(y_longterm_len)
y_longterm_mean_correct_2 = Y_LONGTERM_MU + c2_correct * Y_LONGTERM_SIG / np.sqrt(y_longterm_len)
print()
print()
print("e) Langzeitstabilität")
print()
print("Mittelwert")
mean_flag_warn = False
mean_flag_corr = False
if ((y_longterm_mean > y_longterm_mean_warn_1).all(axis = None) & (y_longterm_mean < y_longterm_mean_warn_2).all(axis = None)):
    print("Warngrenzen nicht überschritten")
else:
    print("Warngrenzen überschritten")
    mean_flag_warn = True
if ((y_longterm_mean > y_longterm_mean_correct_1).all(axis = None) & (y_longterm_mean < y_longterm_mean_correct_2).all(axis = None)):
    print("Eingriffsgrenzen nicht überschritten")    
else:
    print("Eingriffsgrenzen überschritten")
    mean_flag_corr = True

# Hypothesentest für die Standardabweichung
c1_warn = stats.chi2.ppf((1 - GAMMA_WARN) / 2, y_longterm_len - 1)
c2_warn = stats.chi2.ppf((1 + GAMMA_WARN) / 2, y_longterm_len - 1)
c1_correct = stats.chi2.ppf((1 - GAMMA_CORRECT) / 2, y_longterm_len - 1)
c2_correct = stats.chi2.ppf((1 + GAMMA_CORRECT) / 2, y_longterm_len - 1)
y_longterm_sig_warn_1 = np.sqrt(c1_warn / (y_longterm_len - 1)) * Y_LONGTERM_SIG
y_longterm_sig_warn_2 = np.sqrt(c2_warn / (y_longterm_len - 1)) * Y_LONGTERM_SIG
y_longterm_sig_correct_1 = np.sqrt(c1_correct/(y_longterm_len - 1)) * Y_LONGTERM_SIG
y_longterm_sig_correct_2 = np.sqrt(c2_correct/(y_longterm_len - 1)) * Y_LONGTERM_SIG
print()
print("Standardabweichung")
std_flag_warn = False
std_flag_corr = False
if ((y_longterm_std > y_longterm_sig_warn_1).all(axis = None) & (y_longterm_std < y_longterm_sig_warn_2).all(axis = None)):
    print("Warngrenzen nicht überschritten")
else:
    print("Warngrenzen überschritten")
    std_flag_warn = True
if ((y_longterm_std > y_longterm_sig_correct_1).all(axis = None) & (y_longterm_std < y_longterm_sig_correct_2).all(axis = None)):
    print("Eingriffsgrenzen nicht überschritten")
else:
    print("Eingriffsgrenzen überschritten")
    std_flag_corr = True
print()
if (not(mean_flag_warn or std_flag_warn)):
    print('Das Messsystem ist langzeitstabil')
elif (not(mean_flag_corr or std_flag_corr)):
    print('Das Messsystem ist bedingt langzeitstabil')
else:
    print('Das Messsystem ist nicht langzeitstabil')

# Visualisierung
fig3 = plt.figure(3, figsize = (12, 4))
fig3.suptitle('Shewhart Regelkarte')
ax1, ax2 = fig3.subplots(1, 2, gridspec_kw = dict(wspace = 0.3))

#ax1.plot(np.arange(0, y_longterm.shape[0]) + 1, Y_LONGTERM_MU * np.ones(y_longterm.shape[0]), 'k', alpha = 0.7)
ax1.plot(np.arange(0, y_longterm.shape[0]) + 1, y_longterm_mean, 'bo-', label = 'Mittelwert')
ax1.plot(np.arange(0, y_longterm.shape[0]) + 1, y_longterm_mean_correct_1 * np.ones(y_longterm.shape[0]), 'r:', label = 'EG')
ax1.plot(np.arange(0, y_longterm.shape[0]) + 1, y_longterm_mean_correct_2 * np.ones(y_longterm.shape[0]), 'r:')
ax1.plot(np.arange(0, y_longterm.shape[0]) + 1, y_longterm_mean_warn_1 * np.ones(y_longterm.shape[0]), 'g--', label = 'WG')
ax1.plot(np.arange(0, y_longterm.shape[0]) + 1, y_longterm_mean_warn_2 * np.ones(y_longterm.shape[0]), 'g--')

ax1.set_xlabel('Stichprobe')
ax1.set_ylabel(r'Mittelwert $\overline{' + Einheit_sign + '}$ in ' + Einheit_unit)
ax1.set_title('Kontrolle des Mittelwerts')
ax1.grid()
ax1.legend(loc = 9, ncol = 3)

ax2.plot(np.arange(0, y_longterm.shape[0]) + 1, y_longterm_std, 'bo-', label = 'Standardabweichung')
ax2.plot(np.arange(0, y_longterm.shape[0]) + 1, y_longterm_sig_correct_1 * np.ones(y_longterm.shape[0]), 'r:', label = 'EG')
ax2.plot(np.arange(0, y_longterm.shape[0]) + 1, y_longterm_sig_correct_2 * np.ones(y_longterm.shape[0]), 'r:')
ax2.plot(np.arange(0, y_longterm.shape[0]) + 1, y_longterm_sig_warn_1 * np.ones(y_longterm.shape[0]), 'g--', label = 'WG')
ax2.plot(np.arange(0, y_longterm.shape[0]) + 1, y_longterm_sig_warn_2 * np.ones(y_longterm.shape[0]), 'g--')

ax2.set_xlabel('Stichprobe')
ax2.set_ylabel('Standardabweichung s in ' + Einheit_unit)
ax2.set_title('Kontrolle der Standardabweichung')
ax2.grid()
ax2.legend(loc = 9, ncol = 3)


""" f) Toleranzgrenze """

# Die minimale Toleranz ergibt sich aus der größten ermittelten Toleranzgrenze
print()
print()
print('Die gerade noch zulässige Zieltoleranz ergibt sich aus dem Maximum der berechneten Toleranzgrenzen')
print('In diesem Beispiel ist das der Wert: TCGK =', round(y_tol_cgk, 4), Einheit_unit)