"""Βοηθά ο καιρός της νύχτας (άνεμος, νεφοκάλυψη) το μοντέλο διόρθωσης πάνω στο CERRA;

    python3 night_model.py

Σύγκριση στη νύχτα του σταθμού (τρόπος A), έλεγχος σε σταθμούς που δεν είδε (LOSO),
μόνο σταθμοί εδάφους στη βαθμολογία.
"""
import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from validate import TERRAIN, load, miss, stats


def main():
    d = load('cerra')
    w = pd.read_csv('data/night_weather_cerra.csv', parse_dates=['night_date'])
    d = d.merge(w, on=['station_id', 'night_date'], how='inner')
    d['target'] = d.tmin_c - d.cell_night
    d['cell_t'] = d.cell_night
    d['calm_clear'] = ((d.wind_ms < 2) & (d.cloud_pct < 30)).astype(int)
    d['calm_x_tpi'] = d.calm_clear * d.tpi_3k
    d['calm_x_rise'] = d.calm_clear * d.rise_frac150
    d['wind_x_tpi'] = d.wind_ms * d.tpi_3k
    d['cloud_x_tpi'] = d.cloud_pct * d.tpi_3k
    base = TERRAIN + ['dz', 'ground', 'month', 'cell_t', 'lat', 'lon']
    wx = base + ['wind_ms', 'cloud_pct', 'calm_clear', 'calm_x_tpi', 'calm_x_rise', 'wind_x_tpi', 'cloud_x_tpi']
    g = d.ground == 1
    res = {'μετρήσεις': int(len(d)), 'σταθμοί': int(d.station_id.nunique()), 'CERRA χωρίς διόρθωση': stats(d.tmin_c[g], d.cell_night[g])}
    res['CERRA χωρίς διόρθωση']['χάνει_−2,2'] = miss(d.tmin_c[g], d.cell_night[g])[0]
    makers = {
        'ridge': lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
        'gb': lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0),
    }
    for fname, feats in [('ανάγλυφο', base), ('ανάγλυφο + καιρός νύχτας', wx)]:
        for mname, mk in makers.items():
            pred = pd.Series(np.nan, index=d.index)
            for s in d.station_id.unique():
                te = d.station_id == s
                pred[te] = mk().fit(d.loc[~te, feats], d.loc[~te, 'target']).predict(d.loc[te, feats])
            p = d.cell_night + pred
            r = stats(d.tmin_c[g], p[g]); r['χάνει_−2,2'] = miss(d.tmin_c[g], p[g])[0]
            fr = g & (d.tmin_c <= 0); r['rmse_παγετός'] = stats(d.tmin_c[fr], p[fr])['rmse']
            res['%s · %s' % (fname, mname)] = r
    # σφάλμα του CERRA ανά τύπο νύχτας και θέση (σταθμοί εδάφους)
    gg = d[g].assign(err=lambda x: x.tmin_c - x.cell_night,
                     night=lambda x: np.where(x.calm_clear == 1, 'ήρεμη και καθαρή', 'άλλη'),
                     place=lambda x: pd.cut(x.tpi_3k, [-500, -30, 30, 500], labels=['κοιλάδα', 'επίπεδο', 'ύψωμα']))
    res['σφάλμα CERRA ανά νύχτα και θέση'] = {f'{a} / {b}': [int(len(v)), round(float(v.err.mean()), 2)] for (a, b), v in gg.groupby(['night', 'place'], observed=True)}
    json.dump(res, open('results_night_model.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
