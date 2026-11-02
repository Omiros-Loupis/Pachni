"""Έλεγχος του χάρτη (ERA5-Land ή CERRA) απέναντι στους σταθμούς του ΕΑΑ, και πρώτο μοντέλο διόρθωσης.

    python3 validate.py era5_land
    python3 validate.py cerra

Δύο τρόποι σύγκρισης:
  A. «Νύχτα σταθμού»: το ελάχιστο του μήνα στον σταθμό με τον χάρτη την ίδια νύχτα.
     Διαλέγει τη νύχτα από τον σταθμό, άρα φουσκώνει λίγο το λάθος.
  B. «Ελάχιστο με ελάχιστο»: το ελάχιστο του μήνα στον σταθμό με το ελάχιστο του μήνα στον χάρτη.
     Καμία πλευρά δεν διαλέγει νύχτα, άρα δεν έχει αυτή την προκατάληψη.

Μοντέλο: προβλέπει πόσο πρέπει να διορθωθεί ο χάρτης σε ένα σημείο (σταθμός − χάρτης), από
το ανάγλυφο γύρω του. Ελέγχεται σε σταθμούς που δεν είδε (leave-one-station-out) και,
πιο αυστηρά, σε ολόκληρες περιοχές που δεν είδε (μπλοκ 0,5°).
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LAPSE = 0.0065
DAMAGE = -2.2
TERRAIN = ['elev_dem', 'tpi_1k', 'tpi_3k', 'tpi_10k', 'slope_pct', 'rise_med', 'rise_frac150', 'dist_sea_km']


def load(model):
    st = pd.read_csv('data/noa_stations.csv')
    ter = pd.read_csv('data/station_terrain.csv')
    obs = pd.read_csv('data/noa_monthly_min.csv')
    nights = pd.read_csv('data/cell_nights_%s.csv' % model, parse_dates=['night_date'])
    cells = pd.read_csv('data/cells_%s.csv' % model)
    cells = cells[cells.has_data == 1].groupby('station_id', as_index=False)[['cell_lat', 'cell_lon', 'cell_elev_m']].first()

    obs['night_date'] = pd.to_datetime(dict(year=obs.year, month=obs.month, day=obs.day))
    a = obs.merge(nights.rename(columns={'cell_min_c': 'cell_night'}), on=['station_id', 'night_date'], how='inner')

    nights['year'] = nights.night_date.dt.year
    nights['month'] = nights.night_date.dt.month
    mm = nights.loc[nights.groupby(['station_id', 'year', 'month']).cell_min_c.idxmin()]
    mm = mm.rename(columns={'cell_min_c': 'cell_month_min', 'night_date': 'cell_min_date'})[
        ['station_id', 'year', 'month', 'cell_month_min', 'cell_min_date']]
    d = a.merge(mm, on=['station_id', 'year', 'month'], how='inner')
    d = d.merge(st, on='station_id').merge(ter, on='station_id').merge(cells, on='station_id')
    d['dz'] = d.elev_m - d.cell_elev_m
    d['lapse_adj'] = LAPSE * (d.cell_elev_m - d.elev_m)
    return d


def stats(obs, pred):
    e = obs - pred
    return dict(n=int(len(e)), bias=round(float(e.mean()), 2), rmse=round(float(np.sqrt((e ** 2).mean())), 2),
                mae=round(float(e.abs().mean()), 2))


def miss(obs, pred, thr=DAMAGE):
    m = obs <= thr
    return round(float(100 * (pred[m] > thr).mean()), 1), int(m.sum())


def compare(d, col):
    """Σφάλμα του χάρτη (με και χωρίς διόρθωση υψομέτρου) για τη στήλη χάρτη `col`."""
    raw, lap = d[col], d[col] + d.lapse_adj
    g = d[d.ground == 1]
    out = {
        'όλα': [stats(d.tmin_c, raw), stats(d.tmin_c, lap)],
        'παγετός (σταθμός ≤ 0)': [stats(d.tmin_c[d.tmin_c <= 0], raw[d.tmin_c <= 0]),
                                   stats(d.tmin_c[d.tmin_c <= 0], lap[d.tmin_c <= 0])],
        'σταθμοί εδάφους': [stats(g.tmin_c, raw[g.index]), stats(g.tmin_c, lap[g.index])],
        'σταθμοί σε ταράτσα': [stats(d.tmin_c[d.ground == 0], raw[d.ground == 0]),
                               stats(d.tmin_c[d.ground == 0], lap[d.ground == 0])],
        'χάνει το −2,2°C (%, n)': [miss(d.tmin_c, raw), miss(d.tmin_c, lap)],
    }
    bins = [-9999, -300, -150, -50, 50, 150, 300, 9999]
    lab = ['>300 χαμηλότερα', '150–300 χαμ.', '50–150 χαμ.', '±50', '50–150 ψηλ.', '150–300 ψηλ.', '>300 ψηλότερα']
    g = g.assign(b=pd.cut(g.dz, bins, labels=lab))
    out['ανά διαφορά υψομέτρου (εδάφους)'] = {
        str(k): [stats(v.tmin_c, v[col]), stats(v.tmin_c, v[col] + v.lapse_adj)] for k, v in g.groupby('b', observed=True)}
    return out


def cv_predict(d, target_col, feats, groups, make):
    pred = pd.Series(np.nan, index=d.index)
    for gval in groups.unique():
        te = groups == gval
        tr = ~te
        m = make()
        m.fit(d.loc[tr, feats], d.loc[tr, target_col])
        pred[te] = m.predict(d.loc[te, feats])
    return pred


def model_eval(d, col):
    """Διόρθωση = σταθμός − χάρτης. Έλεγχος σε σταθμούς εδάφους που δεν είδε το μοντέλο."""
    d = d.copy()
    d['target'] = d.tmin_c - d[col]
    d['cell_t'] = d[col]
    feats = TERRAIN + ['dz', 'ground', 'month', 'cell_t', 'lat', 'lon']
    loso = d.station_id
    block = (np.floor(d.lat / 0.5) * 100 + np.floor(d.lon / 0.5)).astype(int)
    makers = {
        'σταθερή διόρθωση (μέσος όρος)': None,
        'γραμμικό (ridge)': lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
        'δέντρα (gradient boosting)': lambda: HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0),
    }
    res = {}
    for cv_name, groups in [('LOSO', loso), ('μπλοκ 0,5°', block)]:
        r = {}
        g = d.ground == 1
        r['χάρτης χωρίς διόρθωση'] = stats(d.tmin_c[g], d[col][g])
        r['διόρθωση υψομέτρου'] = stats(d.tmin_c[g], (d[col] + d.lapse_adj)[g])
        for name, mk in makers.items():
            if mk is None:
                corr = pd.Series(np.nan, index=d.index)
                for gv in groups.unique():
                    te = groups == gv
                    corr[te] = d.loc[~te, 'target'].mean()
            else:
                corr = cv_predict(d, 'target', feats, groups, mk)
            pred = d[col] + corr
            r[name] = stats(d.tmin_c[g], pred[g])
            r[name]['χάνει_−2,2'] = miss(d.tmin_c[g], pred[g])[0]
            fr = g & (d.tmin_c <= 0)
            r[name]['rmse_παγετός'] = stats(d.tmin_c[fr], pred[fr])['rmse']
        for k in ['χάρτης χωρίς διόρθωση', 'διόρθωση υψομέτρου']:
            p = d[col] if k.startswith('χάρτης') else d[col] + d.lapse_adj
            r[k]['χάνει_−2,2'] = miss(d.tmin_c[g], p[g])[0]
            fr = g & (d.tmin_c <= 0)
            r[k]['rmse_παγετός'] = stats(d.tmin_c[fr], p[fr])['rmse']
        res[cv_name] = r
    return res


if __name__ == '__main__':
    model = sys.argv[1] if len(sys.argv) > 1 else 'era5_land'
    d = load(model)
    out = {
        'μετρήσεις': int(len(d)), 'σταθμοί': int(d.station_id.nunique()),
        'A_νύχτα_σταθμού': compare(d, 'cell_night'),
        'B_ελάχιστο_με_ελάχιστο': compare(d, 'cell_month_min'),
        'μοντέλο_B': model_eval(d, 'cell_month_min'),
        'μοντέλο_A': model_eval(d, 'cell_night'),
    }
    d.to_csv('data/validation_%s.csv' % model, index=False)
    json.dump(out, open('results_%s.json' % model, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))
