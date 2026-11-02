"""Ανεξάρτητος έλεγχος: μοντέλο εκπαιδευμένο ΜΟΝΟ στους σταθμούς του ΕΑΑ (ελάχιστο με ελάχιστο),
εφαρμοσμένο σε ΟΛΕΣ τις νύχτες 5 σταθμών της ΕΜΥ (GHCN-Daily) που δεν είδε ποτέ.

    python3 independent_test.py era5_land
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from terrain import Terrain
from ghcn_check import stats, table

FEATS = ['elev_dem', 'tpi_1k', 'tpi_3k', 'tpi_10k', 'slope_pct', 'rise_med', 'rise_frac150', 'dist_sea_km',
         'dz', 'ground', 'month', 'cell_t', 'lat', 'lon']


def main(model):
    tr = pd.read_csv('data/validation_%s.csv' % model)
    tr['target'] = tr.tmin_c - tr.cell_month_min
    tr['cell_t'] = tr.cell_month_min
    te = pd.read_csv('data/ghcn_nights_%s.csv' % model, parse_dates=['night_date'])
    st = pd.read_csv('data/ghcn_stations.csv')
    f = Terrain().features(st.lat.values, st.lon.values)
    ter = pd.DataFrame(f).assign(station_id=st.station_id.values, lat=st.lat.values, lon=st.lon.values, ground=1)
    te = te.merge(ter, on='station_id')
    te['dz'] = te.elev_m - te.cell_elev_m
    te['month'] = te.night_date.dt.month
    te['cell_t'] = te.cell_min_c
    out = {'χάρτης χωρίς διόρθωση': te.cell_min_c, 'διόρθωση υψομέτρου': te.lap}
    for name, mk in [('γραμμικό', lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))),
                     ('δέντρα', lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                                       min_samples_leaf=40, l2_regularization=1.0))]:
        m = mk().fit(tr[FEATS], tr.target)
        out[name] = te.cell_min_c + m.predict(te[FEATS])
    res = {}
    for k, p in out.items():
        res[k] = {'όλες': stats(te.tmin_c, p), 'σταθμός≤0': stats(te.tmin_c[te.tmin_c <= 0], p[te.tmin_c <= 0]),
                  '0°C': table(te.tmin_c, p, 0), '−2,2°C': table(te.tmin_c, p, -2.2)}
    json.dump(res, open('results_independent_%s.json' % model, 'w'), ensure_ascii=False, indent=1)
    for k, v in res.items():
        print(k, v)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'era5_land')
