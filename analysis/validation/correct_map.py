"""Διορθωμένος χάρτης CERRA για το demo (νύχτες προς 19 και 20/3/2025).

    python3 correct_map.py ../../data/raw_cerra.txt ../../data

Μοντέλο: ridge πάνω στο ανάγλυφο, εκπαιδευμένο στον τρόπο B (ελάχιστο του μήνα στον σταθμό
με ελάχιστο του μήνα στο CERRA), όπου καμία πλευρά δεν διαλέγει νύχτα. Εκπαιδεύεται στα
2006–2021 χωρίς τους 5 σταθμούς του demo, άρα το demo είναι εκτός δείγματος και στον χρόνο
(2025) και στον χώρο.

Υψόμετρο τετραγώνου CERRA: μέσος όρος του DEM σε 5,5 × 5,5 χλμ. γύρω από το κέντρο του
πλησιέστερου τετραγώνου του GRIB (ελέγχεται απέναντι στο υψόμετρο που δίνει το Open-Meteo).

Σημεία με |υψόμετρο − υψόμετρο τετραγώνου| πάνω από το 95ό εκατοστημόριο των σταθμών
εκπαίδευσης (~350 μ., κυρίως κορυφές και απότομες πλαγιές) σημειώνονται ως εκτός εύρους.

Περιθώριο λάθους: 10ο και 90ό εκατοστημόριο του λάθους σε σταθμούς που το μοντέλο δεν είδε
(leave-one-station-out, σταθμοί εδάφους, μήνας Μάρτιος έως Απρίλιος και Φεβρουάριος).

Έξοδοι στον φάκελο εξόδου:
  raw_cerra_cor.txt   19:...;20:...  (δέκατα °C, ίδιο πλέγμα 36×36 με το raw_cerra.txt)
  cerra_cor_meta.json περιθώριο λάθους και τιμές στους 5 σταθμούς
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
from scipy.spatial import cKDTree
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from terrain import Terrain
from validate import TERRAIN, load, miss, stats

GRIB = 'raw/cerra/cerra_tmin_202503_19-20.grib'
LAT0, LON0, N, STEP = 40.45, 21.85, 36, 0.02
DEMO = {4: 'Αριδαία', 57: 'Κερασιά', 11: 'Καρυώτισσα', 26: 'Νάουσα', 112: 'Βέροια'}
FEATS = TERRAIN + ['dz', 'ground', 'month', 'cell_t', 'lat', 'lon']


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))


def cell_elev_proxy(T, lat, lon):
    """Μέσο υψόμετρο DEM σε τετράγωνο 5,5 χλμ. γύρω από (lat, lon)."""
    return T.sample(T._mean(2750), lat, lon)


def main(raw_cerra, outdir):
    d = load('cerra')
    d['cell_t'] = d.cell_month_min
    d['target'] = d.tmin_c - d.cell_month_min
    tr = d[~d.station_id.isin(DEMO)].copy()

    # περιθώριο λάθους: LOSO στα δεδομένα εκπαίδευσης, σταθμοί εδάφους
    pred = pd.Series(np.nan, index=tr.index)
    for s in tr.station_id.unique():
        te = tr.station_id == s
        pred[te] = ridge().fit(tr.loc[~te, FEATS], tr.loc[~te, 'target']).predict(tr.loc[te, FEATS])
    g = tr.ground == 1
    err = (tr.tmin_c - (tr.cell_month_min + pred))[g]
    q10, q90 = np.percentile(err, [10, 90])
    loso = stats(tr.tmin_c[g], (tr.cell_month_min + pred)[g])
    loso['χάνει_−2,2'] = miss(tr.tmin_c[g], (tr.cell_month_min + pred)[g])[0]
    raw = stats(tr.tmin_c[g], tr.cell_month_min[g])
    raw['χάνει_−2,2'] = miss(tr.tmin_c[g], tr.cell_month_min[g])[0]

    m = ridge().fit(tr[FEATS], tr.target)
    T = Terrain()

    # έλεγχος του υψομέτρου τετραγώνου: DEM γύρω από το κέντρο του τετραγώνου vs Open-Meteo
    ce = cell_elev_proxy(T, d.cell_lat.values, d.cell_lon.values)
    elev_check = stats(d.cell_elev_m, pd.Series(ce, index=d.index))

    # κέντρα τετραγώνων CERRA για κάθε σημείο του πλέγματος του demo
    ds = xr.open_dataset(GRIB, engine='cfgrib', backend_kwargs={'indexpath': ''})
    la2, lo2 = ds.latitude.values, ds.longitude.values
    box = (la2 > 40.2) & (la2 < 41.4) & (lo2 > 21.6) & (lo2 < 22.8)
    cl, co = la2[box], lo2[box]
    k = np.cos(np.radians(40.8))
    tree = cKDTree(np.c_[cl, co * k])
    ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    glat = (LAT0 + ii * STEP).ravel(); glon = (LON0 + jj * STEP).ravel()
    _, nn = tree.query(np.c_[glat, glon * k])
    f = T.features(glat, glon)
    X = pd.DataFrame({kk: v for kk, v in f.items()})
    X['dz'] = X.elev_dem - cell_elev_proxy(T, cl[nn], co[nn])
    X['ground'] = 1; X['month'] = 3; X['lat'] = glat; X['lon'] = glon

    cer = {}
    for part in open(raw_cerra).read().strip().split(';'):
        kk, v = part.split(':'); cer[kk] = np.array([int(x) for x in v.split(',')]) / 10
    out = {}
    for night, v in cer.items():
        X['cell_t'] = v
        out[night] = v + m.predict(X[FEATS])
    with open(os.path.join(outdir, 'raw_cerra_cor.txt'), 'w') as fh:
        fh.write(';'.join('%s:%s' % (kk, ','.join(str(int(round(x * 10))) for x in v)) for kk, v in out.items()))

    # οι 5 σταθμοί: τιμή του μοντέλου στη θέση τους (με τον δικό τους τύπο σταθμού)
    st = pd.read_csv('data/noa_stations.csv').set_index('station_id')
    sid = list(DEMO)
    fs = T.features(st.loc[sid, 'lat'].values, st.loc[sid, 'lon'].values)
    S = pd.DataFrame(fs)
    S['lat'] = st.loc[sid, 'lat'].values; S['lon'] = st.loc[sid, 'lon'].values
    S['ground'] = st.loc[sid, 'ground'].values; S['month'] = 3
    _, sn = tree.query(np.c_[S.lat, S.lon * k])
    S['dz'] = st.loc[sid, 'elev_m'].values - cell_elev_proxy(T, cl[sn], co[sn])
    stv = {}
    for night in cer:
        si = np.clip(np.round((S.lat - LAT0) / STEP).astype(int), 0, N - 1)
        sj = np.clip(np.round((S.lon - LON0) / STEP).astype(int), 0, N - 1)
        S['cell_t'] = cer[night][si * N + sj]
        stv[night] = {DEMO[s]: [round(float(c), 1), round(float(c + p), 1)]
                      for s, c, p in zip(sid, S.cell_t, m.predict(S[FEATS]))}

    # εκτός εύρους εκπαίδευσης: σημεία πολύ ψηλότερα ή χαμηλότερα από το τετράγωνό τους
    lim = float(np.percentile(tr.dz.abs(), 95))
    mask = (X.dz.abs() > lim).astype(int).values
    meta = {'όριο_dz_m': round(lim), 'εκτός_εύρους_σημεία': int(mask.sum()), 'mask': ''.join(map(str, mask)),
            'q10': round(float(q10), 1), 'q90': round(float(q90), 1),
            'loso_σταθμοί_εδάφους': {'CERRA': raw, 'CERRA + διόρθωση': loso},
            'σταθμοί_εκπαίδευσης': int(tr.station_id.nunique()), 'μετρήσεις_εκπαίδευσης': int(len(tr)),
            'υψόμετρο_τετραγώνου_έλεγχος': elev_check,
            'μέση_διόρθωση_πλέγματος': {kk: round(float((out[kk] - cer[kk]).mean()), 2) for kk in out},
            'εύρος_διόρθωσης_πλέγματος': {kk: [round(float((out[kk] - cer[kk]).min()), 1), round(float((out[kk] - cer[kk]).max()), 1)] for kk in out},
            'σταθμοί_demo_[CERRA, με διόρθωση]': stv}
    json.dump(meta, open(os.path.join(outdir, 'cerra_cor_meta.json'), 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
