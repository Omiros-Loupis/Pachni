"""Έλεγχος σε ΟΛΕΣ τις νύχτες Φεβρουαρίου–Απριλίου, με ημερήσια ελάχιστα από 5 σταθμούς της ΕΜΥ
(NOAA GHCN-Daily, δημόσια). Χωρίς επιλογή νυχτών, άρα χωρίς την προκατάληψη του μηνιαίου ελάχιστου.

    python3 ghcn_check.py era5_land
"""
import json
import sys

import numpy as np
import pandas as pd

LAPSE = 0.0065


def load_ghcn(st):
    out = []
    for _, r in st.iterrows():
        d = pd.read_csv('raw/ghcn/%s.csv.gz' % r.ghcn_id, header=None,
                        names=['id', 'date', 'el', 'v', 'm', 'q', 's', 't'], dtype={'date': str})
        d = d[(d.el == 'TMIN') & d.q.isna() & (d.date >= '20060101')]
        out.append(pd.DataFrame({'station_id': r.station_id,
                                 'date': pd.to_datetime(d.date, format='%Y%m%d'),
                                 'tmin_c': d.v / 10}))
    return pd.concat(out)


def stats(o, p):
    e = o - p
    return dict(n=int(len(e)), bias=round(float(e.mean()), 2), rmse=round(float(np.sqrt((e ** 2).mean())), 2))


def table(o, p, thr):
    obs, fc = o <= thr, p <= thr
    hits, miss, fa = int((obs & fc).sum()), int((obs & ~fc).sum()), int((~obs & fc).sum())
    return dict(nights_station_below=int(obs.sum()), nights_map_below=int(fc.sum()),
                hit_rate=round(hits / max(1, hits + miss), 2), false_alarm_ratio=round(fa / max(1, hits + fa), 2))


def main(model):
    st = pd.read_csv('data/ghcn_stations.csv')
    g = load_ghcn(st)
    n = pd.read_csv('data/cell_nights_%s_ghcn.csv' % model, parse_dates=['night_date'])
    c = pd.read_csv('data/cells_%s_ghcn.csv' % model).groupby('station_id', as_index=False).cell_elev_m.first()
    best = None
    for lag in (-1, 0, 1):   # σύμβαση ημερομηνίας του GHCN: βρίσκουμε τη μετατόπιση που ταιριάζει
        m = g.assign(night_date=g.date + pd.Timedelta(days=lag)).merge(n, on=['station_id', 'night_date'])
        mae = float((m.tmin_c - m.cell_min_c).abs().mean())
        if best is None or mae < best[1]:
            best = (lag, mae, m)
    lag, _, m = best
    m = m.merge(st[['station_id', 'name_gr', 'elev_m']], on='station_id').merge(c, on='station_id')
    m['lap'] = m.cell_min_c + LAPSE * (m.cell_elev_m - m.elev_m)
    res = {'lag_days': lag, 'όλες οι νύχτες': [stats(m.tmin_c, m.cell_min_c), stats(m.tmin_c, m.lap)]}
    res['νύχτες με σταθμό ≤ 0'] = [stats(m.tmin_c[m.tmin_c <= 0], m.cell_min_c[m.tmin_c <= 0]),
                                    stats(m.tmin_c[m.tmin_c <= 0], m.lap[m.tmin_c <= 0])]
    res['νύχτες με χάρτη ≤ 0'] = [stats(m.tmin_c[m.cell_min_c <= 0], m.cell_min_c[m.cell_min_c <= 0])]
    for thr in (0, -2.2):
        res['πίνακας %s°C' % thr] = [table(m.tmin_c, m.cell_min_c, thr), table(m.tmin_c, m.lap, thr)]
    res['ανά σταθμό'] = {r: [stats(v.tmin_c, v.cell_min_c), stats(v.tmin_c, v.lap), table(v.tmin_c, v.cell_min_c, 0)]
                         for r, v in m.groupby('name_gr')}
    m.to_csv('data/ghcn_nights_%s.csv' % model, index=False)
    json.dump(res, open('results_ghcn_%s.json' % model, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'era5_land')
