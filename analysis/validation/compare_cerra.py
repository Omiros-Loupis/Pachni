"""ERA5-Land απέναντι σε CERRA, στα ίδια μηνιαία ελάχιστα σταθμών εδάφους του ΕΑΑ (2006–2021).

    python3 compare_cerra.py
"""
import json
from validate import load, stats, miss

e, c = load('era5_land'), load('cerra')
k = ['station_id', 'year', 'month']
m = e.merge(c[k + ['cell_night', 'cell_month_min']], on=k, suffixes=('_e', '_c'))
m = m[m.ground == 1]
res = {'μετρήσεις': int(len(m)), 'σταθμοί': int(m.station_id.nunique())}
for name, s in [('ERA5-Land', '_e'), ('CERRA', '_c')]:
    a, b, fr = m['cell_night' + s], m['cell_month_min' + s], m.tmin_c <= 0
    res[name] = {'A_all': stats(m.tmin_c, a), 'A_frost': stats(m.tmin_c[fr], a[fr]), 'A_miss-2.2': miss(m.tmin_c, a),
                 'B_all': stats(m.tmin_c, b), 'B_frost': stats(m.tmin_c[fr], b[fr]), 'B_miss-2.2': miss(m.tmin_c, b)}
json.dump(res, open('results_era5_vs_cerra_noa.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
