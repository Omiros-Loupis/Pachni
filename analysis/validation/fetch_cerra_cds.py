"""Κατεβάζει από το Copernicus CDS την ελάχιστη θερμοκρασία 2 μ. του CERRA (5,5 χλμ.) για τις νύχτες
που δίνεις. Το κλειδί διαβάζεται από το ~/.cdsapirc.

    .venv/bin/python fetch_cerra_cds.py 2025 03 19 20

Κάθε ώρα της νύχτας: πρόγνωση από τις 18:00 και 00:00 UTC, βήματα 1–6 ωρών, άρα 19:00–06:00 UTC.
Έξοδος: raw/cerra/cerra_tmin_<έτος><μήνας>_<μέρες>.grib
"""
import os
import sys

import cdsapi

year, month, days = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs('raw/cerra', exist_ok=True)
target = 'raw/cerra/cerra_tmin_%s%s_%s.grib' % (year, month, '-'.join(days))
req = {
    'variable': ['minimum_2m_temperature_since_previous_post_processing'],
    'level_type': 'surface_or_atmosphere',
    'data_type': ['reanalysis'],
    'product_type': 'forecast',
    'year': [year], 'month': [month], 'day': days,
    'time': ['00:00', '18:00'],
    'leadtime_hour': ['1', '2', '3', '4', '5', '6'],
    'data_format': 'grib',
}
cdsapi.Client().retrieve('reanalysis-cerra-single-levels', req, target)
print('έτοιμο:', target, os.path.getsize(target), 'bytes')
