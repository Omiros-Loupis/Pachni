"""Καιρός κάθε νύχτας (μέσος άνεμος 10 μ. και νεφοκάλυψη, 18:00–06:00) από το CERRA, στο τετράγωνο
κάθε σταθμού, Φεβρουάριος–Απρίλιος. Δείχνει αν η νύχτα ήταν ήρεμη και καθαρή, δηλαδή αν
μπορούσε να σχηματιστεί λίμνη κρύου αέρα.

    python3 fetch_night_weather.py cerra

Έξοδος: data/night_weather_<model>.csv (station_id, night_date, wind_ms, cloud_pct)
"""
import csv
import os
import sys
import time
from datetime import date, datetime, timedelta

from fetch_openmeteo import API, MAX_PER_MIN, WEIGHT_PER_LOC, get, log, station_years

BATCH = 10


def night_means(times, wind, cloud, year):
    idx = {t: i for i, t in enumerate(times)}
    out = []
    d = date(year, 2, 1)
    while d <= date(year, 4, 30):
        start = datetime(d.year, d.month, d.day) - timedelta(hours=6)
        w, c = [], []
        for h in range(13):          # 18:00 ... 06:00
            i = idx.get((start + timedelta(hours=h)).strftime('%Y-%m-%dT%H:%M'))
            if i is not None and wind[i] is not None and cloud[i] is not None:
                w.append(wind[i]); c.append(cloud[i])
        if w:
            out.append((d.isoformat(), round(sum(w) / len(w), 2), round(sum(c) / len(c), 1)))
        d += timedelta(days=1)
    return out


def main(model):
    out = 'data/night_weather_%s.csv' % model
    done = set()
    if os.path.exists(out):
        done = {(r['station_id'], r['night_date'][:4]) for r in csv.DictReader(open(out))}
    st, by_year = station_years(model, 'data/noa_stations.csv', 'data/noa_monthly_min.csv')
    f = open(out, 'a', newline='')
    w = csv.writer(f)
    if not done:
        w.writerow(['station_id', 'night_date', 'wind_ms', 'cloud_pct'])
    todo = [(y, s) for y, ids in by_year.items() for s in ids if (s, str(y)) not in done]
    log(model, ':', len(todo), 'σταθμοί-χρόνια')
    used, i = [], 0
    while i < len(todo):
        y = todo[i][0]
        batch = [s for yy, s in todo[i:i + BATCH] if yy == y]
        need = len(batch) * WEIGHT_PER_LOC
        now = time.time()
        used = [u for u in used if now - u[0] < 61]
        if sum(u[1] for u in used) + need > MAX_PER_MIN:
            time.sleep(3); continue
        q = {'latitude': ','.join(st[s]['lat'] for s in batch), 'longitude': ','.join(st[s]['lon'] for s in batch),
             'start_date': '%d-01-31' % y, 'end_date': '%d-04-30' % y, 'hourly': 'wind_speed_10m,cloud_cover',
             'wind_speed_unit': 'ms', 'models': model, 'timezone': 'Europe/Athens'}
        js = get(API + '?' + '&'.join('%s=%s' % kv for kv in q.items()))
        used.append((time.time(), need))
        for s, r in zip(batch, js if isinstance(js, list) else [js]):
            h = r['hourly']
            for row in night_means(h['time'], h['wind_speed_10m'], h['cloud_cover'], y):
                w.writerow([s] + list(row))
        f.flush()
        i += len(batch)
        log(model, y, 'έτοιμα', i, '/', len(todo))
    log('τέλος')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'cerra')
