"""Κατεβάζει από το Open-Meteo τη θερμοκρασία του χάρτη (ERA5-Land ή CERRA) στο τετράγωνο
κάθε σταθμού, για κάθε νύχτα Φεβρουαρίου–Απριλίου των χρόνων που ο σταθμός έχει μετρήσεις.

Για κάθε νύχτα κρατά το ελάχιστο από τις 18:00 της προηγούμενης μέρας ως τις 10:00
(ώρα Ελλάδας). Χωρίς διόρθωση υψομέτρου (elevation=nan): η τιμή του τετραγώνου όπως είναι.

Μόνο βιβλιοθήκες της Python (3.9+). Συνεχίζει από εκεί που σταμάτησε αν ξανατρέξει,
και περιμένει μόνο του όταν φτάσει τα όρια του Open-Meteo (ανά λεπτό, ώρα, μέρα).

    python3 fetch_openmeteo.py era5_land
    python3 fetch_openmeteo.py cerra        # το CERRA στο Open-Meteo φτάνει ως το 2021
    python3 fetch_openmeteo.py era5_land data/ghcn_stations.csv data/ghcn_years.csv ghcn

Έξοδος: data/cell_nights_<model>.csv και data/cells_<model>.csv
"""
import csv
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import date, datetime, timedelta

API = 'https://archive-api.open-meteo.com/v1/archive'
BATCH = 10                 # σταθμοί ανά κλήση
WEIGHT_PER_LOC = 90 / 14   # 90 μέρες = 6,4 «κλήσεις» για το Open-Meteo
MAX_PER_MIN = 500          # το όριο είναι 600· αφήνουμε περιθώριο
LAST_YEAR = {'era5_land': 2025, 'cerra': 2021}


def log(*a):
    print(datetime.now().strftime('%H:%M:%S'), *a, flush=True)


def station_years(model, stations_csv, years_csv):
    st = {r['station_id']: r for r in csv.DictReader(open(stations_csv, encoding='utf-8'))}
    sy = defaultdict(set)
    for r in csv.DictReader(open(years_csv)):
        y = int(r['year'])
        if y <= LAST_YEAR[model]:
            sy[y].add(r['station_id'])
    return st, {y: sorted(v, key=int) for y, v in sorted(sy.items())}


def done_keys(path):
    if not os.path.exists(path):
        return set()
    return {(r['station_id'], r['year']) for r in csv.DictReader(open(path))}


def get(url):
    while True:
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                body = r.read()
            try:
                return json.loads(body)
            except ValueError:
                log('απάντηση που δεν είναι JSON (%d bytes), ξαναδοκιμάζω σε 30"' % len(body)); time.sleep(30)
                continue
        except urllib.error.HTTPError as e:
            body = e.read().decode('utf-8', 'replace')
            reason = body
            try:
                reason = json.loads(body).get('reason', body)
            except ValueError:
                pass
            if 'Daily' in reason:
                log('όριο ημέρας, περιμένω 1 ώρα'); time.sleep(3600)
            elif 'Hourly' in reason:
                log('όριο ώρας, περιμένω 10 λεπτά'); time.sleep(600)
            elif 'Minutely' in reason or e.code == 429:
                log('όριο λεπτού, περιμένω 65"'); time.sleep(65)
            else:
                raise RuntimeError('%s %s' % (e.code, reason[:200]))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            log('δίκτυο:', e, '- ξαναδοκιμάζω σε 30"'); time.sleep(30)


def nights(times, temps, year):
    """Ελάχιστο ανά νύχτα: από 18:00 της D-1 ως 10:00 της D, για D από 1/2 ως 30/4."""
    idx = {t: i for i, t in enumerate(times)}
    out = []
    d = date(year, 2, 1)
    while d <= date(year, 4, 30):
        start = datetime(d.year, d.month, d.day) - timedelta(hours=6)
        best, when = None, ''
        for h in range(17):          # 18:00 ... 10:00
            t = (start + timedelta(hours=h)).strftime('%Y-%m-%dT%H:%M')
            i = idx.get(t)
            if i is not None and temps[i] is not None and (best is None or temps[i] < best):
                best, when = temps[i], t[11:13]
        out.append((d.isoformat(), best, when))
        d += timedelta(days=1)
    return out


def main(model, stations_csv='data/noa_stations.csv', years_csv='data/noa_monthly_min.csv', tag=''):
    os.makedirs('data', exist_ok=True)
    sfx = model + ('_' + tag if tag else '')
    out_n = 'data/cell_nights_%s.csv' % sfx
    out_c = 'data/cells_%s.csv' % sfx
    st, by_year = station_years(model, stations_csv, years_csv)
    done = done_keys(out_c)
    new_n, new_c = not os.path.exists(out_n), not os.path.exists(out_c)
    fn, fc = open(out_n, 'a', newline=''), open(out_c, 'a', newline='')
    wn, wc = csv.writer(fn), csv.writer(fc)
    if new_n:
        wn.writerow(['station_id', 'night_date', 'cell_min_c', 'hour_of_min'])
    if new_c:
        wc.writerow(['station_id', 'year', 'cell_lat', 'cell_lon', 'cell_elev_m', 'has_data'])
    todo = [(y, s) for y, ids in by_year.items() for s in ids if (s, str(y)) not in done]
    log(model, ':', len(todo), 'σταθμοί-χρόνια για κατέβασμα (', len(done), 'έτοιμα )')
    used = []
    i = 0
    while i < len(todo):
        y = todo[i][0]
        batch = [s for yy, s in todo[i:i + BATCH] if yy == y]
        need = len(batch) * WEIGHT_PER_LOC
        now = time.time()
        used = [u for u in used if now - u[0] < 61]
        if sum(u[1] for u in used) + need > MAX_PER_MIN:
            time.sleep(3); continue
        q = {
            'latitude': ','.join(st[s]['lat'] for s in batch),
            'longitude': ','.join(st[s]['lon'] for s in batch),
            'elevation': ','.join('nan' for _ in batch),
            'start_date': '%d-01-31' % y, 'end_date': '%d-04-30' % y,
            'hourly': 'temperature_2m', 'models': model, 'timezone': 'Europe/Athens',
        }
        url = API + '?' + '&'.join('%s=%s' % kv for kv in q.items())
        js = get(url)
        used.append((time.time(), need))
        res = js if isinstance(js, list) else [js]
        for s, r in zip(batch, res):
            temps = r['hourly']['temperature_2m']
            has = any(t is not None for t in temps)
            for d, v, h in nights(r['hourly']['time'], temps, y):
                if v is not None:
                    wn.writerow([s, d, v, h])
            wc.writerow([s, y, r['latitude'], r['longitude'], r.get('elevation'), int(has)])
        fn.flush(); fc.flush()
        i += len(batch)
        log(model, y, 'έτοιμα', i, '/', len(todo))
    log(model, 'τέλος')


if __name__ == '__main__':
    a = sys.argv[1:]
    main(*(a or ['era5_land']))
