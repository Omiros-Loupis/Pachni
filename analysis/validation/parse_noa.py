"""Διαβάζει το δημόσιο αρχείο ρεκόρ του ΕΑΑ (RECORDS_STATIONS_METEO_2006-2025.xlsx)
και βγάζει το ελάχιστο κάθε Φεβρουαρίου, Μαρτίου και Απριλίου ανά σταθμό, με την ημέρα του.

Είσοδος:  raw/RECORDS_STATIONS_METEO_2006-2025.xlsx  (https://meteosearch.meteo.gr/Raw%20Materials/)
          data/noa_stations.csv  (σταθμοί Μακεδονίας και Θεσσαλίας από τον κατάλογο του δικτύου)
Έξοδος:   data/noa_monthly_min.csv  (station_id, year, month, tmin_c, day)
"""
import csv, re, unicodedata
from openpyxl import load_workbook

MONTH_COLS = {2: 5, 3: 7, 4: 9}   # στήλη θερμοκρασίας (0-based)· η ημέρα είναι στην επόμενη


def norm(s):
    s = unicodedata.normalize('NFD', str(s or '').lower())
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    return re.sub(r'[^a-z0-9α-ω]', '', s)


def load_stations(path='data/noa_stations.csv'):
    st = list(csv.DictReader(open(path, encoding='utf-8')))
    by_en, by_gr = {}, {}
    for r in st:
        by_en[norm(r['name_en'])] = r
        by_gr.setdefault(norm(r['name_gr']), []).append(r)
    return st, by_en, by_gr


def match(row, year, by_en, by_gr):
    r = by_en.get(norm(row[2]))
    if r:
        return r
    cands = by_gr.get(norm(row[1]), [])
    if len(cands) == 1:
        return cands[0]
    ok = [c for c in cands if int(c['start_year']) <= year]
    ground = [c for c in ok if c['ground'] == '1']
    return (ground or ok or cands or [None])[0]


def parse(xlsx='raw/RECORDS_STATIONS_METEO_2006-2025.xlsx'):
    _, by_en, by_gr = load_stations()
    wb = load_workbook(xlsx, read_only=True, data_only=True)
    out = []
    for name in wb.sheetnames:
        year = int(name)
        rows = [list(r) for r in wb[name].iter_rows(values_only=True)]
        start = next(i for i, r in enumerate(rows)
                     if any(isinstance(x, str) and 'MIN TEMP' in x.upper() for x in r))
        for r in rows[start + 3:]:
            if any(isinstance(x, str) and 'MAX TEMP' in x.upper() for x in r):
                break
            if len(r) < 11 or not r[1]:
                continue
            st = match(r, year, by_en, by_gr)
            if st is None:
                continue
            for m, c in MONTH_COLS.items():
                t, d = r[c], r[c + 1]
                if isinstance(t, (int, float)) and isinstance(d, (int, float)):
                    out.append((int(st['station_id']), year, m, float(t), int(d)))
    return out


if __name__ == '__main__':
    obs = parse()
    with open('data/noa_monthly_min.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['station_id', 'year', 'month', 'tmin_c', 'day'])
        w.writerows(obs)
    print(len(obs), 'μηνιαία ελάχιστα')
