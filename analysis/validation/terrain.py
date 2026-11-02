"""Χαρακτηριστικά ανάγλυφου από το Copernicus DEM GLO-90 (πλακίδια 1°×1°, ~90 μ.).

Κατέβασμα πλακιδίων (δημόσια, χωρίς λογαριασμό):
  https://copernicus-dem-90m.s3.amazonaws.com/Copernicus_DSM_COG_30_N40_00_E022_00_DEM/Copernicus_DSM_COG_30_N40_00_E022_00_DEM.tif
για N39–N41 και E021–E024, στο raw/dem/.

Για κάθε σημείο δίνει:
  elev_dem      υψόμετρο (μ.)
  tpi_1k/3k/10k πόσο ψηλότερα ή χαμηλότερα είναι το σημείο από τον μέσο όρο γύρω του (μ.)
  slope_pct     κλίση (%), εξομαλυμένη σε ~300 μ.
  rise_med      διάμεση άνοδος του εδάφους σε 16 κατευθύνσεις, έως 15 χλμ. (μ.)
  rise_frac150  ποσοστό κατευθύνσεων όπου το έδαφος ανεβαίνει πάνω από 150 μ.
  dist_sea_km   απόσταση από τη θάλασσα (χλμ.)
"""
import glob
import os

import numpy as np
import tifffile
from scipy import ndimage

LAT0, LAT1, LON0, LON1 = 39, 42, 21, 25
N = 1200                       # pixels ανά μοίρα (3")
CACHE = 'raw/dem_mosaic.npy'


def mosaic():
    if os.path.exists(CACHE):
        return np.load(CACHE)
    a = np.zeros(((LAT1 - LAT0) * N, (LON1 - LON0) * N), np.float32)
    for f in glob.glob('raw/dem/*.tif'):
        name = os.path.basename(f)
        la = int(name.split('_N')[1][:2]); lo = int(name.split('_E')[1][:3])
        r0 = (LAT1 - 1 - la) * N; c0 = (lo - LON0) * N
        a[r0:r0 + N, c0:c0 + N] = tifffile.imread(f)
    np.save(CACHE, a)
    return a


def rc(lat, lon):
    return (LAT1 - np.asarray(lat)) * N - 0.5, (np.asarray(lon) - LON0) * N - 0.5


class Terrain:
    def __init__(self):
        self.z = mosaic()
        lat_mid = (LAT0 + LAT1) / 2
        self.dy = 111320 / N                                  # μ. ανά pixel (βορράς-νότος)
        self.dx = 111320 * np.cos(np.radians(lat_mid)) / N    # μ. ανά pixel (ανατολή-δύση)
        self._maps = {}

    def _mean(self, radius_m):
        key = ('mean', radius_m)
        if key not in self._maps:
            sy = max(1, int(round(2 * radius_m / self.dy)) | 1)
            sx = max(1, int(round(2 * radius_m / self.dx)) | 1)
            self._maps[key] = ndimage.uniform_filter(self.z, size=(sy, sx), mode='nearest')
        return self._maps[key]

    def _slope(self):
        if 'slope' not in self._maps:
            zs = ndimage.uniform_filter(self.z, size=3, mode='nearest')
            gy, gx = np.gradient(zs, self.dy, self.dx)
            self._maps['slope'] = 100 * np.hypot(gx, gy)
        return self._maps['slope']

    def _sea(self):
        if 'sea' not in self._maps:
            water = self.z <= 0.0
            lab, n = ndimage.label(water)
            sizes = ndimage.sum(water, lab, range(1, n + 1))
            big = np.isin(lab, 1 + np.flatnonzero(sizes > 20000))   # μόνο θάλασσα, όχι μικρές λίμνες
            d = ndimage.distance_transform_edt(~big, sampling=(self.dy, self.dx))
            self._maps['sea'] = d / 1000
        return self._maps['sea']

    def sample(self, m, lat, lon):
        r, c = rc(lat, lon)
        return ndimage.map_coordinates(m, [np.atleast_1d(r), np.atleast_1d(c)], order=1, mode='nearest')

    def rises(self, lat, lon, max_km=15, ndir=16, step_m=180):
        lat = np.atleast_1d(lat); lon = np.atleast_1d(lon)
        z0 = self.sample(self.z, lat, lon)
        ang = np.arange(ndir) * 2 * np.pi / ndir
        d = np.arange(step_m, max_km * 1000 + 1, step_m)
        out = np.zeros((len(lat), ndir))
        for k, a in enumerate(ang):
            dlat = np.cos(a) * d[None, :] / 111320
            dlon = np.sin(a) * d[None, :] / (111320 * np.cos(np.radians(lat[:, None])))
            zz = self.sample(self.z, (lat[:, None] + dlat).ravel(), (lon[:, None] + dlon).ravel())
            out[:, k] = zz.reshape(len(lat), -1).max(axis=1) - z0
        return out

    def features(self, lat, lon):
        lat = np.atleast_1d(np.asarray(lat, float)); lon = np.atleast_1d(np.asarray(lon, float))
        z = self.sample(self.z, lat, lon)
        f = {'elev_dem': z}
        for r, name in [(1000, 'tpi_1k'), (3000, 'tpi_3k'), (10000, 'tpi_10k')]:
            f[name] = z - self.sample(self._mean(r), lat, lon)
        f['slope_pct'] = self.sample(self._slope(), lat, lon)
        rs = self.rises(lat, lon)
        f['rise_med'] = np.median(rs, axis=1)
        f['rise_frac150'] = (rs > 150).mean(axis=1)
        f['dist_sea_km'] = self.sample(self._sea(), lat, lon)
        return f


if __name__ == '__main__':
    import csv
    st = list(csv.DictReader(open('data/noa_stations.csv', encoding='utf-8')))
    T = Terrain()
    f = T.features([float(r['lat']) for r in st], [float(r['lon']) for r in st])
    keys = list(f)
    with open('data/station_terrain.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['station_id'] + keys)
        for i, r in enumerate(st):
            w.writerow([r['station_id']] + [round(float(f[k][i]), 3) for k in keys])
    print('γράφτηκε data/station_terrain.csv για', len(st), 'σταθμούς')
