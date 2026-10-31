# Retest 19-20/3/2025: hourly ERA5-Land (Open-Meteo archive), event window 18/3 18:00 - 21/3 10:00 local
rows = [  # station, type, obs Tmin, station elev, cell elev, raw cell min, min at station elevation
 ("Aridaia",     "basin", -5.1, 120, 427, -2.8, -0.8),
 ("Platykampos", "plain", -2.3,  66, -13,  2.6,  2.1),
 ("Agia",        "plain", -1.4, 167, 412,  1.6,  3.2),
 ("Giannitsa",   "plain", -1.1,  35, 159, -1.7, -0.8),
 ("Naousa",      "town",   1.2, 365, 661, -2.9, -0.9),
]
print(f"{'station':12} {'dElev':>6} {'raw-obs':>8} {'corr-obs':>9} {'implied lapse':>14}")
for s,t,o,se,ce,raw,cor in rows:
    print(f"{s:12} {ce-se:6d} {raw-o:8.1f} {cor-o:9.1f} {(cor-raw)/((ce-se)/1000) if ce!=se else float('nan'):10.1f} K/km")
for label, idx in (("raw", 5), ("at station elev", 6)):
    d = [r[idx]-r[2] for r in rows]
    bp = [r[idx]-r[2] for r in rows if r[1] in ("basin","plain") and r[0]!="Giannitsa"]
    print(label, "mean", round(sum(d)/5,2), "MAE", round(sum(abs(x) for x in d)/5,2), "basin/plain 3:", [round(x,1) for x in bp], "min", round(min(bp),1))
