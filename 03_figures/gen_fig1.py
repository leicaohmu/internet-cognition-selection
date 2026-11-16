"""Builds project/Main.dc.html (Fig. 1 study overview) for the Design canvas.
Python only computes coordinates (map projection, chart scales); the artwork is SVG/HTML."""

import json
import numpy as np
import pandas as pd

W, H = 1360, 1700
INK, INK2, MUTED, GRID, LAND = "#1a1a1a", "#4d4d4a", "#8a8a85", "#e6e5e0", "#e9e7e1"
EFF, SELC = "#5b2a86", "#9b1b30"
EVT, EVT2 = "#2b2b2b", "#8a8781"  # neutral: going online (A), comparison goes online later (B)
COL = {
    "HRS": "#2a78d6",
    "ELSA": "#eb6834",
    "SHARE": "#1baf7a",
    "CHARLS": "#eda100",
    "LASI": "#e87ba4",
}
FONT = "Inter, 'Helvetica Neue', Arial, sans-serif"

# ---------------------------------------------------------------- icons (Lucide, ISC licence)
ICON = {
    "laptop": '<path d="M20 16V7a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v9m16 0H4m16 0 1.28 2.55a1 1 0 0 1-.9 1.45H3.62a1 1 0 0 1-.9-1.45L4 16"></path>',
    "brain": '<path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"></path><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"></path><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"></path><path d="M17.599 6.5a3 3 0 0 0 .399-1.375"></path><path d="M6.003 5.125A3 3 0 0 0 6.401 6.5"></path><path d="M3.477 10.896a4 4 0 0 1 .585-.396"></path><path d="M19.938 10.5a4 4 0 0 1 .585.396"></path><path d="M6 18a4 4 0 0 1-1.967-.516"></path><path d="M19.967 17.484A4 4 0 0 1 18 18"></path>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M22 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path>',
    "calendar": '<rect width="18" height="18" x="3" y="4" rx="2"></rect><path d="M16 2v4"></path><path d="M3 10h18"></path><path d="M8 2v4"></path><path d="M17 14h-6"></path><path d="M13 18H7"></path><path d="M7 14h.01"></path><path d="M17 18h.01"></path>',
    "clipboard": '<rect width="8" height="4" x="8" y="2" rx="1" ry="1"></rect><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><path d="m9 14 2 2 4-4"></path>',
    "globe": '<circle cx="12" cy="12" r="10"></circle><path d="M12 2a14.5 14.5 0 0 0 0 20 14.5 14.5 0 0 0 0-20"></path><path d="M2 12h20"></path>',
    "hourglass": '<path d="M5 22h14"></path><path d="M5 2h14"></path><path d="M17 22v-4.172a2 2 0 0 0-.586-1.414L12 12l-4.414 4.414A2 2 0 0 0 7 17.828V22"></path><path d="M7 2v4.172a2 2 0 0 0 .586 1.414L12 12l4.414-4.414A2 2 0 0 0 17 6.172V2"></path>',
    "house": '<path d="M15 21v-8a1 1 0 0 0-1-1h-4a1 1 0 0 0-1 1v8"></path><path d="M3 10a2 2 0 0 1 .709-1.528l7-5.999a2 2 0 0 1 2.582 0l7 5.999A2 2 0 0 1 21 10v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path>',
    "heart": '<path d="M19 14c1.49-1.46 3-3.21 3-5.5A5.5 5.5 0 0 0 16.5 3c-1.76 0-3 .5-4.5 2-1.5-1.5-2.74-2-4.5-2A5.5 5.5 0 0 0 2 8.5c0 2.3 1.5 4.05 3 5.5l7 7Z"></path><path d="M3.22 12H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27"></path>',
    "cloud": '<path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242"></path><path d="M16 14v6"></path><path d="M8 14v6"></path><path d="M12 16v6"></path>',
    "calc": '<rect width="16" height="20" x="4" y="2" rx="2"></rect><line x1="8" x2="16" y1="6" y2="6"></line><line x1="16" x2="16" y1="14" y2="18"></line><path d="M16 10h.01"></path><path d="M12 10h.01"></path><path d="M8 10h.01"></path><path d="M12 14h.01"></path><path d="M8 14h.01"></path><path d="M12 18h.01"></path><path d="M8 18h.01"></path>',
    "msg": '<path d="M7.9 20A9 9 0 1 0 4 16.1L2 22Z"></path>',
    "list": '<path d="M10 12h11"></path><path d="M10 18h11"></path><path d="M10 6h11"></path><path d="M4 10h2"></path><path d="M4 6h1v4"></path><path d="M6 18H4c0-1 2-2 2-3s-1-1.5-2-1"></path>',
    "ribbon": '<path d="M12 11.22C11 9.997 10 9 10 8a2 2 0 0 1 4 0c0 1-.998 2.002-2.01 3.22"></path><path d="m12 18 2.57-3.5"></path><path d="M6.243 9.016a7 7 0 0 1 11.507-.009"></path><path d="M9.35 14.53 12 11.22"></path><path d="M9.35 14.53C7.728 12.246 6 10.221 6 7a6 5 0 0 1 12 0c-.005 3.22-1.778 5.235-3.43 7.5l3.557 4.527a1 1 0 0 1-.203 1.43l-1.894 1.36a1 1 0 0 1-1.384-.215L12 18l-2.679 3.593a1 1 0 0 1-1.39.213l-1.865-1.353a1 1 0 0 1-.203-1.422z"></path>',
    "braincirc": '<path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"></path><path d="M9 13a4.5 4.5 0 0 0 3-4"></path><path d="M6.003 5.125A3 3 0 0 0 6.401 6.5"></path><path d="M3.477 10.896a4 4 0 0 1 .585-.396"></path><path d="M6 18a4 4 0 0 1-1.967-.516"></path><path d="M12 13h4"></path><path d="M12 18h6a2 2 0 0 1 2 2v1"></path><path d="M12 8h8"></path><path d="M16 8V5a2 2 0 0 1 2-2"></path><circle cx="16" cy="13" r=".5"></circle><circle cx="18" cy="3" r=".5"></circle><circle cx="20" cy="21" r=".5"></circle><circle cx="20" cy="8" r=".5"></circle>',
    "bike": '<circle cx="18.5" cy="17.5" r="3.5"></circle><circle cx="5.5" cy="17.5" r="3.5"></circle><circle cx="15" cy="5" r="1"></circle><path d="M12 17.5V14l-3-3 4-3 2 3h2"></path>',
    "layers": '<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"></path><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"></path><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"></path>',
    "sigma": '<path d="M18 7V5a1 1 0 0 0-1-1H6.5a.5.5 0 0 0-.4.8l4.5 6a2 2 0 0 1 0 2.4l-4.5 6a.5.5 0 0 0 .4.8H17a1 1 0 0 0 1-1v-2"></path>',
    "chart": '<path d="M3 3v16a2 2 0 0 0 2 2h16"></path><path d="m19 9-5 5-4-4-3 3"></path>',
    "scale": '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"></path><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"></path><path d="M7 21h10"></path><path d="M12 3v18"></path><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"></path>',
    "compare": '<circle cx="5" cy="6" r="3"></circle><path d="M12 6h5a2 2 0 0 1 2 2v7"></path><path d="m15 9-3-3 3-3"></path><circle cx="19" cy="18" r="3"></circle><path d="M12 18H7a2 2 0 0 1-2-2V9"></path><path d="m9 15 3 3-3 3"></path>',
}


def icon(name, size, color, sw=2, extra=""):
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{sw}" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" style="flex: none; {extra}">{ICON[name]}</svg>'
    )


def icon_g(name, x, y, size, color, sw=2):
    s = size / 24
    return f'<g transform="translate({x:.1f},{y:.1f}) scale({s:.4f})" fill="none" stroke="{color}" stroke-width="{sw / s * s:.2f}" stroke-linecap="round" stroke-linejoin="round">{ICON[name]}</g>'


def pos(x, y, w=None, h=None):
    s = f"position: absolute; left: {x}px; top: {y}px;"
    if w is not None:
        s += f" width: {w}px;"
    if h is not None:
        s += f" height: {h}px;"
    return s


def panel_head(x, y, letter, title, w=700):
    return (
        f'<div style="{pos(x, y, w)} display: flex; align-items: baseline; gap: 10px; color: {INK}">'
        f'<span style="font-size: 30px; font-weight: 800; line-height: 32px">{letter}</span>'
        f'<span style="font-size: 21px; font-weight: 650; line-height: 26px">{title}</span></div>'
    )


out = []
# ================================================================ a  map
AX, AY, AW, AH = 40, 82, 880, 455


def ee(lon, lat):
    A1, A2, A3, A4 = 1.340264, -0.081106, 0.000893, 0.003796
    M = np.sqrt(3) / 2
    lam, phi = np.radians(lon), np.radians(lat)
    th = np.arcsin(M * np.sin(phi))
    x = (
        2
        * np.sqrt(3)
        * lam
        * np.cos(th)
        / (3 * (9 * A4 * th**8 + 7 * A3 * th**6 + 3 * A2 * th**2 + A1))
    )
    y = A4 * th**9 + A3 * th**7 + A2 * th**3 + A1 * th
    return x, y


LON0, LON1, LAT0, LAT1 = -128, 150, 4, 75
x0, _ = ee(np.array(LON0), np.array(0))
x1, _ = ee(np.array(LON1), np.array(0))
_, y0 = ee(0, LAT0)
_, y1 = ee(0, LAT1)
sc = AW / (x1 - x0)
MH = (y1 - y0) * sc
SHARE_C = [
    "AUT",
    "DEU",
    "SWE",
    "ESP",
    "ITA",
    "FRA",
    "DNK",
    "GRC",
    "CHE",
    "BEL",
    "ISR",
    "CZE",
    "POL",
    "LUX",
    "SVN",
    "EST",
    "HRV",
]
FILL = {
    "USA": COL["HRS"],
    "GBR": COL["ELSA"],
    "CHN": COL["CHARLS"],
    "TWN": COL["CHARLS"],
    "IND": COL["LASI"],
    **{c: COL["SHARE"] for c in SHARE_C},
}
G = json.load(open("/path/to/project/an/world110.geojson"))
paths_bg, paths_fg = [], []
for f in G["features"]:
    code = f["properties"]["ADM0_A3"]
    if code == "ATA":
        continue
    geom = f["geometry"]
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    d = []
    for poly in polys:
        ring = np.array(poly[0])
        if code == "FRA" and ring[:, 0].max() < -30:
            continue
        if (
            ring[:, 0].max() < LON0 - 30
            or ring[:, 0].min() > LON1 + 30
            or ring[:, 1].max() < LAT0 - 10
        ):
            continue
        px, py = ee(ring[:, 0], ring[:, 1])
        px = (px - x0) * sc
        py = (y1 - py) * sc
        d.append("M" + " L".join(f"{a:.1f} {b:.1f}" for a, b in zip(px, py)) + "Z")
    if not d:
        continue
    fc = FILL.get(code)
    p = f'<path d="{"".join(d)}" fill="{fc or LAND}" stroke="#ffffff" stroke-width="{0.9 if fc else 0.6}" stroke-linejoin="round"></path>'
    (paths_fg if fc else paths_bg).append(p)
P = lambda lon, lat: (lambda q: ((q[0] - x0) * sc, (y1 - q[1]) * sc))(
    ee(np.array(lon), np.array(lat))
)
out.append(panel_head(AX, 40, "a", "Five harmonized ageing cohorts, 21 countries"))
# direct labels on the map (white on filled countries, coloured beside small ones)
labs = [
    ("HRS", -99, 40, "middle", "#ffffff"),
    ("CHARLS", 103, 34, "middle", "#ffffff"),
    ("LASI", 67, 14, "end", COL["LASI"]),
    ("ELSA", -12, 58.5, "end", COL["ELSA"]),
    ("SHARE", -12, 43, "end", COL["SHARE"]),
]
lab_svg = ""
for t, lo, la, anc, col in labs:
    px, py = P(lo, la)
    halo = (
        ""
        if col == "#ffffff"
        else ' paint-order="stroke" stroke="#ffffff" stroke-width="4" stroke-linejoin="round"'
    )
    lab_svg += f'<text x="{px:.1f}" y="{py + 5:.1f}" text-anchor="{anc}" font-size="16" font-weight="700" fill="{col}"{halo}>{t}</text>'
out.append(
    f'<svg width="{AW}" height="{MH:.0f}" viewBox="0 0 {AW} {MH:.0f}" role="img" aria-label="World map of the five cohorts" style="{pos(AX, AY)} overflow: hidden; font-family: {FONT}">'
    + "".join(paths_bg)
    + "".join(paths_fg)
    + lab_svg
    + "</svg>"
)
MAPH = MH
# compact table under the map
TY = AY + MAPH + 14
cols = [
    ("Cohort", 0, "left"),
    ("Country", 120, "left"),
    ("Survey years", 430, "left"),
    ("Waves", 580, "right"),
    ("Person-trials", 880, "right"),
]
rows = [
    ("HRS", "United States", "2002–2020", "10", "65,350"),
    ("ELSA", "England", "2004–2018", "8", "15,163"),
    ("SHARE", "16 European countries and Israel", "2013–2022", "5", "37,525"),
    ("CHARLS", "China", "2011–2021", "6", "47,930"),
    ("LASI", "India", "2017–2019", "1", "cross-sectional, n = 51,758"),
]
RHt = 25
tbl = f'<line x1="0" x2="{AW}" y1="0.5" y2="0.5" stroke="{INK}" stroke-width="1.2"></line><line x1="0" x2="{AW}" y1="{RHt + 0.5}" y2="{RHt + 0.5}" stroke="{GRID}"></line>'
tbl += f'<line x1="0" x2="{AW}" y1="{RHt * 6 + 4.5}" y2="{RHt * 6 + 4.5}" stroke="{GRID}"></line><line x1="0" x2="{AW}" y1="{RHt * 7 + 6.5}" y2="{RHt * 7 + 6.5}" stroke="{INK}" stroke-width="1.2"></line>'
out.append(
    f'<svg width="{AW}" height="{RHt * 7 + 8}" viewBox="0 0 {AW} {RHt * 7 + 8}" aria-hidden="true" style="{pos(AX, TY)}">{tbl}</svg>'
)


def cell(x, y, w, txt, al, bold=False, col=INK):
    left = AX + x if al == "left" else AX + x - w
    return f'<div style="{pos(left, y, w)} text-align: {al}; font-size: 14px; line-height: {RHt}px; color: {col}; font-weight: {700 if bold else 400}">{txt}</div>'


for nm, x, al in cols:
    out.append(cell(x if al == "left" else x, TY + 1, 160, nm, al, True, INK2))
for k, r in enumerate(rows):
    y = TY + RHt * (k + 1) + 2
    out.append(
        f'<div style="{pos(AX, y + 7, 11, 11)} border-radius: 2px; background: {COL[r[0]]}"></div>'
    )
    out.append(cell(18, y, 100, r[0], "left", True))
    out.append(cell(120, y, 300, r[1], "left"))
    out.append(cell(430, y, 140, r[2], "left"))
    out.append(cell(580, y, 60, r[3], "right"))
    out.append(cell(880, y, 260, r[4], "right"))
y = TY + RHt * 6 + 5
out.append(cell(18, y, 100, "Total", "left", True))
out.append(cell(120, y, 300, "70,170 people offline at baseline", "left"))
out.append(cell(880, y, 420, "<b>165,968</b> person-trials · <b>19,220</b> went online", "right"))
KY = TY + RHt * 7 + 8 - 40  # keeps later layout keys valid


def linechart(x, y, w, h, series, xr, yr, xt, yt, xlab, ylab, labels):
    L, R, T, B = 50, w - labels, 10, h - 44
    fx = lambda v: L + (v - xr[0]) / (xr[1] - xr[0]) * (R - L)
    fy = lambda v: T + (1 - (v - yr[0]) / (yr[1] - yr[0])) * (B - T)
    s = ""
    for v in yt:
        s += f'<line x1="{L}" x2="{R}" y1="{fy(v):.1f}" y2="{fy(v):.1f}" stroke="{GRID}"></line><text x="{L - 9}" y="{fy(v) + 5:.1f}" text-anchor="end" font-size="15" fill="{INK2}">{v}</text>'
    for v in xt:
        s += f'<line x1="{fx(v):.1f}" x2="{fx(v):.1f}" y1="{B}" y2="{B + 5}" stroke="{MUTED}"></line><text x="{fx(v):.1f}" y="{B + 21}" text-anchor="middle" font-size="15" fill="{INK2}">{v}</text>'
    s += f'<line x1="{L}" x2="{R}" y1="{B}" y2="{B}" stroke="{MUTED}"></line>'
    for c, xs, ys, lab, dy in series:
        X = [fx(v) for v in xs]
        Y = [fy(v) for v in ys]
        if len(X) > 1:
            s += f'<polyline points="{" ".join(f"{p:.1f},{q:.1f}" for p, q in zip(X, Y))}" fill="none" stroke="{COL[c]}" stroke-width="2.6" stroke-linejoin="round"></polyline>'
        for p, q in zip(X, Y):
            s += f'<circle cx="{p:.1f}" cy="{q:.1f}" r="4.4" fill="{COL[c]}" stroke="#ffffff" stroke-width="1.5"></circle>'
        dx, dy = dy if isinstance(dy, tuple) else (0, dy)
        s += f'<text x="{X[-1] + 9 + dx:.1f}" y="{Y[-1] + 5 + dy:.1f}" font-size="15" font-weight="700" fill="{COL[c]}">{lab}</text>'
    s += f'<text x="13" y="{(T + B) / 2:.0f}" transform="rotate(-90 13 {(T + B) / 2:.0f})" text-anchor="middle" font-size="15" fill="{INK2}">{ylab}</text>'
    s += f'<text x="{(L + R) / 2:.0f}" y="{h - 2}" text-anchor="middle" font-size="15" fill="{INK2}">{xlab}</text>'
    return f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{ylab}" style="{pos(x, y)} overflow: visible; font-family: {FONT}">{s}</svg>'


# ================================================================ b, c  stacked charts on the right
BX = 960
pr = pd.read_csv("/path/to/project/an2/prevalence.csv")
pr = pr[~((pr.cohort == "SHARE") & (pr.wave == 7))]
out.append(panel_head(BX - 20, 40, "b", "Older adults online, by survey wave", 400))
ser = []
for c, dy in [("ELSA", 0), ("HRS", -20), ("SHARE", 4), ("CHARLS", 30), ("LASI", -1)]:
    d = pr[pr.cohort == c].sort_values("year")
    ser.append((c, list(d.year), list(100 * d.prev), c, dy))
out.append(
    linechart(
        BX,
        82,
        380,
        212,
        ser,
        (2001, 2023),
        (0, 90),
        [2002, 2008, 2014, 2020],
        [0, 30, 60, 90],
        "Survey year (each point is one wave)",
        "Online (%)",
        60,
    )
)
ini = pd.read_csv("/path/to/project/an2/initiation.csv")
out.append(panel_head(BX - 20, 316, "c", "Non-users online by the next wave", 400))
NT = {"HRS": "65,350", "ELSA": "15,163", "SHARE": "37,525", "CHARLS": "47,930"}
ser = []
for c, dy in [("SHARE", -10), ("ELSA", (-130, -34)), ("HRS", 12), ("CHARLS", 0)]:
    d = ini[ini.cohort == c].sort_values("year")
    ser.append((c, list(d.year), list(100 * d.rate), c, dy))
out.append(
    linechart(
        BX,
        358,
        380,
        196,
        ser,
        (2001, 2019),
        (0, 30),
        [2002, 2006, 2010, 2014, 2018],
        [0, 10, 20, 30],
        "Baseline year (t0)",
        "Started (%)",
        74,
    )
)

# ================================================================ d  concept: two explanations
CY0 = 568
out.append(
    panel_head(40, CY0, "d", "Two explanations for why internet users have better memory", 780)
)


def concept(x, y, col, title, sub, effect):
    w, h = 360, 290
    L, R, T, B = 40, w - 18, 14, h - 40
    tt = [0, 1, 2, 3]
    fxx = lambda t: L + t / 3 * (R - L)
    non = [0.55, 0.47, 0.39, 0.31]
    ini_ = [0.55, 0.47, 0.39, 0.45] if effect else [0.80, 0.72, 0.64, 0.56]
    fyy = lambda v: T + (1 - v) * (B - T)
    s = f'<rect x="{fxx(2):.1f}" y="{T}" width="{fxx(3) - fxx(2):.1f}" height="{B - T}" fill="{col}" fill-opacity="0.07"></rect>'
    s += f'<line x1="{fxx(2):.1f}" x2="{fxx(2):.1f}" y1="{T}" y2="{B}" stroke="{col}" stroke-width="1.6" stroke-dasharray="5 4"></line>'
    s += icon_g("laptop", fxx(2) - 32, B - 32, 24, col, 2)
    s += f'<text x="{fxx(2) - 38:.1f}" y="{B - 15}" text-anchor="end" font-size="14" fill="{col}" font-weight="600">first use reported</text>'
    s += f'<line x1="{L}" x2="{R}" y1="{B}" y2="{B}" stroke="{MUTED}"></line><line x1="{L}" x2="{L}" y1="{T}" y2="{B}" stroke="{MUTED}"></line>'
    for i, lab in enumerate(["t−1", "t0", "t1", "t2"]):
        s += f'<text x="{fxx(i):.1f}" y="{B + 20}" text-anchor="middle" font-size="15" fill="{INK2}">{lab}</text>'
    s += f'<polyline points="{" ".join(f"{fxx(t):.1f},{fyy(v):.1f}" for t, v in zip(tt, non))}" fill="none" stroke="{MUTED}" stroke-width="3" stroke-linecap="round"></polyline>'
    s += f'<polyline points="{" ".join(f"{fxx(t):.1f},{fyy(v):.1f}" for t, v in zip(tt, ini_))}" fill="none" stroke="{col}" stroke-width="3.4" stroke-linecap="round"></polyline>'
    for t, v in zip(tt, non):
        s += f'<circle cx="{fxx(t):.1f}" cy="{fyy(v):.1f}" r="5" fill="{MUTED}" stroke="#fff" stroke-width="1.5"></circle>'
    for t, v in zip(tt, ini_):
        s += f'<circle cx="{fxx(t):.1f}" cy="{fyy(v):.1f}" r="5.5" fill="{col}" stroke="#fff" stroke-width="1.5"></circle>'
    for t in ([3] if effect else [0, 3]):
        a_, b_ = fyy(ini_[t]), fyy(non[t])
        xx = fxx(t) + (12 if t < 3 else -12)
        s += (
            f'<line x1="{xx:.1f}" x2="{xx:.1f}" y1="{a_ + 6:.1f}" y2="{b_ - 6:.1f}" stroke="{INK}" stroke-width="1.4"></line>'
            f'<line x1="{xx - 4:.1f}" x2="{xx + 4:.1f}" y1="{a_ + 6:.1f}" y2="{a_ + 6:.1f}" stroke="{INK}" stroke-width="1.4"></line>'
            f'<line x1="{xx - 4:.1f}" x2="{xx + 4:.1f}" y1="{b_ - 6:.1f}" y2="{b_ - 6:.1f}" stroke="{INK}" stroke-width="1.4"></line>'
        )
    s += f'<text x="14" y="{(T + B) / 2:.0f}" transform="rotate(-90 14 {(T + B) / 2:.0f})" text-anchor="middle" font-size="15" fill="{INK2}">Memory</text>'
    if effect:
        s += f'<text x="{fxx(3) - 6:.1f}" y="{fyy(ini_[3]) - 12:.1f}" text-anchor="end" font-size="14" font-weight="600" fill="{col}">initiators</text>'
    else:
        s += f'<text x="{fxx(0) + 22:.1f}" y="{fyy(ini_[0]) - 12:.1f}" text-anchor="start" font-size="14" font-weight="600" fill="{col}">initiators</text>'
    s += f'<text x="{fxx(3) - 6:.1f}" y="{fyy(non[3]) + 24:.1f}" text-anchor="end" font-size="14" font-weight="600" fill="{MUTED}">non-initiators</text>'
    return [
        f'<div style="{pos(x, y, w)} font-size: 17px; line-height: 22px; font-weight: 700; color: {col}">{title}</div>',
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{title}" style="{pos(x, y + 30)} overflow: visible; font-family: {FONT}">{s}</svg>',
        f'<div style="{pos(x + 40, y + 30 + h + 2, w - 40)} font-size: 14px; line-height: 18px; color: {INK2}">{sub}</div>',
    ]


out += concept(
    40,
    CY0 + 44,
    EFF,
    "If going online improves memory",
    "The gap opens only <b>after</b> starting; people who start later show no advantage yet.",
    True,
)
out += concept(
    430,
    CY0 + 44,
    SELC,
    "If adopters were already ahead",
    "The gap is present <b>before</b> starting and does not widen; later starters look the same.",
    False,
)

# ================================================================ e  measures x cohorts
EX = 860
out.append(panel_head(EX - 20, CY0, "e", "Outcomes and where they were measured", 480))
COLS = ["HRS", "ELSA", "SHARE", "CHARLS", "LASI"]
CX0, CSTEP = EX + 226, 57
ALL4 = ["HRS", "ELSA", "SHARE", "CHARLS"]
rowsE = [
    (
        "Cognition",
        [
            ("Word recall (primary)", ALL4, ["LASI"]),
            ("Serial 7s", ALL4, []),
            ("Verbal fluency", ["HRS", "ELSA", "SHARE"], []),
            ("Two-domain composite", ALL4, []),
            ("Impairment or dementia", ["HRS"], []),
        ],
    ),
    (
        "Function and health",
        [
            ("IADL difficulty", ALL4, ["LASI"]),
            ("Depressive symptoms", ALL4, ["LASI"]),
            ("Fair or poor self-rated health", ALL4, []),
        ],
    ),
    (
        "Checks on residual confounding",
        [
            ("Cancer (negative control)", ["HRS", "ELSA", "CHARLS"], []),
            ("Death (benchmark)", ["HRS", "ELSA", "CHARLS"], []),
        ],
    ),
    ("Second behaviour", [("Vigorous physical activity", ALL4, [])]),
]
yy = CY0 + 44
for k, c in enumerate(COLS):
    out.append(
        f'<div style="{pos(CX0 + k * CSTEP - 29, yy - 4, 58)} text-align: center; font-size: 14px; font-weight: 700; color: {INK}">{c}</div>'
    )
yy += 26
RW = 24
mat = ""
for gi, (gname, items) in enumerate(rowsE):
    mat += f'<line x1="0" x2="460" y1="{yy - CY0 + 0.5}" y2="{yy - CY0 + 0.5}" stroke="{INK if gi == 0 else GRID}" stroke-width="{1.2 if gi == 0 else 1}"></line>'
    out.append(
        f'<div style="{pos(EX, yy + 5, 300)} font-size: 14px; font-weight: 700; color: {INK}">{gname}</div>'
    )
    yy += 27
    for nm, full, xs in items:
        out.append(
            f'<div style="{pos(EX + 12, yy, 230)} font-size: 14px; line-height: {RW}px; color: {INK}">{nm}</div>'
        )
        for k, c in enumerate(COLS):
            cx = CX0 + k * CSTEP - EX
            cy = yy - CY0 + RW / 2
            if c in full:
                mat += f'<circle cx="{cx}" cy="{cy}" r="6.5" fill="{COL[c]}"></circle>'
            elif c in xs:
                mat += f'<circle cx="{cx}" cy="{cy}" r="5.5" fill="#ffffff" stroke="{COL[c]}" stroke-width="2"></circle>'
            else:
                mat += f'<circle cx="{cx}" cy="{cy}" r="2" fill="#cfcdc6"></circle>'
        yy += RW
    yy += 4
mat += f'<line x1="0" x2="460" y1="{yy - CY0 + 0.5}" y2="{yy - CY0 + 0.5}" stroke="{INK}" stroke-width="1.2"></line>'
out.append(
    f'<svg width="460" height="{yy - CY0 + 4}" viewBox="0 0 460 {yy - CY0 + 4}" role="img" aria-label="Outcomes by cohort" style="{pos(EX, CY0)} overflow: visible">{mat}</svg>'
)
dot = (
    lambda fill, stroke, r: f'<svg width="{2 * r + 4}" height="{2 * r + 4}" viewBox="0 0 {2 * r + 4} {2 * r + 4}" aria-hidden="true"><circle cx="{r + 2}" cy="{r + 2}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"></circle></svg>'
)
out.append(
    f'<div style="{pos(EX, yy + 10, 460)} display: flex; gap: 22px; align-items: center; font-size: 14px; color: {INK}">'
    f'<span style="display: flex; gap: 6px; align-items: center">{dot(INK2, INK2, 5.5)}longitudinal designs</span>'
    f'<span style="display: flex; gap: 6px; align-items: center">{dot("#ffffff", INK2, 5)}cross-sectional only</span></div>'
)
EEND = yy + 40

# ================================================================ f  four comparisons (dot timelines)
EY0 = EEND + 14
out.append(panel_head(40, EY0, "f", "Four comparisons that tell the two explanations apart", 900))
HY = EY0 + 44
ROW0 = HY + 40
RH = 74
DX0, DSTEP = 520, 84
waves = ["t−1", "t0", "t1", "t2", "t3"]
wx = {w: DX0 + i * DSTEP for i, w in enumerate(waves)}
for wv in waves:
    out.append(
        f'<div style="{pos(wx[wv] - 30, HY + 12, 60)} text-align: center; font-size: 16px; font-weight: 700; color: {INK}">{wv}</div>'
    )
PX = {"eff": 935, "sel": 1135}
out.append(
    f'<div style="{pos(PX["eff"], HY, 190)} font-size: 15px; line-height: 18px; font-weight: 700; color: {EFF}">If going online<br>improves memory</div>'
)
out.append(
    f'<div style="{pos(PX["sel"], HY, 190)} font-size: 15px; line-height: 18px; font-weight: 700; color: {SELC}">If adopters were<br>already ahead</div>'
)
designs = [
    ("1", "Target trial", "A starts at t1 · B stays offline", "t1", None, "t2", None, (1, 1)),
    (
        "2",
        "Placebo trial",
        "A starts at t3, after the test · B stays offline",
        "t3",
        None,
        "t2",
        None,
        (0, 1),
    ),
    (
        "3",
        "Before anyone starts",
        "A will start at t1 · B stays offline",
        "t1",
        None,
        "t0",
        "t−1",
        (0, 1),
    ),
    (
        "4",
        "Early vs late adopters",
        "A starts at t1 · B starts at t3",
        "t1",
        "t3",
        "t2",
        None,
        (1, 0),
    ),
]
for i in range(len(designs)):
    out.append(
        f'<div style="{pos(40, ROW0 + i * RH - 3, 1280, 1)} background: {INK if i == 0 else GRID}"></div>'
    )
OFF = "#b9b7b0"


def badge_brain(cx, cy, dashed=False):
    st = (
        f'stroke="{MUTED}" stroke-width="1.8" stroke-dasharray="3 3"'
        if dashed
        else f'stroke="{INK}" stroke-width="1.8"'
    )
    return f'<circle cx="{cx}" cy="{cy}" r="16" fill="#ffffff" {st}></circle>' + icon_g(
        "brain", cx - 11, cy - 11, 22, MUTED if dashed else INK, 1.9
    )


def badge_start(cx, cy, col):
    return f'<circle cx="{cx}" cy="{cy}" r="16" fill="{col}"></circle>' + icon_g(
        "laptop", cx - 10, cy - 10, 20, "#ffffff", 2.2
    )


for i, (num, nm, q, sa, sb, test, adj, pred) in enumerate(designs):
    y = ROW0 + i * RH
    cyr = (RH - 6) / 2
    out.append(
        f'<div style="{pos(56, y + cyr - 21, 400)} font-size: 16px; line-height: 21px; color: {INK}"><b>{nm}</b><br><span style="font-size: 14px; color: {INK2}">{q}</span></div>'
    )
    W_ = DSTEP * 4 + 60
    X = lambda w: wx[w] - DX0 + 30
    cy = 34
    s_ = f'<line x1="{X("t−1")}" x2="{X("t3")}" y1="{cy}" y2="{cy}" stroke="{OFF}" stroke-width="2"></line>'
    s_ += f'<line x1="{X(sa)}" x2="{X("t3") + 22}" y1="{cy}" y2="{cy}" stroke="{EVT}" stroke-opacity="0.18" stroke-width="10" stroke-linecap="round"></line>'
    for wv in waves:
        if wv in (sa, sb, test, adj):
            continue
        s_ += f'<circle cx="{X(wv)}" cy="{cy}" r="5" fill="#ffffff" stroke="{OFF}" stroke-width="2"></circle>'
    if adj:
        s_ += badge_brain(X(adj), cy, dashed=True)
    s_ += badge_brain(X(test), cy)
    s_ += badge_start(X(sa), cy, EVT)
    if sb:
        s_ += badge_start(X(sb), cy, EVT2)
    out.append(
        f'<svg width="{W_}" height="68" viewBox="0 0 {W_} 68" role="img" aria-label="{nm} timeline" style="{pos(DX0 - 30, y)} overflow: visible">{s_}</svg>'
    )
    for key, val in zip(["eff", "sel"], pred):
        col = EFF if key == "eff" else SELC
        ha, hb = (40, 20) if val else (28, 28)
        g = (
            f'<rect x="4" y="{46 - ha}" width="20" height="{ha}" rx="3" fill="{col}"></rect>'
            f'<rect x="28" y="{46 - hb}" width="20" height="{hb}" rx="3" fill="{OFF}"></rect>'
            f'<text x="14" y="61" text-anchor="middle" font-size="14" fill="{INK2}">A</text><text x="38" y="61" text-anchor="middle" font-size="14" fill="{INK2}">B</text>'
        )
        out.append(
            f'<svg width="52" height="64" viewBox="0 0 52 64" role="img" aria-label="{"A higher than B" if val else "no gap between A and B"}" style="{pos(PX[key], y + 2)} overflow: visible; font-family: {FONT}">{g}</svg>'
        )
        out.append(
            f'<div style="{pos(PX[key] + 62, y + cyr - 11, 120)} font-size: 16px; font-weight: {700 if val else 400}; color: {col}">{"A higher" if val else "no gap"}</div>'
        )
out.append(f'<div style="{pos(40, ROW0 + 4 * RH - 3, 1280, 1)} background: {INK}"></div>')
LY = ROW0 + 4 * RH + 10


def chip(inner, txt):
    return f'<span style="display: flex; gap: 8px; align-items: center; white-space: nowrap">{inner}{txt}</span>'


def mini(svg_inner, w=36):
    return f'<svg width="{w}" height="36" viewBox="0 0 {w} 36" aria-hidden="true">{svg_inner}</svg>'


leg = (
    f'<div style="{pos(56, LY, 1264)} display: flex; gap: 26px; align-items: center; font-size: 15px; color: {INK}">'
    + chip(mini(badge_start(18, 18, EVT)), "group A goes online")
    + chip(
        mini(
            f'<line x1="6" x2="40" y1="18" y2="18" stroke="{EVT}" stroke-opacity="0.18" stroke-width="10" stroke-linecap="round"></line>',
            46,
        ),
        "A online",
    )
    + chip(mini(badge_start(18, 18, EVT2)), "group B goes online")
    + chip(mini(badge_brain(18, 18)), "memory test, A vs B")
    + chip(mini(badge_brain(18, 18, dashed=True)), "earlier memory adjusted for")
    + chip(
        mini(
            f'<circle cx="18" cy="18" r="5" fill="#ffffff" stroke="{OFF}" stroke-width="2"></circle>'
        ),
        "survey wave",
    )
    + "</div>"
)
out.append(leg)
H = int(LY + 50)

body = "\n".join(out)
html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Fig. 1 – Study overview (NHB)</title>
<script src="./support.js"></script>
</head>
<body>
<x-dc>
<helmet>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&amp;display=swap" rel="stylesheet">
<style>
body{{margin:0}}
a{{color:#4d4d4a}}a:hover{{color:#1a1a1a}}
</style>
</helmet>
<div style="position: relative; width: {W}px; height: {H}px; background: #ffffff; font-family: {FONT}; color: {INK}; overflow: hidden">
{body}
</div>
</x-dc>
<script type="text/x-dc" data-dc-script data-props='{{"$preview":{{"width":{W},"height":{H}}}}}'>
class Component extends DCLogic {{
renderVals() {{
return {{}};
}}
}}
</script>
</body>
</html>
"""
open("/path/to/project/an2/fig1design/project/Main.dc.html", "w").write(html)
canvas = {
    "v": 3,
    "createdOnFiles": {"v": 1, "at": "2026-11-08T11:20:00Z"},
    "title": "Fig. 1 – Study overview (NHB)",
    "launch": {"view": "canvas"},
    "pages": [],
    "boards": {
        "Main.dc.html": {"x": 0, "y": 0, "w": W, "h": H, "title": "Fig. 1 (180 mm wide at 2×)"}
    },
    "order": ["Main.dc.html"],
    "notes": {},
    "designSystems": [],
}
json.dump(
    canvas,
    open("/path/to/project/an2/fig1design/project/canvas.json", "w"),
    ensure_ascii=False,
    indent=1,
)
print("ok", len(html) // 1024, "KB; map height", round(MAPH))
