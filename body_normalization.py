"""
Normalizacija keypointa relativno na telo (poziciju i velicinu ramena).

PROBLEM koji ovo resava:
MediaPipe koordinate su normalizovane na CEO KADAR kamere (0 do 1 preko cele
slike). To znaci da ista fizicka gestikulacija izgleda DRUGACIJE modelu
zavisno od toga koliko si blizu/daleko od kamere - kada si dalje, ruka
zauzima manji opseg tih koordinata, sto je za model "druga gesta".

RESENJE:
Normalizuj sve koordinate relativno na poziciju i velicinu RAMENA:
  1. Centar = sredina izmedju levog i desnog ramena (referentna tacka tela)
  2. Skala  = rastojanje izmedju ramena (referentna jedinica velicine)
  3. Svaka tacka (pose, leva ruka, desna ruka) se transformise:
       nova_tacka = (tacka - centar) / skala

Posle ovoga, isti fizicki pokret izgleda priblizno isto bez obzira na
udaljenost od kamere ili poziciju u kadru, jer su i ruka i ramena
skalirani/pomereni zajedno.

VAZNO: normalizuju se samo X i Y koordinate. Z (dubina) i visibility/
ostale vrednosti se NE skaliraju na isti nacin (Z je vec relativna mera
u MediaPipe-u, ne treba dvostruko skaliranje).
"""
import numpy as np

# MediaPipe POSE landmark indeksi
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

POSE_LEN = 33 * 4   # x,y,z,visibility po tacki
HAND_LEN = 21 * 3   # x,y,z po tacki
LEFT_HAND_START = POSE_LEN
RIGHT_HAND_START = POSE_LEN + HAND_LEN

MIN_SCALE = 1e-4  # zastita od deljenja sa ~0 ako ramena nisu detektovana


def _get_pose_xy(frame, landmark_idx):
    """Izvuci (x, y) za dati pose landmark indeks iz 258-vrednosnog frejma."""
    base = landmark_idx * 4
    return frame[base], frame[base + 1]


def _compute_center_and_scale(frame):
    """
    Vraca (center_x, center_y, scale) na osnovu polozaja ramena.
    Ako ramena nisu detektovana (sve nule), vraca None - frejm se preskace
    (ne normalizuje), sto se kasnije rukuje van ove funkcije.
    """
    lx, ly = _get_pose_xy(frame, LEFT_SHOULDER)
    rx, ry = _get_pose_xy(frame, RIGHT_SHOULDER)

    if (lx == 0 and ly == 0) or (rx == 0 and ry == 0):
        return None

    center_x = (lx + rx) / 2.0
    center_y = (ly + ry) / 2.0
    scale = np.sqrt((rx - lx) ** 2 + (ry - ly) ** 2)

    if scale < MIN_SCALE:
        return None

    return center_x, center_y, scale


def normalize_frame(frame):
    """
    frame: (258,) sirov keypoint frejm.
    Vraca: (258,) normalizovan frejm - X,Y koordinate centrirane na sredinu
    ramena i skalirane razmakom izmedju ramena. Z i visibility ostaju
    netaknuti (Z se samo skalira istom skalom radi konzistentnosti,
    visibility se ne dira).

    Ako ramena nisu detektovana, vraca frejm NEPROMENJEN (fallback) -
    ovo se desava retko i model i dalje ima ostale podatke na raspolaganju.
    """
    result = frame.copy()

    cs = _compute_center_and_scale(frame)
    if cs is None:
        return result  # ne mozemo normalizovati - vrati original

    center_x, center_y, scale = cs

    # --- POSE deo (33 tacke x 4: x,y,z,visibility) ---
    for i in range(33):
        base = i * 4
        x, y, z, vis = result[base], result[base + 1], result[base + 2], result[base + 3]
        if x == 0 and y == 0 and z == 0:
            continue  # tacka nije detektovana, ostavi nule
        result[base]     = (x - center_x) / scale
        result[base + 1] = (y - center_y) / scale
        result[base + 2] = z / scale
        # visibility (base+3) ostaje netaknut

    # --- LEVA RUKA (21 tacka x 3: x,y,z) ---
    for i in range(21):
        base = LEFT_HAND_START + i * 3
        x, y, z = result[base], result[base + 1], result[base + 2]
        if x == 0 and y == 0 and z == 0:
            continue
        result[base]     = (x - center_x) / scale
        result[base + 1] = (y - center_y) / scale
        result[base + 2] = z / scale

    # --- DESNA RUKA (21 tacka x 3: x,y,z) ---
    for i in range(21):
        base = RIGHT_HAND_START + i * 3
        x, y, z = result[base], result[base + 1], result[base + 2]
        if x == 0 and y == 0 and z == 0:
            continue
        result[base]     = (x - center_x) / scale
        result[base + 1] = (y - center_y) / scale
        result[base + 2] = z / scale

    return result


def normalize_sequence(sequence):
    """
    sequence: (n_frames, 258)
    Vraca: (n_frames, 258) normalizovana sekvenca, frejm po frejm.
    """
    return np.array([normalize_frame(frame) for frame in sequence])