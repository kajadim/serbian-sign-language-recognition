"""
Zagladjivanje (temporal smoothing) curl feature-a kroz susedne frejmove.

PROBLEM koji ovo resava:
Curl feature (savijenost prsta) se racuna kao odnos dva rastojanja,
deljen sa 'scale' (razmak ramena). Kada je osoba daleko od kamere,
scale je mali broj, pa se SVAKA mala greska MediaPipe detekcije (par
piksela suma na vrhu prsta) UVECAVA nakon deljenja. To pravi curl
vrednosti nestabilnim bas kada je najpotrebnije da budu pouzdane.

RESENJE:
Umesto da racunamo curl iz JEDNOG frejma, racunamo prosek curl
vrednosti kroz prozor suseднih frejmova (npr. trenutni +- 2 frejma).
Sum (slucajna greska po frejmu) se delom ponisti usrednjavanjem,
dok se pravi signal (oblik sake, koji se ne menja naglo iz frejma u
frejm) zadrzava.
"""
import numpy as np


def smooth_curl_sequence(curl_sequence, window=5):
    """
    curl_sequence: (n_frames, 10) - curl vrednosti po frejmu (pre zagladjivanja)
    window: velicina prozora za rolling average (mora biti neparan broj)

    Vraca: (n_frames, 10) zagladjena sekvenca, ista duzina kao original
    (koristi se 'centered' rolling average sa edge-padding na ivicama).
    """
    if window % 2 == 0:
        window += 1  # osiguraj neparan broj radi simetricnog prozora

    n_frames = curl_sequence.shape[0]
    half = window // 2
    smoothed = np.zeros_like(curl_sequence)

    for i in range(n_frames):
        start = max(0, i - half)
        end = min(n_frames, i + half + 1)
        smoothed[i] = curl_sequence[start:end].mean(axis=0)

    return smoothed