"""
Interpolacija nedostajucih frejmova (rupa) u keypoint sekvencama.

Resava problem: MediaPipe ponekad "izgubi" ruku usred pokreta
(npr. ruka prolazi blizu lica, brz pokret), sto ostavlja nule
usred inace validne sekvence. Te nule ne predstavljaju stvarni
"nedostatak ruke" nego gresku detekcije, i kvare model jer
sekvenca onda izgleda "isprekidano" umesto neprekidno.

VAZNO razlikovanje:
  - Ruka NIJE KORISCENA za ceo znak (npr. jednorucni znak)
    -> ostaje SVE NULE kroz citavu sekvencu -> NE DIRAMO
  - Ruka JE KORISCENA ali je nakratko izgubljena usred pokreta
    -> nule okruzene validnim frejmovima -> INTERPOLIRAMO

Pravilo: interpoliramo frejm SAMO ako:
  1. Taj frejm ima sve nule za datu ruku, I
  2. Postoji bar jedan validan (ne-nula) frejm PRE njega za tu ruku, I
  3. Postoji bar jedan validan (ne-nula) frejm POSLE njega za tu ruku

Ako su SVI frejmovi za tu ruku nula (cela sekvenca), ne diramo nista -
to znaci da ruka prirodno nije ucestvovala u znaku.
"""
import numpy as np

HAND_LEN = 21 * 3
POSE_LEN = 33 * 4
LEFT_HAND_START = POSE_LEN
RIGHT_HAND_START = POSE_LEN + HAND_LEN


def _is_zero_hand(hand_vec):
    """hand_vec: (63,) - True ako su sve vrednosti nula (ruka nije detektovana)."""
    return np.all(hand_vec == 0)


def _interpolate_hand_sequence(hand_seq):
    """
    hand_seq: (n_frames, 63) - sekvenca koordinata jedne ruke kroz vreme.
    Vraca novu sekvencu gde su 'rupe' (nule okruzene validnim podacima)
    popunjene linearnom interpolacijom. Nule na pocetku/kraju ili
    kompletno nula-sekvenca ostaju netaknute.
    """
    n_frames = hand_seq.shape[0]
    is_zero = np.array([_is_zero_hand(hand_seq[i]) for i in range(n_frames)])

    # Ako je sve nula (ruka se uopste ne koristi u ovom znaku) - ne diraj
    if is_zero.all():
        return hand_seq, 0

    # Ako nema nijedne nule - nema sta da se interpolira
    if not is_zero.any():
        return hand_seq, 0

    result = hand_seq.copy()
    filled_count = 0

    i = 0
    while i < n_frames:
        if is_zero[i]:
            # Pronadji pocetak i kraj rupe
            gap_start = i
            gap_end = i
            while gap_end < n_frames and is_zero[gap_end]:
                gap_end += 1
            # gap je [gap_start, gap_end)

            prev_idx = gap_start - 1  # poslednji validan frejm pre rupe
            next_idx = gap_end        # prvi validan frejm posle rupe

            has_prev = prev_idx >= 0 and not is_zero[prev_idx]
            has_next = next_idx < n_frames and not is_zero[next_idx]

            if has_prev and has_next:
                # Linearna interpolacija izmedju prev_idx i next_idx
                prev_vals = hand_seq[prev_idx]
                next_vals = hand_seq[next_idx]
                gap_len = next_idx - prev_idx

                for offset, frame_idx in enumerate(range(gap_start, gap_end), start=1):
                    t = offset / gap_len
                    result[frame_idx] = prev_vals * (1 - t) + next_vals * t
                    filled_count += 1
            # else: rupa je na pocetku ili kraju sekvence - ostaje nula
            # (nemamo oba "kraja" da bismo pouzdano interpolirali)

            i = gap_end
        else:
            i += 1

    return result, filled_count


def interpolate_sequence_gaps(sequence):
    """
    sequence: (n_frames, 258) - sirova keypoint sekvenca (pose + leva + desna ruka)
    Vraca: (interpolisana_sekvenca, broj_popunjenih_frejmova_leva, broj_popunjenih_frejmova_desna)

    Interpolira SAMO ruke (leva, desna), ne dira pose deo (telo se redje gubi
    i manje je kriticno za razliku medju znacima).
    """
    sequence = sequence.copy()

    left_hand_seq  = sequence[:, LEFT_HAND_START:LEFT_HAND_START + HAND_LEN]
    right_hand_seq = sequence[:, RIGHT_HAND_START:RIGHT_HAND_START + HAND_LEN]

    left_interp,  n_left_filled  = _interpolate_hand_sequence(left_hand_seq)
    right_interp, n_right_filled = _interpolate_hand_sequence(right_hand_seq)

    sequence[:, LEFT_HAND_START:LEFT_HAND_START + HAND_LEN]  = left_interp
    sequence[:, RIGHT_HAND_START:RIGHT_HAND_START + HAND_LEN] = right_interp

    return sequence, n_left_filled, n_right_filled