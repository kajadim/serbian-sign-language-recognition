import numpy as np

HAND_LEN = 21 * 3
POSE_LEN = 33 * 4
LEFT_HAND_START = POSE_LEN
RIGHT_HAND_START = POSE_LEN + HAND_LEN


def _is_zero_hand(hand_vec):
    return np.all(hand_vec == 0)


def _interpolate_hand_sequence(hand_seq):
    n_frames = hand_seq.shape[0]
    is_zero = np.array([_is_zero_hand(hand_seq[i]) for i in range(n_frames)])

    if is_zero.all():
        return hand_seq, 0

    if not is_zero.any():
        return hand_seq, 0

    result = hand_seq.copy()
    filled_count = 0

    i = 0
    while i < n_frames:
        if is_zero[i]:
            gap_start = i
            gap_end = i
            while gap_end < n_frames and is_zero[gap_end]:
                gap_end += 1

            prev_idx = gap_start - 1  
            next_idx = gap_end        

            has_prev = prev_idx >= 0 and not is_zero[prev_idx]
            has_next = next_idx < n_frames and not is_zero[next_idx]

            if has_prev and has_next:
                prev_vals = hand_seq[prev_idx]
                next_vals = hand_seq[next_idx]
                gap_len = next_idx - prev_idx

                for offset, frame_idx in enumerate(range(gap_start, gap_end), start=1):
                    t = offset / gap_len
                    result[frame_idx] = prev_vals * (1 - t) + next_vals * t
                    filled_count += 1

            i = gap_end
        else:
            i += 1

    return result, filled_count


def interpolate_sequence_gaps(sequence):
    sequence = sequence.copy()

    left_hand_seq  = sequence[:, LEFT_HAND_START:LEFT_HAND_START + HAND_LEN]
    right_hand_seq = sequence[:, RIGHT_HAND_START:RIGHT_HAND_START + HAND_LEN]

    left_interp,  n_left_filled  = _interpolate_hand_sequence(left_hand_seq)
    right_interp, n_right_filled = _interpolate_hand_sequence(right_hand_seq)

    sequence[:, LEFT_HAND_START:LEFT_HAND_START + HAND_LEN]  = left_interp
    sequence[:, RIGHT_HAND_START:RIGHT_HAND_START + HAND_LEN] = right_interp

    return sequence, n_left_filled, n_right_filled