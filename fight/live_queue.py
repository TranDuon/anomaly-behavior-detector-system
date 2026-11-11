"""Discard expired live samples rather than building seconds of inference lag."""


def fresh_live_samples(samples, now, maximum_age=.8):
    fresh = [sample for sample in samples if now - sample[1] <= maximum_age]
    return fresh, len(samples) - len(fresh)
