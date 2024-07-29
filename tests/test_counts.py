import pytest


@pytest.mark.parametrize(
    "agcounts,agcounts_serial",
    [
        ((30, 30), (30, 30)),
        ((30, 40), (30, 40)),
        ((10, 30), (10, 30)),
        ((10, 40), (10, 40)),
    ],
    indirect=True,
)
@pytest.mark.slow
def test_counts_fast_and_slow(agcounts, agcounts_serial):
    assert (agcounts_serial == agcounts).all()


@pytest.mark.parametrize(
    "agcounts,cp_counts",
    [
        ((30, 30), (30, 30)),
        ((30, 40), (30, 40)),
        ((10, 30), (10, 30)),
        ((10, 40), (10, 40)),
    ],
    indirect=True,
)
def test_counts_cp(agcounts, cp_counts):
    assert not (abs(agcounts.flatten()[6:] - cp_counts.flatten()[6:]) > 1).any()


@pytest.mark.parametrize(
    "agcounts,al_counts",
    [
        ((30, 30), (30, 30)),
        ((30, 40), (30, 40)),
        ((10, 30), (10, 30)),
        ((10, 40), (10, 40)),
    ],
    indirect=True,
)
def test_counts_al(agcounts, al_counts):
    assert (agcounts == al_counts).all()


@pytest.mark.parametrize(
    "agcounts_better_lpf,al_counts",
    [
        ((30, 40), (30, 40)),
        ((10, 40), (10, 40)),
    ],
    indirect=True,
)
def test_counts_al_better_lpf(agcounts_better_lpf, al_counts):
    assert agcounts_better_lpf.shape == al_counts.shape


@pytest.mark.parametrize(
    "noisy_agcounts,agcounts",
    [
        ((30, 30, 0.01), (30, 30)),
        ((30, 30, 0.02), (30, 30)),
        ((30, 30, 0.05), (30, 30)),
        ((30, 30, 0.06), (30, 30)),
        ((30, 30, 0.07), (30, 30)),
        ((30, 30, 0.5), (30, 30)),
        ((30, 30, 50), (30, 30)),
    ],
    indirect=True,
)
def test_counts_noisy_accel(noisy_agcounts, agcounts):
    assert not (abs(noisy_agcounts.flatten()[6:] - agcounts.flatten()[6:]) > 1).any()
