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
    assert not (abs(agcounts - al_counts) > 1).any()
