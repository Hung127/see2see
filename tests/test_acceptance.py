"""DESIGN §17 acceptance criteria that have no other named home.

Each test here is named exactly as §17 names it. The other §17 criteria are
covered by tests that already carry the canonical name (or were renamed to it):
``test_mass_and_energy_conservation``, ``test_predation_split``,
``test_plant_competition``, ``test_drought_increases_mortality``,
``test_stats_schema``, ``test_resume_equals_continuation``,
``test_headless_run``.

These are behaviour checks, not documentation: each one would fail if the
corresponding mechanism were removed or replaced by a rule.
"""
import numpy as np

from evolve.actions import Actions, decode
from evolve.brain import forward
from evolve.config import (
    Config, EnergyConfig, EnvironmentConfig, GeneticsConfig, OrganismConfig,
    PlantConfig, StatsConfig, WorldConfig,
)
from evolve.engine import Engine
from evolve.genome import N_TRAITS, TRAIT_NAMES, founders, mutate, phenotype


def make_engine(mode="asexual", n=4, seed=1, max_pop=50, maturity=50):
    cfg = Config(
        seed=seed, ticks=100,
        world=WorldConfig(width=500.0, height=500.0, n_field_cells=10, cell_size=50.0),
        plants=PlantConfig(initial=50, max=200),
        organisms=OrganismConfig(initial=n, max=max_pop, maturity_age=maturity),
        energy=EnergyConfig(),
        genetics=GeneticsConfig(mode=mode),
        environment=EnvironmentConfig(rain_rate=0.0),
        stats=StatsConfig(),
    )
    return Engine(cfg)


def force_ready(eng, frac=0.9):
    pop = eng.population
    pop.compact()
    phen = phenotype(pop.genome)
    emax = 10.0 * phen[:, 0] ** 0.9
    pop.energy[:] = frac * emax
    pop.age[:] = 100
    return phen, emax


def all_reproduce(n, z=None):
    z = np.zeros(n) if z is None else z
    return Actions(z, z, np.zeros(n, bool), np.zeros(n, bool), np.ones(n),
                   np.ones(n, bool), z)


# -- §17: genome -> organism -------------------------------------------------
def test_founders_are_genome_derived():
    """Founder state is computed from its own genome, not assigned externally."""
    eng = make_engine(n=8)
    pop = eng.population
    pop.compact()  # live slots only; freshly allocated capacity is padded
    phen = phenotype(pop.genome)
    emax = 10.0 * phen[:, 0] ** 0.9
    # body, reserves and every phenotype come from the genome
    assert np.allclose(pop.body, phen[:, 0])
    assert np.allclose(pop.energy, 0.5 * emax)
    assert np.all(phen[:, 0] >= 0.3) and np.all(phen[:, 0] <= 5.0)
    # different genomes give different phenotypes (not one shared organism)
    assert len({tuple(np.round(row, 9)) for row in phen}) > 1
    # recomputing from the genome reproduces the same phenotype exactly
    assert np.array_equal(phenotype(pop.genome.copy()), phen)


# -- §17: individual networks ------------------------------------------------
def test_individual_weights():
    """Each organism's neural block is its own and drives its own output."""
    rng = __import__("evolve.rng", fromlist=["Rng"]).Rng(3)
    genomes = founders(4, rng)
    weights = genomes[:, N_TRAITS:]
    # distinct genomes -> distinct parameter blocks
    assert len({tuple(np.round(w, 9)) for w in weights}) == 4

    sensors = np.linspace(-1.0, 1.0, 16)[None, :].repeat(4, axis=0)
    batched = forward(sensors, weights)
    # the batch really is one forward pass per organism: row i equals the
    # single-organism forward pass with organism i's weights
    for i in range(4):
        single = forward(sensors[i:i + 1], weights[i:i + 1])
        assert np.allclose(batched[i], single[0])
    # and different weights produce different outputs from identical sensors
    assert len({tuple(np.round(row, 9)) for row in batched}) > 1


# -- §17: network controls action --------------------------------------------
def test_action_from_network_only():
    """Holding the sensors fixed, changing only the network changes the action.

    The engine's own forward pass must equal ``forward(sensors, weights)`` and
    the decoded action must be a pure function of (outputs, phenotype): there is
    no other input that could be choosing the action.
    """
    eng = make_engine(n=6)
    pop = eng.population
    pop.compact()  # live slots only
    actions = eng._think()
    sensors = eng.last_sensors
    weights = pop.genome[:, N_TRAITS:]
    phen = eng._phen
    # the engine used exactly the network it says it used
    assert np.allclose(eng.last_outputs, forward(sensors, weights))
    rebuilt = decode(eng.last_outputs, phen)
    for name in ("throttle", "turn", "eat", "attack", "sprint", "reproduce", "rest"):
        a, b = getattr(actions, name), getattr(rebuilt, name)
        assert np.array_equal(np.asarray(a), np.asarray(b)), name

    # same sensors, different weights -> a different decoded action vector
    changed = weights.copy()
    changed[0, :] = -changed[0, :]  # flip one organism's whole first layer
    out = forward(sensors, changed)
    assert not np.allclose(out[0], eng.last_outputs[0])
    new = decode(out, phen)
    assert (not np.array_equal(np.asarray(new.eat), np.asarray(actions.eat))
            or not np.array_equal(np.asarray(new.attack), np.asarray(actions.attack))
            or not np.allclose(np.asarray(new.throttle), np.asarray(actions.throttle)))


# -- §17: weight inheritance -------------------------------------------------
def test_weights_inherited():
    """An asexual child inherits the parent's network, mutated at few loci."""
    eng = make_engine(n=2, seed=5)
    pop = eng.population
    pop.compact()
    # make mutation visible: a low mutation-rate draw can leave 380 loci intact
    pop.genome[0, 11] = 1.0  # mapped to the 5e-2 ceiling
    force_ready(eng)
    parent = pop.genome[0, N_TRAITS:].copy()
    parent_id = int(pop.id[0])
    n0 = pop.n_alive
    eng._reproduce(all_reproduce(pop.n_alive))
    assert pop.n_alive > n0
    child = n0
    child_w = pop.genome[child, N_TRAITS:]
    changed = float(np.mean(child_w != parent))
    assert 0.0 < changed < 0.2        # mutated, but overwhelmingly inherited
    assert int(pop.parent_a[child]) == parent_id
    # and the child's weights are its parent's, not a fresh random draw
    assert np.allclose(np.median(np.abs(child_w - parent)), 0.0, atol=0.5)


# -- §17: mutation changes behaviour ----------------------------------------
def test_mutation_changes_action():
    """Mutating the network changes what the organism does on identical input."""
    rng = __import__("evolve.rng", fromlist=["Rng"]).Rng(11)
    genome = founders(1, rng)[0]
    sensors = np.zeros((1, 16))
    sensors[0, 8] = 0.7  # a fixed, non-trivial sensor row
    before = forward(sensors, genome[None, N_TRAITS:])
    child, n_mut = mutate(genome[None, :], np.array([0.5]), rng,
                          GeneticsConfig())
    assert int(n_mut.sum()) > 0
    after = forward(sensors, child[:, N_TRAITS:])
    assert not np.allclose(before, after)


# -- §17: birth and death ----------------------------------------------------
def test_reproduce_and_die():
    """A rich, willing organism reproduces; a dead one is removed."""
    eng = make_engine(n=3, seed=7)
    pop = eng.population
    force_ready(eng)
    n0 = pop.n_alive
    eng._reproduce(all_reproduce(pop.n_alive))
    n_after_births = pop.n_alive
    assert n_after_births > n0                    # birth happened

    pop.compact()
    eng._refresh_phen()  # births changed the row set; stages need a fresh cache
    victim = int(pop.id[0])
    pop.health[0] = 0.0                           # health-destroyed organism
    eng._mortality()
    assert pop.n_alive == n_after_births - 1      # exactly the victim is gone
    assert victim not in set(int(i) for i in pop.id[pop.alive])
