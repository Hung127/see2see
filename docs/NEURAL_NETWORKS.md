# Neural networks

## Shape and layout

Every organism runs the same network shape with its own parameters:
**16 → 12 → 8 → 8**, three weight matrices, 380 parameters total
(`brain.N_PARAMS`). The parameters are the neural block of the genome
(loci 12–391), laid out row-major:

```
[0:192)    W1 (16x12)    [192:204)  b1 (12)
[204:300)  W2 (12x8)     [300:308)  b2 (8)
[308:372)  W3 (8x8)      [372:380)  b3 (8)
```

The whole population is one batch: three `einsum` products per tick, so 5,000
organisms cost the same *number* of matrix multiplications as one organism.

## Activations

* Hidden layers: `tanh`.
* Output channels 0–1 (forward, turn): `tanh` — signed, continuous.
* Output channels 2–7 (eat, attack, sprint, reproduce, rest): sigmoid,
  computed as `0.5·(1 + tanh(0.5·z))` for numerical stability.

## Sensors (16 inputs)

Sensors only report. "Nothing detected" is encoded as −1. Threat/prey/mate are
*perceptual categories* derived from observable phenotype — body size, attack
locus, age — never species labels, and they never imply an action.

| # | Input | Encoding |
|---|-------|----------|
| 0–1 | nearest food (plant or carcass) | `1 − d/r_vision`, bearing/π |
| 2–3 | nearest threat (visible, `M_t > 1.15·M`, `attack_t > 1`) | same |
| 4–5 | nearest prey (visible, `M_t < 0.85·M`) | same |
| 6–7 | nearest mate (visible, mature, `0.7·M ≤ M_t ≤ 1.43·M`) | same |
| 8 | own energy | `E / E_max` |
| 9 | own health | `health / health_max` |
| 10 | own age | `age / lifespan` |
| 11 | local temperature | `(T − 20)/15` |
| 12 | local moisture | `2·m − 1` |
| 13 | local plant biomass | `2·B_cell/10 − 1` |
| 14 | local organism count | `2·n_cell/20 − 1` |
| 15 | reproduction drive | `(E − threshold)/(E_max − threshold)` |

Vision is a cone: a target is visible when `d ≤ r_vision` and
`|bearing| ≤ vision_angle/2`. The nearest visible target wins; ties go to the
lowest target index. `sensors.py` must not import `actions.py` — a test
enforces that sensors cannot imply actions.

## Action decoding

`actions.decode(outputs, phen)` turns the 8 outputs into physical attempts:

| Output | Action | Rule |
|--------|--------|------|
| o0 | forward throttle | `abs(o0) · o2 · (1 − o7)` × `speed_max` |
| o1 | turn | `o1 · turn_rate_max`, `turn_rate_max = 3.0 / M^(1/3)` |
| o2 | throttle gate | multiplies forward speed |
| o3 | eat | `o3 > 0.5` and not resting |
| o4 | attack | `o4 > 0.5` and not resting |
| o5 | sprint | `1 + 1.5·o5` (scales speed and movement cost) |
| o6 | reproduce | `o6 > 0.5` (still gated by maturity and energy) |
| o7 | rest | `o7 > 0.5` (throttle → 0, blocks eating and attacking) |

`escape` is not a separate channel: sprinting (`o5`) scales speed and movement
cost, and it can never steer — turning is `o1` only. There is no "flee" input
and no rule that maps a threat to a heading.

## Decision-point audit

The design requires an explicit audit of every point in the code where the
system decides anything on an organism's behalf. The complete list:

1. **`actions.decode` threshold comparisons** — the *only* decision point that
   is not the network itself. Three fixed comparisons (`o3 > 0.5`,
   `o4 > 0.5`, `o6 > 0.5`, `o7 > 0.5`) turn continuous outputs into discrete
   physical attempts. They encode no preference over targets or situations:
   they are the same for every organism, every tick, and every state of the
   world. The network can and does learn to sit on either side of any of them.

That is the whole list. In particular:

* **Sensors do not decide.** Threat/prey/mate classification selects *what is
  reported*, not what is done; a threat sensor value has no privileged path to
  any action.
* **Feeding does not decide.** `_feed` bites the nearest resource in reach
  when the *network* said "eat"; there is no plant-vs-carcass preference rule
  beyond "nearest wins, ties by kind then ID".
* **Predation does not decide.** `_predation` strikes the nearest organism in
  reach when the network said "attack"; damage and kill credit are physical
  bookkeeping.
* **Reproduction does not decide.** The network's `o6` is necessary but not
  sufficient: maturity and the genome's own energy threshold gate it. The
  threshold is a *trait* the genome carries, not a rule the engine imposes.
* **Mortality does not decide.** Death is physics: health, starvation,
  senescence.
* **Events do not decide.** Environmental events change fields and health;
  they never touch a genome (asserted by test) and never inject behaviour.

## What the network cannot do

* It cannot change its own traits; the trait loci are inherited, not computed.
* It cannot sense another organism's genome, energy or health — only its
  phenotype-derived perceptual class, distance and bearing.
* It cannot act outside the eight channels; there is no "cheat" output.
