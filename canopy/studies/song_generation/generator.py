"""
Song generator, stage 1: the loop of the predictor of the next phrase on recorded phrases, without the GAN.

  - Predictor: the reservoirs of a study of the next phrase (canopy.studies.next_phrase.task: the MFCC of the
    song in input, no reset between the phrases) and their ridge readout of the next phrase ("last" -> "last": the
    state at the end of each phrase), trained on all the train songs with the coefficient of the best trial of the
    study (seed: the draw of the reservoirs).
  - Choice at the end of each phrase (and before the first one), with end="gate" (default): first the end of the
    song, END with the probability of the gate of the end, the output END of the readout calibrated by a logistic
    function (Platt's scaling) on its out-of-fold outputs at the points of the train songs (N_FOLDS folds by song, as
    the cross-validation of the studies): the output END of the readout ranks the ends of the test songs well (AUC
    0.86 for random_ei 1000 with calls and leak) but is rarely the largest (14 % of the ends found by the argmax), and
    the songs of the argmax were too long; then, if the song goes on, the next phrase among the labels the voice can
    sing: drawn with probabilities proportional to the positive outputs of the readout raised to 1 / temperature
    (temperature 0: the largest output). With end="readout", END is one of the labels of this choice.
  - Working memory: the label of the phrase being sung, for each song, written when its gate opens (here at the end
    of each phrase sung: the next choice) and read by the voice.
  - Voice ("recorded"): a recorded phrase of the label in memory, drawn among those of the train songs, with the
    silence that preceded it in its song: the MFCC frames between two prediction points of a train song, those that
    the reservoir hears between two decisions of the readout in canopy.studies.next_phrase.task. A song starts with the frames of a
    train song before its first prediction point (the silence and noise before its first phrase), drawn too, and ends
    when END is chosen or after max_phrases phrases (default twice the longest train song).
  - Reservoirs run phrase by phrase from the state of each song (advance), the songs in parallel.

Replaying the phrases of a train song in their order gives back its states at the prediction points (the frames of the
song, in the same order), so the predictor hears in the generated songs what it heard in training, except the
transitions between phrases of different songs.

Comparison with marron1 (compare): number of phrases per song, songs ended by END, and total variation distances of
the distributions of the n-grams of the labels (n = 1, 2, 3, with START before the first phrase and END after the last
one) from those of the test songs, for the generated songs, the train songs (the floor of the sampling noise) and the
songs of a Markov chain of order 1 fitted on the train songs (markov_songs: what the generator must beat on the 3-grams
if it uses more than the last phrase); novel_2grams: the part of the 2-grams of the songs that the train songs do not
have.

Reservoirs: the studies of canopy.studies.next_phrase.study, with the calls (HAG_CANARY_CALLS=1) and leaky
reservoirs (HAG_CANARY_LEAK=1) by default, the studies available locally; the calls are phrases too.

Results: outputs/song_generation/<prefix of the studies><model>_<size>_songs.csv (one row per phrase of the
generated songs: end, temperature, song, position, label, the source song and frames of the recorded phrase; label
END after the last phrase of a song ended by END) and ..._comparison.csv (one row per end, temperature and source).

Usage: python -m canopy.studies.song_generation.generator [model] [size] [nb_songs] [temperatures] [seed]
  (default random_ei 1000 200 0,0.25,0.5,1 0)
"""
import math
import os
import sys
from collections import Counter

os.environ.setdefault("HAG_CANARY_CALLS", "1")
os.environ.setdefault("HAG_CANARY_LEAK", "1")

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from hag.performances.batched_reservoir import PARAMETERS, _run  # noqa: E402
from hag.performances.esn_model_evaluation import to_jax_node  # noqa: E402

from canopy.common import pretrain_phrases  # noqa: E402
from canopy.studies.common import conditions  # noqa: E402
from canopy.models.reservoirs import build  # noqa: E402
from canopy.settings import FOLD_SEED, N_FOLDS, PREFIXES, outputs_dir  # noqa: E402
from canopy.studies.next_phrase.study import choose_ridge  # noqa: E402
from canopy.studies.next_phrase.task import END, START, fit_readout, load_task_dataset, point_states  # noqa: E402

RESULTS = outputs_dir("song_generation")
LENGTH_BUCKET = 64  # (series padded to a multiple of this length: few shapes compiled by JAX)


# ------------------------------------------------- reservoirs -----------------------------------------------------

def jax_nodes(reservoirs):
    """[(reservoirpy.jax node, its parameters, input gains)] of the reservoirs (canopy.models.reservoirs.build)."""
    gains = getattr(reservoirs, "input_gains", ()) or (None,) * len(reservoirs)
    result = []
    for reservoir, g in zip(reservoirs, gains):
        node = to_jax_node(reservoir)
        if node is None or set(node.state) != {"out"}:
            raise ValueError(f"{type(reservoir).__name__}: not a reservoirpy.jax node whose state is its output")
        parameters = {name: jnp.asarray(getattr(node, name)) for name in PARAMETERS
                      if getattr(node, name, None) is not None}
        result.append((node, parameters, g))
    return result


def advance(nodes, state, inputs):
    """
    States (len(inputs), units) of the reservoirs (jax_nodes, side by side as in canopy.models.reservoirs.states) after
    the input series (T_i, features), each from its own state, the row i of state; an empty series keeps it.
    """
    lengths = np.array([len(x) for x in inputs])
    rows = 2 ** math.ceil(math.log2(len(inputs)))  # (rows padded to a power of two, as hag's run_batched)
    steps = max(1, math.ceil(lengths.max() / LENGTH_BUCKET)) * LENGTH_BUCKET
    padded_lengths = np.zeros(rows, dtype=int)
    padded_lengths[:len(inputs)] = lengths
    result, offset = [], 0
    for node, parameters, gains in nodes:
        n = node.output_dim
        x = np.zeros((rows, steps, node.input_dim))
        for i, series in enumerate(inputs):
            x[i, :len(series)] = series if gains is None else series * gains
        state0 = np.zeros((rows, n))
        state0[:len(inputs)] = state[:, offset:offset + n]
        out = _run(parameters, jnp.asarray(x), jnp.asarray(padded_lengths), {"out": jnp.asarray(state0, dtype=float)},
                   node_type=type(node), activation=node.activation, return_all=False)
        result.append(np.asarray(out, dtype=np.float64)[:len(inputs)])
        offset += n
    return np.hstack(result)


class Predictor:
    """The reservoirs of the study of the model and their ridge readout of the next phrase, trained on the train
    songs; data: the result of canopy.studies.next_phrase.task.load_task_dataset (loaded if None)."""

    def __init__(self, model="random_ei", size=1000, seed=0, data=None):
        self.songs, self.train_songs, self.test_songs, self.frame, self.classes = (
            data if data is not None else load_task_dataset())
        found = [c for c in conditions(size, "next_phrase") if c[0] == model]
        if not found:
            raise ValueError(f"no study {PREFIXES['next_phrase']}{model}_{size}")
        _, model, params, ridge, _ = found[0]
        if ridge is None:  # (the conditions without study: the coefficient chosen by cross-validation)
            ridge = choose_ridge(model, params, size, self.songs, self.train_songs, self.frame, self.classes)[0]
        self.model, self.size, self.ridge = model, size, ridge
        np.random.seed(seed)
        self.reservoirs = build(model, params, size, pretrain_phrases(self.songs, self.train_songs, self.frame))
        states = point_states(self.reservoirs, self.songs, self.train_songs)
        self.readout = fit_readout(states, self.songs, self.train_songs, self.classes, ridge)
        self.nodes = jax_nodes(self.reservoirs)
        self.units = sum(node.output_dim for node, _, _ in self.nodes)
        self.end = list(self.classes).index(END)
        self.gate = self.end_gate(states)

    def end_gate(self, states):
        """Logistic function of the output END of the readout (Platt's scaling) fitted on its out-of-fold outputs at
        the points of the train songs (states: {wave: (points, units)}), readouts of the same coefficient trained on
        the other folds."""
        outputs, ends = [], []
        waves = self.train_songs
        for fit_index, val_index in KFold(N_FOLDS, shuffle=True, random_state=FOLD_SEED).split(waves):
            fit, val = [waves[i] for i in fit_index], [waves[i] for i in val_index]
            readout = fit_readout({w: states[w] for w in fit}, self.songs, fit, self.classes, self.ridge)
            outputs.append(np.asarray(readout.run(np.vstack([states[w] for w in val])))[:, self.end])
            ends.append(np.concatenate([self.songs[w]["next"] == END for w in val]))
        return LogisticRegression(C=1e6).fit(np.concatenate(outputs)[:, None], np.concatenate(ends))

    def outputs(self, state):
        """Outputs (songs, classes) of the readout on the states (songs, units)."""
        return np.asarray(self.readout.run(state), dtype=np.float64)

    def end_probability(self, outputs):
        """Probability of the end of the song (songs,) from the outputs of the readout (the gate of the end)."""
        return self.gate.predict_proba(outputs[:, self.end:self.end + 1])[:, 1]


# ------------------------------------------------- generation -----------------------------------------------------

class RecordedVoice:
    """
    Recorded phrases of songs (songs[w]["mfcc"], ["points"] and ["next"] of prediction.prepare), by label: the frames
    after a prediction point up to the next one (the silence before the phrase and the phrase), (wave, first, last)
    frames first to last included; and the beginnings of the songs, their frames up to their first point.
    """

    def __init__(self, songs, waves):
        self.songs, self.phrases, self.beginnings = songs, {}, []
        for w in waves:
            points, labels = songs[w]["points"], songs[w]["next"]
            self.beginnings.append((w, 0, int(points[0])))  # (no frame if the first point is the zero state, -1)
            for k in range(len(points) - 1):
                self.phrases.setdefault(labels[k], []).append((w, int(points[k]) + 1, int(points[k + 1])))

    def frames(self, source):
        w, first, last = source
        return self.songs[w]["mfcc"][first:last + 1]

    def phrase(self, label, rng):
        options = self.phrases[label]
        return options[rng.integers(len(options))]

    def beginning(self, rng):
        return self.beginnings[rng.integers(len(self.beginnings))]


class WorkingMemory:
    """The label of the phrase being sung by each song (START before the first one), written when its gate opens."""

    def __init__(self, nb_songs):
        self.content = np.full(nb_songs, START, dtype=object)

    def write(self, gate, labels):
        """Writes the labels in the memory of the songs gate (indices), where the gate is open."""
        self.content[gate] = labels

    def read(self, songs):
        return self.content[songs]


def choose(outputs, allowed, temperature, rng):
    """Indices of the classes chosen from the outputs (songs, classes), among the allowed ones (classes,): drawn with
    probabilities proportional to the positive outputs raised to 1 / temperature, the largest output with temperature
    0 (or when no allowed output is positive)."""
    best = np.where(allowed, outputs, -np.inf).argmax(axis=1)
    if temperature == 0:
        return best
    weights = np.where(allowed, np.clip(outputs, 0, None), 0) ** (1 / temperature)
    return np.array([rng.choice(len(w), p=w / w.sum()) if w.sum() > 0 else b for w, b in zip(weights, best)])


def generate(predictor, voice, nb_songs, temperature=1.0, seed=0, max_phrases=None, end="gate"):
    """DataFrame of the phrases of nb_songs songs generated by the loop predictor -> working memory -> voice ->
    reservoirs (columns song, position, label, wave, first, last; an END row after the last phrase of the songs ended
    by END); end: "gate" (the gate of the end, then the next phrase) or "readout" (END one of the labels chosen)."""
    rng = np.random.default_rng(seed)
    classes = predictor.classes
    if max_phrases is None:
        max_phrases = 2 * max(len(predictor.songs[w]["next"]) for w in predictor.train_songs)
    allowed = np.array([c in voice.phrases or (c == END and end == "readout") for c in classes])
    beginnings = [voice.beginning(rng) for _ in range(nb_songs)]
    state = advance(predictor.nodes, np.zeros((nb_songs, predictor.units)), [voice.frames(b) for b in beginnings])
    memory, active, rows = WorkingMemory(nb_songs), np.arange(nb_songs), []
    for position in range(max_phrases):
        outputs = predictor.outputs(state[active])
        labels = classes[choose(outputs, allowed, temperature, rng)]
        if end == "gate":
            labels[rng.random(len(active)) < predictor.end_probability(outputs)] = END
        memory.write(active, labels)  # (the gate opens at the end of the phrase being sung: the next choice)
        for i in active[labels == END]:
            rows.append(dict(song=i, position=position, label=END, wave=None, first=-1, last=-1))
        active = active[labels != END]
        if not len(active):
            break
        sources = [voice.phrase(label, rng) for label in memory.read(active)]
        rows += [dict(song=i, position=position, label=label, wave=w, first=first, last=last)
                 for i, label, (w, first, last) in zip(active, memory.read(active), sources)]
        state[active] = advance(predictor.nodes, state[active], [voice.frames(s) for s in sources])
    return pd.DataFrame(rows).sort_values(["song", "position"], kind="stable").reset_index(drop=True)


# ------------------------------------------------- comparison -----------------------------------------------------

def sequences_of(phrases):
    """[(labels of the phrases of the song, ended by END)] of the songs of a DataFrame of generate."""
    return [([label for label in song.label if label != END], bool((song.label == END).any()))
            for _, song in phrases.groupby("song", sort=True)]


def real_sequences(songs, waves):
    """[(labels of the phrases, True)] of the songs (songs[w]["next"]: the labels and END)."""
    return [(list(songs[w]["next"][:-1]), True) for w in waves]


def markov_songs(train, nb_songs, max_phrases, seed=0):
    """[(labels, ended)] of nb_songs songs of a Markov chain of order 1 (START, the labels, END) fitted on the
    sequences train."""
    rng = np.random.default_rng(seed)
    transitions = {}
    for labels, _ in train:
        padded = [START] + labels + [END]
        for a, b in zip(padded, padded[1:]):
            transitions.setdefault(a, Counter())[b] += 1
    result = []
    for _ in range(nb_songs):
        labels, current = [], START
        while len(labels) < max_phrases:
            nexts, counts = zip(*transitions[current].items())
            current = nexts[rng.choice(len(nexts), p=np.array(counts) / sum(counts))]
            if current == END:
                break
            labels.append(current)
        result.append((labels, current == END))
    return result


def ngrams(sequences, n):
    """Distribution (Counter normalized) of the n-grams of the labels of the songs, START before and END after (n = 1:
    the labels and END)."""
    counts = Counter()
    for labels, ended in sequences:
        padded = ([START] if n > 1 else []) + labels + ([END] if ended else [])
        counts.update(tuple(padded[i:i + n]) for i in range(len(padded) - n + 1))
    total = sum(counts.values())
    return {gram: c / total for gram, c in counts.items()}


def total_variation(p, q):
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))


def compare(generated, train, test, nb_songs, max_phrases, seed=0):
    """DataFrame of the statistics of the songs generated, train and of the Markov chain of order 1 (markov_songs),
    their n-grams compared with those of the test songs (total variation distances tv_1, tv_2, tv_3)."""
    sources = dict(generated=generated, train=train, markov_order1=markov_songs(train, nb_songs, max_phrases, seed),
                   test=test)
    seen = set(ngrams(train, 2))
    rows = []
    for name, sequences in sources.items():
        lengths = np.array([len(labels) for labels, _ in sequences])
        bigrams = Counter()
        for labels, ended in sequences:
            padded = [START] + labels + ([END] if ended else [])
            bigrams.update(zip(padded, padded[1:]))
        rows.append(dict(source=name, songs=len(sequences), phrases_mean=lengths.mean(), phrases_sd=lengths.std(),
                         ended=np.mean([ended for _, ended in sequences]),
                         novel_2grams=sum(c for g, c in bigrams.items() if g not in seen) / sum(bigrams.values()),
                         **{f"tv_{n}": total_variation(ngrams(sequences, n), ngrams(test, n)) for n in (1, 2, 3)}))
    return pd.DataFrame(rows)


TEMPERATURES = (0, 0.25, 0.5, 1)
ENDS = ("gate", "readout")


def name_of(model, size):
    return f"{PREFIXES['next_phrase']}{model}_{size}"


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "random_ei"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    nb_songs = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    temperatures = [float(t) for t in sys.argv[4].split(",")] if len(sys.argv) > 4 else TEMPERATURES
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    predictor = Predictor(model, size, seed)
    voice = RecordedVoice(predictor.songs, predictor.train_songs)
    max_phrases = 2 * max(len(predictor.songs[w]["next"]) for w in predictor.train_songs)
    train = real_sequences(predictor.songs, predictor.train_songs)
    test = real_sequences(predictor.songs, predictor.test_songs)
    all_phrases, comparisons = [], []
    for end in ENDS:
        for temperature in temperatures:
            phrases = generate(predictor, voice, nb_songs, temperature, seed, max_phrases, end)
            comparison = compare(sequences_of(phrases), train, test, nb_songs, max_phrases, seed)
            all_phrases.append(phrases.assign(end=end, temperature=temperature))
            comparisons.append(comparison.assign(end=end, temperature=temperature))
            print(f"end {end}, temperature {temperature:g}: {' '.join(sequences_of(phrases)[0][0][:15])} ...",
                  flush=True)
    comparison = pd.concat(comparisons)
    pd.set_option("display.width", 200)
    print(comparison.round(3).to_string(index=False))
    RESULTS.mkdir(parents=True, exist_ok=True)
    name = name_of(model, size)
    pd.concat(all_phrases).to_csv(RESULTS / f"{name}_songs.csv", index=False)
    comparison.to_csv(RESULTS / f"{name}_comparison.csv", index=False)
    print(f"-> {RESULTS / name}_songs.csv, _comparison.csv")
