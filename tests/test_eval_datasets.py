"""Benchmark tooling: corpus adapters, manifests, trial generation and summaries."""

import random

from scripts.eval import datasets as ds
from scripts.eval.public_benchmark import sampled_trials, summaries


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_kathbath_reads_speaker_gender_and_noisy_split(tmp_path):
    _touch(tmp_path / "kb_data_clean_m4a" / "hindi" / "test" / "audio" / "844424930464239-1161-m.m4a")
    _touch(tmp_path / "kb_data_noisy_m4a" / "tamil" / "test" / "audio" / "844424930000001-70-f.m4a")
    rows = {r.language: r for r in ds.kathbath(str(tmp_path))}
    assert rows["hindi"].speaker == "kb-hindi-1161" and rows["hindi"].gender == "m"
    assert rows["hindi"].group == "Hindi (Kathbath)" and rows["hindi"].dataset == "kathbath"
    assert rows["tamil"].group == "Tamil (Kathbath noisy)" and rows["tamil"].dataset == "kathbath_noisy"


def test_crowdsourced_groups_by_dialect_prefix(tmp_path):
    _touch(tmp_path / "wef_12484_01482829612.wav")
    _touch(tmp_path / "irm_02484_00001.wav")
    rows = {r.speaker: r for r in ds.crowdsourced(str(tmp_path))}
    assert rows["gc-wef_12484"].group == "English, Welsh" and rows["gc-wef_12484"].gender == "f"
    assert rows["gc-irm_02484"].group == "English, Irish" and rows["gc-irm_02484"].gender == "m"


def test_commonvoice_uses_accent_when_present(tmp_path):
    (tmp_path / "clips").mkdir()
    (tmp_path / "validated.tsv").write_text(
        "client_id\tpath\tsentence\taccents\tlocale\tgender\n"
        "abc123\ta.mp3\thello\tIndia and South Asia\ten\tfemale\n"
        "def456\tb.mp3\tnamaste\t\thi\tmale\n",
        encoding="utf-8",
    )
    rows = list(ds.commonvoice(str(tmp_path)))
    assert rows[0].group == "en: India and South Asia" and rows[0].gender == "f"
    assert rows[1].group == "hi" and rows[1].language == "hi"


def test_folder_layout_and_manifest_round_trip(tmp_path):
    _touch(tmp_path / "family" / "mom" / "hindi" / "1.wav")
    _touch(tmp_path / "family" / "dad" / "2.wav")
    rows = list(ds.folder(str(tmp_path / "family"), group="Our family"))
    assert {r.speaker for r in rows} == {"fd-mom", "fd-dad"}
    assert {r.language for r in rows if r.speaker == "fd-mom"} == {"hindi"}

    out = tmp_path / "m.csv"
    assert ds.write_manifest(rows[:1], str(out)) == 1
    assert ds.write_manifest(rows[1:], str(out), append=True) == 1
    back = ds.read_manifests([str(out)])
    assert [r.path for r in back] == [r.path for r in rows]


def test_sampled_trials_are_balanced_and_labelled_correctly():
    by_spk = {f"s{i}": [f"s{i}_{j}" for j in range(4)] for i in range(5)}
    trials = sampled_trials(by_spk, 40, random.Random(0))
    target = [t for t in trials if t[0] == 1]
    impostor = [t for t in trials if t[0] == 0]
    assert len(target) == len(impostor) == 20
    assert all(a.split("_")[0] == b.split("_")[0] and a != b for _, a, b in target)
    assert all(a.split("_")[0] != b.split("_")[0] for _, a, b in impostor)


def test_household_streams_have_requested_length_and_noise_level():
    import numpy as np

    from scripts.eval.simulate_households import SR, add_noise, build_stream, turn_taking

    rng = random.Random(0)
    a = [np.full(SR, 0.1, dtype=np.float32)]
    b = [np.full(SR, -0.1, dtype=np.float32)]
    stream = build_stream(a, rng, 10.0)
    assert len(stream) == 10 * SR

    convo = turn_taking(a, b, rng, 10.0)
    assert len(convo) == 10 * SR and (convo > 0).any() and (convo < 0).any()

    noise = [np.random.default_rng(0).standard_normal(3 * SR).astype(np.float32)]
    noisy = add_noise(stream, noise, 10.0, rng)
    voiced = np.abs(stream) > 1e-4
    snr = 10 * np.log10(np.mean(stream[voiced] ** 2) / np.mean((noisy - stream) ** 2))
    assert abs(snr - 10.0) < 0.5


def test_audit_flags_an_id_that_holds_two_voices_only():
    import numpy as np

    from scripts.eval.audit_speakers import is_mixed, two_voice_split

    rng = np.random.default_rng(0)

    def voice(centre, n):
        v = centre + 0.35 * rng.standard_normal((n, 64))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    a, b = rng.standard_normal(64), rng.standard_normal(64)
    two = np.vstack([voice(a, 7), voice(b, 7)])
    labels, wa, wb, between = two_voice_split(two)
    assert sorted([int(labels.sum()), len(labels) - int(labels.sum())]) == [7, 7]
    assert is_mixed(7, 7, wa, wb, between)

    one = voice(a, 14)
    labels, wa, wb, between = two_voice_split(one)
    assert not is_mixed(len(labels) - int(labels.sum()), int(labels.sum()), wa, wb, between)


def test_summaries_add_pooled_rows_and_measure_eer():
    scores = []
    for g in ("A", "B"):
        scores += [("pairs", g, "utterance pairs (official)", 1, 0.8 + k / 100) for k in range(5)]
        scores += [("pairs", g, "utterance pairs (official)", 0, 0.1 + k / 100) for k in range(5)]
    rows = summaries(scores)
    groups = [r["group"] for r in rows]
    assert groups == ["A", "B", "ALL GROUPS (pooled)"]
    pooled = rows[-1]
    assert pooled["condition"] == "utterance pairs" and pooled["n_target"] == 10
    assert pooled["eer"] == 0.0
