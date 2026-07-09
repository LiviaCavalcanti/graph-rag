"""
Offline tests for the declarative prompt registry (src.agents.prompt_registry).

Verifies that prompt variants are loaded from prompts/registry.yaml, expose
stable content fingerprints + metadata, drive graph-context serialization, and
stay in sync with the patcher variant table the harness depends on. No network.
"""

from __future__ import annotations

import pytest

from src.agents import patcher, prompt_registry

EXPECTED_VARIANTS = {"default", "default_v2", "graph", "graph_v2"}


def test_list_variants_matches_manifest():
    assert set(prompt_registry.list_variants()) == EXPECTED_VARIANTS


def test_load_variants_shape():
    variants = prompt_registry.load_variants()
    assert set(variants) == EXPECTED_VARIANTS
    for name, templates in variants.items():
        assert templates, f"{name} has no messages"
        for role, text in templates:
            assert role in {"system", "user", "assistant"}
            assert isinstance(text, str) and text


def test_patcher_variants_sourced_from_registry():
    # The harness (src.agents.harness.registry) reads patcher._VARIANTS directly,
    # so the manifest variants must seed it. Other tests may register extra
    # variants at runtime, so assert a subset rather than exact equality.
    loaded = prompt_registry.load_variants()
    assert set(loaded) == EXPECTED_VARIANTS
    assert EXPECTED_VARIANTS <= set(patcher._VARIANTS)
    assert patcher._VARIANTS["default"] == loaded["default"]


def test_fingerprint_is_stable_and_well_formed():
    fp1 = prompt_registry.variant_fingerprint("default")
    fp2 = prompt_registry.variant_fingerprint("default")
    assert fp1 == fp2
    assert len(fp1) == 12
    assert all(c in "0123456789abcdef" for c in fp1)


def test_distinct_variants_have_distinct_fingerprints():
    fps = {v: prompt_registry.variant_fingerprint(v) for v in EXPECTED_VARIANTS}
    assert len(set(fps.values())) == len(fps)


def test_needs_graph_context_flags():
    assert prompt_registry.needs_graph_context("graph") is True
    assert prompt_registry.needs_graph_context("graph_v2") is True
    assert prompt_registry.needs_graph_context("default") is False
    assert prompt_registry.needs_graph_context("default_v2") is False


def test_needs_graph_context_unknown_falls_back_to_name():
    # Runtime-registered variants absent from the manifest fall back to a heuristic.
    assert prompt_registry.needs_graph_context("graph_experimental") is True
    assert prompt_registry.needs_graph_context("seq_only_thing") is False


def test_variant_meta_fields():
    meta = prompt_registry.variant_meta("graph")
    assert meta["name"] == "graph"
    assert meta["needs_graph_context"] is True
    assert meta["n_messages"] == len(patcher._VARIANTS["graph"])
    assert meta["fingerprint"] == prompt_registry.variant_fingerprint("graph")
    assert isinstance(meta["files"], list) and meta["files"]


def test_variant_meta_unknown_raises():
    with pytest.raises(KeyError):
        prompt_registry.variant_meta("does_not_exist")


def test_fingerprint_matches_harness_registry():
    # prompt_registry and the harness registry must agree on fingerprints so a
    # run's recorded prompt hash matches what evolve/compare tooling reports.
    from src.agents.harness import registry as harness_registry

    for v in EXPECTED_VARIANTS:
        assert (
            prompt_registry.variant_fingerprint(v)
            == harness_registry.describe(v)["fingerprint"]
        )
