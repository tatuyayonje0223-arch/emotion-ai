"""データ駆動モジュールのテスト。"""

import tempfile
from pathlib import Path

import pytest

from src.data_driven.circuit_spec import CircuitSpec, PopulationSpec, ConnectionSpec, InputSpec
from src.data_driven.allen_connectivity import (
    build_literature_matrix, save_matrix, load_matrix, ALLEN_STRUCTURE_IDS,
)
from src.data_driven.build_fear_spec import build_fear_circuit_spec
from src.data_driven.spec_to_brian2 import build_and_run


class TestCircuitSpec:
    def test_create_minimal(self):
        spec = CircuitSpec(
            name="test",
            populations=[PopulationSpec(name="exc", cell_type="RS", n=10, region="test")],
            connections=[],
        )
        assert spec.total_neurons == 10

    def test_yaml_roundtrip(self):
        spec = CircuitSpec(
            name="roundtrip_test",
            populations=[
                PopulationSpec(name="a", cell_type="RS", n=5, region="X"),
                PopulationSpec(name="b", cell_type="PV", n=3, region="Y"),
            ],
            connections=[
                ConnectionSpec(source="a", target="b", probability=0.3, weight_mean=2.0),
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_spec.yaml"
            spec.to_yaml(path)
            loaded = CircuitSpec.from_yaml(path)
            assert loaded.name == "roundtrip_test"
            assert loaded.total_neurons == 8
            assert len(loaded.connections) == 1

    def test_get_connections(self):
        spec = CircuitSpec(
            name="test",
            populations=[
                PopulationSpec(name="a", cell_type="RS", n=5, region="X"),
                PopulationSpec(name="b", cell_type="RS", n=5, region="Y"),
            ],
            connections=[
                ConnectionSpec(source="a", target="b", probability=0.3),
                ConnectionSpec(source="b", target="a", probability=0.2),
            ],
        )
        assert len(spec.get_connections_to("b")) == 1
        assert len(spec.get_connections_from("a")) == 1


class TestAllenConnectivity:
    def test_build_literature_matrix(self):
        matrix = build_literature_matrix()
        assert len(matrix.regions) > 5
        assert len(matrix.projections) > 20

    def test_get_projection(self):
        matrix = build_literature_matrix()
        proj = matrix.get_projection("BLA", "NAc")
        assert proj is not None
        assert proj.normalized_projection_volume > 0
        assert proj.n_experiments > 0

    def test_nonexistent_projection(self):
        matrix = build_literature_matrix()
        proj = matrix.get_projection("NONEXISTENT", "ALSO_NOT")
        assert proj is None

    def test_to_weight(self):
        matrix = build_literature_matrix()
        w = matrix.to_weight("VTA", "NAc", scale=10.0)
        assert w > 0  # VTA→NAcは強い投射

    def test_to_probability(self):
        matrix = build_literature_matrix()
        p = matrix.to_probability("CeA", "PAG", max_prob=0.5)
        assert 0 < p <= 0.5

    def test_save_load_roundtrip(self):
        matrix = build_literature_matrix()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "matrix.json"
            save_matrix(matrix, path)
            loaded = load_matrix(path)
            assert len(loaded.projections) == len(matrix.projections)
            assert loaded.regions == matrix.regions

    def test_structure_ids(self):
        assert ALLEN_STRUCTURE_IDS["BLA"] == 295
        assert ALLEN_STRUCTURE_IDS["VTA"] == 749
        assert len(ALLEN_STRUCTURE_IDS) >= 12


class TestBuildFearSpec:
    def test_builds_valid_spec(self):
        spec = build_fear_circuit_spec()
        assert spec.name == "fear_circuit_data_driven"
        assert spec.total_neurons > 100
        assert len(spec.connections) > 10
        assert len(spec.populations) == 11

    def test_data_driven_connections_exist(self):
        spec = build_fear_circuit_spec()
        data_conns = [c for c in spec.connections if "allen" in c.data_source]
        assert len(data_conns) > 0

    def test_confidence_tracked(self):
        spec = build_fear_circuit_spec()
        for conn in spec.connections:
            assert 0 <= conn.confidence <= 1

    def test_scale_factor(self):
        small = build_fear_circuit_spec(scale=0.5)
        large = build_fear_circuit_spec(scale=2.0)
        assert large.total_neurons > small.total_neurons

    def test_yaml_export(self):
        spec = build_fear_circuit_spec()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fear_spec.yaml"
            spec.to_yaml(path)
            loaded = CircuitSpec.from_yaml(path)
            assert loaded.total_neurons == spec.total_neurons


class TestSpecToBrian2:
    def test_minimal_circuit(self):
        spec = CircuitSpec(
            name="minimal",
            populations=[
                PopulationSpec(name="exc", cell_type="RS", n=10, region="test"),
                PopulationSpec(name="inh", cell_type="PV", n=5, region="test"),
            ],
            connections=[
                ConnectionSpec(source="exc", target="inh", probability=0.3, weight_mean=3.0),
                ConnectionSpec(source="inh", target="exc", probability=0.4, weight_mean=4.0,
                               conn_type="inhibitory"),
            ],
            inputs=[
                InputSpec(name="stim", target_population="exc", amplitude=10.0,
                          onset_ms=20.0, duration_ms=50.0),
            ],
            simulation={"dt_ms": 0.5, "duration_ms": 100.0, "background_noise": 3.0},
        )
        result = build_and_run(spec)
        assert result.total_spikes >= 0
        assert "exc" in result.population_rates
        assert "inh" in result.population_rates
        assert result.population_rates["exc"] >= 0

    def test_fear_spec_runs(self):
        """データ駆動版恐怖回路がBrian2で実行できること。"""
        spec = build_fear_circuit_spec(scale=0.5)
        result = build_and_run(spec)
        assert result.total_spikes > 0
        assert "la_exc" in result.population_rates
        assert "cem" in result.population_rates
        assert "bnst" in result.population_rates

    def test_all_populations_have_rates(self):
        spec = build_fear_circuit_spec(scale=0.5)
        result = build_and_run(spec)
        for pop in spec.populations:
            assert pop.name in result.population_rates
            assert result.population_rates[pop.name] >= 0
