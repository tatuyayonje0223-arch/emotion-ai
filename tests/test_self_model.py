"""自己モデルのテスト。"""

from src.self_model.model import SelfModel, StableTraits, SituationModel
from src.schemas.affect_state import AffectState


class TestSelfModel:
    def test_default_creation(self):
        model = SelfModel()
        assert model.traits.openness == 0.5
        assert model.situation.context_type == "unknown"

    def test_assess_positive(self):
        model = SelfModel()
        state = AffectState(valence=0.6, trust=0.7)
        assessment = model.assess(state)
        assert "ポジティブ" in assessment.current_state_summary
        assert assessment.confidence_in_assessment > 0.3

    def test_assess_negative(self):
        model = SelfModel()
        state = AffectState(valence=-0.5, threat_load=0.6, uncertainty=0.7)
        assessment = model.assess(state)
        assert "ネガティブ" in assessment.current_state_summary
        assert "脅威" in assessment.current_state_summary
        assert len(assessment.active_goals) > 1

    def test_assess_with_high_fatigue(self):
        model = SelfModel()
        state = AffectState(fatigue=0.8)
        assessment = model.assess(state)
        assert "負荷" in assessment.current_state_summary
        assert any("負荷" in g or "簡潔" in g for g in assessment.active_goals)

    def test_trait_influence(self):
        model = SelfModel(StableTraits(threat_sensitivity=0.8, trust_tendency=0.2))
        state = AffectState()
        influence = model.get_trait_influence(state)
        assert influence["threat_sensitivity_mod"] > 0
        assert influence["trust_tendency_mod"] < 0

    def test_update_situation(self):
        model = SelfModel()
        model.update_situation(context_type="emotional_support", interaction_count=5)
        assert model.situation.context_type == "emotional_support"
        assert model.situation.interaction_count == 5

    def test_assessment_history(self):
        model = SelfModel()
        for i in range(5):
            model.assess(AffectState(valence=i * 0.1))
        assert model.last_assessment is not None
        assert len(model._assessment_history) == 5

    def test_known_limitations_always_present(self):
        model = SelfModel()
        assessment = model.assess(AffectState())
        assert len(assessment.known_limitations) >= 2
        assert any("主観的感情ではない" in l for l in assessment.known_limitations)

    def test_regulation_rationale(self):
        model = SelfModel()
        state = AffectState(regulation_mode="reappraisal")
        assessment = model.assess(state)
        assert "再評価" in assessment.regulation_rationale
