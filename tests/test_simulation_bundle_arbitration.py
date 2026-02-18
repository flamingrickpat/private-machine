from types import SimpleNamespace

from pm.csm.csm import CSMItem, CSMManager, CSMState
from pm.data_structures import Feature, FeatureType
from pm.ghosts.procedures.arbitration import ArbitrationProc, arbitrate_sim_bundle
from pm.ghosts.procedures.simulations import _build_sim_bundle
from pm.ghosts.schemas import SimBundle, SimLensOutcome, SimulationCodeletResult


def _mk_ms(interlocus: float, ego: float, aperture: float, energy: float, processing: float):
    return SimpleNamespace(
        state_cognition=SimpleNamespace(
            interlocus=interlocus,
            ego_strength=ego,
            mental_aperture=aperture,
        ),
        state_needs=SimpleNamespace(
            energy_stability=energy,
            processing_power=processing,
        ),
    )


def test_build_sim_bundle_produces_ranked_outcomes_and_consolidated_lists():
    codelets = [
        SimulationCodeletResult(
            scenario="consequence",
            predicted_outcome="clear support and better alignment",
            risk=0.2,
            benefit=0.8,
            emotional_forecast="calm",
        ),
        SimulationCodeletResult(
            scenario="social",
            predicted_outcome="reduced conflict and safer interaction",
            risk=0.3,
            benefit=0.7,
            emotional_forecast="positive",
        ),
    ]

    bundle = _build_sim_bundle(
        tick=8,
        context_text="Need to respond clearly and safely.",
        proposed_action="Offer concise plan and validate concerns.",
        core_outputs={
            "world": "Likely improves coordination and trust (positive)",
            "self": "Should reduce internal tension and keep coherence (positive)",
            "meta": "Stays consistent with guardrails and persona tone (neutral)",
        },
        codelet_results=codelets,
    )

    assert isinstance(bundle, SimBundle)
    assert len(bundle.outcomes) == 3
    assert bundle.winner_lens in {"world", "self", "meta"}
    assert bundle.consolidated_attractors
    assert bundle.dissonance >= 0.0
    assert bundle.dissonance <= 1.0


def test_arbitrate_sim_bundle_prefers_world_when_external_weighted():
    bundle = SimBundle(
        tick=20,
        context_excerpt="context",
        proposed_action="action",
        outcomes=[
            SimLensOutcome(
                lens="world",
                summary="world outcome",
                polarity="positive",
                confidence=0.75,
                utility=0.78,
                attractors=["clarity", "safety"],
                repellers=["conflict"],
                evidence=["e1"],
            ),
            SimLensOutcome(
                lens="self",
                summary="self outcome",
                polarity="positive",
                confidence=0.85,
                utility=0.62,
                attractors=["coherence"],
                repellers=["fatigue"],
                evidence=["e2"],
            ),
            SimLensOutcome(
                lens="meta",
                summary="meta outcome",
                polarity="neutral",
                confidence=0.66,
                utility=0.57,
                attractors=["guardrails"],
                repellers=["drift"],
                evidence=["e3"],
            ),
        ],
    )

    ghost = SimpleNamespace(
        current_state=SimpleNamespace(
            latent_mental_state=_mk_ms(
                interlocus=0.8,
                ego=0.6,
                aperture=0.2,
                energy=0.8,
                processing=0.8,
            )
        )
    )
    decision = arbitrate_sim_bundle(bundle, ghost=ghost)
    assert decision.winner_lens == "world"
    assert decision.runner_up_lens in {"self", "meta"}
    assert decision.directive
    assert 0.0 <= decision.stochasticity <= 1.0


def test_arbitration_proc_sets_directive_and_modulates_csm():
    class _Ghost:
        def __init__(self):
            self.current_tick_id = 33
            self.current_state = SimpleNamespace(
                latent_mental_state=_mk_ms(
                    interlocus=-0.8,
                    ego=0.7,
                    aperture=0.5,
                    energy=0.3,
                    processing=0.3,
                )
            )
            self._next = 100
            self.all_knoxels = {}
            self.all_features = []

            f1 = Feature(
                id=1,
                tick_id=30,
                content="Need more coherence and calm in response",
                feature_type=FeatureType.Thought,
                source="test",
                interlocus=-1,
                causal=False,
            )
            f2 = Feature(
                id=2,
                tick_id=30,
                content="Risk of conflict escalation if too sharp",
                feature_type=FeatureType.Thought,
                source="test",
                interlocus=-1,
                causal=False,
            )
            self.all_knoxels = {1: f1, 2: f2}
            self.all_features = [f1, f2]
            self.csm_manager = CSMManager(
                self,
                state=CSMState(
                    csm_item_states={
                        1: CSMItem(knoxel_id=1, first_tick=30, last_tick=30, activation=0.50, peak_activation=0.50),
                        2: CSMItem(knoxel_id=2, first_tick=30, last_tick=30, activation=0.50, peak_activation=0.50),
                    }
                ),
            )

            self.simulation_bundle_last_model = SimBundle(
                tick=33,
                context_excerpt="ctx",
                proposed_action="act",
                outcomes=[
                    SimLensOutcome(
                        lens="self",
                        summary="internal coherence improves",
                        polarity="positive",
                        confidence=0.8,
                        utility=0.85,
                        attractors=["coherence", "calm"],
                        repellers=["conflict"],
                        evidence=["e1"],
                    ),
                    SimLensOutcome(
                        lens="world",
                        summary="world outcome",
                        polarity="neutral",
                        confidence=0.6,
                        utility=0.55,
                        attractors=["clarity"],
                        repellers=["ambiguity"],
                        evidence=["e2"],
                    ),
                ],
            )

        def add_knoxel(self, knoxel):
            if getattr(knoxel, "id", -1) == -1:
                knoxel.id = self._next
                self._next += 1
            self.all_knoxels[knoxel.id] = knoxel
            if isinstance(knoxel, Feature):
                self.all_features.append(knoxel)
            return knoxel.id

        def get_knoxel_by_id(self, kid):
            return self.all_knoxels.get(kid)

    ghost = _Ghost()
    before_1 = ghost.csm_manager.state.csm_item_states[1].activation
    before_2 = ghost.csm_manager.state.csm_item_states[2].activation

    ArbitrationProc.run(ghost)

    assert ghost.ego_directive
    assert ghost.ego_decision_last["winner_lens"] == "self"
    assert ghost.csm_manager.state.csm_item_states[1].activation > before_1
    assert ghost.csm_manager.state.csm_item_states[2].activation < before_2
    assert any(f.source == "ArbitrationProc" for f in ghost.all_features)
