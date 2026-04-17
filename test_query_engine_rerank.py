import unittest
import numpy as np
import tempfile
from pathlib import Path

from memory.graph_db import EventNode, SessionNode
from memory.graph_db import LinkType
from memory.query_engine import QueryEngine
from memory.vector_db import FAISS_AVAILABLE, FAISSVectorDB


class DummyTRG:
    def __init__(self):
        self.graph_db = type(
            "GraphDB",
            (),
            {
                "links": {},
                "nodes": {},
                "get_neighbors": lambda self_ref, node_id: [],
                "get_node": lambda self_ref, node_id: self_ref.nodes.get(node_id),
            },
        )()
        self.encoder = type("Encoder", (), {"encode": lambda self_ref, _: np.array([1.0, 0.0], dtype=np.float32)})()


def make_node(node_id: str, text: str, similarity: float = 0.0) -> EventNode:
    node = EventNode(node_id=node_id, content_narrative=text, attributes={})
    node.similarity_score = similarity
    return node


class QueryEngineRerankTests(unittest.TestCase):
    def test_coerce_embedding_parses_string_payload(self):
        vector = QueryEngine._coerce_embedding("[1.0 2.0 3.5]")

        self.assertIsNotNone(vector)
        self.assertEqual((3,), vector.shape)
        self.assertAlmostEqual(3.5, float(vector[2]), places=5)

    @unittest.skipUnless(FAISS_AVAILABLE, "FAISS not available")
    def test_faiss_vector_db_supports_legacy_single_file_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            legacy_path = Path(tmpdir) / "vectors"
            legacy_path.write_text("{}", encoding="utf-8")

            db = FAISSVectorDB(dimension=2)
            db.add_vector("n1", np.array([1.0, 0.0], dtype=np.float32), {"kind": "test"})
            db.save(str(legacy_path))

            reloaded = FAISSVectorDB(dimension=2)
            reloaded.load(str(legacy_path))

            self.assertTrue(legacy_path.is_file())
            self.assertIsNotNone(reloaded.get_vector("n1"))

    def test_gap_driven_expansion_adds_temporal_evidence_when_missing(self):
        trg = DummyTRG()
        anchor = make_node("n1", "Caroline adopted a dog.", similarity=0.8)
        temporal_neighbor = make_node("n2", "This happened in 2022.", similarity=0.2)
        trg.graph_db.nodes[anchor.node_id] = anchor
        trg.graph_db.nodes[temporal_neighbor.node_id] = temporal_neighbor

        link = type(
            "Link",
            (),
            {
                "link_type": LinkType.TEMPORAL,
                "properties": {"sub_type": "PRECEDES"},
            },
        )()
        trg.graph_db.get_neighbors = lambda node_id: [(temporal_neighbor, link)] if node_id == "n1" else []

        engine = QueryEngine(trg, node_index={})
        profile = engine.build_query_profile("When did Caroline adopt the dog in 2022?")

        expanded = engine._gap_driven_evidence_expansion(
            question="When did Caroline adopt the dog in 2022?",
            nodes=[anchor],
            profile=profile,
            top_k=5,
        )

        expanded_ids = [node.node_id for node in expanded]
        self.assertIn("n2", expanded_ids)

    def test_gap_driven_expansion_adds_actor_evidence_when_missing(self):
        trg = DummyTRG()
        anchor = make_node("n1", "Someone adopted a dog.", similarity=0.8)
        actor_neighbor = make_node("n2", "I adopted a dog.", similarity=0.2)
        actor_neighbor.attributes["speaker"] = "caroline"
        trg.graph_db.nodes[anchor.node_id] = anchor
        trg.graph_db.nodes[actor_neighbor.node_id] = actor_neighbor

        link = type(
            "Link",
            (),
            {
                "link_type": LinkType.SEMANTIC,
                "properties": {"sub_type": "SAME_ENTITY"},
            },
        )()
        trg.graph_db.get_neighbors = lambda node_id: [(actor_neighbor, link)] if node_id == "n1" else []

        engine = QueryEngine(trg, node_index={})
        profile = engine.build_query_profile("What did Caroline adopt?")

        expanded = engine._gap_driven_evidence_expansion(
            question="What did Caroline adopt?",
            nodes=[anchor],
            profile=profile,
            top_k=5,
        )

        expanded_ids = [node.node_id for node in expanded]
        self.assertIn("n2", expanded_ids)

    def test_sparse_retrieval_uses_index_without_full_scan(self):
        trg = DummyTRG()
        node = make_node("n1", "Caroline adopted a dog and shared updates.")
        trg.graph_db.nodes[node.node_id] = node
        node_index = {
            "caroline": {"n1"},
            "adopted": {"n1"},
            "dog": {"n1"},
            "caroline adopted": {"n1"},
        }

        engine = QueryEngine(trg, node_index=node_index)
        results = engine._sparse_retrieval("What did Caroline adopt?", top_k=5)

        self.assertGreaterEqual(len(results), 1)
        self.assertEqual("n1", results[0].node_id)

    def test_full_scan_only_used_when_recall_is_low(self):
        engine = QueryEngine(DummyTRG(), node_index={})
        profile = engine.build_query_profile("Why did Caroline adopt the dog?")

        self.assertFalse(
            engine._should_use_full_scan(
                ranked_lists=[[object()] * 6, [object()] * 6, [object()] * 6],
                candidate_count=18,
                top_k=5,
                profile=profile,
            )
        )
        self.assertTrue(
            engine._should_use_full_scan(
                ranked_lists=[[object()] * 2],
                candidate_count=3,
                top_k=5,
                profile=profile,
            )
        )

    def test_identify_target_sessions_uses_session_summaries(self):
        trg = DummyTRG()
        session1 = SessionNode(node_id="s1", session_id=1, summary="Caroline adopted a dog and shared updates.")
        session2 = SessionNode(node_id="s2", session_id=2, summary="Melanie discussed painting projects.")
        trg.graph_db.nodes[session1.node_id] = session1
        trg.graph_db.nodes[session2.node_id] = session2

        engine = QueryEngine(trg, node_index={})
        sessions = engine._identify_target_sessions("What did Caroline adopt?", nodes=[])

        self.assertEqual([1], sessions)

    def test_identify_target_sessions_uses_candidate_event_support(self):
        trg = DummyTRG()
        session1 = SessionNode(node_id="s1", session_id=1, summary="General catch-up session.")
        session2 = SessionNode(node_id="s2", session_id=2, summary="Another general session.")
        trg.graph_db.nodes[session1.node_id] = session1
        trg.graph_db.nodes[session2.node_id] = session2

        event1 = make_node("n1", "Caroline adopted a dog.", similarity=0.9)
        event1.attributes["session_id"] = 2
        event2 = make_node("n2", "Unrelated event.", similarity=0.1)
        event2.attributes["session_id"] = 1

        engine = QueryEngine(trg, node_index={})
        sessions = engine._identify_target_sessions("What did Caroline adopt?", nodes=[event1, event2])

        self.assertGreaterEqual(len(sessions), 1)
        self.assertEqual(2, sessions[0])

    def test_query_profile_unifies_type_and_intent(self):
        engine = QueryEngine(DummyTRG(), node_index={})

        profile = engine.build_query_profile("When did Caroline adopt the dog in 2022?")

        self.assertEqual("temporal", profile.primary_type)
        self.assertGreater(profile.temporal_focus, 0.45)
        self.assertGreaterEqual(profile.entity_focus, 0.45)
        self.assertEqual("WHEN", engine.detect_query_intent("When did Caroline adopt the dog in 2022?"))
        self.assertEqual("temporal", engine.detect_query_type("When did Caroline adopt the dog in 2022?"))

    def test_rerank_filters_by_person_name_before_scoring(self):
        engine = QueryEngine(DummyTRG(), node_index={})
        relevant = make_node("n1", "Caroline adopted a dog and shared the news.", similarity=0.1)
        distractor = make_node("n2", "Melanie adopted a dog and shared the news.", similarity=0.9)

        result = engine._rerank_and_filter(
            nodes=[relevant, distractor],
            question="What did Caroline adopt?",
            top_k=1,
            query_type="activity",
        )

        self.assertEqual(1, len(result))
        self.assertEqual("n1", result[0].node_id)

    def test_actor_consistency_gate_removes_conflicting_speaker_node(self):
        engine = QueryEngine(DummyTRG(), node_index={})
        relevant = make_node("n1", "I adopted a dog.", similarity=0.2)
        relevant.attributes["speaker"] = "caroline"
        distractor = make_node("n2", "I adopted a dog.", similarity=0.9)
        distractor.attributes["speaker"] = "melanie"

        result = engine._rerank_and_filter(
            nodes=[relevant, distractor],
            question="What did Caroline adopt?",
            top_k=2,
            query_type="activity",
        )

        result_ids = [node.node_id for node in result]
        self.assertIn("n1", result_ids)
        self.assertNotIn("n2", result_ids)

    def test_actor_consistency_gate_keeps_matching_entity_node(self):
        engine = QueryEngine(DummyTRG(), node_index={})
        node = make_node("n1", "Caroline adopted a dog and shared the news.", similarity=0.8)
        node.attributes["entities"] = ["Caroline"]

        result = engine._rerank_and_filter(
            nodes=[node],
            question="What did Caroline adopt?",
            top_k=1,
            query_type="activity",
        )

        self.assertEqual(1, len(result))
        self.assertEqual("n1", result[0].node_id)

    def test_answer_turn_first_detects_category5_style_prompt(self):
        engine = QueryEngine(DummyTRG(), node_index={})
        profile = engine.build_query_profile("What does Caroline say running has been great for?")

        self.assertTrue(engine._is_answer_turn_first_query("What does Caroline say running has been great for?", profile))

    def test_answer_turn_first_prefers_other_speaker_answer(self):
        trg = DummyTRG()
        question_turn = make_node("n1", "What got you into running?")
        question_turn.attributes["speaker"] = "caroline"
        question_turn.attributes["session_id"] = 7
        question_turn.attributes["original_text"] = "[Caroline]: What got you into running?"

        answer_turn = make_node("n2", "This has been great for my mental health.")
        answer_turn.attributes["speaker"] = "melanie"
        answer_turn.attributes["session_id"] = 7
        answer_turn.attributes["original_text"] = "[Melanie]: Thanks, Caroline! This has been great for my mental health."

        response_link = type(
            "Link",
            (),
            {
                "link_type": LinkType.SEMANTIC,
                "properties": {"sub_type": "RESPONSE_TO"},
                "source_node_id": "n2",
                "target_node_id": "n1",
            },
        )()

        trg.graph_db.nodes["n1"] = question_turn
        trg.graph_db.nodes["n2"] = answer_turn
        trg.graph_db.get_neighbors = lambda node_id: ([(question_turn, response_link)] if node_id == "n2" else [])

        engine = QueryEngine(trg, node_index={})
        result = engine._rerank_and_filter(
            nodes=[question_turn, answer_turn],
            question="What does Caroline say running has been great for?",
            top_k=1,
            query_type="activity",
        )

        self.assertEqual(1, len(result))
        self.assertEqual("n2", result[0].node_id)

    def test_promote_same_session_answer_turns_boosts_other_speaker_answer(self):
        trg = DummyTRG()
        question_turn = make_node("n1", "What got you into running?", similarity=0.9)
        question_turn.attributes["speaker"] = "caroline"
        question_turn.attributes["session_id"] = 7
        question_turn.attributes["original_text"] = "[Caroline]: What got you into running?"

        answer_turn = make_node("n2", "This has been great for my mental health.", similarity=0.2)
        answer_turn.attributes["speaker"] = "melanie"
        answer_turn.attributes["session_id"] = 7
        answer_turn.attributes["original_text"] = "[Melanie]: Thanks, Caroline! This has been great for my mental health."

        response_link = type(
            "Link",
            (),
            {
                "link_type": LinkType.SEMANTIC,
                "properties": {"sub_type": "RESPONSE_TO"},
                "source_node_id": "n2",
                "target_node_id": "n1",
            },
        )()

        trg.graph_db.nodes["n1"] = question_turn
        trg.graph_db.nodes["n2"] = answer_turn
        trg.graph_db.get_neighbors = lambda node_id: ([(question_turn, response_link)] if node_id == "n2" else [])

        engine = QueryEngine(trg, node_index={})
        profile = engine.build_query_profile("What does Caroline say running has been great for?")
        promoted = engine._promote_same_session_answer_turns(
            question="What does Caroline say running has been great for?",
            candidates=[question_turn, answer_turn],
            profile=profile,
        )

        self.assertEqual("n2", promoted[0].node_id)

    def test_query_evidence_constraints_extract_subject_action_and_object(self):
        engine = QueryEngine(DummyTRG(), node_index={})

        constraints = engine._extract_query_evidence_constraints("What did Caroline adopt from the shelter?")

        self.assertEqual("caroline", constraints["subject"])
        self.assertIn("adopt", constraints["action_terms"])
        self.assertIn("shelter", constraints["object_terms"])

    def test_rerank_filters_by_temporal_constraint_before_scoring(self):
        engine = QueryEngine(DummyTRG(), node_index={})
        relevant = make_node("n1", "In 2022 Caroline started counseling.", similarity=0.1)
        distractor = make_node("n2", "In 2021 Caroline started counseling.", similarity=0.9)

        result = engine._rerank_and_filter(
            nodes=[relevant, distractor],
            question="When did Caroline start counseling in 2022?",
            top_k=1,
            query_type="temporal",
        )

        self.assertEqual(1, len(result))
        self.assertEqual("n1", result[0].node_id)

    def test_rerank_falls_back_when_filter_is_too_strict(self):
        engine = QueryEngine(DummyTRG(), node_index={})
        fallback = make_node("n1", "A completely unrelated but high-similarity node.", similarity=0.9)
        filtered_out = make_node("n2", "Another unrelated node.", similarity=0.1)

        result = engine._rerank_and_filter(
            nodes=[fallback, filtered_out],
            question="What did Caroline adopt?",
            top_k=4,
            query_type="activity",
        )

        result_ids = [node.node_id for node in result]
        self.assertIn("n1", result_ids)
        self.assertIn("n2", result_ids)

    def test_query_profile_marks_actor_consistency_for_action_questions(self):
        engine = QueryEngine(DummyTRG(), node_index={})

        profile = engine.build_query_profile("What did Caroline adopt?")

        self.assertTrue(profile.needs_actor_consistency)
        self.assertGreater(profile.action_focus, 0.45)

    def test_probabilistic_beam_search_accepts_query_profile(self):
        trg = DummyTRG()
        node = make_node("n1", "Caroline adopted a dog.", similarity=0.5)
        node.embedding_vector = [1.0, 0.0]
        trg.graph_db.nodes[node.node_id] = node

        engine = QueryEngine(trg, node_index={})
        profile = engine.build_query_profile("Why did Caroline adopt the dog?")

        results = engine._probabilistic_beam_search(
            anchor_nodes=[node],
            question="Why did Caroline adopt the dog?",
            profile=profile,
        )

        self.assertEqual(1, len(results))
        self.assertEqual("n1", results[0][0].node_id)
        self.assertGreater(results[0][1], 0.0)

    def test_probabilistic_beam_search_handles_string_embedding_payload(self):
        trg = DummyTRG()
        anchor = make_node("n1", "Anchor node.", similarity=0.5)
        anchor.embedding_vector = [1.0, 0.0]
        neighbor = make_node("n2", "Neighbor node.", similarity=0.2)
        neighbor.embedding_vector = "[1.0 0.0]"
        trg.graph_db.nodes[anchor.node_id] = anchor
        trg.graph_db.nodes[neighbor.node_id] = neighbor

        link = type(
            "Link",
            (),
            {
                "link_type": LinkType.SEMANTIC,
                "properties": {"sub_type": "RELATED_TO"},
            },
        )()
        trg.graph_db.get_neighbors = lambda node_id: [(neighbor, link)] if node_id == "n1" else []

        engine = QueryEngine(trg, node_index={})
        profile = engine.build_query_profile("What is related to Caroline?")

        results = engine._probabilistic_beam_search(
            anchor_nodes=[anchor],
            question="What is related to Caroline?",
            profile=profile,
        )

        result_ids = [node.node_id for node, _ in results]
        self.assertIn("n2", result_ids)


if __name__ == "__main__":
    unittest.main()
