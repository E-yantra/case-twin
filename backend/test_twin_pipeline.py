"""Regression coverage for extraction, hybrid search fusion and text-only MedGemma calls."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
from PIL import Image

import extraction
import local_ai
import main
import manifest
import qdrant_service


def _png(color: str = "white") -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format="PNG")
    return output.getvalue()


async def _post(path: str, **kwargs) -> httpx.Response:
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(path, **kwargs)


class ManifestTests(unittest.TestCase):
    def test_normalize_keeps_extra_fields_and_new_sections(self):
        profile = manifest.normalize_profile({"extra_fields": {"smoking_status": "smoker"},
                                              "management": {"treatments": "antibiotics"}})
        self.assertEqual(profile["extra_fields"], {"smoking_status": "smoker"})
        self.assertEqual(profile["management"]["treatments"], ["antibiotics"])
        self.assertIsNone(profile["summary"]["conclusion"])

    def test_case_document_uses_report_fields_not_outcome(self):
        profile = manifest.normalize_profile({
            "patient": {"age_years": 58, "sex": "male"},
            "assessment": {"diagnosis_primary": "lobar pneumonia"},
            "outcome": {"detail": "recovered fully"},
        })
        document = manifest.case_document(profile)
        self.assertIn("58-year-old male", document)
        self.assertIn("lobar pneumonia", document)
        self.assertNotIn("recovered", document)


class ExtractionTests(unittest.TestCase):
    def test_schema_constrained_gemma_call(self):
        reply = [{"generated_text": json.dumps({"patient": {"age_years": 40}})}]
        with patch.object(extraction, "query_local_model", return_value=reply) as model:
            fields, step = extraction.extract_profile("prompt")
        self.assertEqual(fields["patient"]["age_years"], 40)
        kwargs = model.call_args.kwargs
        self.assertIs(kwargs["json_schema"], extraction.PROFILE_SCHEMA)
        self.assertFalse(kwargs["thinking"])
        self.assertEqual(step["model"], local_ai.GEMMA_MODEL)

    def test_text_only_request_has_plain_string_content(self):
        response = Mock(status_code=200)
        response.raise_for_status.return_value = None
        response.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        with patch.object(local_ai, "GATEWAY_BASE_URL", "http://gw/v1"), \
                patch.object(local_ai, "GATEWAY_API_KEY", "k"), \
                patch.object(local_ai.httpx, "post", return_value=response) as post:
            local_ai.query_text("Explain atelectasis", history=[{"role": "user", "content": "hi"}])
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["messages"][-1], {"role": "user", "content": "Explain atelectasis"})
        self.assertEqual(payload["messages"][0]["content"], "hi")


class ExtractEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_gemma_profile_reported_with_trace_and_no_outcome(self):
        fields = {"patient": {"age_years": 61, "sex": "female"}, "outcome": {"detail": "died"},
                  "assessment": {"diagnosis_primary": "pneumonia"}}
        with patch.object(main, "extract_profile", return_value=(fields, {"model": "gemma-4", "task": "x", "ms": 1})):
            response = await _post("/extract", data={"notes": "61 year old woman with fever"})
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["method"], "gemma-4")
        self.assertEqual(body["profile"]["patient"]["age_years"], 61)
        self.assertIsNone(body["profile"]["outcome"]["detail"])
        self.assertEqual(body["trace"][-1]["model"], "gemma-4")

    def test_image_read_prompt_names_the_routed_image_type(self):
        self.assertIn("clinical photograph of a skin lesion", main._image_read_prompt("derm"))
        self.assertIn("chest X-ray", main._image_read_prompt("cxr"))
        self.assertEqual(main._image_read_prompt(None), main.IMAGE_READ_PROMPT)

    async def test_regex_fallback_is_labelled(self):
        with patch.object(main, "extract_profile", side_effect=ValueError("down")):
            response = await _post("/extract", data={"notes": "A 70-year-old man with COPD presenting with dyspnea."})
        body = response.json()
        self.assertEqual(body["method"], "regex-fallback")
        self.assertEqual(body["profile"]["patient"]["age_years"], 70)
        self.assertIn("COPD", body["profile"]["patient"]["comorbidities"])

    async def test_empty_input_is_rejected(self):
        response = await _post("/extract", data={"notes": "  "})
        self.assertEqual(response.status_code, 400)


class SearchFusionTests(unittest.TestCase):
    def _point(self, point_id, image, text, diagnosis):
        return SimpleNamespace(id=point_id, vector={"image": image, "text": text},
                               payload={"collection": "cxr", "profile": {"assessment": {"diagnosis_primary": diagnosis}}})

    def test_every_candidate_scored_on_every_channel_then_reranked(self):
        points = [self._point("a", [1.0, 0.0], [0.0, 1.0], "visual twin"),
                  self._point("b", [0.6, 0.8], [1.0, 0.0], "text twin")]
        client = Mock()
        client.query_points.side_effect = [SimpleNamespace(points=[SimpleNamespace(id="a")]),
                                           SimpleNamespace(points=[SimpleNamespace(id="b")])]
        client.retrieve.return_value = points
        reranker = Mock(return_value=[0.0, 0.0])
        with patch.object(qdrant_service, "_get_client", return_value=client), \
                patch.object(qdrant_service, "_calibration", {}):
            matches, trace = qdrant_service.search_similar(
                image_vector=[1.0, 0.0], text_vector=[1.0, 0.0], query_document="query", reranker=reranker)
        by_id = {match["id"]: match for match in matches}
        self.assertEqual(by_id["a"]["scores"]["text"], 0.0)
        self.assertEqual(by_id["b"]["scores"]["image"], 0.6)
        self.assertEqual(by_id["a"]["scores"]["rerank"], 0.5)
        # b: 0.5*0.6 + 0.3*1 + 0.2*0.5 = 0.70 beats a: 0.5*1 + 0 + 0.1 = 0.60
        self.assertEqual([match["id"] for match in matches], ["b", "a"])
        self.assertEqual(trace[-1]["model"], "bge-reranker-v2-m3")

    def test_calibration_maps_channel_range_to_unit_interval(self):
        with patch.object(qdrant_service, "_calibration", {"image": {"lo": 0.4, "hi": 0.8}}):
            self.assertAlmostEqual(qdrant_service.calibrated("image", 0.6), 0.5)
            self.assertEqual(qdrant_service.calibrated("image", 0.95), 1.0)
            self.assertEqual(qdrant_service.calibrated("text", 0.3), 0.3)

    def test_search_requires_image_or_case(self):
        with self.assertRaises(ValueError):
            qdrant_service.search_similar()


class SearchEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_rejects_search_with_nothing_to_match(self):
        response = await _post("/search", data={"profile": json.dumps({})})
        self.assertEqual(response.status_code, 400)

    async def test_text_only_search_uses_crossmodal_and_text_channels(self):
        profile = {"study": {"modality": "CXR"}, "findings": {"imaging_findings": ["right pleural effusion"]},
                   "assessment": {"diagnosis_primary": "parapneumonic effusion"}}
        with patch.object(main, "text_embeddings", return_value=[[1.0]]), \
                patch.object(main, "medsiglip_text_embedding", return_value=[1.0]) as crossmodal, \
                patch.object(main, "search_similar", return_value=([], [])) as search:
            response = await _post("/search", data={"profile": json.dumps(profile)})
        self.assertEqual(response.status_code, 200)
        crossmodal.assert_called_once()
        kwargs = search.call_args.kwargs
        self.assertIsNone(kwargs["image_vector"])
        self.assertEqual(kwargs["text_vector"], [1.0])


class RoutingTests(unittest.TestCase):
    def test_near_tie_with_distractor_routes_to_collection(self):
        ranked = [{"label": "a barium fluoroscopy study", "score": 7.4e-5}, {"label": "a chest X-ray radiograph", "score": 7.3e-5}]
        with patch.object(main, "medsiglip_classify", return_value=ranked):
            chosen, scores = main._route_collection(Image.new("RGB", (8, 8)))
        self.assertEqual(chosen, "cxr")
        self.assertEqual(scores[0]["collection"], "cxr")

    def test_clear_distractor_win_searches_all(self):
        ranked = [{"label": "an electrocardiogram tracing", "score": 9e-4}, {"label": "a chest X-ray radiograph", "score": 1e-4}]
        with patch.object(main, "medsiglip_classify", return_value=ranked):
            chosen, _ = main._route_collection(Image.new("RGB", (8, 8)))
        self.assertIsNone(chosen)


class CrossmodalTests(unittest.IsolatedAsyncioTestCase):
    def test_long_text_is_shortened_until_the_encoder_accepts_it(self):
        request = httpx.Request("POST", "http://m/v1/embed_text")
        too_long = httpx.HTTPStatusError("500", request=request, response=httpx.Response(500, request=request))
        sent = []

        def fake_post(path, payload):
            sent.append(payload["texts"][0])
            if len(sent) == 1:
                raise too_long
            return {"embeddings": [[3.0, 4.0]]}

        with patch.object(local_ai, "_medsiglip_post", side_effect=fake_post):
            vector = local_ai.medsiglip_text_embedding("consolidation " * 40)
        self.assertEqual(vector, [0.6, 0.8])
        self.assertLessEqual(len(sent[0]), 180)
        self.assertLess(len(sent[1]), len(sent[0]))

    async def test_search_continues_when_crossmodal_fails(self):
        profile = {"findings": {"imaging_findings": ["right lower lobe consolidation"]},
                   "assessment": {"diagnosis_primary": "pneumonia"}}
        with patch.object(main, "text_embeddings", return_value=[[1.0]]), \
                patch.object(main, "medsiglip_text_embedding", side_effect=RuntimeError("500")), \
                patch.object(main, "search_similar", return_value=([], [])) as search:
            response = await _post("/search", data={"profile": json.dumps(profile)})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(search.call_args.kwargs["crossmodal_vector"])
        self.assertIn("skipped", response.json()["trace"][-1]["task"])


class TextEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def test_explain_is_text_only_medgemma(self):
        with patch.object(main, "query_text", return_value="Fluid in the pleural space. More text. Extra. Four.") as model:
            response = await _post("/explain_selection", data={"selected_text": "pleural effusion", "audience": "patient"})
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("image", model.call_args.kwargs)
        self.assertIn("plain language", model.call_args.args[0])
        self.assertTrue(response.json()["explanation"].startswith("Fluid"))

    async def test_local_language_chains_medgemma_then_gemma(self):
        replies = ["Fluid has collected around the lung.", "फुफ्फुसाभोवती पाणी साचले आहे (pleural effusion)."]
        with patch.object(main, "query_text", side_effect=replies) as model:
            response = await _post("/explain_selection", data={"selected_text": "pleural effusion", "language": "mr"})
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["explanation"], replies[1])
        self.assertEqual(body["explanation_en"], replies[0])
        first, second = model.call_args_list
        self.assertEqual(first.kwargs["model"], main.MEDGEMMA_MODEL)
        self.assertIn("plain language", first.args[0])
        self.assertEqual(second.kwargs["model"], main.GEMMA_MODEL)
        self.assertIn("Marathi", second.args[0])
        self.assertIn(replies[0], second.args[0])
        self.assertEqual([step["model"] for step in body["trace"]], [main.MEDGEMMA_MODEL, main.GEMMA_MODEL])

    async def test_unknown_language_is_rejected(self):
        response = await _post("/explain_selection", data={"selected_text": "x", "language": "fr"})
        self.assertEqual(response.status_code, 400)

    async def test_chat_twin_passes_history_and_twin_outcome(self):
        twin = {"outcome": {"detail": "Recovered after drainage"}, "summary": {"conclusion": "Empyema"}}
        history = [{"role": "user", "content": "Earlier question"}, {"role": "assistant", "content": "Earlier answer"}]
        with patch.object(main, "query_text", return_value="**Drainage** helped.") as model:
            response = await _post("/chat_twin", data={"query": "What happened?", "case_text": "narrative",
                                                        "twin_profile": json.dumps(twin), "history": json.dumps(history)})
        self.assertEqual(response.json()["reply"], "**Drainage** helped.")
        self.assertIn("Recovered after drainage", model.call_args.args[0])
        self.assertEqual(model.call_args.kwargs["history"], history)

    async def test_non_cxr_comparison_skips_localization(self):
        with tempfile.TemporaryDirectory() as directory:
            matched = Path(directory) / "m.png"
            matched.write_bytes(_png("black"))
            with patch.object(main, "asset_path", return_value=matched), \
                    patch.object(main, "query_medgemma_read", return_value=[{"generated_text": "Silvery plaque."}]) as read, \
                    patch.object(main, "query_text", return_value="Both show similar plaques.") as text, \
                    patch.object(main, "query_medgemma_localization") as localize:
                response = await _post("/compare_insights", files={"original_image": ("u.png", _png(), "image/png")},
                                       data={"match_diagnosis": "psoriasis", "match_asset_id": "m.png",
                                             "match_collection": "derm"})
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["match_box"])
        self.assertIn("Dermatology photo", read.call_args.args[1])
        text.assert_called_once()
        localize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
