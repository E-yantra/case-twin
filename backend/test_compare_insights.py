"""Regression coverage for the match comparison endpoint and gateway request."""

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
from PIL import Image

import local_ai
import main


def _reader(current_read: str, historical_read: str):
    """Fake single-image reader: the white upload is current, the black asset historical."""
    def read(image, prompt, max_tokens=220):
        return [{"generated_text": current_read if image.getpixel((0, 0)) == (255, 255, 255) else historical_read}]
    return read


class CompareInsightsTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _image_bytes(color: str) -> bytes:
        output = io.BytesIO()
        Image.new("RGB", (8, 8), color).save(output, format="PNG")
        return output.getvalue()

    async def _compare(self, image_bytes: bytes, *, asset_id: str | None = "matched.png",
                       caption: str | None = None) -> httpx.Response:
        data = {
            "match_diagnosis": "pneumothorax",
            "match_payload": json.dumps({
                "presentation": {"hpi": "Historical cough"},
                "outcome": {"detail": "Recovered"},
            }),
        }
        if asset_id is not None:
            data["match_asset_id"] = asset_id
        if caption is not None:
            data["match_caption"] = caption
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/compare_insights",
                files={"original_image": ("upload.png", image_bytes, "image/png")},
                data=data,
            )

    async def _run(self, current_read, historical_read, text_replies, *, caption=None, localization=None):
        with tempfile.TemporaryDirectory() as directory:
            matched_image = Path(directory) / "matched.png"
            matched_image.write_bytes(self._image_bytes("black"))
            with patch.object(main, "asset_path", return_value=matched_image), \
                    patch.object(main, "query_medgemma_read", side_effect=_reader(current_read, historical_read)) as read, \
                    patch.object(main, "query_text", side_effect=text_replies) as text, \
                    patch.object(main, "query_medgemma_localization", return_value=localization) as localize:
                response = await self._compare(self._image_bytes("white"), caption=caption)
        return response, read, text, localize

    async def test_each_image_is_read_alone_without_the_historical_record(self):
        response, read, text, _ = await self._run(
            "The lungs are clear.", "There is a visible right pleural line.",
            ["The historical image shows a pleural line that the current one lacks.",
             '{"current_finding": null, "historical_finding": null}'])

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body.pop("trace")[0]["model"], local_ai.MEDGEMMA_COMPARISON_MODEL)
        self.assertEqual(body, {
            "insights_text": ("Current: The lungs are clear.\nHistorical: There is a visible right pleural line.\n"
                              "Visual comparison: The historical image shows a pleural line that the current one lacks."),
            "original_box": None,
            "match_box": None,
        })
        self.assertEqual(read.call_count, 2)
        colours = sorted(call.args[0].getpixel((0, 0)) for call in read.call_args_list)
        self.assertEqual(colours, [(0, 0, 0), (255, 255, 255)])
        for call in read.call_args_list:
            prompt = call.args[1]
            self.assertNotIn("Historical cough", prompt)
            self.assertNotIn("Recovered", prompt)
            self.assertNotIn("pneumothorax", prompt)
        self.assertIn("independent reads", text.call_args_list[0].args[0])

    async def test_genuinely_similar_images_get_a_real_comparison(self):
        read = "Large left pleural effusion obscuring the left hemidiaphragm with mediastinal shift to the right."
        response, _, _, _ = await self._run(read, read, ["Both show a large left pleural effusion.",
                                                         '{"current_finding": null, "historical_finding": null}'])
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("inconclusive", response.json()["insights_text"])
        self.assertIn("Both show a large left pleural effusion.", response.json()["insights_text"])

    async def test_preamble_is_removed_from_each_read(self):
        response, _, _, _ = await self._run("Here's a description of the findings:\nThe lungs are clear.",
                                            "Right lower lobe consolidation.",
                                            ["They differ.", '{"current_finding": null, "historical_finding": null}'])
        self.assertTrue(response.json()["insights_text"].startswith("Current: The lungs are clear."))

    async def test_localizes_only_the_visible_historical_finding(self):
        response, _, _, localize = await self._run(
            "The lungs are clear.", "There is a left pneumothorax.",
            ["Different pleural markings.", '{"current_finding": null, "historical_finding": "left pneumothorax"}'],
            localization=[{"generated_text": 'thought... Final Answer: ```json[{"box_2d":[110,450,770,870],"label":"left pneumothorax"}]```'}])

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["original_box"], None)
        self.assertEqual(response.json()["match_box"], [110, 450, 770, 870])
        localize.assert_called_once()
        self.assertEqual(localize.call_args.args[0].getpixel((0, 0)), (0, 0, 0))

    async def test_screenshot_pair_does_not_show_false_normal_read_or_effusion_boxes(self):
        caption = (
            'Chest X-ray shows a continuous diaphragm sign caused by mediastinal gas '
            'and Naclerio\'s V sign (red arrowheads).'
        )
        response, _, _, localize = await self._run(
            "The lungs are clear. There is no pleural effusion or pneumothorax.",
            "The lungs are clear and the mediastinum is unremarkable. There is no pleural effusion or pneumothorax.",
            ["Both images appear normal."], caption=caption)

        self.assertEqual(response.status_code, 200)
        self.assertIn("reported mediastinal gas", response.json()["insights_text"])
        self.assertIn("inconclusive", response.json()["insights_text"])
        self.assertNotIn("unremarkable", response.json()["insights_text"])
        self.assertIsNone(response.json()["original_box"])
        self.assertIsNone(response.json()["match_box"])
        localize.assert_not_called()

    async def test_classifier_cannot_box_finding_denied_by_narrative(self):
        response, _, text, localize = await self._run(
            "No pleural effusion or pneumothorax.", "No pleural effusion or pneumothorax.",
            ["Neither image shows pleural effusion.",
             '{"current_finding":"Pleural effusion","historical_finding":"Pleural effusion"}'])

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["original_box"])
        self.assertIsNone(response.json()["match_box"])
        self.assertEqual(text.call_count, 2)
        localize.assert_not_called()

    def test_resolved_caption_is_not_treated_as_visible_pneumothorax(self):
        self.assertIsNone(main._reported_finding("Chest X-ray revealed complete resolution of pneumothorax."))

    def test_localizer_label_must_match_the_finding_not_just_anatomy(self):
        reply = 'Final Answer: [{"box_2d":[100,200,500,600],"label":"pleural thickening"}]'
        self.assertIsNone(main._localization_box(reply, "pleural effusion"))

    async def test_malformed_localization_does_not_invent_an_overlay(self):
        response, _, _, _ = await self._run(
            "The lungs are clear.", "There is a left pneumothorax.",
            ["Different pleural markings.", '{"current_finding": null, "historical_finding": "left pneumothorax"}'],
            localization=[{"generated_text": 'Final Answer: [{"box_2d":[900,450,770,870],"label":"left pneumothorax"}]'}])

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["match_box"])

    async def test_missing_historical_image_returns_clear_error_without_model_call(self):
        current = self._image_bytes("white")
        with patch.object(main, "query_medgemma_read") as model:
            no_asset = await self._compare(current, asset_id=None)
            with patch.object(main, "asset_path", return_value=Path("/missing/matched.png")):
                missing_file = await self._compare(current)

        self.assertEqual(no_asset.status_code, 400)
        self.assertIn("historical match image is required", no_asset.json()["detail"])
        self.assertEqual(missing_file.status_code, 404)
        self.assertIn("Historical match image was not found", missing_file.json()["detail"])
        model.assert_not_called()

    async def test_model_failure_and_empty_response_return_retryable_errors(self):
        current = self._image_bytes("white")
        with tempfile.TemporaryDirectory() as directory:
            matched_image = Path(directory) / "matched.png"
            matched_image.write_bytes(self._image_bytes("black"))
            with patch.object(main, "asset_path", return_value=matched_image), patch.object(
                main, "query_medgemma_read", side_effect=RuntimeError("gateway unavailable")
            ):
                unavailable = await self._compare(current)
            with patch.object(main, "asset_path", return_value=matched_image), patch.object(
                main, "query_medgemma_read", return_value=[{"generated_text": "  "}]
            ):
                empty = await self._compare(current)

        self.assertEqual(unavailable.status_code, 502)
        self.assertIn("Please retry", unavailable.json()["detail"])
        self.assertEqual(empty.status_code, 502)
        self.assertIn("Please retry", empty.json()["detail"])

    async def test_rejects_same_patient_progression_claim(self):
        response, _, _, _ = await self._run(
            "Right lower lobe consolidation.", "Clear lungs.",
            ["The current image represents resolution of the historical case."])

        self.assertEqual(response.status_code, 502)
        self.assertIn("needs review", response.json()["detail"])


class GatewayComparisonTests(unittest.TestCase):
    def test_comparison_uses_27b_and_boxes_use_medgemma_15_by_default(self):
        self.assertEqual(local_ai.MEDGEMMA_COMPARISON_MODEL, "medgemma")
        self.assertEqual(local_ai.MEDGEMMA_LOCALIZATION_MODEL, "medgemma-1.5")
        with patch.object(local_ai, "query_local_model", return_value=[{"generated_text": "x"}]) as model:
            local_ai.query_medgemma_localization(Image.new("RGB", (8, 8)), prompt="Where?")
        self.assertEqual(model.call_args.kwargs["model"], "medgemma-1.5")

    def test_sends_current_then_historical_image_in_one_gateway_request(self):
        gateway_response = Mock()
        gateway_response.json.return_value = {
            "choices": [{"message": {"content": "Visible differences noted."}}]
        }
        with patch.object(local_ai, "GATEWAY_BASE_URL", "http://gateway.test/v1"), patch.object(
            local_ai, "_gateway_headers", return_value={"Authorization": "Bearer test"}
        ), patch.object(local_ai.httpx, "post", return_value=gateway_response) as post:
            result = local_ai.query_medgemma_comparison(
                Image.new("RGB", (8, 8), "white"),
                Image.new("RGB", (8, 8), "black"),
                prompt="Compare independent cases.",
            )

        self.assertEqual(result, [{"generated_text": "Visible differences noted."}])
        post.assert_called_once()
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], local_ai.MEDGEMMA_COMPARISON_MODEL)
        self.assertEqual(payload["temperature"], 0)
        content = payload["messages"][0]["content"]
        self.assertEqual([part["type"] for part in content], ["text", "image_url", "image_url"])
        images = [Image.open(io.BytesIO(base64.b64decode(part["image_url"]["url"].split(",", 1)[1])))
                  for part in content[1:]]
        self.assertEqual(images[0].getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(images[1].getpixel((0, 0)), (0, 0, 0))

    def test_single_image_read_uses_comparison_model(self):
        with patch.object(local_ai, "query_local_model", return_value=[{"generated_text": "x"}]) as model:
            local_ai.query_medgemma_read(Image.new("RGB", (8, 8)), "Describe")
        self.assertEqual(model.call_args.kwargs["model"], local_ai.MEDGEMMA_COMPARISON_MODEL)
        self.assertIsNotNone(model.call_args.kwargs["image"])

    def test_localization_sends_image_before_prompt(self):
        gateway_response = Mock()
        gateway_response.json.return_value = {"choices": [{"message": {"content": "Final Answer: []"}}]}
        with patch.object(local_ai, "GATEWAY_BASE_URL", "http://gateway.test/v1"), patch.object(
            local_ai, "_gateway_headers", return_value={"Authorization": "Bearer test"}
        ), patch.object(local_ai.httpx, "post", return_value=gateway_response) as post:
            local_ai.query_medgemma_localization(Image.new("RGB", (8, 8), "black"), "Locate finding")

        content = post.call_args.kwargs["json"]["messages"][0]["content"]
        self.assertEqual([item["type"] for item in content], ["image_url", "text"])


if __name__ == "__main__":
    unittest.main()
