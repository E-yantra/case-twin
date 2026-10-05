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

    async def test_returns_text_and_null_boxes_for_two_distinct_cases(self):
        current = self._image_bytes("white")
        with tempfile.TemporaryDirectory() as directory:
            matched_image = Path(directory) / "matched.png"
            matched_image.write_bytes(self._image_bytes("black"))
            with patch.object(main, "asset_path", return_value=matched_image), patch.object(
                main, "query_medgemma_comparison",
                side_effect=[
                    [{"generated_text": "The current image is clear. The historical image has a visible pleural line."}],
                    [{"generated_text": '{"current_finding": null, "historical_finding": null}'}],
                ],
            ) as model:
                response = await self._compare(current)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "insights_text": "The current image is clear. The historical image has a visible pleural line.",
            "original_box": None,
            "match_box": None,
        })
        self.assertEqual(model.call_count, 2)
        args, kwargs = model.call_args_list[0]
        self.assertEqual(args[0].getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(args[1].getpixel((0, 0)), (0, 0, 0))
        self.assertIn("separate patients' cases", kwargs["prompt"])
        self.assertNotIn("Historical cough", kwargs["prompt"])
        self.assertNotIn("Recovered", kwargs["prompt"])
        self.assertNotIn("pneumothorax", kwargs["prompt"])

    async def test_localizes_only_the_visible_historical_finding(self):
        current = self._image_bytes("white")
        with tempfile.TemporaryDirectory() as directory:
            matched_image = Path(directory) / "matched.png"
            matched_image.write_bytes(self._image_bytes("black"))
            with patch.object(main, "asset_path", return_value=matched_image), patch.object(
                main, "query_medgemma_comparison", side_effect=[
                    [{"generated_text": "Current: clear. Historical: left pneumothorax. Visual comparison: different pleural markings."}],
                    [{"generated_text": '{"current_finding": null, "historical_finding": "left pneumothorax"}'}],
                ]
            ), patch.object(
                main, "query_medgemma_localization",
                return_value=[{"generated_text": 'thought... Final Answer: ```json[{"box_2d":[110,450,770,870],"label":"left pneumothorax"}]```'}],
            ) as localize:
                response = await self._compare(current)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["original_box"], None)
        self.assertEqual(response.json()["match_box"], [110, 450, 770, 870])
        localize.assert_called_once()
        self.assertEqual(localize.call_args.args[0].getpixel((0, 0)), (0, 0, 0))

    async def test_screenshot_pair_does_not_show_false_normal_read_or_effusion_boxes(self):
        current = self._image_bytes("white")
        caption = (
            'Chest X-ray shows a continuous diaphragm sign caused by mediastinal gas '
            'and Naclerio\'s V sign (red arrowheads).'
        )
        text = (
            'Current: The lungs are clear. There is no pleural effusion or pneumothorax.\n'
            'Historical: The lungs are clear and the mediastinum is unremarkable. '
            'There is no pleural effusion or pneumothorax.\n'
            'Visual comparison: Both images appear normal.'
        )
        with tempfile.TemporaryDirectory() as directory:
            matched_image = Path(directory) / "matched.png"
            matched_image.write_bytes(self._image_bytes("black"))
            with patch.object(main, "asset_path", return_value=matched_image), patch.object(
                main, "query_medgemma_comparison", return_value=[{"generated_text": text}]
            ) as model, patch.object(main, "query_medgemma_localization") as localize:
                response = await self._compare(current, caption=caption)

        self.assertEqual(response.status_code, 200)
        self.assertIn("reported mediastinal gas", response.json()["insights_text"])
        self.assertIn("inconclusive", response.json()["insights_text"])
        self.assertNotIn("unremarkable", response.json()["insights_text"])
        self.assertIsNone(response.json()["original_box"])
        self.assertIsNone(response.json()["match_box"])
        model.assert_called_once()
        localize.assert_not_called()

    async def test_classifier_cannot_box_finding_denied_by_narrative(self):
        current = self._image_bytes("white")
        with tempfile.TemporaryDirectory() as directory:
            matched_image = Path(directory) / "matched.png"
            matched_image.write_bytes(self._image_bytes("black"))
            with patch.object(main, "asset_path", return_value=matched_image), patch.object(
                main, "query_medgemma_comparison", side_effect=[
                    [{"generated_text": (
                        "Current: No pleural effusion or pneumothorax.\n"
                        "Historical: No pleural effusion or pneumothorax.\n"
                        "Visual comparison: Neither image shows pleural effusion."
                    )}],
                    [{"generated_text": '{"current_finding":"Pleural effusion","historical_finding":"Pleural effusion"}'}],
                ]
            ) as model, patch.object(main, "query_medgemma_localization") as localize:
                response = await self._compare(current)

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["original_box"])
        self.assertIsNone(response.json()["match_box"])
        self.assertEqual(model.call_count, 2)
        localize.assert_not_called()

    def test_resolved_caption_is_not_treated_as_visible_pneumothorax(self):
        self.assertIsNone(main._reported_finding("Chest X-ray revealed complete resolution of pneumothorax."))

    def test_localizer_label_must_match_the_finding_not_just_anatomy(self):
        reply = 'Final Answer: [{"box_2d":[100,200,500,600],"label":"pleural thickening"}]'
        self.assertIsNone(main._localization_box(reply, "pleural effusion"))

    async def test_malformed_localization_does_not_invent_an_overlay(self):
        current = self._image_bytes("white")
        with tempfile.TemporaryDirectory() as directory:
            matched_image = Path(directory) / "matched.png"
            matched_image.write_bytes(self._image_bytes("black"))
            with patch.object(main, "asset_path", return_value=matched_image), patch.object(
                main, "query_medgemma_comparison", side_effect=[
                    [{"generated_text": "Current: clear. Historical: left pneumothorax. Visual comparison: different pleural markings."}],
                    [{"generated_text": '{"current_finding": null, "historical_finding": "left pneumothorax"}'}],
                ]
            ), patch.object(
                main, "query_medgemma_localization",
                return_value=[{"generated_text": 'Final Answer: [{"box_2d":[900,450,770,870],"label":"left pneumothorax"}]'}],
            ):
                response = await self._compare(current)

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.json()["match_box"])

    async def test_missing_historical_image_returns_clear_error_without_model_call(self):
        current = self._image_bytes("white")
        with patch.object(main, "query_medgemma_comparison") as model:
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
                main, "query_medgemma_comparison", side_effect=RuntimeError("gateway unavailable")
            ) as model:
                unavailable = await self._compare(current)
                model.assert_called_once()
            with patch.object(main, "asset_path", return_value=matched_image), patch.object(
                main, "query_medgemma_comparison", return_value=[{"generated_text": "  "}]
            ):
                empty = await self._compare(current)

        self.assertEqual(unavailable.status_code, 502)
        self.assertIn("Please retry", unavailable.json()["detail"])
        self.assertEqual(empty.status_code, 502)
        self.assertIn("Please retry", empty.json()["detail"])

    async def test_rejects_same_patient_progression_claim(self):
        current = self._image_bytes("white")
        with tempfile.TemporaryDirectory() as directory:
            matched_image = Path(directory) / "matched.png"
            matched_image.write_bytes(self._image_bytes("black"))
            with patch.object(main, "asset_path", return_value=matched_image), patch.object(
                main, "query_medgemma_comparison",
                return_value=[{"generated_text": "The current image represents resolution of the historical case."}],
            ):
                response = await self._compare(current)

        self.assertEqual(response.status_code, 502)
        self.assertIn("needs review", response.json()["detail"])


class GatewayComparisonTests(unittest.TestCase):
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
        self.assertEqual(payload["model"], "medgemma-1.5")
        self.assertEqual(payload["temperature"], 0)
        content = payload["messages"][0]["content"]
        self.assertEqual([part["type"] for part in content], ["text", "image_url", "image_url"])
        images = [Image.open(io.BytesIO(base64.b64decode(part["image_url"]["url"].split(",", 1)[1])))
                  for part in content[1:]]
        self.assertEqual(images[0].getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(images[1].getpixel((0, 0)), (0, 0, 0))

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
