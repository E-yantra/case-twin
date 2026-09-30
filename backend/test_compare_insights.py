"""Regression coverage for the match comparison endpoint."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from PIL import Image

import main


class CompareInsightsTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _image_bytes() -> bytes:
        output = io.BytesIO()
        Image.new("RGB", (8, 8), "white").save(output, format="PNG")
        return output.getvalue()

    async def _compare(self, image_bytes: bytes) -> httpx.Response:
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post(
                "/compare_insights",
                files={"original_image": ("upload.png", image_bytes, "image/png")},
                data={
                    "match_diagnosis": "opacity",
                    "match_asset_id": "matched.png",
                    "match_payload": json.dumps({
                        "presentation": {"hpi": "Historical cough"},
                        "outcome": {"detail": "Recovered"},
                    }),
                },
            )

    async def test_returns_both_boxes_and_analysis(self):
        image_bytes = self._image_bytes()
        replies = [
            [{"generated_text": "[100, 200, 400, 500]"}],
            [{"generated_text": "[300, 400, 600, 700]"}],
            [{"generated_text": "The current image shows an opacity."}],
        ]
        with tempfile.TemporaryDirectory() as directory:
            matched_image = Path(directory) / "matched.png"
            matched_image.write_bytes(image_bytes)
            with patch.object(main, "asset_path", return_value=matched_image), patch.object(
                main, "query_medgemma", side_effect=replies
            ) as model:
                response = await self._compare(image_bytes)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "insights_text": "The current image shows an opacity.",
            "original_box": [100, 200, 400, 500],
            "match_box": [300, 400, 600, 700],
        })
        self.assertEqual(model.call_count, 3)

    async def test_missing_localization_returns_422_without_analysis(self):
        image_bytes = self._image_bytes()
        replies = [
            [{"generated_text": "No box found"}],
            [{"generated_text": "[300, 400, 600, 700]"}],
        ]
        with tempfile.TemporaryDirectory() as directory:
            matched_image = Path(directory) / "matched.png"
            matched_image.write_bytes(image_bytes)
            with patch.object(main, "asset_path", return_value=matched_image), patch.object(
                main, "query_medgemma", side_effect=replies
            ) as model:
                response = await self._compare(image_bytes)

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "The local model did not return localization boxes for this comparison.")
        self.assertEqual(model.call_count, 2)


if __name__ == "__main__":
    unittest.main()
