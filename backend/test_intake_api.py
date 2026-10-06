"""Endpoint regression: unknown findings never become search or chat facts."""
import json
import unittest
from unittest.mock import patch

import httpx
from PIL import Image
import io

import main
from evidence import new_current_profile

PNEUMO = "findings.pleura.pneumothorax_present"

class IntakeTests(unittest.IsolatedAsyncioTestCase):
    async def extract(self, note):
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/extract", data={"notes": note})

    async def test_failed_extraction_retains_note_and_unknown_finding(self):
        with patch.object(main, "extract_claims", return_value=([], "failed", ["model unavailable"])):
            response = await self.extract("No pneumothorax.")
        data = response.json()
        self.assertEqual(data["extraction_status"], "failed")
        self.assertEqual(data["profile"]["presentation"]["hpi"], "No pneumothorax.")
        self.assertIsNone(data["profile"]["assessment"]["diagnosis_primary"])
        self.assertIsNone(data["profile"]["findings"]["pleura"]["pneumothorax_present"])
        self.assertEqual(data["evidence"], [])

    async def test_negative_extraction_sets_only_supported_absence(self):
        claim = {"field": PNEUMO, "value": "no", "status": "absent", "quote": "No pneumothorax.", "source": "note"}
        with patch.object(main, "extract_claims", return_value=([claim], "complete", [])):
            response = await self.extract("No pneumothorax.")
        data = response.json()
        self.assertEqual(data["profile"]["findings"]["pleura"]["pneumothorax_present"], "no")
        self.assertIsNone(data["profile"]["findings"]["pleura"]["effusion_present"])
        self.assertIsNone(data["profile"]["assessment"]["diagnosis_primary"])

    async def test_case_twin_chat_ignores_unsupported_note_text(self):
        profile = new_current_profile()
        profile["presentation"]["hpi"] = "Pneumothorax mentioned without a current assertion."
        profile["findings"]["pleura"]["pneumothorax_present"] = "yes"  # untrusted without evidence
        with patch.object(main, "query_local_model", return_value=[{"generated_text": "No supported assertion."}]) as model:
            transport = httpx.ASGITransport(app=main.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post("/chat_twin", data={"query": "What is current?", "case_text": "Historical case.",
                                                                   "current_profile": json.dumps(profile)})
        self.assertEqual(response.status_code, 200)
        prompt = model.call_args.args[0]
        self.assertIn("Supported current positive assertions: none documented", prompt)
        self.assertNotIn("Pneumothorax", prompt)

    async def test_visual_search_rejects_placeholder_image(self):
        image = io.BytesIO()
        Image.new("RGB", (1, 1)).save(image, format="PNG")
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/search", files={"file": ("pixel.png", image.getvalue(), "image/png")})
        self.assertEqual(response.status_code, 400)
        self.assertIn("placeholder", response.json()["detail"])

if __name__ == "__main__":
    unittest.main()
