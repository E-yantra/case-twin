"""Quote and status validation must keep unsupported clinical assertions unknown."""
import json
import unittest
from unittest.mock import patch

import evidence
import qdrant_service

DX = "assessment.diagnosis_primary"
PNEUMO = "findings.pleura.pneumothorax_present"

class EvidenceTests(unittest.TestCase):
    def test_negative_resolved_suspected_and_historical_are_not_positive(self):
        examples = [
            ("No pneumothorax.", "absent", "no"),
            ("Chest X-ray revealed complete resolution of pneumothorax.", "resolved", "no"),
            ("Suspected pneumothorax on referral.", "suspected", "yes"),
            ("History of pneumothorax.", "historical", "yes"),
        ]
        for text, status, value in examples:
            with self.subTest(text=text):
                claim = {"field": PNEUMO, "value": value, "status": status, "quote": text, "source": "note"}
                updates, warnings = evidence.validate_updates(json.dumps({"updates": [claim]}), {"note": text})
                self.assertEqual(warnings, [])
                profile = evidence.apply_updates(evidence.new_current_profile(), updates)
                self.assertNotEqual(profile["findings"]["pleura"]["pneumothorax_present"], "yes")
                self.assertIsNone(profile["assessment"]["diagnosis_primary"])
                self.assertFalse(evidence.supported_positive(updates))

    def test_negated_or_timed_positive_is_rejected(self):
        for text in ("No pneumothorax.", "Complete resolution of pneumothorax.",
                     "Possible pneumothorax.", "History of pneumothorax."):
            update = {"field": PNEUMO, "value": "yes", "status": "present", "quote": "pneumothorax", "source": "note"}
            accepted, warnings = evidence.validate_updates(json.dumps({"updates": [update]}), {"note": text})
            self.assertEqual(accepted, [])
            self.assertTrue(warnings)

    def test_unrelated_negation_does_not_reject_a_positive_assertion(self):
        text = "Chest X-ray revealed a large left-sided pneumothorax with no tension."
        claim = {"field": PNEUMO, "value": "yes", "status": "present",
                 "quote": "large left-sided pneumothorax", "source": "caption"}
        accepted, warnings = evidence.validate_updates(
            json.dumps({"updates": [claim]}), {"caption": text}
        )
        self.assertEqual(warnings, [])
        self.assertEqual(len(accepted), 1)

    def test_negation_for_another_condition_does_not_veto_diagnosis(self):
        text = "No fever, but pneumonia is present."
        claim = {"field": DX, "value": "pneumonia", "status": "present",
                 "quote": "pneumonia", "source": "note"}
        accepted, warnings = evidence.validate_updates(
            json.dumps({"updates": [claim]}), {"note": text}
        )
        self.assertEqual(warnings, [])
        self.assertEqual(len(accepted), 1)

    def test_quote_must_preserve_negation_and_timing(self):
        cases = [
            ("No pneumothorax.", "pneumothorax", "absent", "no"),
            ("History of pneumothorax.", "pneumothorax", "historical", "yes"),
            ("Complete resolution of pneumothorax.", "pneumothorax", "resolved", "yes"),
            ("Possible pneumothorax.", "pneumothorax", "suspected", "yes"),
        ]
        for text, quote, status, value in cases:
            update = {"field": PNEUMO, "value": value, "status": status,
                      "quote": quote, "source": "note"}
            accepted, warnings = evidence.validate_updates(json.dumps({"updates": [update]}), {"note": text})
            self.assertEqual(accepted, [])
            self.assertTrue(warnings)

    def test_quote_type_and_field_validation(self):
        items = [
            {"field": DX, "value": "pneumothorax", "status": "present", "quote": "made up", "source": "note"},
            {"field": PNEUMO, "value": True, "status": "present", "quote": "pneumothorax", "source": "note"},
            {"field": "unknown.field", "value": "yes", "status": "present", "quote": "pneumothorax", "source": "note"},
        ]
        accepted, warnings = evidence.validate_updates(json.dumps({"updates": items}), {"note": "pneumothorax"})
        self.assertEqual(accepted, [])
        self.assertEqual(len(warnings), 3)

    def test_sex_alias_is_quote_checked_and_canonicalized(self):
        claim = {"field": "patient.sex", "value": "woman", "status": "present",
                 "quote": "The 60-year-old woman presented.", "source": "note"}
        accepted, warnings = evidence.validate_updates(json.dumps({"updates": [claim]}),
                                                       {"note": "The 60-year-old woman presented."})
        self.assertEqual(warnings, [])
        self.assertEqual(accepted[0]["value"], "female")

    def test_conflicting_claims_leave_field_unknown(self):
        updates = [
            {"field": PNEUMO, "value": "yes", "status": "present", "quote": "Pneumothorax seen.", "source": "note"},
            {"field": PNEUMO, "value": "no", "status": "absent", "quote": "No pneumothorax.", "source": "note"},
        ]
        profile = evidence.apply_updates(evidence.new_current_profile(), updates)
        self.assertIsNone(profile["findings"]["pleura"]["pneumothorax_present"])
        self.assertFalse(evidence.supported_positive(updates))

    def test_malformed_json_and_model_failure_are_reported(self):
        for reply in ("not json", '{"updates": "wrong"}'):
            with patch.object(evidence, "query_local_model", return_value=[{"generated_text": reply}]):
                updates, status, warnings = evidence.extract_claims({"note": "No pneumothorax."})
                self.assertEqual((updates, status), ([], "failed"))
                self.assertTrue(warnings)
        with patch.object(evidence, "query_local_model", side_effect=RuntimeError("offline")):
            updates, status, warnings = evidence.extract_claims({"note": "No pneumothorax."})
            self.assertEqual((updates, status), ([], "failed"))
            self.assertTrue(warnings)

    def test_extractor_requests_json_mode_from_the_configured_gateway(self):
        with patch.object(evidence, "query_local_model",
                          return_value=[{"generated_text": '{"updates":[]}'}]) as model:
            updates, status, warnings = evidence.extract_claims({"note": "No pneumothorax."})
        self.assertEqual((updates, status, warnings), ([], "complete", []))
        self.assertEqual(model.call_args.kwargs["response_format"], {"type": "json_object"})

    def test_search_context_requires_supported_current_and_caption_positive(self):
        candidate = {"evidence": [{"field": PNEUMO, "value": "yes", "status": "present"}]}
        self.assertEqual(qdrant_service._context_score(candidate, {"findings": {"pleura": {"pneumothorax_present": "yes"}}}), 0)
        self.assertEqual(qdrant_service._context_score(candidate, {"evidence": [{"field": PNEUMO, "value": "yes", "status": "resolved"}]}), 0)
        self.assertEqual(qdrant_service._context_score(candidate, {"evidence": [{"field": PNEUMO, "value": "yes", "status": "present"}]}), 1)
        normalized = __import__("manifest").normalize_profile({"case_id": "x", "image_id": "y", "evidence": candidate["evidence"]})
        self.assertEqual(qdrant_service._context_score(normalized, normalized), 1)

if __name__ == "__main__":
    unittest.main()
