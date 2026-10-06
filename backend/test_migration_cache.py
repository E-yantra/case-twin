"""Failed model responses resume without repeat calls unless requested."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_pipeline import rebuild_evidence_manifest as migration  # noqa: E402


class MigrationCacheTests(unittest.TestCase):
    def test_failed_cache_is_reused_and_can_be_explicitly_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            items = {"article-1": {"abstract": "Retained abstract."}}
            allowed = {"patient.age_years"}
            path, digest = migration._cache_file(cache_dir, "article", "article-1", items["article-1"], allowed)
            migration._write_cache(path, digest, [], "failed", ["gateway offline"])

            with patch.object(migration, "extract_claims") as extract:
                cached = migration._collect_claims(items, cache_dir, "article", allowed, 3, 1)
            extract.assert_not_called()
            self.assertEqual(cached["article-1"]["status"], "failed")

            with patch.object(migration, "extract_claims", return_value=([], "complete", [])) as extract:
                retried = migration._collect_claims(items, cache_dir, "article", allowed, 3, 1,
                                                    retry_failed=True)
            extract.assert_called_once()
            self.assertEqual(retried["article-1"]["status"], "complete")

    def test_batched_source_ids_are_short_and_mapped_back_to_each_caption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            batch = []
            for index, key in enumerate(("point-a", "point-b"), start=1):
                sources = {"caption": f"Caption {index}."}
                path, digest = migration._cache_file(root, "caption", key, sources,
                                                     migration.CAPTION_FIELDS,
                                                     migration.CAPTION_EXTRACTION_VERSION)
                batch.append((key, sources, path, digest))

            def fake_extract(sources, allowed, max_tokens):
                self.assertEqual(set(sources), {"record_1_caption", "record_2_caption"})
                return ([{"field": "findings.pleura.pneumothorax_present", "value": "yes",
                          "status": "present", "quote": "Caption 1.", "source": "record_1_caption"},
                         {"field": "findings.pleura.pneumothorax_present", "value": "yes",
                          "status": "present", "quote": "Caption 2.", "source": "record_2_caption"}],
                        "complete", [])

            with patch.object(migration, "extract_claims", side_effect=fake_extract):
                result = migration._claim_batch(batch, migration.CAPTION_FIELDS,
                                                migration.CAPTION_EXTRACTION_VERSION)
            self.assertEqual(result["point-a"]["updates"][0]["source"], "caption")
            self.assertEqual(result["point-b"]["updates"][0]["source"], "caption")
            self.assertEqual(result["point-a"]["version"], migration.CAPTION_EXTRACTION_VERSION)

    def test_clean_legacy_cache_is_promoted_but_warning_cache_is_reextracted(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            items = {"clean": {"caption": "No pneumothorax."},
                     "warning": {"caption": "Possible pneumothorax."}}
            allowed = migration.CAPTION_FIELDS
            for key, sources, warnings in (("clean", items["clean"], []),
                                           ("warning", items["warning"], ["invalid source id"])):
                path, digest = migration._cache_file(cache_dir, "caption", key, sources, allowed,
                                                     migration.ARTICLE_EXTRACTION_VERSION)
                migration._write_cache(path, digest, [], "complete", warnings,
                                       migration.ARTICLE_EXTRACTION_VERSION)

            with patch.object(migration, "extract_claims", return_value=([], "complete", [])) as extract:
                result = migration._collect_claims(
                    items, cache_dir, "caption", allowed, 5, 1,
                    version=migration.CAPTION_EXTRACTION_VERSION,
                    previous_versions=(migration.ARTICLE_EXTRACTION_VERSION,),
                )
            extract.assert_called_once()
            self.assertEqual(result["clean"]["version"], migration.CAPTION_EXTRACTION_VERSION)
            self.assertEqual(result["warning"]["status"], "complete")

    def test_batch_validation_warnings_are_scoped_to_matching_caption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = {
                "pneumo": {"caption": "Chest X-ray shows pneumothorax."},
                "opacity": {"caption": "Chest X-ray shows bilateral opacities."},
            }
            batch = []
            for key, sources in inputs.items():
                path, digest = migration._cache_file(root, "caption", key, sources,
                                                     migration.CAPTION_FIELDS,
                                                     migration.CAPTION_EXTRACTION_VERSION)
                batch.append((key, sources, path, digest))
            warnings = ["Update 1 field='findings.lungs.opacity_present' status='present': does not name its finding"]
            with patch.object(migration, "extract_claims", return_value=([], "complete", warnings)):
                result = migration._claim_batch(batch, migration.CAPTION_FIELDS,
                                                migration.CAPTION_EXTRACTION_VERSION)
            self.assertEqual(result["pneumo"]["warnings"], [])
            self.assertEqual(result["opacity"]["warnings"], warnings)


if __name__ == "__main__":
    unittest.main()
