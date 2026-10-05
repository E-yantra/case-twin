"""Twin-library collections: MultiCaRe slices that match MedSigLIP's training domains.

``zero_shot`` is the text MedSigLIP compares an uploaded image against to decide
which collection to search (zero-shot classification: no training, just
image-text similarity in a shared embedding space).
"""

COLLECTIONS: dict[str, dict] = {
    "cxr": {"label": "Chest X-ray", "image_type": "radiology", "image_subtype": "x_ray",
            "radiology_region": "thorax", "modality": "CXR", "body_region": "chest",
            "zero_shot": "a chest X-ray radiograph"},
    "chest_ct": {"label": "Chest CT", "image_type": "radiology", "image_subtype": "ct",
                 "radiology_region": "thorax", "modality": "CT", "body_region": "chest",
                 "zero_shot": "an axial CT scan of the chest"},
    "derm": {"label": "Dermatology photo", "image_type": "medical_photograph", "image_subtype": "skin_photograph",
             "radiology_region": None, "modality": "Clinical photograph", "body_region": "skin",
             "zero_shot": "a clinical photograph of a skin lesion"},
    "fundus": {"label": "Fundus photo", "image_type": "ophthalmic_imaging", "image_subtype": "fundus_photograph",
               "radiology_region": None, "modality": "Fundus photograph", "body_region": "eye",
               "zero_shot": "a color fundus photograph of the retina"},
    "histopath": {"label": "H&E histopathology", "image_type": "pathology", "image_subtype": "h&e",
                  "radiology_region": None, "modality": "Histopathology (H&E)", "body_region": None,
                  "zero_shot": "an H&E stained histopathology slide"},
}


# Non-target image types MedSigLIP must out-score the collection label against.
# MultiCaRe's image-type labels are model-predicted and noisy (e.g. pedigree charts
# filed as X-rays), so preparation keeps an image only if MedSigLIP's top zero-shot
# label is its own collection.
QUALITY_DISTRACTORS = [
    "a medical chart, diagram or table", "an endoscopy image", "a barium fluoroscopy study", "an MRI scan",
    "an ultrasound image", "an intraoperative surgical photograph", "an immunohistochemistry stained slide",
    "an electrocardiogram tracing",
]


def quality_labels() -> list[str]:
    return [spec["zero_shot"] for spec in COLLECTIONS.values()] + QUALITY_DISTRACTORS
