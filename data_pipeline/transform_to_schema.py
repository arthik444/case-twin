"""
Produces two dataset variants from cxr_schema_dataset/dataset.json:

1. dataset.json  (per-case)
   One record per case (750). `images` contains ALL images from the case
   (CXR, CT, pathology, photos, etc.), each tagged with `is_chest_xray`.

2. dataset_cxr_primary.json  (per-CXR-image, CXR-primary)
   One record per CXR image (1 002). The primary CXR fields mirror
   schema.json. A `related_images` array holds the non-CXR images from
   the same case for multi-modal context.

Case-level clinical fields come from the Gemini-enriched flat records
already stored in cxr_schema_dataset/dataset.json (the per-case version
is written first and then used as input for the second variant).
"""

import json
import ast
import re
import uuid
from pathlib import Path

FLAT_PATH            = Path("cxr_schema_dataset/dataset.json")   # Gemini-enriched CXR flat records
SOURCE_PATH          = Path("cxr_full_dataset/dataset.json")     # original all-image source
OUTPUT_PATH          = Path("cxr_schema_dataset/dataset.json")
OUTPUT_CXR_PRIMARY   = Path("cxr_schema_dataset/dataset_cxr_primary.json")


def parse_label_list(raw) -> list:
    """Convert a stringified Python list to a real list, or return as-is."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            result = ast.literal_eval(raw)
            return result if isinstance(result, list) else []
        except Exception:
            return []
    return []


def safe_int(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return val


def split_medications(meds_raw) -> list:
    """Split a comma-separated medications string into a list."""
    if isinstance(meds_raw, list):
        return meds_raw
    if isinstance(meds_raw, str) and meds_raw:
        return [m.strip() for m in re.split(r",(?![^()]*\))", meds_raw) if m.strip()]
    return []


VIEW_MAP = {
    "frontal":  "PA",
    "sagittal": "LATERAL",
    "axial":    "UNKNOWN",
    "oblique":  "UNKNOWN",
    None:       "UNKNOWN",
}

def normalise_view(v):
    if v is None:
        return "UNKNOWN"
    return VIEW_MAP.get(v.lower(), v.upper())


def build_image_entry(img: dict) -> dict:
    """Build a single image object from a cxr_full_dataset image record."""
    file_key = img.get("file_id") or img.get("file") or ""
    image_id = str(uuid.uuid5(uuid.NAMESPACE_URL, file_key))
    return {
        "image_id":         image_id,
        "local_image_path": img.get("local_image_path", ""),
        "file_type":        Path(img.get("file", "")).suffix.lstrip(".") or "webp",
        "image_type":       img.get("image_type", ""),
        "image_subtype":    img.get("image_subtype", ""),
        "is_chest_xray":    img.get("is_chest_xray", False),
        "view_position":    normalise_view(img.get("radiology_view")),
        "radiology_region": img.get("radiology_region"),
        "caption":          img.get("caption", ""),
        "text_references":  img.get("text_references", []),
        "ml_labels":        parse_label_list(img.get("ml_labels")),
        "gt_labels":        parse_label_list(img.get("gt_labels")),
    }


def build_case_record(flat: dict, src_images: list) -> dict:
    """
    Merge Gemini-enriched case fields (from flat CXR record) with the full
    image list from the original source.

    flat       – one representative flat record for this case (all case-level
                 Gemini fields are identical across CXR records of the same case)
    src_images – all images for this case from cxr_full_dataset/dataset.json
    """
    pat   = flat.get("patient") or {}
    pres  = flat.get("presentation") or {}
    study = flat.get("study") or {}
    asmt  = flat.get("assessment") or {}
    find  = flat.get("findings") or {}
    summ  = flat.get("summary") or {}
    oc    = flat.get("outcome") or {}
    prov  = flat.get("provenance") or {}
    tags  = flat.get("tags") or {}

    return {
        "case_id": flat.get("case_id"),

        "patient": {
            "age_years":         pat.get("age_years"),
            "sex":               pat.get("sex"),
            "immunocompromised": pat.get("immunocompromised"),
            "weight_kg":         pat.get("weight_kg"),
            "comorbidities":     pat.get("comorbidities") or [],
            "medications":       pat.get("medications") or [],
            "allergies":         pat.get("allergies"),
        },

        "presentation": {
            "chief_complaint":  pres.get("chief_complaint"),
            "symptom_duration": pres.get("symptom_duration"),
            "hpi":              pres.get("hpi"),
            "pmh":              pres.get("pmh"),
        },

        "study": {
            "modality":      "CXR",
            "body_region":   study.get("body_region") or "thorax",
            "view_position": study.get("view_position"),
        },

        "images": [build_image_entry(img) for img in src_images],

        "assessment": {
            "diagnosis_primary": asmt.get("diagnosis_primary"),
            "suspected_primary": asmt.get("suspected_primary") or [],
            "differential":      asmt.get("differential") or [],
            "urgency":           asmt.get("urgency"),
            "infectious_concern": asmt.get("infectious_concern"),
            "icu_candidate":     asmt.get("icu_candidate"),
        },

        "findings": find,

        "summary": {
            "one_liner":  summ.get("one_liner"),
            "key_points": summ.get("key_points") or [],
            "red_flags":  summ.get("red_flags") or [],
        },

        "outcome": {
            "success": oc.get("success"),
            "detail":  oc.get("detail"),
        },

        "provenance": {
            "dataset_name":  prov.get("dataset_name"),
            "pmc_id":        prov.get("pmc_id"),
            "pmid":          prov.get("pmid"),
            "doi":           prov.get("doi"),
            "article_title": prov.get("article_title"),
            "journal":       prov.get("journal"),
            "year":          prov.get("year"),
            "authors":       prov.get("authors") or [],
            "license":       prov.get("license"),
            "source_url":    prov.get("source_url"),
        },

        "tags": {
            "ml_labels":  tags.get("ml_labels") or [],
            "gt_labels":  tags.get("gt_labels") or [],
            "keywords":   tags.get("keywords") or [],
            "mesh_terms": tags.get("mesh_terms") or [],
        },

        "embeddings": [],
    }


def build_cxr_primary_record(case_record: dict, non_cxr_images: list) -> list:
    """
    Given one per-case record (already in final schema) and the list of
    non-CXR source images, return a list of flat per-CXR-image records.
    Each has the CXR as the primary image (top-level `study` fields) and
    all non-CXR images nested under `related_images`.
    """
    related = [
        {
            "image_id":         img["image_id"],
            "local_image_path": img["local_image_path"],
            "file_type":        img["file_type"],
            "image_type":       img["image_type"],
            "image_subtype":    img["image_subtype"],
            "view_position":    img["view_position"],
            "radiology_region": img["radiology_region"],
            "caption":          img["caption"],
            "ml_labels":        img["ml_labels"],
            "gt_labels":        img["gt_labels"],
        }
        for img in non_cxr_images
    ]

    records = []
    for img in case_record["images"]:
        if not img["is_chest_xray"]:
            continue
        records.append({
            "profile_id": f"{case_record['case_id']}:{img['image_id']}",
            "case_id":    case_record["case_id"],
            "image_id":   img["image_id"],

            "patient":      case_record["patient"],
            "presentation": case_record["presentation"],

            "study": {
                "modality":         img["image_subtype"],
                "body_region":      img["radiology_region"] or case_record["study"].get("body_region"),
                "view_position":    img["view_position"],
                "radiology_region": img["radiology_region"],
                "caption":          img["caption"],
                "image_type":       img["image_type"],
                "image_subtype":    img["image_subtype"],
                "storage_path":     img["local_image_path"],
            },

            "related_images": related,

            "assessment": case_record["assessment"],
            "findings":   case_record["findings"],
            "summary":    case_record["summary"],
            "outcome":    case_record["outcome"],
            "provenance": case_record["provenance"],

            "tags": {
                "ml_labels":  img["ml_labels"],
                "gt_labels":  img["gt_labels"],
                "keywords":   case_record["tags"].get("keywords") or [],
                "mesh_terms": case_record["tags"].get("mesh_terms") or [],
            },

            "embeddings": [],
        })
    return records


def main():
    # ── Load Gemini-enriched flat records (CXR only) and group by PMC ID ──
    print(f"Loading flat Gemini-enriched records from {FLAT_PATH} ...")
    with open(FLAT_PATH, "r") as f:
        flat_records = json.load(f)

    # Index: pmc_id → first flat record for that case (case-level fields are
    # identical across all CXR records belonging to the same case)
    flat_by_pmc: dict[str, dict] = {}
    for rec in flat_records:
        pmc = rec.get("provenance", {}).get("pmc_id")
        if pmc and pmc not in flat_by_pmc:
            flat_by_pmc[pmc] = rec

    # ── Load original source (all images) ──
    print(f"Loading original full-image source from {SOURCE_PATH} ...")
    with open(SOURCE_PATH, "r") as f:
        source_cases = json.load(f)

    # ── Merge ──
    output = []
    missing = []
    for src_case in source_cases:
        pmc_id = src_case.get("pmc_id")
        flat = flat_by_pmc.get(pmc_id)
        if flat is None:
            missing.append(pmc_id)
            continue
        record = build_case_record(flat, src_case.get("images", []))
        output.append(record)

    total_images = sum(len(r["images"]) for r in output)
    cxr_images   = sum(sum(1 for img in r["images"] if img["is_chest_xray"]) for r in output)

    print(f"Cases produced  : {len(output)}")
    print(f"Cases missing   : {len(missing)}")
    print(f"Total images    : {total_images}  ({cxr_images} CXR, {total_images - cxr_images} non-CXR)")

    print(f"Writing {OUTPUT_PATH} ...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ── Build CXR-primary variant ──
    print("\nBuilding CXR-primary variant ...")

    # Index source cases by pmc_id for quick lookup of non-CXR images
    src_by_pmc = {c["pmc_id"]: c for c in source_cases}

    cxr_primary_records = []
    for case_record in output:
        pmc_id = case_record["provenance"].get("pmc_id")
        src_case = src_by_pmc.get(pmc_id, {})
        non_cxr = [
            build_image_entry(img)
            for img in src_case.get("images", [])
            if not img.get("is_chest_xray")
        ]
        cxr_primary_records.extend(build_cxr_primary_record(case_record, non_cxr))

    print(f"CXR-primary records: {len(cxr_primary_records)}")
    print(f"Writing {OUTPUT_CXR_PRIMARY} ...")
    with open(OUTPUT_CXR_PRIMARY, "w") as f:
        json.dump(cxr_primary_records, f, indent=2, ensure_ascii=False)

    print("Done.")


if __name__ == "__main__":
    main()
