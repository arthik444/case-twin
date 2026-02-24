"""
Build a chest X-ray dataset from the whole_multicare_dataset.

Inclusion rule: any PMCID that has at least one chest X-ray image.
For those PMCIDs, ALL images (CXR, CT, pathology, etc.) are included.
Each image has an "is_chest_xray" boolean flag.

Output structure:
    cxr_full_dataset/
        dataset.json       <- array of JSON objects, one per PMCID
        images/            <- copies of ALL images for qualifying PMCIDs
"""

import os
import json
import shutil
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE         = Path("medical_datasets/whole_multicare_dataset")
OUTPUT_DIR   = Path("cxr_full_dataset")
IMAGES_DIR   = OUTPUT_DIR / "images"
DATASET_JSON = OUTPUT_DIR / "dataset.json"

os.makedirs(IMAGES_DIR, exist_ok=True)
print(f"Output folder: {OUTPUT_DIR.resolve()}")

# ── Load all source files ──────────────────────────────────────────────────
print("Loading source files...")
captions_df  = pd.read_csv(BASE / "captions_and_labels.csv")
metadata_df  = pd.read_parquet(BASE / "metadata.parquet")
abstracts_df = pd.read_parquet(BASE / "abstracts.parquet")
cases_df     = pd.read_parquet(BASE / "cases.parquet")
ci_df        = pd.read_parquet(BASE / "case_images.parquet")

# ── Filter for chest X-rays only ──────────────────────────────────────────
cxr_mask = (
    (captions_df["image_subtype"] == "x_ray") &
    (captions_df["image_type"] == "radiology") &
    (captions_df["caption"].str.contains(r"chest\s*x-?ray|cxr", case=False, na=False))
)
cxr_df = captions_df[cxr_mask].copy()
print(f"Chest X-ray image rows found: {len(cxr_df)}")

pmc_ids = sorted(cxr_df["patient_id"].str.extract(r"(PMC\d+)")[0].unique())
print(f"Unique PMCIDs with chest X-rays: {len(pmc_ids)}")

# ── Build image file index (filename -> full path) ─────────────────────────
# Use the naming convention from data_dictionary: PMC1/PMC10/filename
def image_path_from_filename(fname: str) -> Path:
    return BASE / fname[:4] / fname[:5] / fname

# ── Build per-image text_references lookup from case_images.parquet ────────
# Flatten: image_id -> text_references list
text_ref_lookup: dict[str, list[str]] = {}
for _, ci_row in ci_df.iterrows():
    for img in ci_row["case_images"]:
        image_id = img.get("image_id", "")
        refs = img.get("text_references", [])
        text_ref_lookup[image_id] = list(refs) if refs is not None else []

# ── Helpers to safely convert numpy/pandas types to plain Python ───────────
def to_list(val):
    if val is None:
        return []
    try:
        return list(val)
    except TypeError:
        return []

def to_python(val):
    """Recursively convert numpy scalars / NaN to plain Python types."""
    import numpy as np
    if isinstance(val, float) and pd.isna(val):
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, (np.ndarray,)):
        return [to_python(v) for v in val]
    if isinstance(val, list):
        return [to_python(v) for v in val]
    if isinstance(val, dict):
        return {k: to_python(v) for k, v in val.items()}
    return val

# ── Main loop ─────────────────────────────────────────────────────────────
dataset = []
copied_images = 0
missing_images = 0

for pmc_id in pmc_ids:

    # -- Metadata --
    meta_row = metadata_df[metadata_df["article_id"] == pmc_id]
    if len(meta_row):
        m = meta_row.iloc[0]["article_metadata"]
        meta = {
            "title":           m.get("title", ""),
            "authors":         to_list(m.get("authors")),
            "journal":         m.get("journal", ""),
            "journal_detail":  m.get("journal_detail", ""),
            "year":            m.get("year", ""),
            "doi":             m.get("doi", ""),
            "pmid":            m.get("pmid", ""),
            "license":         m.get("license", ""),
            "keywords":        to_list(m.get("keywords")),
            "mesh_terms":      to_list(m.get("mesh_terms")),
            "major_mesh_terms":to_list(m.get("major_mesh_terms")),
            "link":            m.get("link", ""),
            "case_amount":     int(m.get("case_amount", 0)),
        }
    else:
        meta = {}

    # -- Abstract --
    abs_row = abstracts_df[abstracts_df["article_id"] == pmc_id]
    abstract = abs_row.iloc[0]["abstract"] if len(abs_row) else ""

    # -- Cases --
    cases_row = cases_df[cases_df["article_id"] == pmc_id]
    cases_out = []
    if len(cases_row):
        for case in cases_row.iloc[0]["cases"]:
            cases_out.append({
                "case_id":   case.get("case_id", ""),
                "age":       to_python(case.get("age")),
                "gender":    case.get("gender", ""),
                "case_text": case.get("case_text", ""),
            })

    # -- All images for this PMCID (CXR cases include all modalities) --
    # Build a set of CXR file_ids for flagging
    cxr_file_ids = set(cxr_df[cxr_df["patient_id"].str.startswith(pmc_id)]["file_id"].values)
    all_pmc_images = captions_df[captions_df["patient_id"].str.startswith(pmc_id)]
    images_out = []

    for _, img_row in all_pmc_images.iterrows():
        filename  = img_row["file"]
        src_path  = image_path_from_filename(filename)
        dst_fname = filename
        dst_path  = IMAGES_DIR / dst_fname
        local_rel = f"images/{dst_fname}"

        # Copy image
        if src_path.exists():
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)
            copied_images += 1
        else:
            missing_images += 1

        # text_references via main_image id
        main_image_id = img_row.get("main_image", "")
        text_refs = text_ref_lookup.get(main_image_id, [])

        images_out.append({
            "file_id":            img_row.get("file_id", ""),
            "file":               filename,
            "main_image":         main_image_id,
            "image_component":    img_row.get("image_component", ""),
            "patient_id":         img_row.get("patient_id", ""),
            "license":            img_row.get("license", ""),
            "file_size":          int(img_row.get("file_size", 0)),
            "image_type":         img_row.get("image_type", ""),
            "image_subtype":      img_row.get("image_subtype", ""),
            "is_chest_xray":      img_row.get("file_id", "") in cxr_file_ids,
            "caption":            img_row.get("caption", ""),
            "case_substring":     img_row.get("case_substring", ""),
            "radiology_region":   to_python(img_row.get("radiology_region")),
            "radiology_region_granular": to_python(img_row.get("radiology_region_granular")),
            "radiology_view":     to_python(img_row.get("radiology_view")),
            "ml_labels":          img_row.get("ml_labels_for_supervised_classification", ""),
            "gt_labels":          img_row.get("gt_labels_for_semisupervised_classification", ""),
            "text_references":    text_refs,
            "local_image_path":   local_rel,
        })

    # -- Assemble PMCID record --
    record = {
        "pmc_id":   pmc_id,
        **meta,
        "abstract": abstract,
        "cases":    cases_out,
        "images":   images_out,
    }
    dataset.append(record)

# ── Write JSON ─────────────────────────────────────────────────────────────
print(f"\nWriting dataset.json ({len(dataset)} PMCIDs)...")
with open(DATASET_JSON, "w") as f:
    json.dump(dataset, f, indent=2, default=str)

print(f"\nDone.")
print(f"  PMCIDs:         {len(dataset)}")
print(f"  Images copied:  {copied_images}")
print(f"  Images missing: {missing_images}")
print(f"  Output:         {OUTPUT_DIR.resolve()}")
