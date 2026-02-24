"""
Extract structured fields from multicare CXR dataset using Gemini API
and produce a final dataset.json conforming to the target schema.

Fields covered:
  - Directly mapped  : source, study, images, patient age/sex, abstract→summary
  - Gemini-extracted : chief_complaint, symptom_duration, comorbidities,
                       medications, immunocompromised, findings_structured,
                       radiology_labels, outcomes, clinical_text structured note
  - Removed (N/A)    : dicom, facility, routing, embeddings, audit.models_used,
                       pregnancy, smoking_status, family_history, portable,
                       laterality_marker_present, transfer_memo
"""

import os, json, uuid, time, re
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from tqdm import tqdm

# ── Config ─────────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=Path("/blue/gtyson.fsu/tp22o.fsu/medgemma/.env"), override=True)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL_NAME = "gemini-2.0-flash"

INPUT_JSON  = Path("cxr_full_dataset/dataset.json")
OUTPUT_DIR  = Path("cxr_schema_dataset")
OUTPUT_JSON = OUTPUT_DIR / "dataset.json"
CACHE_DIR   = OUTPUT_DIR / ".cache"          # per-record cache to allow resume

OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# ── Gemini extraction prompt ───────────────────────────────────────────────
EXTRACT_PROMPT = """
You are a medical NLP assistant. Given a clinical case report text and abstract, extract the following fields and return ONLY valid JSON — no markdown, no explanation.

Return exactly this structure (use null for unknown/missing values, use the exact enum strings shown):

{{
  "chief_complaint": "<string or null>",
  "symptom_duration": "<string or null>",
  "comorbidities": ["<list of conditions, lowercase, e.g. hypertension, copd, diabetes>"],
  "medications_used": "<string summarising medications or null>",
  "immunocompromised": "unknown | yes | no",
  "clinical_note_hpi": "<1-3 sentence history of present illness or null>",
  "clinical_note_pmh": "<past medical history summary or null>",
  "clinical_note_meds": "<medications string or null>",
  "clinical_note_allergies": "<allergies string or null>",
  "primary_suspected": ["<list of diagnoses, lowercase>"],
  "differential": ["<alternative diagnoses, lowercase>"],
  "infectious_concern": "unknown | yes | no",
  "icu_candidate": "unknown | yes | no",
  "lungs_consolidation_present": "unknown | yes | no",
  "lungs_consolidation_location": ["<e.g. RLL, LUL>"],
  "lungs_consolidation_extent": "mild | moderate | severe | unknown",
  "lungs_atelectasis_present": "unknown | yes | no",
  "lungs_atelectasis_location": [],
  "lungs_edema_present": "unknown | yes | no",
  "lungs_edema_pattern": "interstitial | alveolar | mixed | unknown",
  "pleura_effusion_present": "unknown | yes | no",
  "pleura_effusion_side": "left | right | bilateral | unknown",
  "pleura_effusion_size": "small | moderate | large | unknown",
  "pleura_pneumothorax_present": "unknown | yes | no",
  "pleura_pneumothorax_side": "left | right | bilateral | unknown",
  "cardiomegaly": "unknown | yes | no",
  "mediastinal_widening": "unknown | yes | no",
  "lines_tubes_present": "unknown | yes | no",
  "device_list": ["<e.g. ett, central_line, chest_tube>"],
  "summary_1_2_lines": "<1-2 sentence case summary>",
  "bullets": ["<key finding 1>", "<key finding 2>", "<key finding 3>"],
  "red_flags": ["<string>"],
  "uncertainties": ["<string>"],
  "urgency": "routine | urgent | emergent",
  "outcome_success": "unknown | yes | no",
  "outcome_detail": "<string describing what happened to the patient or null>",
  "ground_truth_diagnosis": "<final confirmed diagnosis or null>",
  "ground_truth_source": "case_text | abstract | unknown"
}}

ABSTRACT:
{abstract}

CASE TEXT:
{case_text}
"""

def call_gemini(abstract: str, case_text: str) -> dict:
    prompt = EXTRACT_PROMPT.format(
        abstract=abstract or "Not available",
        case_text=case_text or "Not available"
    )
    for attempt in range(4):
        try:
            resp = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            raw = resp.text.strip()
            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            return json.loads(raw)
        except json.JSONDecodeError:
            time.sleep(2 ** attempt)
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                time.sleep(30)
            else:
                time.sleep(2 ** attempt)
    return {}

# ── View position normalisation ────────────────────────────────────────────
VIEW_MAP = {
    "frontal":   "PA",
    "sagittal":  "LATERAL",
    "axial":     "UNKNOWN",
    "oblique":   "UNKNOWN",
    None:        "UNKNOWN",
}

def normalise_view(v):
    return VIEW_MAP.get(v, "UNKNOWN")

# ── Build one schema record ────────────────────────────────────────────────
def build_record(rec: dict, extracted: dict) -> dict:
    pmc_id    = rec["pmc_id"]
    cases     = rec.get("cases", [])
    case0     = cases[0] if cases else {}
    images    = rec.get("images", [])
    cxr_imgs  = [img for img in images if img.get("is_chest_xray")]
    first_cxr = cxr_imgs[0] if cxr_imgs else (images[0] if images else {})
    e = extracted  # shorthand

    return {
        "case_id": str(uuid.uuid5(uuid.NAMESPACE_URL, pmc_id)),
        "source": {
            "dataset_name": "multicare",
            "dataset_record_id": pmc_id,
            "pmc_link": rec.get("link", ""),
            "doi": rec.get("doi", ""),
            "pmid": rec.get("pmid", ""),
            "title": rec.get("title", ""),
            "authors": rec.get("authors", []),
            "journal": rec.get("journal", ""),
            "journal_detail": rec.get("journal_detail", ""),
            "year": rec.get("year", ""),
            "license": rec.get("license", ""),
        },

        "patient_context": {
            "age_years": case0.get("age"),
            "sex": (case0.get("gender") or "unknown").lower(),
            "immunocompromised": e.get("immunocompromised", "unknown"),
            "chief_complaint": e.get("chief_complaint"),
            "symptom_duration": e.get("symptom_duration"),
            "comorbidities": e.get("comorbidities", []),
            "medications_used": e.get("medications_used"),
        },

        "study": {
            "modality": "CXR",
            "view_position": normalise_view(first_cxr.get("radiology_view")),
            "body_region": first_cxr.get("radiology_region") or "chest",
        },

        "images": [
            {
                "image_id": str(uuid.uuid5(uuid.NAMESPACE_URL, img.get("file_id", img.get("file", "")))),
                "local_image_path": img.get("local_image_path", ""),
                "file_type": Path(img.get("file", "")).suffix.lstrip(".") or "webp",
                "image_type": img.get("image_type", ""),
                "image_subtype": img.get("image_subtype", ""),
                "is_chest_xray": img.get("is_chest_xray", False),
                "view_position": normalise_view(img.get("radiology_view")),
                "radiology_region": img.get("radiology_region"),
                "caption": img.get("caption", ""),
                "text_references": img.get("text_references", []),
                "ml_labels": img.get("ml_labels", ""),
                "gt_labels": img.get("gt_labels", ""),
            }
            for img in images
        ],

        "clinical_text": {
            "raw_note": case0.get("case_text", ""),
            "structured_note": {
                "hpi":       e.get("clinical_note_hpi"),
                "pmh":       e.get("clinical_note_pmh"),
                "meds":      e.get("clinical_note_meds"),
                "allergies": e.get("clinical_note_allergies"),
            },
        },

        "radiology_labels": {
            "primary_suspected": e.get("primary_suspected", []),
            "differential":      e.get("differential", []),
            "urgency":           e.get("urgency", "routine"),
            "infectious_concern": e.get("infectious_concern", "unknown"),
            "icu_candidate":     e.get("icu_candidate", "unknown"),
        },

        "findings_structured": {
            "lungs": {
                "consolidation": {
                    "present":  e.get("lungs_consolidation_present", "unknown"),
                    "location": e.get("lungs_consolidation_location", []),
                    "extent":   e.get("lungs_consolidation_extent", "unknown"),
                },
                "atelectasis": {
                    "present":  e.get("lungs_atelectasis_present", "unknown"),
                    "location": e.get("lungs_atelectasis_location", []),
                },
                "edema": {
                    "present":  e.get("lungs_edema_present", "unknown"),
                    "pattern":  e.get("lungs_edema_pattern", "unknown"),
                },
            },
            "pleura": {
                "effusion": {
                    "present": e.get("pleura_effusion_present", "unknown"),
                    "side":    e.get("pleura_effusion_side", "unknown"),
                    "size":    e.get("pleura_effusion_size", "unknown"),
                },
                "pneumothorax": {
                    "present": e.get("pleura_pneumothorax_present", "unknown"),
                    "side":    e.get("pleura_pneumothorax_side", "unknown"),
                },
            },
            "cardiomediastinal": {
                "cardiomegaly":         e.get("cardiomegaly", "unknown"),
                "mediastinal_widening": e.get("mediastinal_widening", "unknown"),
            },
            "devices": {
                "lines_tubes_present": e.get("lines_tubes_present", "unknown"),
                "device_list":         e.get("device_list", []),
            },
        },

        "text_outputs": {
            "case_card": {
                "summary_1_2_lines": e.get("summary_1_2_lines") or rec.get("abstract", ""),
                "bullets":           e.get("bullets", []),
                "red_flags":         e.get("red_flags", []),
                "uncertainties":     e.get("uncertainties", []),
            },
        },

        "outcomes": {
            "label_type":             "real",
            "success":                e.get("outcome_success", "unknown"),
            "outcome_detail":         e.get("outcome_detail"),
            "ground_truth_diagnosis": e.get("ground_truth_diagnosis"),
            "ground_truth_source":    e.get("ground_truth_source", "case_text"),
        },

        "audit": {
            "created_at": f"{rec.get('year', '2023')}-01-01T00:00:00Z",
            "keywords":   rec.get("keywords", []),
            "mesh_terms": rec.get("mesh_terms", []),
        },
    }

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    with open(INPUT_JSON) as f:
        source_data = json.load(f)

    print(f"Processing {len(source_data)} records with Gemini extraction...")
    results = []
    errors  = []

    for rec in tqdm(source_data, unit="record"):
        pmc_id     = rec["pmc_id"]
        cache_file = CACHE_DIR / f"{pmc_id}.json"

        # Use cache if already processed
        if cache_file.exists():
            with open(cache_file) as f:
                extracted = json.load(f)
        else:
            cases    = rec.get("cases", [])
            case0    = cases[0] if cases else {}
            case_text = case0.get("case_text", "")
            abstract  = rec.get("abstract", "")

            extracted = call_gemini(abstract, case_text)

            # Save to cache
            with open(cache_file, "w") as f:
                json.dump(extracted, f)

            time.sleep(0.3)  # gentle rate limiting

        try:
            schema_record = build_record(rec, extracted)
            results.append(schema_record)
        except Exception as ex:
            errors.append({"pmc_id": pmc_id, "error": str(ex)})

    print(f"\nWriting {len(results)} records to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if errors:
        err_path = OUTPUT_DIR / "errors.json"
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"⚠  {len(errors)} errors written to {err_path}")

    print(f"\nDone. Output: {OUTPUT_JSON.resolve()}")
    print(f"  Records: {len(results)}")

if __name__ == "__main__":
    main()
