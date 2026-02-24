"""
Cluster the 750 cases in cxr_schema_dataset/dataset_per_case.json by clinical
similarity and write two output files:

  cxr_schema_dataset/clusters.json
      Array of cluster objects:
      {
        "cluster_id":          int,
        "cluster_name":        str,   # auto-named from dominant features
        "dominant_diagnosis":  str,
        "size":                int,
        "case_ids":            [str, ...]
      }
      Outliers (HDBSCAN label -1) are collected into a single "Miscellaneous" cluster.

  cxr_schema_dataset/dataset_per_case.json   (updated in-place)
  cxr_schema_dataset/dataset_cxr_primary.json (updated in-place)
      The `embeddings` field of every record is populated with the 384-dim
      sentence-transformer vector for that case.
"""

import json
import os
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# Redirect HuggingFace cache to project dir (avoids home-dir space limits on HPC)
os.environ.setdefault(
    "HF_HOME",
    str(Path(__file__).parent / ".hf_cache"),
)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE             = Path("cxr_schema_dataset")
PER_CASE_PATH    = BASE / "dataset_per_case.json"
CXR_PRIMARY_PATH = BASE / "dataset_cxr_primary.json"
CLUSTERS_PATH    = BASE / "clusters.json"

# ── Embedding model ────────────────────────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"   # 384-dim, fast, good semantic similarity

# ── UMAP / HDBSCAN tuning ──────────────────────────────────────────────────
UMAP_N_COMPONENTS   = 20
UMAP_N_NEIGHBORS    = 15
UMAP_MIN_DIST       = 0.0
HDBSCAN_MIN_CLUSTER = 5    # minimum cases per cluster
HDBSCAN_MIN_SAMPLES = 3


# ── Text representation ────────────────────────────────────────────────────
def case_to_text(r: dict) -> str:
    """
    Build a single clinical text for a case by concatenating the most
    semantically informative fields.
    """
    parts = []

    pat  = r.get("patient") or {}
    pres = r.get("presentation") or {}
    asmt = r.get("assessment") or {}
    find = r.get("findings") or {}
    summ = r.get("summary") or {}

    # Demographics + chief complaint
    age = pat.get("age_years")
    sex = pat.get("sex", "")
    if age:
        parts.append(f"{int(age)}-year-old {sex}.")

    if pres.get("chief_complaint"):
        parts.append(f"Chief complaint: {pres['chief_complaint']}.")

    # HPI (most information-dense field)
    if pres.get("hpi"):
        parts.append(pres["hpi"])

    # PMH
    if pres.get("pmh"):
        parts.append(f"PMH: {pres['pmh']}.")

    # Comorbidities
    comorbids = pat.get("comorbidities") or []
    if comorbids:
        parts.append(f"Comorbidities: {', '.join(comorbids)}.")

    # Diagnosis + differentials
    dx = asmt.get("diagnosis_primary")
    if dx:
        parts.append(f"Diagnosis: {dx}.")
    suspected = asmt.get("suspected_primary") or []
    if suspected:
        parts.append(f"Suspected: {', '.join(suspected)}.")
    diff = asmt.get("differential") or []
    if diff:
        parts.append(f"Differential: {', '.join(diff)}.")

    # Urgency / infection
    if asmt.get("urgency"):
        parts.append(f"Urgency: {asmt['urgency']}.")
    if asmt.get("infectious_concern") == "yes":
        parts.append("Infectious concern.")
    if asmt.get("icu_candidate") == "yes":
        parts.append("ICU candidate.")

    # Findings summary
    lungs = find.get("lungs") or {}
    pleura = find.get("pleura") or {}
    cm = find.get("cardiomediastinal") or {}

    findings_parts = []
    if lungs.get("consolidation_present") == "yes":
        locs = ", ".join(lungs.get("consolidation_locations") or [])
        findings_parts.append(f"consolidation ({locs})" if locs else "consolidation")
    if lungs.get("atelectasis_present") == "yes":
        findings_parts.append("atelectasis")
    if lungs.get("edema_present") == "yes":
        findings_parts.append(f"edema ({lungs.get('edema_pattern', 'unknown')})")
    if pleura.get("effusion_present") == "yes":
        findings_parts.append(f"pleural effusion ({pleura.get('effusion_side', '')})")
    if pleura.get("pneumothorax_present") == "yes":
        findings_parts.append("pneumothorax")
    if cm.get("cardiomegaly") == "yes":
        findings_parts.append("cardiomegaly")
    if findings_parts:
        parts.append(f"Findings: {', '.join(findings_parts)}.")

    # Summary
    if summ.get("one_liner"):
        parts.append(summ["one_liner"])

    key_pts = summ.get("key_points") or []
    if key_pts:
        parts.append(" ".join(key_pts))

    return " ".join(parts)


# ── Cluster naming ─────────────────────────────────────────────────────────
def name_cluster(cases) -> tuple:
    """
    Return (cluster_name, dominant_diagnosis) for a group of case records.
    Naming strategy: most common diagnosis + most common finding/urgency modifier.
    """
    diagnoses = [
        r["assessment"]["diagnosis_primary"]
        for r in cases
        if r["assessment"].get("diagnosis_primary")
    ]
    findings_tokens = []
    for r in cases:
        lungs = (r.get("findings") or {}).get("lungs") or {}
        pleura = (r.get("findings") or {}).get("pleura") or {}
        cm = (r.get("findings") or {}).get("cardiomediastinal") or {}
        if lungs.get("consolidation_present") == "yes":
            findings_tokens.append("consolidation")
        if lungs.get("edema_present") == "yes":
            findings_tokens.append("edema")
        if pleura.get("effusion_present") == "yes":
            findings_tokens.append("effusion")
        if cm.get("cardiomegaly") == "yes":
            findings_tokens.append("cardiomegaly")

    infectious = sum(
        1 for r in cases if r["assessment"].get("infectious_concern") == "yes"
    )

    if diagnoses:
        top_dx = Counter(diagnoses).most_common(1)[0][0]
        dominant = top_dx
    else:
        top_dx = "unknown"
        dominant = "unknown"

    # Build a concise label
    label_parts = [top_dx.title()]
    if findings_tokens:
        top_finding = Counter(findings_tokens).most_common(1)[0][0]
        label_parts.append(f"({top_finding})")
    if infectious > len(cases) // 2:
        label_parts.append("[infectious]")

    return " ".join(label_parts), dominant


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("Loading per-case dataset ...")
    with open(PER_CASE_PATH) as f:
        cases = json.load(f)

    # Build text representations
    print("Building text representations ...")
    texts = [case_to_text(r) for r in cases]
    case_ids = [r["case_id"] for r in cases]

    # Embed
    print(f"Embedding {len(texts)} cases with '{EMBED_MODEL}' ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
    print(f"Embedding shape: {embeddings.shape}")

    # UMAP dimensionality reduction
    print(f"UMAP reduction to {UMAP_N_COMPONENTS} dims ...")
    import umap
    reducer = umap.UMAP(
        n_components=UMAP_N_COMPONENTS,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=42,
    )
    reduced = reducer.fit_transform(embeddings)

    # HDBSCAN clustering
    print("HDBSCAN clustering ...")
    import hdbscan
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(reduced)

    n_clusters  = len(set(labels)) - (1 if -1 in labels else 0)
    n_outliers  = int(np.sum(labels == -1))
    print(f"Found {n_clusters} clusters, {n_outliers} outliers → 'Miscellaneous'")

    # Build case_id → embedding mapping (for backfill)
    embed_map = {cid: emb.tolist() for cid, emb in zip(case_ids, embeddings)}

    # Build cluster → list of (case_id, case_record) pairs
    cluster_map: dict[int, list] = {}
    for idx, label in enumerate(labels):
        cluster_map.setdefault(label, []).append(idx)

    # Assemble output clusters (sorted by size desc, outliers last)
    cluster_objects = []
    cluster_id_counter = 0

    for label in sorted(cluster_map.keys(), key=lambda l: (-len(cluster_map[l]), l)):
        indices = cluster_map[label]
        cluster_cases = [cases[i] for i in indices]
        cluster_case_ids = [case_ids[i] for i in indices]

        if label == -1:
            cname = "Miscellaneous"
            dominant = "various"
        else:
            cname, dominant = name_cluster(cluster_cases)

        cluster_objects.append({
            "cluster_id":         cluster_id_counter,
            "cluster_name":       cname,
            "dominant_diagnosis": dominant,
            "size":               len(cluster_case_ids),
            "case_ids":           cluster_case_ids,
        })
        cluster_id_counter += 1

    print(f"\nCluster summary:")
    for c in cluster_objects:
        print(f"  [{c['cluster_id']:2d}] {c['cluster_name']:<50s}  n={c['size']}")

    # Write clusters.json
    print(f"\nWriting {CLUSTERS_PATH} ...")
    with open(CLUSTERS_PATH, "w") as f:
        json.dump(cluster_objects, f, indent=2, ensure_ascii=False)

    # Backfill embeddings into per-case dataset
    print(f"Backfilling embeddings into {PER_CASE_PATH} ...")
    for r in cases:
        r["embeddings"] = embed_map.get(r["case_id"], [])
    with open(PER_CASE_PATH, "w") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)

    # Backfill embeddings into cxr-primary dataset
    print(f"Backfilling embeddings into {CXR_PRIMARY_PATH} ...")
    with open(CXR_PRIMARY_PATH) as f:
        cxr_primary = json.load(f)
    for r in cxr_primary:
        r["embeddings"] = embed_map.get(r["case_id"], [])
    with open(CXR_PRIMARY_PATH, "w") as f:
        json.dump(cxr_primary, f, indent=2, ensure_ascii=False)

    print("\nDone.")
    print(f"  clusters.json            → {len(cluster_objects)} clusters")
    print(f"  dataset_per_case.json    → embeddings backfilled")
    print(f"  dataset_cxr_primary.json → embeddings backfilled")


if __name__ == "__main__":
    main()
