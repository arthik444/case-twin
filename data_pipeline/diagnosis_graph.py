"""
MedGemma Differential Diagnosis System using LangGraph + MultiCaRe Dataset
"""

import json
import base64
import os
from pathlib import Path
from typing import TypedDict, Annotated, List, Optional
import operator

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# ─────────────────────────────────────────────
# State Definition
# ─────────────────────────────────────────────

class DiagnosisState(TypedDict):
    case_id: str
    case_data: dict
    images_b64: List[dict]          # [{image_id, b64, caption, subtype}]
    initial_diagnosis: Optional[str]
    differential_diagnoses: Optional[List[dict]]  # [{condition, confidence, reasoning}]
    bias_check_notes: Optional[str]
    alternative_hypotheses: Optional[List[dict]]
    final_report: Optional[str]
    messages: Annotated[List, operator.add]


# ─────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────

def load_case_node(state: DiagnosisState) -> DiagnosisState:
    """Load case details from parsed MultiCaRe JSON."""
    case_id = state["case_id"]
    
    # Load from your parsed dataset — adjust path as needed
    dataset_path = Path(os.getenv("MULTICARE_DATASET_PATH", "multicare_parsed.json"))
    
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            "Set MULTICARE_DATASET_PATH env var to your parsed JSON file."
        )
    
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    # Support both list and dict-keyed formats
    if isinstance(dataset, list):
        cases = {c["case_id"]: c for c in dataset}
    else:
        cases = dataset
    
    if case_id not in cases:
        raise ValueError(f"Case ID '{case_id}' not found in dataset.")
    
    case_data = cases[case_id]
    print(f"[load_case] Loaded case: {case_data['provenance']['article_title']}")
    
    return {
        **state,
        "case_data": case_data,
        "messages": [HumanMessage(content=f"Starting diagnosis for case {case_id}")]
    }


def load_images_node(state: DiagnosisState) -> DiagnosisState:
    """Load and base64-encode images for the case."""
    case_data = state["case_data"]
    images_b64 = []
    
    base_path = Path(os.getenv("MULTICARE_IMAGES_PATH", "."))
    
    for img_meta in case_data.get("images", []):
        img_path = base_path / img_meta["local_image_path"]
        
        if not img_path.exists():
            print(f"[load_images] Warning: Image not found: {img_path}")
            continue
        
        with open(img_path, "rb") as f:
            raw = f.read()
        
        b64 = base64.b64encode(raw).decode("utf-8")
        
        # Determine MIME type
        ext = img_meta["file_type"].lower()
        mime_map = {
            "webp": "image/webp",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
        }
        mime = mime_map.get(ext, "image/jpeg")
        
        images_b64.append({
            "image_id": img_meta["image_id"],
            "b64": b64,
            "mime": mime,
            "caption": img_meta.get("caption", ""),
            "subtype": img_meta.get("image_subtype", "unknown"),
            "image_type": img_meta.get("image_type", "unknown"),
        })
        print(f"[load_images] Loaded image: {img_meta['image_subtype']} ({img_meta['image_id'][:8]}...)")
    
    return {**state, "images_b64": images_b64}


# ─────────────────────────────────────────────
# MedGemma Helpers
# ─────────────────────────────────────────────
def _build_medgemma_prompt(case_data: dict, images_b64: list, task: str) -> list:
    """Build a multimodal message list for MedGemma."""
    import json
    
    patient = case_data.get("patient", {})
    presentation = case_data.get("presentation", {})
    findings = case_data.get("findings", {})
    study = case_data.get("study", {})
    
    # Clinical summary text
    clinical_text = f"""
=== CLINICAL CASE ===

PATIENT:
- Age: {patient.get('age_years', 'unknown')} years, {patient.get('sex', 'unknown')}
- Immunocompromised: {patient.get('immunocompromised', 'unknown')}
- Comorbidities: {', '.join(patient.get('comorbidities', [])) or 'None documented'}
- Medications: {', '.join(patient.get('medications', [])) or 'None documented'}
- Allergies: {patient.get('allergies', 'None documented')}

PRESENTATION:
- Chief Complaint: {presentation.get('chief_complaint', 'N/A')}
- Duration: {presentation.get('symptom_duration', 'N/A')}
- HPI: {presentation.get('hpi', 'N/A')}
- Past Medical History: {presentation.get('pmh', 'N/A')}

IMAGING STUDY:
- Modality: {study.get('modality', 'N/A')}
- Region: {study.get('body_region', 'N/A')}
- View: {study.get('view_position', 'N/A')}

RADIOLOGICAL FINDINGS:
"""
    # Dynamically inject ALL available findings, regardless of body system
    if findings:
        for region, details in findings.items():
            clinical_text += f"- {region.upper()}: {json.dumps(details)}\n"
    else:
        clinical_text += "- No specific radiological findings documented.\n"

    # Image captions context
    if images_b64:
        clinical_text += "\nAVAILABLE IMAGES:\n"
        for i, img in enumerate(images_b64, 1):
            clinical_text += f"  Image {i} ({img['subtype']}): {img['caption']}\n"

    content = [{"type": "text", "text": clinical_text + f"\n\n{task}"}]
    
    # Attach images
    for img in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{img['mime']};base64,{img['b64']}"}
        })
    
    return [HumanMessage(content=content)]

def preload_model() -> None:
    """
    Pre-warm the local MedGemma model so the first pipeline call isn't slow.
    Call this once before graph.invoke() to load weights into GPU memory.
    """
    use_local = os.getenv("USE_LOCAL_MEDGEMMA", "0") == "1"
    if not use_local:
        print("[preload] Using HF Inference API — no local model to preload.")
        return
    model_id = os.getenv("MEDGEMMA_MODEL", "google/medgemma-1.5-4b-it")
    if hasattr(_call_medgemma_local, "_model") and _call_medgemma_local._model_id == model_id:
        print(f"[preload] Model already loaded: {model_id}")
        return
    # Trigger the lazy-load by making a minimal text-only call
    _call_medgemma_local(
        [HumanMessage(content="Hello")],
        system_prompt="You are a helpful assistant.",
        max_new_tokens=8,
    )
    print(f"[preload] Model ready: {model_id}")


def _call_medgemma(messages: list, system_prompt: str, max_new_tokens: int = 1024) -> str:
    """
    Call MedGemma via HuggingFace Inference or local pipeline.

    Supports two backends:
      1. HuggingFace Inference API (set HF_TOKEN env var)
      2. Local transformers pipeline (set USE_LOCAL_MEDGEMMA=1)
    """
    use_local = os.getenv("USE_LOCAL_MEDGEMMA", "0") == "1"

    if use_local:
        return _call_medgemma_local(messages, system_prompt, max_new_tokens=max_new_tokens)
    else:
        return _call_medgemma_hf_api(messages, system_prompt, max_new_tokens=max_new_tokens)


def _call_medgemma_hf_api(messages: list, system_prompt: str, max_new_tokens: int = 1024) -> str:
    """Call MedGemma via HuggingFace Inference API."""
    from huggingface_hub import InferenceClient

    token = os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set.")

    model = os.getenv("MEDGEMMA_MODEL", "google/medgemma-1.5-4b-it")
    client = InferenceClient(model=model, token=token)
    
    # Build HF-compatible message format
    hf_messages = [{"role": "system", "content": system_prompt}]
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            if isinstance(msg.content, list):
                # Multimodal
                hf_content = []
                for part in msg.content:
                    if part["type"] == "text":
                        hf_content.append({"type": "text", "text": part["text"]})
                    elif part["type"] == "image_url":
                        hf_content.append({
                            "type": "image_url",
                            "image_url": part["image_url"]
                        })
                hf_messages.append({"role": "user", "content": hf_content})
            else:
                hf_messages.append({"role": "user", "content": msg.content})
    
    response = client.chat_completion(
        messages=hf_messages,
        max_tokens=max_new_tokens,
        temperature=0.3,
    )
    
    return response.choices[0].message.content


def _call_medgemma_local(messages: list, system_prompt: str, max_new_tokens: int = 1024) -> str:
    """
    Call MedGemma locally using AutoProcessor + AutoModelForImageTextToText.
    Requires a GPU with sufficient VRAM (≥16 GB for 4b, ≥40 GB for 27b).
    Set MEDGEMMA_MODEL env var to the HuggingFace model ID.
    If the model is gated, set HF_TOKEN env var for authentication.
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from PIL import Image
    import torch
    import io

    model_id = os.getenv("MEDGEMMA_MODEL", "google/medgemma-1.5-4b-it")
    hf_token = os.getenv("HF_TOKEN", None)

    # ── Lazy-load model + processor (cached on the function object) ──────────
    if not hasattr(_call_medgemma_local, "_model") or _call_medgemma_local._model_id != model_id:
        print(f"[medgemma] Loading local model: {model_id}  (this may take a minute)")
        _call_medgemma_local._processor = AutoProcessor.from_pretrained(
            model_id, token=hf_token
        )
        _call_medgemma_local._model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            token=hf_token,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        _call_medgemma_local._model_id = model_id
        _call_medgemma_local._model.eval()
        print("[medgemma] Model loaded.")

    processor = _call_medgemma_local._processor
    model     = _call_medgemma_local._model

    # ── Collect PIL images and build chat messages ────────────────────────────
    pil_images = []
    chat_messages = [{"role": "system", "content": system_prompt}]

    for msg in messages:
        if isinstance(msg, HumanMessage):
            if isinstance(msg.content, list):
                content = []
                for part in msg.content:
                    if part["type"] == "text":
                        content.append({"type": "text", "text": part["text"]})
                    elif part["type"] == "image_url":
                        data_url = part["image_url"]["url"]
                        b64_data = data_url.split(",", 1)[1]
                        img_bytes = base64.b64decode(b64_data)
                        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                        pil_images.append(pil_img)
                        content.append({"type": "image"})
                chat_messages.append({"role": "user", "content": content})
            else:
                chat_messages.append(
                    {"role": "user", "content": [{"type": "text", "text": msg.content}]}
                )

    # ── Tokenise ─────────────────────────────────────────────────────────────
    prompt_text = processor.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=prompt_text,
        images=pil_images if pil_images else None,
        return_tensors="pt",
    ).to(model.device)

    # ── Generate ─────────────────────────────────────────────────────────────
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[-1]
    generated = output_ids[0][input_len:]
    return processor.decode(generated, skip_special_tokens=True)


# ─────────────────────────────────────────────
# Diagnosis Nodes
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert AI medical diagnostic assistant.
You analyze clinical cases with the rigor of a senior physician, considering all
available information: patient history, symptoms, medications, imaging, and lab findings.
Be systematic, evidence-based, and transparent about your reasoning and uncertainty.
When asked for JSON output, respond with ONLY valid JSON — no markdown fences, no commentary before or after."""


def _extract_text_diagnosis(text: str) -> dict | None:
    """
    Fallback parser for when model outputs prose instead of JSON.
    Attempts to extract diagnosis info from markdown/text format.
    """
    import re
    
    result = {}
    
    # Extract primary/main condition
    condition_patterns = [
        r'(?:PRIMARY DIAGNOSIS|Condition|Main Diagnosis)[:\s]*[-\*]?\s*([^\n]+)',
        r'(?:The (?:primary|main) diagnosis is)[:\s]*([^\n\.]+)',
    ]
    for pat in condition_patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            condition = m.group(1).strip().strip('*').strip()
            if condition and len(condition) > 2:
                result['acute_complication'] = {'condition': condition, 'confidence': 85, 'reasoning': 'Extracted from text output'}
                break
    
    # Extract confidence if present
    conf_match = re.search(r'Confidence[:\s]*([\d]+)', text, re.IGNORECASE)
    if conf_match and 'acute_complication' in result:
        try:
            result['acute_complication']['confidence'] = int(conf_match.group(1))
        except ValueError:
            pass
    
    # Extract differentials from numbered or bulleted lists
    diff_section = re.search(r'DIFFERENTIAL DIAGNOSES?[:\s]*(.+?)(?:CRITICAL|CLINICAL REASONING|$)', text, re.DOTALL | re.IGNORECASE)
    if diff_section:
        diff_text = diff_section.group(1)
        # Find numbered items like "1. Condition" or "- Condition"
        diff_items = re.findall(r'(?:\d+\.\s*\*\*|\d+\.\s*|[-\*]\s*\*\*)([^\*\n]+)', diff_text)
        differentials = []
        for item in diff_items[:5]:  # Limit to 5
            cond = item.strip().strip('*').strip()
            if cond and len(cond) > 2 and not cond.lower().startswith(('confidence', 'supporting', 'against')):
                differentials.append({
                    'condition': cond,
                    'confidence': 50,
                    'supporting_evidence': [],
                    'against_evidence': []
                })
        if differentials:
            result['differentials'] = differentials
    
    # Extract critical findings
    crit_section = re.search(r'CRITICAL FINDINGS?[:\s]*(.+?)(?:CLINICAL REASONING|$)', text, re.DOTALL | re.IGNORECASE)
    if crit_section:
        findings = re.findall(r'[-\*]\s*\*\*([^\*]+)\*\*|[-\*]\s*([^\n]+)', crit_section.group(1))
        result['critical_findings'] = [f[0] or f[1] for f in findings if (f[0] or f[1]).strip()][:5]
    
    return result if result else None


def _extract_json_object(text: str) -> dict | None:
    """Robustly extract a JSON object from model output."""
    import re
    # 1. Try fenced ```json ... ``` block
    fenced = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    # 2. Find all '{' positions and try parsing from each (last to first)
    candidates = [m.start() for m in re.finditer(r'\{', text)]
    for start in reversed(candidates):
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            # Try finding the matching closing brace
            depth = 0
            for i, ch in enumerate(text[start:]):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:start + i + 1])
                        except json.JSONDecodeError:
                            break
    
    # 3. Fallback: try to extract from prose/text format
    text_result = _extract_text_diagnosis(text)
    if text_result:
        return text_result
    
    return None


def _extract_json_array(text: str) -> list | None:
    """Robustly extract a JSON array from model output."""
    import re
    # 1. Try fenced ```json ... ``` block
    fenced = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    # 2. Find all '[' positions and try parsing from each (last to first)
    candidates = [m.start() for m in re.finditer(r'\[', text)]
    for start in reversed(candidates):
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            depth = 0
            for i, ch in enumerate(text[start:]):
                if ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:start + i + 1])
                        except json.JSONDecodeError:
                            break
    return None

def initial_diagnosis_node(state: DiagnosisState) -> DiagnosisState:
    """Generate initial diagnosis with differential and confidence scores."""
    print("[diagnosis] Generating initial diagnosis...")

    task = """Analyze the clinical case and images above. Respond with ONLY a JSON object (no other text).

CRITICAL INSTRUCTIONS:
1. Separate the underlying etiology (root cause) from the acute complication or primary focus that requires immediate intervention.
2. MEDICAL REALITY CHECK: Never assign 100% confidence to any single condition. Always distribute probabilities to leave room for uncertainty and differentials.

Required JSON structure:
{
  "underlying_etiology": {"condition": "<name>", "confidence": <0-99>},
  "acute_complication": {"condition": "<name>", "confidence": <0-99>, "requires_intervention": true, "reasoning": "<1-2 sentences>"},
  "differentials": [
    {"condition": "<name>", "confidence": <0-99>, "supporting_evidence": ["<evidence>"], "against_evidence": ["<evidence>"]}
  ],
  "critical_findings": ["<finding>"],
  "clinical_reasoning": "<step-by-step reasoning>"
}

Provide 3-5 differentials. Be specific and evidence-based."""

    # Retry logic: try up to 2 times if JSON parsing fails
    max_retries = 2
    diagnosis_data = None
    response = ""
    
    for attempt in range(max_retries):
        messages = _build_medgemma_prompt(state["case_data"], state["images_b64"], task)
        response = _call_medgemma(messages, SYSTEM_PROMPT, max_new_tokens=2048)

        # Debug: show raw response
        print(f"[diagnosis] Attempt {attempt + 1} - Raw response ({len(response)} chars):")
        print(response[:500])
        if len(response) > 500:
            print(f"... ({len(response) - 500} more chars)")

        # Robust JSON extraction (includes text fallback)
        diagnosis_data = _extract_json_object(response)
        
        if diagnosis_data is not None and diagnosis_data.get('acute_complication') or diagnosis_data.get('differentials'):
            print(f"[diagnosis] JSON extraction successful on attempt {attempt + 1}")
            break
        elif attempt < max_retries - 1:
            print(f"[diagnosis] JSON extraction failed, retrying with stricter prompt...")
            # Make the task prompt stricter for retry
            task = """IMPORTANT: You MUST respond with ONLY valid JSON. No explanatory text before or after.

Analyze the clinical case. Return this exact JSON structure:
{"underlying_etiology": {"condition": "<name>", "confidence": <0-99>}, "acute_complication": {"condition": "<name>", "confidence": <0-99>, "requires_intervention": true, "reasoning": "<1-2 sentences>"}, "differentials": [{"condition": "<name>", "confidence": <0-99>, "supporting_evidence": ["<evidence>"], "against_evidence": ["<evidence>"]}], "critical_findings": ["<finding>"], "clinical_reasoning": "<reasoning>"}

Provide 3-5 differentials. Start your response with { and end with }."""

    if diagnosis_data is None:
        print("[diagnosis] WARNING: Could not parse JSON from response after retries, storing raw text.")
        diagnosis_data = {"raw_response": response}

    return {
        **state,
        "initial_diagnosis": json.dumps(diagnosis_data, indent=2),
        "differential_diagnoses": diagnosis_data.get("differentials", []),
        "messages": [AIMessage(content=f"Initial diagnosis generated: {diagnosis_data.get('acute_complication', {}).get('condition', 'unknown')}")]
    }
    
def bias_check_node(state: DiagnosisState) -> DiagnosisState:
    """Check for diagnostic anchoring bias and cognitive shortcuts."""
    print("[bias_check] Performing cognitive bias check...")
    
    initial = state["initial_diagnosis"]
    
    task = f"""
You have generated an initial diagnosis. Now perform a COGNITIVE BIAS AUDIT:

Initial diagnosis was:
{initial}

Check for these specific biases:
1. ANCHORING BIAS: Are you too fixated on the first or most obvious finding?
2. AVAILABILITY BIAS: Are common diagnoses being over-weighted?
3. PREMATURE CLOSURE: Have you stopped considering alternatives too early?
4. FRAMING EFFECT: How is the case presentation framing your thinking?
5. REPRESENTATIVE HEURISTIC: Are you pattern-matching too quickly?
6. RARE DISEASE NEGLECT: What uncommon but serious conditions are being missed?
7. CONFIRMATION BIAS: Are you selectively weighing evidence?

For each bias found, explain:
- What the bias is in this case
- How it might be skewing the diagnosis
- What to reconsider

Also identify: What diagnoses might a physician MISS due to these biases?
"""
    
    messages = _build_medgemma_prompt(state["case_data"], [], task)  # No images needed for bias check
    response = _call_medgemma(messages, SYSTEM_PROMPT, max_new_tokens=1024)

    print(f"[bias_check] Response ({len(response)} chars)")

    return {
        **state,
        "bias_check_notes": response,
        "messages": [AIMessage(content="Bias check completed.")]
    }


def _extract_text_alternatives(text: str) -> list | None:
    """
    Fallback parser for extracting alternative diagnoses from prose output.
    """
    import re
    
    alternatives = []
    
    # Look for numbered items like "1. Condition Name" or "**Condition Name**"
    patterns = [
        r'\d+\.\s*\*\*([^\*]+)\*\*',  # 1. **Condition**
        r'\d+\.\s*([^:\n]+?)(?::|\n)',  # 1. Condition:
        r'[-\*]\s*\*\*([^\*]+)\*\*',  # - **Condition**
    ]
    
    for pat in patterns:
        matches = re.findall(pat, text)
        for match in matches:
            cond = match.strip()
            # Filter out common non-diagnosis phrases
            if (cond and len(cond) > 3 and len(cond) < 100 and 
                not any(skip in cond.lower() for skip in 
                       ['confidence', 'evidence', 'missed', 'why', 'risk', 'test', 'supporting'])):
                # Try to extract confidence for this condition
                conf_match = re.search(rf'{re.escape(cond)}.*?(?:confidence|Confidence)[:\s]*(\d+)', text, re.DOTALL)
                confidence = int(conf_match.group(1)) if conf_match else 50
                
                # Try to extract risk level
                risk_match = re.search(rf'{re.escape(cond)}.*?(?:risk|Risk)[:\s]*([\w]+)', text, re.DOTALL)
                risk = risk_match.group(1).lower() if risk_match else 'medium'
                if risk not in ['low', 'medium', 'high', 'critical']:
                    risk = 'medium'
                
                alternatives.append({
                    'condition': cond,
                    'confidence': confidence,
                    'why_missed': 'Extracted from text output',
                    'supporting_evidence': [],
                    'confirmatory_tests': [],
                    'risk_if_missed': risk
                })
        if alternatives:
            break
    
    return alternatives[:5] if alternatives else None


def alternative_hypotheses_node(state: DiagnosisState) -> DiagnosisState:
    """Generate alternative diagnoses that might be missed."""
    print("[alternatives] Generating alternative hypotheses...")

    bias_notes = state.get("bias_check_notes", "")
    initial = state["initial_diagnosis"]

    task = f"""Given this initial diagnosis and bias analysis, generate 3-5 alternative diagnoses that may have been missed.

Initial diagnosis:
{initial}

Bias analysis highlights:
{bias_notes[:800] if bias_notes else 'N/A'}

Consider: atypical presentations, rare but serious conditions, dual pathology, mimickers, and systemic diseases.

Respond with ONLY a JSON array (no other text). Each element must have:
{{"condition":"<name>","why_missed":"<reason>","supporting_evidence":["<evidence>"],"confirmatory_tests":["<test>"],"risk_if_missed":"high|medium|low|critical","confidence":<0-100>}}"""

    messages = _build_medgemma_prompt(state["case_data"], state["images_b64"], task)
    response = _call_medgemma(messages, SYSTEM_PROMPT, max_new_tokens=2048)

    # Debug: show raw response
    print(f"[alternatives] Raw response ({len(response)} chars):")
    print(response[:500])
    if len(response) > 500:
        print(f"... ({len(response) - 500} more chars)")

    # Robust JSON extraction
    alts = _extract_json_array(response)
    if alts is None:
        print("[alternatives] WARNING: Could not parse JSON array, trying object fallback.")
        obj = _extract_json_object(response)
        if obj and isinstance(obj, dict):
            # Model may have wrapped array in an object
            for v in obj.values():
                if isinstance(v, list):
                    alts = v
                    break
        if alts is None:
            # Try text extraction fallback
            print("[alternatives] Trying text extraction fallback...")
            alts = _extract_text_alternatives(response)
        if alts is None:
            alts = [{"raw_response": response, "condition": "See raw output", "confidence": 0, "risk_if_missed": "unknown"}]

    return {
        **state,
        "alternative_hypotheses": alts,
        "messages": [AIMessage(content=f"Generated {len(alts)} alternative hypotheses.")]
    }


def final_report_node(state: DiagnosisState) -> DiagnosisState:
    """Compile everything into a final structured diagnostic report."""
    print("[report] Compiling final diagnostic report...")
    
    case = state["case_data"]
    patient = case.get("patient", {})
    prov = case.get("provenance", {})
    
    initial_data = {}
    try:
        initial_data = json.loads(state.get("initial_diagnosis", "{}"))
    except:
        pass
    
    alts = state.get("alternative_hypotheses", [])
    
    # Format differentials
    def fmt_list(items, key):
        return "\n".join(f"  - {i.get(key, str(i))}" for i in items) if items else "  None"
    
    report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║           MEDGEMMA DIFFERENTIAL DIAGNOSIS REPORT                    ║
╚══════════════════════════════════════════════════════════════════════╝

CASE ID: {state['case_id']}
SOURCE:  {prov.get('article_title', 'N/A')} ({prov.get('journal', 'N/A')}, {prov.get('year', 'N/A')})
PMC ID:  {prov.get('pmc_id', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATIENT SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{case.get('summary', {}).get('one_liner', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PRIMARY DIAGNOSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Condition:   {initial_data.get('acute_complication', initial_data.get('underlying_etiology', {})).get('condition', 'See raw output')}
Confidence:  {initial_data.get('acute_complication', initial_data.get('underlying_etiology', {})).get('confidence', '?')}%
Reasoning:   {initial_data.get('acute_complication', initial_data.get('underlying_etiology', {})).get('reasoning', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
UNDERLYING ETIOLOGY (if different from primary)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Condition:   {initial_data.get('underlying_etiology', {}).get('condition', 'Same as primary or N/A')}
Confidence:  {initial_data.get('underlying_etiology', {}).get('confidence', 'N/A')}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIFFERENTIAL DIAGNOSES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
    
    for i, diff in enumerate(state.get("differential_diagnoses", []), 1):
        report += f"""
  [{i}] {diff.get('condition', 'Unknown')} — Confidence: {diff.get('confidence', '?')}%
      Supporting: {'; '.join(diff.get('supporting_evidence', [])[:2])}
      Against:    {'; '.join(diff.get('against_evidence', [])[:2])}"""
    
    report += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
COGNITIVE BIAS ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{state.get('bias_check_notes', 'Not performed')[:800]}...

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MISSED DIAGNOSES / ALTERNATIVE HYPOTHESES  
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
    
    for i, alt in enumerate(alts, 1):
        if isinstance(alt, dict) and "condition" in alt:
            report += f"""
  [{i}] {alt.get('condition')} (Risk if missed: {alt.get('risk_if_missed', '?').upper()})
      Why missed:  {alt.get('why_missed', 'N/A')}
      Evidence:    {'; '.join(alt.get('supporting_evidence', [])[:2])}
      Confirm via: {'; '.join(alt.get('confirmatory_tests', [])[:3])}"""
    
    # Ground truth comparison
    actual = case.get("assessment", {})
    report += f"""

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GROUND TRUTH (from case record)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Actual Primary Diagnosis: {actual.get('diagnosis_primary', 'N/A')}
Suspected:                {', '.join(actual.get('suspected_primary', []))}
Documented Differentials: {', '.join(actual.get('differential', []))}
Urgency:                  {actual.get('urgency', 'N/A')}
Outcome:                  {case.get('outcome', {}).get('detail', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠  DISCLAIMER: For research/educational use only. Not for clinical decisions.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    return {
        **state,
        "final_report": report,
        "messages": [AIMessage(content="Final report compiled.")]
    }


# ─────────────────────────────────────────────
# Graph Construction
# ─────────────────────────────────────────────

def build_diagnosis_graph() -> StateGraph:
    graph = StateGraph(DiagnosisState)
    
    graph.add_node("load_case", load_case_node)
    graph.add_node("load_images", load_images_node)
    graph.add_node("initial_diagnosis", initial_diagnosis_node)
    graph.add_node("bias_check", bias_check_node)
    graph.add_node("alternative_hypotheses", alternative_hypotheses_node)
    graph.add_node("final_report", final_report_node)
    
    graph.set_entry_point("load_case")
    graph.add_edge("load_case", "load_images")
    graph.add_edge("load_images", "initial_diagnosis")
    graph.add_edge("initial_diagnosis", "bias_check")
    graph.add_edge("bias_check", "alternative_hypotheses")
    graph.add_edge("alternative_hypotheses", "final_report")
    graph.add_edge("final_report", END)
    
    return graph.compile()


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

def run_diagnosis(case_id: str) -> dict:
    """Run the full diagnosis pipeline for a given case ID."""
    graph = build_diagnosis_graph()
    
    initial_state: DiagnosisState = {
        "case_id": case_id,
        "case_data": {},
        "images_b64": [],
        "initial_diagnosis": None,
        "differential_diagnoses": None,
        "bias_check_notes": None,
        "alternative_hypotheses": None,
        "final_report": None,
        "messages": [],
    }
    
    final_state = graph.invoke(initial_state)
    
    print("\n" + "="*70)
    print(final_state["final_report"])
    print("="*70)
    
    return final_state


if __name__ == "__main__":
    import sys
    case_id = sys.argv[1] if len(sys.argv) > 1 else "e0a8f078-fb8f-5281-ae67-256e060d0ef0"
    run_diagnosis(case_id)


# ─────────────────────────────────────────────
# Interactive Chat with Case Context
# ─────────────────────────────────────────────

class CaseChat:
    """
    Interactive chat session with patient case context.
    Maintains conversation history for multi-turn dialogue.
    """

    def __init__(self, final_state: dict):
        """
        Initialize chat with a completed diagnosis pipeline state.

        Args:
            final_state: The dict returned by graph.invoke() containing
                         case_data, images_b64, initial_diagnosis, etc.
        """
        self.state = final_state
        self.case_data = final_state["case_data"]
        self.images_b64 = final_state.get("images_b64", [])
        self.diagnosis = final_state.get("initial_diagnosis", "{}")
        self.differentials = final_state.get("differential_diagnoses", [])
        self.alternatives = final_state.get("alternative_hypotheses", [])
        self.history: List[dict] = []  # [{"role": "user/assistant", "content": "..."}]

    def _build_context_prompt(self) -> str:
        """Build a compact case summary for chat context."""
        patient = self.case_data.get("patient", {})
        presentation = self.case_data.get("presentation", {})
        assessment = self.case_data.get("assessment", {})

        return f"""You are a medical AI assistant discussing a specific patient case. Use the case context below to answer questions accurately.

=== PATIENT CASE CONTEXT ===
Patient: {patient.get('age_years', '?')} y/o {patient.get('sex', '?')}
Chief Complaint: {presentation.get('chief_complaint', 'N/A')}
HPI: {presentation.get('hpi', 'N/A')[:500]}
Comorbidities: {', '.join(patient.get('comorbidities', [])) or 'None'}
Medications: {', '.join(patient.get('medications', [])) or 'None'}

=== AI DIAGNOSIS ===
{self.diagnosis}

=== GROUND TRUTH ===
Actual Diagnosis: {assessment.get('diagnosis_primary', 'N/A')}
Differentials: {', '.join(assessment.get('differential', []))}

Answer the user's questions about this case. Be specific, cite findings from the case, and explain your reasoning. If asked about treatment or prognosis, note that this is for educational discussion only."""

    def ask(self, question: str, include_images: bool = False) -> str:
        """
        Send a question about the case and get a response.

        Args:
            question: User's question about the case
            include_images: Whether to include case images in the context (slower but more accurate for imaging questions)

        Returns:
            AI response string
        """
        # Build messages with conversation history
        system_prompt = self._build_context_prompt()

        # Construct content - text only or multimodal
        if include_images and self.images_b64:
            content = [{"type": "text", "text": question}]
            for img in self.images_b64[:4]:  # Limit to 4 images
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{img['mime']};base64,{img['b64']}"}
                })
            user_msg = HumanMessage(content=content)
        else:
            user_msg = HumanMessage(content=question)

        # Include recent history (last 6 turns to avoid context overflow)
        messages = []
        for turn in self.history[-6:]:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            else:
                messages.append(AIMessage(content=turn["content"]))
        messages.append(user_msg)

        # Call MedGemma
        response = _call_medgemma(messages, system_prompt, max_new_tokens=1024)

        # Update history
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": response})

        return response

    def clear_history(self):
        """Reset conversation history."""
        self.history = []
        print("Chat history cleared.")

    def get_diagnosis_summary(self) -> dict:
        """Return a structured summary of diagnoses for display."""
        try:
            diag_data = json.loads(self.diagnosis)
        except:
            diag_data = {}

        # Support both schema variants: acute_complication (correct) or primary_diagnosis (legacy)
        primary = diag_data.get("acute_complication", diag_data.get("primary_diagnosis", {}))
        # Also check for underlying_etiology as a secondary primary source
        if not primary.get("condition"):
            primary = diag_data.get("underlying_etiology", {})
        diffs = diag_data.get("differentials", [])

        # Build ranked list
        ranked = []
        if primary.get("condition"):
            ranked.append({
                "rank": 1,
                "condition": primary["condition"],
                "confidence": primary.get("confidence", 0),
                "reasoning": primary.get("reasoning", ""),
                "type": "primary"
            })

        for i, d in enumerate(diffs, start=2):
            ranked.append({
                "rank": i,
                "condition": d.get("condition", "Unknown"),
                "confidence": d.get("confidence", 0),
                "reasoning": "; ".join(d.get("supporting_evidence", [])[:2]),
                "type": "differential"
            })

        # Add alternatives
        for alt in self.alternatives:
            if isinstance(alt, dict) and alt.get("condition"):
                ranked.append({
                    "rank": len(ranked) + 1,
                    "condition": alt["condition"],
                    "confidence": alt.get("confidence", 0),
                    "reasoning": alt.get("why_missed", ""),
                    "type": "alternative",
                    "risk_if_missed": alt.get("risk_if_missed", "unknown")
                })

        return {
            "primary": primary,
            "ranked_list": ranked,
            "ground_truth": self.case_data.get("assessment", {}).get("diagnosis_primary", "N/A")
        }


def create_chat_session(final_state: dict) -> CaseChat:
    """
    Create an interactive chat session from a completed diagnosis pipeline.

    Usage:
        final_state = graph.invoke(initial_state)
        chat = create_chat_session(final_state)
        response = chat.ask("Why did you rule out tuberculosis?")
    """
    return CaseChat(final_state)
