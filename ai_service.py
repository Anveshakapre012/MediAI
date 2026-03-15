#!/usr/bin/env python3
"""Simple local Python NLP service for consultation note generation."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse


HOST = "127.0.0.1"
PORT = 8765
AI_MODEL_NAME = "MediaiMini-NLP-Python"
KEY_POINT_LIMIT = 5
UNCLEAR = "[UNCLEAR - needs clarification]"
CLINICIAN_DEFAULT = "Dr. Meera Nair"


SYMPTOM_TERMS: Sequence[str] = (
    "pain",
    "ache",
    "headache",
    "fever",
    "cough",
    "sore throat",
    "fatigue",
    "dizziness",
    "nausea",
    "vomit",
    "shortness of breath",
    "chest pain",
    "diarrhea",
    "stomach",
)

PLAN_TERMS: Sequence[str] = (
    "follow up",
    "follow-up",
    "monitor",
    "return",
    "recheck",
    "continue",
    "start",
    "stop",
    "prescribed",
    "refer",
    "schedule",
    "advised",
    "recommend",
    "test",
    "order",
)

ASSESSMENT_TERMS: Sequence[str] = (
    "diagnosed",
    "assessment",
    "likely",
    "probable",
    "differential",
    "impression",
    "consistent with",
    "appears to be",
    "suggests",
    "consider",
)

RED_FLAG_TERMS: Sequence[str] = (
    "chest pain",
    "shortness of breath",
    "difficulty breathing",
    "loss of consciousness",
    "severe",
    "unresponsive",
    "bleeding",
    "urgent",
    "emergency",
)

ALLERGY_TERMS: Sequence[str] = (
    "allergy",
    "allergies",
    "allergic",
    "angioedema",
    "anaphylaxis",
    "rash",
    "hives",
    "urticaria",
    "critical history",
)

WORSENING_TERMS: Sequence[str] = (
    "worse",
    "worsening",
    "deteriorating",
    "progressively",
    "increasing",
    "more severe",
    "getting worse",
    "not improving",
    "spreading",
)

MED_LIBRARY: Sequence[str] = (
    "paracetamol",
    "acetaminophen",
    "ibuprofen",
    "amoxicillin",
    "metformin",
    "lisinopril",
    "atorvastatin",
    "omeprazole",
    "aspirin",
)

VITAL_PATTERNS = (
    (r"\b(?:blood pressure|bp)\s*(?:is|:)?\s*(\d{2,3}\s*/\s*\d{2,3})\b", "blood pressure"),
    (r"\b(?:heart rate|pulse|hr)\s*(?:is|:)?\s*(\d{2,3})\s*(?:bpm)?\b", "heart rate"),
    (r"\b(?:temperature|temp)\s*(?:is|:)?\s*(\d{2,3}(?:\.\d)?)\b", "temperature"),
    (r"\b(?:oxygen saturation|spo2|o2 sat|oxygen)\s*(?:is|:)?\s*(\d{2,3})\s*%?\b", "oxygen saturation"),
    (r"\b(?:respiratory rate|respiration)\s*(?:is|:)?\s*(\d{1,2})\b", "respiratory rate"),
)

FOLLOW_UP_RE = re.compile(
    r"\b(?:follow[\s-]?up|return|review|recheck)\b.*?(?:in|after)\s+(\d+)\s+(day|days|week|weeks|month|months)\b",
    re.IGNORECASE,
)


def chunk_long_sentence(sentence: str, max_words: int = 40, chunk_size: int = 32) -> List[str]:
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]
    return [" ".join(words[i : i + chunk_size]).strip() for i in range(0, len(words), chunk_size)]


def split_transcript_sentences(transcript: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", transcript or "").strip()
    if not normalized:
        return []
    fragments = [fragment.strip() for fragment in re.split(r"[.!?;,]+|\r\n|\r|\n", normalized) if fragment.strip()]
    if not fragments:
        return []
    chunks: List[str] = []
    for fragment in fragments:
        chunks.extend(chunk_long_sentence(fragment))
    return [chunk for chunk in chunks if chunk.strip()]


def speaker_label_for_sentence(sentence: str) -> str:
    normalized = sentence.lower()
    doctor_indicators = (
        "i suggest",
        "i recommend",
        "i prescribed",
        "prescribe",
        "advised",
        "plan",
        "please take",
        "start taking",
        "we will start",
        "we will adjust",
        "i will start",
        "review",
        "order",
        "referred",
        "x-ray",
        "ecg",
        "diagnosis",
        "assessment",
        "follow up",
        "follow-up",
        "followed up",
        "schedule",
        "continue",
        "continue taking",
        "increase",
        "decrease",
        "dose",
        "dosage",
        "imaging",
        "lab results show",
    )
    patient_indicators = (
        "i have",
        "i'm",
        "i am",
        "i feel",
        "i feel like",
        "my pain",
        "symptom",
        "symptoms",
        "chest pain",
        "headache",
        "fever",
        "cough",
        "nausea",
        "sore throat",
        "dizziness",
        "my",
        "i have been",
        "hurt",
        "hurts",
        "lasted",
        "since",
        "for the past",
        "for the last",
    )

    doctor_score = 0
    patient_score = 0
    for phrase in doctor_indicators:
        if phrase in normalized:
            doctor_score += 1
    for phrase in patient_indicators:
        if phrase in normalized:
            patient_score += 1

    if doctor_score == patient_score:
        return "UNKNOWN"
    return "DOCTOR" if doctor_score > patient_score else "PATIENT"

SPEAKER_LABEL_RE = re.compile(r"^\s*(?:\[(doctor|patient)\]|(doctor|patient)\s*:)\s*", re.IGNORECASE)


def tag_transcript(sentences: List[str]) -> List[Dict[str, str]]:
    last_label = "PATIENT"
    tagged: List[Dict[str, str]] = []
    for sentence in sentences:
        label_match = SPEAKER_LABEL_RE.match(sentence)
        cleaned_sentence = sentence
        if label_match:
            matched_label = label_match.group(1) or label_match.group(2) or ""
            label = "DOCTOR" if matched_label.lower().startswith("doctor") else "PATIENT"
            cleaned_sentence = SPEAKER_LABEL_RE.sub("", sentence).strip() or sentence
        else:
            label = speaker_label_for_sentence(sentence)
        final_label = label if label != "UNKNOWN" else last_label
        last_label = final_label
        tagged.append({
            "speaker": final_label,
            "text": cleaned_sentence,
            "lower_text": cleaned_sentence.lower(),
        })
    return tagged


def normalize_list(values: Any) -> List[str]:
    if isinstance(values, (list, tuple)):
        return unique_keep_order([str(value).strip() for value in values if str(value).strip()])
    if isinstance(values, str):
        text = values.strip()
        return [text] if text else []
    return []


def parse_patient_identity_from_text(lines: List[Dict[str, str]]) -> Dict[str, str]:
    for line in lines:
        text = line["text"]
        age_match = re.search(r"\b(\d{1,3})\s*(?:y(?:ears?)?\s*)?old\b", text, flags=re.IGNORECASE)
        name_match = re.search(r"(?:patient name|name)\s*[:\-]?\s*([a-z][a-z.'\s]+)", text, flags=re.IGNORECASE)
        patient_age = age_match.group(1) if age_match else ""
        patient_name = name_match.group(1).strip() if name_match else ""
        if patient_name or patient_age:
            return {
                "patient_name": patient_name,
                "patient_age": patient_age,
            }
    return {
        "patient_name": "",
        "patient_age": "",
    }


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        text = item.strip()
        if not text:
            continue
        if text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def contains_any(text: str, terms: Sequence[str]) -> bool:
    lower_text = text.lower()
    return any(term in lower_text for term in terms)


def gather_vital_mentions(sentence: str) -> List[str]:
    findings: List[str] = []
    for pattern, name in VITAL_PATTERNS:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match and match.group(1):
            findings.append(f"{name}: {match.group(1)}")
    return findings


def gather_medication_mentions(sentence: str) -> List[str]:
    findings: List[str] = []
    dose_pattern = re.findall(
        r"\b([a-z][a-z0-9\s\-]{1,30}?)\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|iu|ml|units?)\b",
        sentence,
        flags=re.IGNORECASE,
    )
    if dose_pattern:
        findings.extend(dose_pattern)

    lower_sentence = sentence.lower()
    for med in MED_LIBRARY:
        if med in lower_sentence:
            findings.append(f"possible {med}")

    return unique_keep_order(findings)


def score_sentence(sentence: str, lower_sentence: str) -> int:
    score = 0
    if not sentence:
        return score

    if len(sentence) > 160:
        score += 2
    elif len(sentence) > 90:
        score += 1

    if re.search(r"\d", sentence):
        score += 1

    if re.search(r"\b(i|i'm|patient|history|complaints?|symptoms?|condition|medication|prescribed|follow up|follow-up|referred|recommended|plan)\b", lower_sentence, flags=re.IGNORECASE):
        score += 2

    return score


def rank_sentences_by_score(sentences: Iterable[str], sentence_scores: Dict[str, int], sentence_order: Dict[str, int], limit: int = KEY_POINT_LIMIT) -> List[str]:
    normalized = unique_keep_order(list(sentences))
    if not normalized:
        return []

    ranked: List[tuple] = []
    for index, sentence in enumerate(normalized):
        ranked.append((sentence, sentence_scores.get(sentence, 0), sentence_order.get(sentence, index)))

    total_sentences = max(len(ranked), 1)
    buckets: List[List[tuple]] = [[], [], []]
    for item in ranked:
        sentence_order_index = item[2]
        bucket = int((sentence_order_index / total_sentences) * 3)
        if bucket < 0:
            bucket = 0
        elif bucket > 2:
            bucket = 2
        buckets[bucket].append(item)

    for bucket in buckets:
        bucket.sort(key=lambda item: (-item[1], item[2]))

    selected: List[tuple] = []
    max_rounds = max(len(bucket) for bucket in buckets)
    for round_index in range(max_rounds):
        if len(selected) >= limit:
            break
        for bucket in buckets:
            if len(selected) >= limit:
                break
            if round_index < len(bucket):
                selected.append(bucket[round_index])

    if len(selected) >= limit:
        return [sentence for sentence, _, _ in selected[:limit]]

    remaining = [item for item in ranked if item not in selected]
    remaining.sort(key=lambda item: (-item[1], item[2]))
    return [sentence for sentence, _, _ in (selected + remaining)[:limit]]


def evaluate_risk_level(red_flags: Sequence[str]) -> str:
    count = len(red_flags)
    if count == 0:
        return "low"
    if count >= 3:
        return "high"
    return "moderate"


def clean_clinical_text(text: str, max_words: int = 22) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"^(doctor|patient)\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)
    words = cleaned.split()
    if len(words) > max_words:
        return f"{' '.join(words[:max_words])}..."
    return cleaned


def extract_family_history_text(lines: List[Dict[str, str]]) -> List[str]:
    family_terms = (
        "father",
        "mother",
        "sister",
        "brother",
        "family history",
        "family",
        "hypertension",
        "diabetes",
        "heart attack",
        "stroke",
        "asthma",
        "cancer",
    )
    return unique_keep_order(
        [
            clean_clinical_text(line["text"])
            for line in lines
            if any(term in line["lower_text"] for term in family_terms)
        ]
    )[:KEY_POINT_LIMIT]


def extract_allergies(lines: List[Dict[str, str]]) -> List[str]:
    return unique_keep_order(
        [
            clean_clinical_text(line["text"])
            for line in lines
            if any(term in line["lower_text"] for term in ALLERGY_TERMS)
        ]
    )[:KEY_POINT_LIMIT]


def extract_worsening_symptoms(lines: List[Dict[str, str]]) -> List[str]:
    worsening_mentions: List[str] = []
    for line in lines:
        lower_text = line["lower_text"]
        if not any(term in lower_text for term in WORSENING_TERMS):
            continue
        match = re.search(
            r"\b(chest pain|shortness of breath|headache|fatigue|cough|fever|dizziness|nausea|abdominal pain|diarrhea|stomach|pain)\b",
            lower_text,
        )
        if match:
            worsening_mentions.append(clean_clinical_text(line["text"]))
    return unique_keep_order(worsening_mentions)[:KEY_POINT_LIMIT]


def build_worsening_vs_prior(current_worsening: List[str], prior_sessions: List[Dict[str, Any]]) -> List[str]:
    if not current_worsening or not prior_sessions:
        return []

    prior_symptoms = set()
    for prior in prior_sessions[:5]:
        prior_summary = prior.get("aiSummary") or {}
        prior_symptoms.update(map(str.lower, normalize_list(prior_summary.get("symptomsDescribed") or prior_summary.get("symptoms") or [])))
        prior_symptoms.update(
            map(str.lower, normalize_list(prior_summary.get("worseningSymptoms") or prior_summary.get("worsening") or []))
        )

    matched = []
    for line in current_worsening:
        lower_line = line.lower()
        if any(previous in lower_line for previous in prior_symptoms):
            matched.append(line)

    return unique_keep_order(matched)[:KEY_POINT_LIMIT]


def extract_follow_up_instructions(lines: List[Dict[str, str]]) -> List[str]:
    direct_terms = (
        "follow up",
        "follow-up",
        "revisit",
        "return for review",
        "recheck",
        "repeat",
        "next visit",
    )

    mentions: List[str] = []
    for line in lines:
        text = line["text"]
        lowered = line["lower_text"]
        match = FOLLOW_UP_RE.search(text)
        if match:
            mentions.append(clean_clinical_text(match.group(0)))
            continue
        if any(term in lowered for term in direct_terms):
            mentions.append(clean_clinical_text(text))
    return unique_keep_order(mentions)[:KEY_POINT_LIMIT]


def parse_follow_up_warning(prior_sessions: Optional[List[Dict[str, Any]]]) -> str:
    if not prior_sessions:
        return ""

    latest = prior_sessions[0]
    latest_summary = latest.get("aiSummary") or {}
    followup = (
        latest_summary.get("followUpInstructions")
        or latest_summary.get("followUp")
        or latest_summary.get("nextSteps")
    )
    followup_text = " ".join(normalize_list(followup))
    match = re.search(r"\b(\d+)\s*(day|days|week|weeks|month|months)\b", followup_text, flags=re.IGNORECASE)
    if not match:
        return ""

    amount = int(match.group(1))
    unit = match.group(2).lower()
    due_date = latest.get("savedAt") or datetime.now().isoformat()
    due = None
    try:
        due = datetime.fromisoformat(due_date)
    except (TypeError, ValueError):
        for fmt in ("%m/%d/%Y, %I:%M:%S %p", "%m/%d/%Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
            try:
                due = datetime.strptime(due_date, fmt)
                break
            except (TypeError, ValueError):
                due = None
    if due is None:
        due = datetime.now()

    if unit.startswith("week"):
        days = amount * 7
    elif unit.startswith("month"):
        # approximate month as 30 days for this demo
        days = amount * 30
    else:
        days = amount

    target = due + timedelta(days=days)
    delta = target - datetime.now()
    if delta.days >= 0:
        return f"Follow-up due in {delta.days} day(s)."
    return f"⚠ Follow-up overdue by {abs(delta.days)} day(s) from prior session ({latest.get('sessionId', 'prior session')})."


def extract_symptoms_from_patient_lines(lines: List[Dict[str, str]]) -> List[str]:
    symptom_terms = (
        "pain",
        "ache",
        "headache",
        "fever",
        "cough",
        "sore throat",
        "fatigue",
        "dizziness",
        "nausea",
        "vomit",
        "shortness of breath",
        "chest pain",
        "diarrhea",
        "stomach",
    )
    severity_terms = (
        "mild",
        "moderate",
        "severe",
        "worsening",
        "sudden",
        "constant",
        "intermittent",
        "acute",
    )
    trigger_terms = (
        "after",
        "while",
        "during",
        "triggered by",
        "on exertion",
        "on movement",
        "after eating",
        "in the morning",
        "at night",
    )

    fragments: List[str] = []
    for line in lines:
        text = line["text"]
        lower_text = line["lower_text"]
        if not any(term in lower_text for term in symptom_terms):
            continue
        duration_match = re.search(r"\b(last|for)\s+(\d+\s*(?:day|days|week|weeks|month|months|hour|hours|minute|minutes))\b", lower_text)
        duration = duration_match.group(0) if duration_match else ""
        severities = [term for term in severity_terms if term in lower_text]
        triggers = [term for term in trigger_terms if term in lower_text]
        descriptors = []
        if duration:
            descriptors.append(duration)
        if severities:
            descriptors.append(f"severity: {severities[0]}")
        if triggers:
            descriptors.append(f"trigger: {triggers[0]}")
        if descriptors:
            fragments.append(clean_clinical_text(f"{text} ({', '.join(descriptors)})"))
        else:
            fragments.append(clean_clinical_text(text))
    return unique_keep_order(fragments)[:KEY_POINT_LIMIT]


def extract_tests_from_lines(lines: List[Dict[str, str]]) -> List[str]:
    terms = (
        "ecg",
        "electrocardiogram",
        "xray",
        "x-ray",
        "chest xray",
        "blood test",
        "cbc",
        "ct",
        "mri",
        "ultrasound",
        "x ray",
        "urine",
        "lft",
        "esr",
        "crp",
        "glucose",
    )
    return unique_keep_order(
        [clean_clinical_text(line["text"]) for line in lines if any(term in line["lower_text"] for term in terms)]
    )[:KEY_POINT_LIMIT]


def extract_prescriptions_from_doctor(lines: List[Dict[str, str]]) -> List[str]:
    med_pattern = re.compile(
        r"\b([a-z][a-z0-9\s\-]{1,30}?)\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|iu|ml|units?)\b",
        flags=re.IGNORECASE,
    )
    freq_pattern = re.compile(r"\b(once\s+daily|twice\s+daily|thrice\s+daily|daily|bid|tid|qid|prn|at\s+bedtime)\b", flags=re.IGNORECASE)
    prescriptions: List[str] = []

    for line in lines:
        text = line["text"]
        lower_text = line["lower_text"]
        matches = med_pattern.findall(text)
        freq_match = freq_pattern.search(lower_text)
        freq = f" - {freq_match.group(0)}" if freq_match else ""

        if matches:
            for med in matches:
                prescriptions.append(f"{med.strip()}{freq}")
        else:
            mentions = gather_medication_mentions(text)
            if mentions:
                for med in mentions:
                    prescriptions.append(f"{med}{freq}")
            elif re.search(r"\b(rx|prescribed|start|take)\b", lower_text):
                prescriptions.append(clean_clinical_text(text))

    return unique_keep_order(prescriptions)[:KEY_POINT_LIMIT]


def build_transcript_summary(transcript: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    options = options or {}
    clean_transcript = transcript.strip()
    if not clean_transcript:
        patient = options.get("patient", {}) or {}
        return {
            "model": AI_MODEL_NAME,
            "sessionId": options.get("sessionId", f"MS-{int(datetime.now().timestamp())}"),
            "sessionDate": options.get("sessionDate", datetime.now().strftime("%B %d, %Y")),
            "generatedAt": datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p"),
            "transcriptLength": 0,
            "sentenceCount": 0,
            "patientName": patient.get("name", UNCLEAR),
            "patientAge": patient.get("age", UNCLEAR),
            "patientGender": patient.get("gender", UNCLEAR),
            "doctorName": options.get("doctorName", CLINICIAN_DEFAULT),
            "chiefComplaint": UNCLEAR,
            "symptomsDescribed": [UNCLEAR],
            "familyHistory": [UNCLEAR],
            "vitalsRecorded": [UNCLEAR],
            "doctorAssessment": [UNCLEAR],
            "medicationsPrescribed": [UNCLEAR],
            "testsOrdered": [UNCLEAR],
            "followUpInstructions": [UNCLEAR],
            "doctorsNotes": [UNCLEAR],
            "summaryHighlights": ["No speech detected."],
            "allergies": [],
            "criticalHistory": [UNCLEAR],
            "worseningSymptoms": [UNCLEAR],
            "riskLevel": "low",
            "confidenceScore": "0%",
            "recurringSymptoms": [],
            "redFlags": [],
            "speakerTranscript": "",
            "followUpStatus": "",
        }

    sentences = split_transcript_sentences(clean_transcript)
    tagged = tag_transcript(sentences)
    patient_record = options.get("patient", {})
    patient = parse_patient_identity_from_text(tagged)
    patient_name = patient.get("patient_name") or patient_record.get("name", UNCLEAR)
    patient_age = patient.get("patient_age") or str(patient_record.get("age", UNCLEAR))
    patient_gender = patient_record.get("gender", UNCLEAR)

    patient_lines = [line for line in tagged if line["speaker"] == "PATIENT"]
    doctor_lines = [line for line in tagged if line["speaker"] == "DOCTOR"]
    all_lines = tagged
    patient_scope = patient_lines if patient_lines else all_lines
    doctor_scope = doctor_lines if doctor_lines else all_lines
    lower_sentences = [line["lower_text"] for line in tagged]
    transcript_words = re.findall(r"[a-z']+", clean_transcript.lower())

    sentence_scores: Dict[str, int] = {}
    sentence_order: Dict[str, int] = {}
    for index, line in enumerate(all_lines):
        sentence = line["text"]
        lower_sentence = line["lower_text"]
        sentence_scores[sentence] = score_sentence(sentence, lower_sentence)
        sentence_order[sentence] = index

    family_history = extract_family_history_text(all_lines)
    vitals = unique_keep_order([mention for line in all_lines for mention in gather_vital_mentions(line["text"])])
    symptoms = extract_symptoms_from_patient_lines(patient_scope)
    if not symptoms and patient_scope is not all_lines:
        symptoms = extract_symptoms_from_patient_lines(all_lines)
    doctor_assessment = unique_keep_order(
        [
            clean_clinical_text(line["text"])
            for line in doctor_scope
            if any(phrase in line["lower_text"] for phrase in ("diagnosed", "assessment", "consistent with", "likely", "probable", "consider", "appears to be", "impression"))
        ]
    )[:KEY_POINT_LIMIT]
    if not doctor_assessment and doctor_scope is not all_lines:
        doctor_assessment = unique_keep_order(
            [
                clean_clinical_text(line["text"])
                for line in all_lines
                if any(phrase in line["lower_text"] for phrase in ("diagnosed", "assessment", "consistent with", "likely", "probable", "consider", "appears to be", "impression"))
            ]
        )[:KEY_POINT_LIMIT]
    prescriptions = extract_prescriptions_from_doctor(doctor_scope)
    if not prescriptions and doctor_scope is not all_lines:
        prescriptions = extract_prescriptions_from_doctor(all_lines)
    tests_ordered = extract_tests_from_lines(doctor_scope)
    if not tests_ordered and doctor_scope is not all_lines:
        tests_ordered = extract_tests_from_lines(all_lines)
    follow_up = extract_follow_up_instructions(doctor_scope)
    if not follow_up and doctor_scope is not all_lines:
        follow_up = extract_follow_up_instructions(all_lines)
    prior_sessions = options.get("priorSessionContext")
    doctor_notes = unique_keep_order(
        [
            clean_clinical_text(line["text"])
            for line in doctor_scope
            if not any(token in line["lower_text"] for token in ("follow up", "follow-up", "rx", "prescribe", "assessment", "diagnos", "test"))
        ]
    )[:KEY_POINT_LIMIT]
    if not doctor_notes and doctor_scope is not all_lines:
        doctor_notes = unique_keep_order(
            [
                clean_clinical_text(line["text"])
                for line in all_lines
                if not any(token in line["lower_text"] for token in ("follow up", "follow-up", "rx", "prescribe", "assessment", "diagnos", "test"))
            ]
        )[:KEY_POINT_LIMIT]
    allergies = extract_allergies(all_lines)
    critical_history = unique_keep_order(extract_allergies(all_lines) + extract_family_history_text(all_lines))
    worsening = extract_worsening_symptoms(all_lines)
    worsening_vs_prior = build_worsening_vs_prior(worsening, prior_sessions if isinstance(prior_sessions, list) else [])
    red_flags = unique_keep_order(
        [
            clean_clinical_text(line["text"])
            for line in all_lines
            if any(term in line["lower_text"] for term in RED_FLAG_TERMS)
        ]
    )
    chief_complaint = symptoms[0] if symptoms else (doctor_assessment[0] if doctor_assessment else UNCLEAR)

    summary_highlights = [
        clean_clinical_text(item, 28)
        for item in rank_sentences_by_score(
            [line["text"] for line in all_lines],
            sentence_scores,
            sentence_order,
            KEY_POINT_LIMIT,
        )
    ]
    confidence_score = min(
        (35 if symptoms else 0)
        + (25 if vitals else 0)
        + (25 if doctor_assessment else 0)
        + (15 if prescriptions else 0)
        + (15 if follow_up else 0),
        100,
    )

    session_id = options.get("sessionId", f"MS-{int(datetime.now().timestamp())}")
    session_date = options.get("sessionDate", datetime.now().strftime("%B %d, %Y"))
    prior_symptom_terms = ("chest pain", "shortness of breath", "headache", "fever", "cough", "dizziness", "nausea", "fatigue", "diarrhea", "abdominal pain", "stomach")
    recurring_symptoms: List[str] = []
    if symptoms and prior_sessions:
        for term in prior_symptom_terms:
            if any(term in symptom.lower() for symptom in symptoms):
                was_seen_before = False
                for prior in prior_sessions[:5]:
                    prior_summary = prior.get("aiSummary") or {}
                    prior_values = normalize_list(
                        prior_summary.get("symptomsDescribed")
                        or prior_summary.get("symptoms")
                        or []
                    )
                    if any(term in str(item).lower() for item in prior_values):
                        was_seen_before = True
                        break
                if was_seen_before:
                    recurring_symptoms.append(term)

    dosage_pattern = re.compile(
        r"\b([a-z][a-z0-9\s\-]{1,30}?)\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|iu|ml|units?)\b",
        flags=re.IGNORECASE,
    )
    medicine_highlights = unique_keep_order(
        [mention for line in all_lines for mention in gather_medication_mentions(line["text"])]
    )
    dosage_highlights = unique_keep_order(
        [match.group(0).strip() for line in all_lines for match in dosage_pattern.finditer(line["text"])]
    )
    condition_highlights = unique_keep_order(doctor_assessment)
    symptom_highlights = unique_keep_order(symptoms)
    medical_history_highlights = unique_keep_order(family_history + allergies + critical_history)

    return {
        "model": AI_MODEL_NAME,
        "sessionId": session_id,
        "sessionDate": session_date,
        "generatedAt": datetime.now().strftime("%m/%d/%Y, %I:%M:%S %p"),
        "transcriptLength": len(transcript_words),
        "sentenceCount": len(all_lines),
        "patientName": patient_name,
        "patientAge": patient_age,
        "patientGender": patient_gender,
        "doctorName": options.get("doctorName", "Dr. Meera Nair"),
        "chiefComplaint": chief_complaint,
        "symptomsDescribed": symptoms or [UNCLEAR],
        "familyHistory": family_history or [UNCLEAR],
        "vitalsRecorded": vitals or [UNCLEAR],
        "doctorAssessment": doctor_assessment or [UNCLEAR],
        "medicationsPrescribed": prescriptions or [UNCLEAR],
        "testsOrdered": tests_ordered or [UNCLEAR],
        "followUpInstructions": follow_up or [UNCLEAR],
        "doctorsNotes": doctor_notes or [UNCLEAR],
        "allergies": allergies or [UNCLEAR],
        "criticalHistory": critical_history or [UNCLEAR],
        "worseningSymptoms": worsening_vs_prior or worsening or [UNCLEAR],
        "summaryHighlights": summary_highlights or [UNCLEAR],
        "riskLevel": evaluate_risk_level(red_flags),
        "confidenceScore": f"{confidence_score}%",
        "recurringSymptoms": recurring_symptoms or [],
        "redFlags": red_flags,
        "followUpStatus": parse_follow_up_warning(prior_sessions if isinstance(prior_sessions, list) else []),
        "speakerTranscript": "\n".join(f"[{line['speaker']}] {line['text']}" for line in all_lines),
        "subjective": symptoms or [UNCLEAR],
        "objective": vitals or [UNCLEAR],
        "assessment": doctor_assessment or [UNCLEAR],
        "plan": (prescriptions + tests_ordered + follow_up)[:KEY_POINT_LIMIT] if (prescriptions or tests_ordered or follow_up) else [UNCLEAR],
        "nextSteps": follow_up or [UNCLEAR],
        "symptomHighlights": symptom_highlights,
        "medicineHighlights": medicine_highlights,
        "dosageHighlights": dosage_highlights,
        "conditionHighlights": condition_highlights,
        "medicalHistoryHighlights": medical_history_highlights,
    }


def build_note(transcript: str, summary: Dict[str, Any], patient: Optional[Dict[str, Any]]) -> str:
    patient_name = summary.get("patientName") or (patient.get("name") if patient else "Unknown Patient")
    patient_age = summary.get("patientAge") or (patient.get("age") if patient else "N/A")
    patient_gender = summary.get("patientGender") or (patient.get("gender") if patient else "N/A")
    transcript_date = summary.get("sessionDate") or datetime.now().strftime("%B %d, %Y")
    subjective = normalize_list(summary.get("subjective") or summary.get("symptomsDescribed") or [])
    objective = normalize_list(summary.get("objective") or summary.get("vitalsRecorded") or [])
    assessment = normalize_list(summary.get("assessment") or summary.get("doctorAssessment") or [])
    plan = normalize_list(
        summary.get("plan")
        or normalize_list(summary.get("medicationsPrescribed"))
        + normalize_list(summary.get("testsOrdered"))
        + normalize_list(summary.get("followUpInstructions"))
    )
    family_history = normalize_list(summary.get("familyHistory") or [])
    vitals = normalize_list(summary.get("vitalsRecorded") or [])
    critical_history = normalize_list(summary.get("criticalHistory") or summary.get("allergies") or [])
    worsening = normalize_list(summary.get("worseningSymptoms") or [])
    symptom_highlights = normalize_list(summary.get("symptomHighlights") or summary.get("symptomsDescribed") or [])
    condition_highlights = normalize_list(summary.get("conditionHighlights") or summary.get("doctorAssessment") or [])
    medication_highlights = normalize_list(summary.get("medicineHighlights") or summary.get("medicationHighlights") or summary.get("medicationsPrescribed") or [])
    dosage_highlights = normalize_list(summary.get("dosageHighlights") or [])
    risk = summary.get("riskLevel", "low")
    recurring = ", ".join(summary.get("recurringSymptoms", []) or ["None identified"])
    follow_up_status = summary.get("followUpStatus") or "No follow-up pattern found"

    return (
        "AI CONSULTATION NOTE (SOAP)\n"
        "======================================\n"
        f"Session ID      : {summary.get('sessionId', 'N/A')}\n"
        f"Patient         : {patient_name} ({patient_age}y, {patient_gender})\n"
        f"Doctor          : {summary.get('doctorName', 'N/A')}\n"
        f"Date of Session : {transcript_date}\n\n"
        f"Chief Complaint: {summary.get('chiefComplaint', UNCLEAR)}\n\n"
        "Subjective:\n"
        f"{to_bullet_list(subjective, KEY_POINT_LIMIT)}\n\n"
        "Objective:\n"
        f"{to_bullet_list(objective, KEY_POINT_LIMIT)}\n\n"
        "Assessment:\n"
        f"{to_bullet_list(assessment, KEY_POINT_LIMIT)}\n\n"
        "Plan:\n"
        f"{to_bullet_list(plan, KEY_POINT_LIMIT)}\n\n"
        "Medical Highlights:\n"
        f"- Family & Medical History: {to_bullet_list(family_history, KEY_POINT_LIMIT)}\n"
        f"- Allergies / Critical History: {to_bullet_list(critical_history, KEY_POINT_LIMIT)}\n"
        f"- Vitals / Findings: {to_bullet_list(vitals, KEY_POINT_LIMIT)}\n"
        f"- Worsening Conditions: {to_bullet_list(worsening, KEY_POINT_LIMIT)}\n\n"
        "Detected Medical Terms:\n"
        f"- Symptoms / Complaints: {to_bullet_list(symptom_highlights, KEY_POINT_LIMIT)}\n"
        f"- Conditions: {to_bullet_list(condition_highlights, KEY_POINT_LIMIT)}\n"
        f"- Medicines: {to_bullet_list(medication_highlights, KEY_POINT_LIMIT)}\n"
        f"- Dosages: {to_bullet_list(dosage_highlights, KEY_POINT_LIMIT)}\n\n"
        "Safety & Follow-up:\n"
        f"- Recurring Symptoms: {recurring}\n"
        f"- Risk Level: {risk}\n"
        f"- Follow-up Status: {follow_up_status}\n"
        f"- Confidence: {summary.get('confidenceScore', '0%')}\n\n"
        f"Transcript Snapshot: {summary.get('transcriptLength', 0)} words, {summary.get('sentenceCount', 0)} fragments\n\n"
        f"TRANSCRIPT (LABELED LINES):\n{summary.get('speakerTranscript', '')}\n\n"
        "Note: Requires doctor verification before clinical action."
    )


def to_bullet_list(lines: Sequence[str], limit: int = KEY_POINT_LIMIT) -> str:
    if not lines:
        return "- Not stated in transcript."
    normalized = list(lines)[:max(0, limit)] if limit else list(lines)
    if not normalized:
        return "- Not stated in transcript."
    return "\n".join(f"- {line}" for line in normalized)


def parse_json_body(length: int, body: bytes) -> Dict[str, Any]:
    if length <= 0:
        return {}
    try:
        return json.loads(body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


class ConsultationHandler(BaseHTTPRequestHandler):
    def _set_cors_headers(self, status_code: int = 200) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self) -> None:
        self._set_cors_headers(204)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route != "/api/analyze-consultation":
            self._set_cors_headers(404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode("utf-8"))
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        data = parse_json_body(length, body)
        transcript = (data.get("transcript") or "").strip() if isinstance(data, dict) else ""
        patient = data.get("patient") if isinstance(data, dict) else None
        session_time = data.get("sessionTime") if isinstance(data, dict) else None

        if not transcript:
            self._set_cors_headers(400)
            self.wfile.write(json.dumps({"error": "transcript is required"}).encode("utf-8"))
            return

        summary = build_transcript_summary(
            transcript,
            {
                "sessionTime": session_time,
                "sessionId": data.get("sessionId"),
                "sessionDate": data.get("sessionDate"),
                "doctorName": data.get("doctorName"),
                "patient": patient,
                "priorSessionContext": data.get("priorSessionContext"),
            },
        )
        if session_time:
            summary["sessionTime"] = session_time
        else:
            summary["sessionTime"] = "N/A"

        note = build_note(transcript, summary, patient)
        response = {
            "source": "python-nlp",
            "model": AI_MODEL_NAME,
            "summary": summary,
            "note": note,
        }

        self._set_cors_headers(200)
        self.wfile.write(json.dumps(response).encode("utf-8"))

    def do_GET(self) -> None:
        self._set_cors_headers(200)
        self.wfile.write(
            json.dumps({"message": "MediaAI Python NLP service is running"}).encode("utf-8")
        )


def run_server() -> None:
    server = HTTPServer((HOST, PORT), ConsultationHandler)
    print(f"MediaAI Python NLP server listening on http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run_server()


