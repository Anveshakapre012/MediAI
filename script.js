// ==========================================
// Recording Functionality
// ==========================================

const CONSULTATION_STORAGE_KEY = 'mediaiConsultationRecords';
const AI_SIMULATED_MODEL = 'MediaiMini-NLP-v1';
const AI_SUMMARY_ENDPOINT = 'http://127.0.0.1:8765/api/analyze-consultation';
const AI_REQUEST_TIMEOUT_MS = 8000;
const MAX_LOCAL_RECORDS = 10;
const KEY_POINT_LIMIT = 5;
const SUMMARY_HIGHLIGHT_LIMIT = 4;
const INTERIM_FLUSH_MS = 1200; // flush interim text into final when silence is detected
const MAX_SPEECH_ALTERNATIVES = 3;
const SESSION_ID_PREFIX = 'MS';
const SESSION_ID_PAD = 5;
const CLINICIAN_DEFAULT = 'Dr. Meera Nair';
const UNCLEAR = '[UNCLEAR - needs clarification]';
const CRITICAL_TERM_SET = [
    'allergy',
    'allergies',
    'angioedema',
    'anaphylaxis',
    'rash',
    'hives',
    'shortness of breath',
    'chest pain',
    'bleeding',
    'unresponsive',
    'difficulty breathing'
];
const WORSENING_TERMS = [
    'worse',
    'worsening',
    'deteriorating',
    'progressively',
    'increasing',
    'more severe',
    'getting worse',
    'not improving',
    'spreading'
];
const MOCK_PATIENT_RECORD = {
    id: 'PT-DEMO-0001',
    name: 'Demo Patient',
    age: 38,
    gender: 'Female',
    medicalRecordNumber: 'MRN-442900'
};

let isRecording = false;
let recordingTime = 0;
let recordingTimer = null;
let recognition = null;
let finalTranscript = '';
let interimTranscript = '';
let finalSegments = [];
let interimFlushTimer = null;
let savePending = false;
let shouldFinalizeOnStop = false;
let isPaused = false;
let isFinalizing = false;
let recordingDurationAtStop = 0;

const recordBtn = document.getElementById('recordBtn');
const pauseBtn = document.getElementById('pauseBtn');
const restartBtn = document.getElementById('restartBtn');
const recordingTimeDisplay = document.getElementById('recordingTime');
const recordingStatus = document.getElementById('recordingStatus');
const recordingIndicator = document.getElementById('recordingIndicator');
const recordingMetadata = document.getElementById('recordingMetadata');
const transcriptionBox = document.getElementById('transcriptionBox');
const notesBox = document.getElementById('notesBox');

function toggleRecording() {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
}

function formatRecordingTime(totalSeconds) {
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
}

function clearRecordingTimer() {
    if (recordingTimer) {
        clearInterval(recordingTimer);
        recordingTimer = null;
    }
}

function resetRecordingTime() {
    recordingTime = 0;
    if (recordingTimeDisplay) {
        recordingTimeDisplay.textContent = formatRecordingTime(recordingTime);
    }
}

function appendFinalSegment(text) {
    const cleaned = (text || '').replace(/\s+/g, ' ').trim();
    if (!cleaned) {
        return;
    }
    const last = finalSegments[finalSegments.length - 1];
    if (last && last.toLowerCase() === cleaned.toLowerCase()) {
        return;
    }
    finalSegments.push(cleaned);
    finalTranscript = finalSegments.join(' ').trim();
}

function updateLiveTranscriptDisplay() {
    const liveTranscript = `${finalTranscript} ${interimTranscript}`.trim();
    if (transcriptionBox) {
        transcriptionBox.textContent = liveTranscript || transcriptionBox.textContent;
    }
}

function flushInterimToFinal(reason = 'silence') {
    if (interimTranscript) {
        appendFinalSegment(interimTranscript);
        interimTranscript = '';
        updateLiveTranscriptDisplay();
    }
    if (interimFlushTimer && reason !== 'silence') {
        clearTimeout(interimFlushTimer);
    }
    interimFlushTimer = null;
}

function scheduleInterimFlush() {
    if (interimFlushTimer) {
        clearTimeout(interimFlushTimer);
    }
    interimFlushTimer = setTimeout(() => flushInterimToFinal('silence'), INTERIM_FLUSH_MS);
}

function getFullTranscript() {
    flushInterimToFinal('finalize');
    return `${finalTranscript}`.trim();
}

function buildSpeakerTranscript(transcript) {
    const lines = tagConversation(transcript);
    if (!lines.length) {
        return '';
    }
    return lines.map(item => `[${item.speaker}] ${item.text}`).join('\n');
}

function extractMedicalTerms(lines) {
    const medicalTerms = [];
    lines.forEach(item => {
        if (containsAnyPhrase(item.lowerText, [
            'pain',
            'fever',
            'cough',
            'headache',
            'nausea',
            'dizziness',
            'shortness of breath',
            'chest pain',
            'fatigue'
        ])) {
            medicalTerms.push(cleanClinicalText(item.text));
        }
    });

    const medicineMentions = lines.flatMap(item => gatherMedicationMentions(item.text));
    const dosageMentions = lines.flatMap(item => {
        const matches = item.text.match(/\b([a-z][a-z0-9\s\-]{1,30}?)\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|iu|ml|units?)\b/gi) || [];
        return matches.map(match => match.trim());
    });
    const patientLines = lines.filter(item => item.speaker === 'PATIENT');
    const symptomSource = patientLines.length ? patientLines : lines;
    const conditionMentions = extractSymptomsFromPatientLines(symptomSource)
        .concat(allConditionPhrases(lines).map(cleanClinicalText));

    return {
        symptoms: uniqueKeepOrder(medicalTerms),
        medicines: uniqueKeepOrder(medicineMentions),
        dosages: uniqueKeepOrder(dosageMentions),
        conditions: uniqueKeepOrder(conditionMentions),
        medicalHistory: uniqueKeepOrder(extractFamilyHistoryText(lines).concat(extractAllergyLines(lines)).concat(extractCriticalHistoryFromLines(lines)))
    };
}

function allConditionPhrases(lines) {
    const conditions = [];
    const conditionMarkers = [
        'diagnosed',
        'diagnosis',
        'assessment',
        'consistent with',
        'likely',
        'probable',
        'impression',
        'appears to be'
    ];
    lines.forEach(item => {
        if (containsAnyPhrase(item.lowerText, conditionMarkers)) {
            conditions.push(item.text);
        }
    });
    return conditions;
}

function updateRecordingControls() {
    if (!recordBtn) {
        return;
    }

    if (isRecording && !isPaused) {
        recordBtn.innerHTML = '<span class="mic-icon">&#127908;</span><span class="mic-text">End Recording</span>';
        recordBtn.classList.add('recording');
    } else {
        recordBtn.innerHTML = '<span class="mic-icon">&#127908;</span><span class="mic-text">Start Recording</span>';
        recordBtn.classList.remove('recording');
    }

    if (pauseBtn) {
        pauseBtn.disabled = !isRecording;
        pauseBtn.textContent = isPaused ? 'Resume Recording' : 'Pause Recording';
        pauseBtn.classList.toggle('pause-active', isPaused);
    }

    if (restartBtn) {
        restartBtn.textContent = isRecording ? 'Restart Recording' : 'Restart Recording';
    }
}

function getSpeechRecognizer() {
    return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

function getSavedConsultations() {
    try {
        const records = localStorage.getItem(CONSULTATION_STORAGE_KEY);
        return records ? JSON.parse(records) : [];
    } catch (error) {
        console.error('Unable to read saved consultations:', error);
        return [];
    }
}

function saveConsultationRecord(record) {
    const records = getSavedConsultations();
    records.unshift(record);
    const latest = records.slice(0, MAX_LOCAL_RECORDS);
    localStorage.setItem(CONSULTATION_STORAGE_KEY, JSON.stringify(latest));
    return record;
}

function formatErrorSummaryText(errorMessage) {
    return `⚠ AI service error: ${errorMessage}`;
}

function splitTranscriptSentences(transcript) {
    const normalized = transcript.replace(/\s+/g, ' ').trim();
    if (!normalized) {
        return [];
    }

    const fragments = normalized.split(/[.!?;,]+|\r\n|\r|\n/).map(sentence => sentence.trim()).filter(Boolean);
    const chunks = [];
    const MAX_CHUNK_WORDS = 32;

    fragments.forEach(fragment => {
        const words = fragment.split(' ');
        if (words.length <= 40) {
            chunks.push(fragment);
            return;
        }

        for (let i = 0; i < words.length; i += MAX_CHUNK_WORDS) {
            chunks.push(words.slice(i, i + MAX_CHUNK_WORDS).join(' '));
        }
    });

    return chunks.filter(Boolean);
}

function scoreSentenceForInsights(sentence, lowerSentence) {
    let score = 0;

    if (!sentence.length) {
        return score;
    }

    if (sentence.length > 160) {
        score += 2;
    } else if (sentence.length > 90) {
        score += 1;
    }

    if (/\d/.test(sentence)) {
        score += 1;
    }

    if (/\b(i|i'm|patient|doctor|history|reports|denies|complains|complaint|symptoms?|condition|medication|prescribed|follow up|follow-up|referred|recommended|plan)\b/i.test(lowerSentence)) {
        score += 2;
    }

    return score;
}

function pickTopSentencesByScore(sentences, scoreBySentence, sentenceOrderByIndex, limit = KEY_POINT_LIMIT) {
    const uniqueSentences = uniqueKeepOrder(sentences);
    if (!uniqueSentences.length || !limit) {
        return [];
    }

    const ranked = uniqueSentences
        .map(sentence => ({
            sentence,
            score: scoreBySentence.get(sentence) || 0,
            order: sentenceOrderByIndex.get(sentence) ?? Number.MAX_SAFE_INTEGER
        }))
        .filter(item => Number.isFinite(item.order));

    const totalSentences = Math.max(ranked.length, 1);
    const buckets = [[], [], []];
    ranked.forEach(item => {
        const bucketIndex = Math.min(2, Math.max(0, Math.floor((item.order / totalSentences) * 3)));
        buckets[bucketIndex].push(item);
    });

    buckets.forEach(bucket => {
        bucket.sort((a, b) => {
            if (b.score !== a.score) {
                return b.score - a.score;
            }
            return a.order - b.order;
        });
    });

    const selected = [];
    const rounds = Math.max(...buckets.map(bucket => bucket.length));
    for (let round = 0; round < rounds && selected.length < limit; round++) {
        for (let bucketIndex = 0; bucketIndex < buckets.length && selected.length < limit; bucketIndex++) {
            if (buckets[bucketIndex][round]) {
                selected.push(buckets[bucketIndex][round].sentence);
            }
        }
    }

    if (selected.length >= limit) {
        return selected.slice(0, limit);
    }

    const remaining = ranked
        .filter(item => !selected.includes(item.sentence))
        .sort((a, b) => {
            if (b.score !== a.score) {
                return b.score - a.score;
            }
            return a.order - b.order;
        });

    return selected.concat(remaining.map(item => item.sentence)).slice(0, limit);
}

function extractAllergyLines(lines) {
    const allergyTerms = [
        'allergic to',
        'allergy',
        'allergic',
        'angioedema',
        'anaphylaxis'
    ];

    return uniqueKeepOrder(
        lines
            .filter(item => allergyTerms.some(term => item.lowerText.includes(term)))
            .map(item => cleanClinicalText(item.text))
    ).slice(0, KEY_POINT_LIMIT);
}

function extractCriticalHistoryFromLines(lines) {
    return uniqueKeepOrder(
        lines
            .filter(item => CRITICAL_TERM_SET.some(term => item.lowerText.includes(term)))
            .map(item => cleanClinicalText(item.text))
    ).slice(0, KEY_POINT_LIMIT);
}

function extractWorseningSymptoms(lines) {
    const worseningSymptoms = [];
    lines.forEach(item => {
        if (!WORSENING_TERMS.some(term => item.lowerText.includes(term))) {
            return;
        }

        const symptomMatch = item.lowerText.match(/\b(chest pain|shortness of breath|headache|fatigue|cough|fever|dizziness|nausea|abdominal pain|diarrhea|stomach|pain)\b/);
        const condition = symptomMatch ? symptomMatch[0] : '';
        const line = condition ? `${item.text}` : '';
        if (line) {
            worseningSymptoms.push(cleanClinicalText(line));
        }
    });

    return uniqueKeepOrder(worseningSymptoms);
}

function buildWorseningComparedToPrior(currentWorsening, priorSessions) {
    if (!priorSessions.length || !currentWorsening.length) {
        return [];
    }

    const currentEntries = currentWorsening.map(line => ({
        line,
        lower: line.toLowerCase()
    }));
    const priorSymptoms = new Set();
    priorSessions.slice(0, 5).forEach(session => {
        const summary = session.aiSummary || {};
        const priorSymptomsHistory = normalizeList(summary.symptomsDescribed || summary.symptoms || []);
        priorSymptomsHistory.forEach(symptom => priorSymptoms.add(symptom.toLowerCase()));
        const priorWorsening = normalizeList(summary.worseningSymptoms || summary.worsening || []);
        priorWorsening.forEach(symptom => priorSymptoms.add(symptom.toLowerCase()));
    });

    return uniqueKeepOrder(
        currentEntries
            .filter(entry => {
                return Array.from(priorSymptoms).some(prior => entry.lower.includes(prior));
            })
            .map(entry => entry.line)
    ).slice(0, KEY_POINT_LIMIT);
}

function uniqueKeepOrder(values) {
    return [...new Set(values.map(value => value.trim()).filter(Boolean))];
}

function containsAnyPhrase(text, terms) {
    return terms.some(term => text.includes(term));
}

function gatherVitalMentions(sentence) {
    const findings = [];
    const patterns = [
        { key: 'blood pressure', re: /\b(?:blood pressure|bp)\s*(?:is|:)?\s*(\d{2,3}\s*\/\s*\d{2,3})\b/i },
        { key: 'heart rate', re: /\b(?:heart rate|pulse|hr)\s*(?:is|:)?\s*(\d{2,3})\s*(?:bpm)?\b/i },
        { key: 'temperature', re: /\b(?:temperature|temp)\s*(?:is|:)?\s*(\d{2,3}(?:\.\d)?)\s*(?:°?f|°?c)?\b/i },
        { key: 'oxygen saturation', re: /\b(?:oxygen saturation|spo2|o2 sat|oxygen)\s*(?:is|:)?\s*(\d{2,3})\s*%?\b/i },
        { key: 'respiratory rate', re: /\b(?:respiratory rate|respiration)\s*(?:is|:)?\s*(\d{1,2})\b/i }
    ];

    patterns.forEach(pattern => {
        const match = sentence.match(pattern.re);
        if (match && match[1]) {
            findings.push(`${pattern.key}: ${match[1]}`);
        }
    });

    return findings;
}

function gatherMedicationMentions(sentence) {
    const meds = [];
    const explicitDose = sentence.match(/\b([a-z][a-z0-9\s\-]{1,30}?)\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|iu|ml|units?)\b/gi);
    if (explicitDose) {
        meds.push(...explicitDose);
    }

    const medLibrary = [
        'paracetamol',
        'acetaminophen',
        'ibuprofen',
        'amoxicillin',
        'metformin',
        'lisinopril',
        'atorvastatin',
        'omeprazole',
        'aspirin'
    ];
    const sentenceLower = sentence.toLowerCase();
    medLibrary.forEach(med => {
        if (sentenceLower.includes(med)) {
            meds.push(`possible ${med}`);
        }
    });

    return uniqueKeepOrder(meds);
}

function cleanClinicalText(text, maxWords = 22) {
    const cleaned = (text || '').replace(/\s+/g, ' ').trim();
    if (!cleaned) {
        return '';
    }
    const withoutLabels = cleaned.replace(/^(doctor|patient)\s*[:\-]\s*/i, '');
    const words = withoutLabels.split(' ');
    if (words.length > maxWords) {
        return `${words.slice(0, maxWords).join(' ')}...`;
    }
    return withoutLabels;
}

function evaluateRiskLevel(redFlags) {
    if (!redFlags.length) {
        return 'low';
    }
    if (redFlags.length >= 3) {
        return 'high';
    }
    return 'moderate';
}

function toBulletList(lines) {
    if (!lines.length) {
        return '- Not stated in transcript.';
    }
    return lines.map(line => `- ${line}`).join('\n');
}

function toBulletListWithLimit(lines, limit = KEY_POINT_LIMIT) {
    const safeLines = Array.isArray(lines) ? lines : [];
    if (!safeLines.length) {
        return '- Not stated in transcript.';
    }
    const safeLimit = Math.max(0, Number.isInteger(limit) ? limit : KEY_POINT_LIMIT);
    if (!safeLimit) {
        return '- Not stated in transcript.';
    }
    return toBulletList(safeLines.slice(0, safeLimit));
}

function normalizeList(value) {
    if (Array.isArray(value)) {
        return uniqueKeepOrder(value.map(item => String(item).trim()).filter(Boolean));
    }

    if (typeof value === 'string' && value.trim()) {
        return [value.trim()];
    }

    return [];
}

function nextSessionId(records) {
    const maxId = records.reduce((max, record) => {
        const current = String(record.sessionId || record.id || '').match(/(\d+)$/);
        if (!current) {
            return max;
        }
        const value = Number(current[1]);
        return Number.isFinite(value) && value > max ? value : max;
    }, 0);
    return `${SESSION_ID_PREFIX}-${String(maxId + 1).padStart(SESSION_ID_PAD, '0')}`;
}

function getSessionsForCurrentPatient() {
    const records = getSavedConsultations();
    return records
        .filter(record => (record.patient && record.patient.id) === MOCK_PATIENT_RECORD.id)
        .sort((a, b) => {
            const left = new Date(a.savedAt || 0).getTime();
            const right = new Date(b.savedAt || 0).getTime();
            return right - left;
        });
}

function parsePatientNameAgeFromText(sentence) {
    const text = sentence.toLowerCase();
    const nameMatch = sentence.match(/(?:patient name|name)\s*[:\-]?\s*([a-z][a-z.'\s]+)/i);
    const ageMatch = text.match(/\b(\d{1,3})\s*(?:y(?:ears?)?\s*)?old\b/);
    return {
        patientName: nameMatch ? nameMatch[1].trim() : null,
        patientAge: ageMatch ? ageMatch[1] : null
    };
}

function speakerLabelForSentence(sentence) {
    const normalized = sentence.toLowerCase();
    const doctorIndicators = [
        'i suggest',
        'i recommend',
        'i prescribed',
        'prescribe',
        'advised',
        'plan',
        'please take',
        'start taking',
        'we will start',
        'we will adjust',
        'i will start',
        'review',
        'order',
        'referred',
        'x-ray',
        'ecg',
        'diagnosis',
        'assessment',
        'follow up',
        'follow-up',
        'followed up',
        'schedule',
        'continue',
        'continue taking',
        'increase',
        'decrease',
        'dose',
        'dosage',
        'imaging',
        'lab results show'
    ];
    const patientIndicators = [
        'i have',
        "i'm",
        'i am',
        'i feel',
        'i feel like',
        'my pain',
        'symptom',
        'symptoms',
        'chest pain',
        'headache',
        'fever',
        'cough',
        'nausea',
        'sore throat',
        'dizziness',
        'my',
        'i have been',
        'hurt',
        'hurts',
        'lasted',
        'since',
        'for the past',
        'for the last'
    ];

    let doctorScore = 0;
    let patientScore = 0;

    doctorIndicators.forEach(phrase => {
        if (normalized.includes(phrase)) {
            doctorScore += 1;
        }
    });

    patientIndicators.forEach(phrase => {
        if (normalized.includes(phrase)) {
            patientScore += 1;
        }
    });

    if (doctorScore === patientScore) {
        return 'UNKNOWN';
    }
    return doctorScore > patientScore ? 'DOCTOR' : 'PATIENT';
}

function tagConversation(transcript) {
    const lines = splitTranscriptSentences(transcript).map(line => line.trim()).filter(Boolean);
    let lastLabel = 'DOCTOR';
    return lines.map(line => {
        const label = speakerLabelForSentence(line);
        const finalLabel = label === 'UNKNOWN' ? lastLabel : label;
        lastLabel = finalLabel;
        return {
            speaker: finalLabel,
            text: line,
            lowerText: line.toLowerCase()
        };
    });
}

function extractFamilyHistoryText(lines) {
    const familyIndicators = [
        'father',
        'mother',
        'sister',
        'brother',
        'family history',
        'family',
        'hypertension',
        'diabetes',
        'heart attack',
        'stroke',
        'asthma',
        'cancer'
    ];

    const matches = lines.filter(item => {
        return familyIndicators.some(term => item.lowerText.includes(term));
    }).map(item => cleanClinicalText(item.text));

    return uniqueKeepOrder(matches).slice(0, KEY_POINT_LIMIT);
}

function extractFollowUpInstructionText(lines) {
    const followUpPatterns = [
        /\bfollow[\s-]up\b.*?(?:in|after)\s+\d+\s+(?:day|days|week|weeks|month|months)\b/i,
        /\breturn\b.*?(?:in|after)\s+\d+\s+(?:day|days|week|weeks|month|months)\b/i,
        /\breview\b.*?(?:in|after)\s+\d+\s+(?:day|days|week|weeks|month|months)\b/i,
        /\brecheck\b.*?(?:in|after)\s+\d+\s+(?:day|days|week|weeks|month|months)\b/i
    ];

    const directTerms = [
        'follow up',
        'follow-up',
        'revisit',
        'return for review',
        'recheck',
        'repeat',
        'next visit'
    ];

    const mentions = [];
    lines.forEach(item => {
        for (const pattern of followUpPatterns) {
            const match = item.text.match(pattern);
            if (match) {
                mentions.push(cleanClinicalText(match[0]));
                return;
            }
        }

        if (directTerms.some(term => item.lowerText.includes(term))) {
            mentions.push(cleanClinicalText(item.text));
        }
    });

    return uniqueKeepOrder(mentions).slice(0, KEY_POINT_LIMIT);
}

function extractSymptomsFromPatientLines(lines) {
    const symptomTerms = [
        'pain',
        'ache',
        'headache',
        'fever',
        'cough',
        'sore throat',
        'fatigue',
        'dizziness',
        'nausea',
        'vomit',
        'shortness of breath',
        'chest pain',
        'diarrhea',
        'stomach'
    ];
    const severityTerms = [
        'mild',
        'moderate',
        'severe',
        'worsening',
        'sudden',
        'constant',
        'intermittent',
        'acute'
    ];
    const triggerTerms = [
        'after',
        'while',
        'during',
        'triggered by',
        'on exertion',
        'on movement',
        'after eating',
        'in the morning',
        'at night'
    ];

    const symptomFragments = [];

    lines.forEach(item => {
        const hasSymptom = symptomTerms.some(term => item.lowerText.includes(term));
        if (!hasSymptom) {
            return;
        }

        const durations = item.lowerText.match(/\b(last|for)\s+(\d+\s*(?:day|days|week|weeks|month|months|hour|hours|minute|minutes))\b/);
        const duration = durations ? durations[0] : '';
        const severity = severityTerms.filter(term => item.lowerText.includes(term));
        const triggers = triggerTerms.filter(term => item.lowerText.includes(term));

        const descriptors = [];
        if (duration) {
            descriptors.push(duration);
        }
        if (severity.length) {
            descriptors.push(`severity: ${severity[0]}`);
        }
        if (triggers.length) {
            descriptors.push(`trigger: ${triggers[0]}`);
        }

        const line = `${item.text}${descriptors.length ? ` (${descriptors.join(', ')})` : ''}`;
        symptomFragments.push(cleanClinicalText(line));
    });

    return uniqueKeepOrder(symptomFragments).slice(0, KEY_POINT_LIMIT);
}

function extractTestsOrderedFromLines(lines) {
    const terms = [
        'ecg',
        'electrocardiogram',
        'xray',
        'x-ray',
        'chest xray',
        'blood test',
        'cbc',
        'ct',
        'mri',
        'ultrasound',
        'x ray',
        'urine',
        'lft',
        'esr',
        'crp',
        'glucose'
    ];

    return uniqueKeepOrder(
        lines
            .filter(item => terms.some(term => item.lowerText.includes(term)))
            .map(item => cleanClinicalText(item.text))
    ).slice(0, KEY_POINT_LIMIT);
}

function extractObjectiveObservationsFromLines(lines) {
    const objectiveIndicators = [
        'exam',
        'physical',
        'appears',
        'blood pressure',
        'heart rate',
        'pulse',
        'temperature',
        'oxygen',
        'respiratory',
        'weight',
        'bp',
        'o2 sat',
        'spo2',
        'findings'
    ];

    return uniqueKeepOrder(
        lines
            .filter(item => objectiveIndicators.some(term => item.lowerText.includes(term)))
            .map(item => cleanClinicalText(item.text))
    ).slice(0, KEY_POINT_LIMIT);
}

function extractPrescriptionLinesFromDoctor(lines) {
    const prescriptions = [];
    const medPattern = /\b([a-z][a-z0-9\s\-]{1,30}?)\s+\d+(?:\.\d+)?\s*(?:mg|mcg|g|iu|ml|units?)\b/gi;
    const frequencyPattern = /\b(once\s+daily|twice\s+daily|thrice\s+daily|daily|bid|tid|qid|qid|prn|at\s+bedtime)\b/i;

    lines.forEach(item => {
        const medMatches = item.text.match(medPattern) || [];
        const freq = item.lowerText.match(frequencyPattern);
        const freqLabel = freq ? ` — ${freq[0]}` : '';

        medMatches.forEach(match => {
            prescriptions.push(`${match.trim()}${freqLabel}`.trim());
        });

        const mentions = gatherMedicationMentions(item.text);
        mentions.forEach(medication => {
            prescriptions.push(`${medication}${freqLabel}`.trim());
        });

        const explicitDoseMatch = item.lowerText.match(/\b(rx|prescribed|start|take)\b/);
        if (!medMatches.length && !mentions.length && explicitDoseMatch) {
            prescriptions.push(cleanClinicalText(item.text));
        }
    });

    return uniqueKeepOrder(prescriptions).slice(0, KEY_POINT_LIMIT);
}

function buildRecurringSymptoms(currentSymptoms, priorSessions) {
    if (!priorSessions.length) {
        return [];
    }

    const symptomPool = new Set();
    currentSymptoms.forEach(item => {
        const normalized = item.toLowerCase();
        [
            'chest pain',
            'shortness of breath',
            'headache',
            'fever',
            'cough',
            'dizziness',
            'nausea',
            'fatigue',
            'diarrhea',
            'abdominal pain',
            'stomach'
        ].forEach(term => {
            if (normalized.includes(term)) {
                symptomPool.add(term);
            }
        });
    });

    if (!symptomPool.size) {
        return [];
    }

    const priorSymptomHits = new Set();
    priorSessions.slice(0, 5).forEach(session => {
        const priorSymptoms = (session.aiSummary && Array.isArray(session.aiSummary.symptomsDescribed)
            ? session.aiSummary.symptomsDescribed
            : Array.isArray(session.aiSummary?.symptomsDescribed)
                ? session.aiSummary.symptomsDescribed
                : [])
            .concat(Array.isArray(session.aiSummary?.symptoms)
                ? session.aiSummary.symptoms
                : []);

        priorSymptoms.forEach(symptomText => {
            const lower = String(symptomText || '').toLowerCase();
            Array.from(symptomPool).forEach(term => {
                if (lower.includes(term)) {
                    priorSymptomHits.add(term);
                }
            });
        });
    });

    return Array.from(priorSymptomHits);
}

function getFollowUpWarningText(lastSessions) {
    if (!lastSessions.length) {
        return '';
    }

    const latest = lastSessions[0];
    const summary = latest.aiSummary || {};
    const instruction = summary.followUpInstructions || summary.followUp || summary.nextSteps;
    const text = Array.isArray(instruction)
        ? instruction.join(' ')
        : String(instruction || '');
    const match = text.match(/\b(\d+)\s*(day|days|week|weeks|month|months)\b/i);
    if (!match) {
        return '';
    }

    const amount = Number(match[1]);
    const unit = match[2].toLowerCase();
    const dueDate = new Date(latest.savedAt || Date.now());
    if (unit.startsWith('week')) {
        dueDate.setDate(dueDate.getDate() + amount * 7);
    } else if (unit.startsWith('month')) {
        dueDate.setMonth(dueDate.getMonth() + amount);
    } else {
        dueDate.setDate(dueDate.getDate() + amount);
    }

    const today = new Date();
    if (today <= dueDate) {
        const remaining = Math.ceil((dueDate.getTime() - today.getTime()) / (1000 * 60 * 60 * 24));
        return `Follow-up due in ${remaining} day(s).`;
    }

    const overdue = Math.ceil((today.getTime() - dueDate.getTime()) / (1000 * 60 * 60 * 24));
    return `⚠️ Follow-up overdue by ${overdue} day(s) from prior session (${latest.sessionId || latest.id || 'prior session'}).`;
}

function formatHistoricalBrief() {
    const sessions = getSessionsForCurrentPatient();
    if (!sessions.length) {
        return 'Previous sessions: none found for this patient.';
    }

    const latest = sessions[0];
    const latestSummary = latest.aiSummary || {};
    const chiefComplaint = latestSummary.chiefComplaint || latestSummary.chiefComplaintDescription || UNCLEAR;
    const medicationLine = latestSummary.medicationsPrescribed?.[0] || latestSummary.prescriptions?.[0] || UNCLEAR;
    const followText = latestSummary.followUpInstructions?.[0] || latestSummary.followUp || UNCLEAR;
    const followWarn = getFollowUpWarningText(sessions);
    const criticalHistory = latestSummary.criticalHistory?.[0] || latestSummary.allergies?.[0] || UNCLEAR;
    const recurring = latestSummary.recurringSymptoms?.length
        ? latestSummary.recurringSymptoms.join(', ')
        : 'No recurring issues were flagged.';
    const riskLevel = latestSummary.riskLevel || 'low';
    const warningBits = [];
    if (followWarn) {
        warningBits.push(followWarn);
    }
    if (criticalHistory && criticalHistory !== UNCLEAR) {
        warningBits.push(`Allergy/Critical history: ${criticalHistory}`);
    }

    return `Patient last visited on ${latest.savedAt || latestSummary.sessionDate || 'N/A'} for ${chiefComplaint}. ` +
        `They were prescribed: ${medicationLine}. ` +
        `Recurring flags: ${recurring}. ` +
        `Follow-up note: ${followText}. ` +
        `${warningBits.join(' ')}`.trim();
}

function generateAiConsultationSummary(transcript, options = {}) {
    const cleanTranscript = transcript.trim();
    const patientRecord = options.patient || MOCK_PATIENT_RECORD;
    const doctorName = options.doctorName || CLINICIAN_DEFAULT;
    const sessionDate = options.sessionDate || new Date().toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
    const sessionId = options.sessionId || `MS-${Date.now()}`;

    if (!cleanTranscript) {
        return {
            model: AI_SIMULATED_MODEL,
            generatedAt: new Date().toLocaleString('en-US'),
            sessionId,
            sessionDate,
            savedAt: options.savedAt || new Date().toLocaleString('en-US'),
            transcriptLength: 0,
            sentenceCount: 0,
            patientName: patientRecord.name || UNCLEAR,
            patientAge: patientRecord.age || UNCLEAR,
            patientGender: patientRecord.gender || UNCLEAR,
            doctorName,
            chiefComplaint: UNCLEAR,
            symptomsDescribed: [UNCLEAR],
            familyHistory: [UNCLEAR],
            vitalsRecorded: [UNCLEAR],
            doctorAssessment: [UNCLEAR],
            medicationsPrescribed: [UNCLEAR],
            testsOrdered: [UNCLEAR],
            followUpInstructions: [UNCLEAR],
            doctorsNotes: [UNCLEAR],
            summaryHighlights: ['No speech detected.'],
            riskLevel: evaluateRiskLevel([]),
            confidenceScore: '0%',
            recurringSymptoms: [],
            redFlags: [],
            speakerTranscript: '',
            source: 'local-heuristic'
        };
    }

    const transcriptLines = tagConversation(cleanTranscript);
    const words = cleanTranscript.toLowerCase().match(/\b[a-z']+\b/g) || [];
    const priorSessions = Array.isArray(options.priorSessionContext) && options.priorSessionContext.length
        ? options.priorSessionContext
        : getSessionsForCurrentPatient();
    const sentenceScores = new Map();
    const sentenceOrder = new Map();
    transcriptLines.forEach((item, index) => {
        sentenceScores.set(item.text, scoreSentenceForInsights(item.text, item.lowerText));
        sentenceOrder.set(item.text, index);
    });

    const patientLines = transcriptLines.filter(line => line.speaker === 'PATIENT');
    const doctorLines = transcriptLines.filter(line => line.speaker === 'DOCTOR');
    const allLines = [...transcriptLines];
    const patientScope = patientLines.length ? patientLines : allLines;
    const doctorScope = doctorLines.length ? doctorLines : allLines;

    const patientIdentityHints = allLines.map(parsePatientNameAgeFromText).filter(Boolean);
    const patientNameFromConversation = patientIdentityHints.find(item => item.patientName)?.patientName || '';
    const patientAgeFromConversation = patientIdentityHints.find(item => item.patientAge)?.patientAge || '';

    const patientName = patientNameFromConversation || patientRecord.name || UNCLEAR;
    const patientAge = patientAgeFromConversation || patientRecord.age || UNCLEAR;
    const patientGender = patientRecord.gender || UNCLEAR;

    let symptomsDescribed = extractSymptomsFromPatientLines(patientScope);
    if (!symptomsDescribed.length && patientScope !== allLines) {
        symptomsDescribed = extractSymptomsFromPatientLines(allLines);
    }
    const familyHistory = extractFamilyHistoryText(allLines);
    const vitals = uniqueKeepOrder(allLines.flatMap(item => gatherVitalMentions(item.text)));
    let doctorAssessment = uniqueKeepOrder(
        doctorScope.filter(item => containsAnyPhrase(
            item.lowerText,
            ['diagnosed', 'assessment', 'consistent with', 'likely', 'probable', 'consider', 'appears to be', 'impression']
        )).map(item => cleanClinicalText(item.text))
    ).slice(0, KEY_POINT_LIMIT);
    if (!doctorAssessment.length && doctorScope !== allLines) {
        doctorAssessment = uniqueKeepOrder(
            allLines.filter(item => containsAnyPhrase(
                item.lowerText,
                ['diagnosed', 'assessment', 'consistent with', 'likely', 'probable', 'consider', 'appears to be', 'impression']
            )).map(item => cleanClinicalText(item.text))
        ).slice(0, KEY_POINT_LIMIT);
    }

    let prescriptions = extractPrescriptionLinesFromDoctor(doctorScope);
    if (!prescriptions.length && doctorScope !== allLines) {
        prescriptions = extractPrescriptionLinesFromDoctor(allLines);
    }

    let testsOrdered = extractTestsOrderedFromLines(doctorScope);
    if (!testsOrdered.length && doctorScope !== allLines) {
        testsOrdered = extractTestsOrderedFromLines(allLines);
    }

    let followUpInstructions = extractFollowUpInstructionText(doctorScope);
    if (!followUpInstructions.length && doctorScope !== allLines) {
        followUpInstructions = extractFollowUpInstructionText(allLines);
    }

    let doctorNotes = uniqueKeepOrder(
        doctorScope.filter(item => {
            const lineText = item.lowerText;
            if (
                lineText.includes('follow up') ||
                lineText.includes('follow-up') ||
                lineText.includes('rx') ||
                lineText.includes('prescribe') ||
                lineText.includes('assessment') ||
                lineText.includes('diagnos') ||
                lineText.includes('test')
            ) {
                return false;
            }
            return true;
        }).map(item => cleanClinicalText(item.text))
    ).slice(0, KEY_POINT_LIMIT);
    if (!doctorNotes.length && doctorScope !== allLines) {
        doctorNotes = uniqueKeepOrder(
            allLines.filter(item => {
                const lineText = item.lowerText;
                if (
                    lineText.includes('follow up') ||
                    lineText.includes('follow-up') ||
                    lineText.includes('rx') ||
                    lineText.includes('prescribe') ||
                    lineText.includes('assessment') ||
                    lineText.includes('diagnos') ||
                    lineText.includes('test')
                ) {
                    return false;
                }
                return true;
            }).map(item => cleanClinicalText(item.text))
        ).slice(0, KEY_POINT_LIMIT);
    }
    const allergies = extractAllergyLines(allLines);
    const criticalHistory = extractCriticalHistoryFromLines(allLines);
    const worsening = extractWorseningSymptoms(allLines);
    const worseningComparedToPrior = buildWorseningComparedToPrior(worsening, priorSessions);
    const redFlagTerms = [
        'chest pain',
        'shortness of breath',
        'difficulty breathing',
        'loss of consciousness',
        'severe',
        'unresponsive',
        'bleeding',
        'urgent',
        'emergency'
    ];
    const redFlags = uniqueKeepOrder(
        allLines.filter(item => redFlagTerms.some(term => item.lowerText.includes(term))).map(item => cleanClinicalText(item.text))
    );

    const chiefComplaint = symptomsDescribed.length
        ? symptomsDescribed[0]
        : (doctorAssessment.length ? doctorAssessment[0] : UNCLEAR);
    const recurringSymptoms = buildRecurringSymptoms(symptomsDescribed, priorSessions);
    const summaryHighlights = pickTopSentencesByScore(
        allLines.map(item => item.text),
        sentenceScores,
        sentenceOrder,
        SUMMARY_HIGHLIGHT_LIMIT
    ).map(line => cleanClinicalText(line, 28));

    const confidenceScore = `${Math.min(
        (symptomsDescribed.length ? 35 : 0) +
        (vitals.length ? 25 : 0) +
        (doctorAssessment.length ? 20 : 0) +
        (prescriptions.length ? 15 : 0) +
        (testsOrdered.length ? 5 : 0),
        100
    )}%`;

    const objectiveFindings = extractObjectiveObservationsFromLines(allLines);
    const medicalTerms = extractMedicalTerms(allLines);
    const subjective = uniqueKeepOrder(
        [chiefComplaint]
            .concat(symptomsDescribed)
            .concat(medicalTerms.symptoms)
    ).filter(Boolean).slice(0, KEY_POINT_LIMIT);
    const objective = uniqueKeepOrder(
        uniqueKeepOrder(vitals)
            .concat(objectiveFindings)
    ).filter(Boolean).slice(0, KEY_POINT_LIMIT);
    const assessment = uniqueKeepOrder(
        uniqueKeepOrder(doctorAssessment)
            .concat(medicalTerms.conditions)
    ).filter(Boolean).slice(0, KEY_POINT_LIMIT);
    const plan = uniqueKeepOrder(
        uniqueKeepOrder(prescriptions)
            .concat(testsOrdered)
            .concat(followUpInstructions)
    ).filter(Boolean).slice(0, KEY_POINT_LIMIT);

    return {
        model: AI_SIMULATED_MODEL,
        generatedAt: new Date().toLocaleString('en-US'),
        sessionId,
        sessionDate,
        savedAt: options.savedAt || new Date().toLocaleString('en-US'),
        transcriptLength: words.length,
        sentenceCount: allLines.length,
        patientName,
        patientAge,
        patientGender,
        doctorName,
        chiefComplaint,
        summaryHighlights: summaryHighlights.length ? summaryHighlights : [UNCLEAR],
        symptomsDescribed: symptomsDescribed.length ? symptomsDescribed : [UNCLEAR],
        familyHistory: familyHistory.length ? familyHistory : [UNCLEAR],
        vitalsRecorded: vitals.length ? vitals : [UNCLEAR],
        doctorAssessment: doctorAssessment.length ? doctorAssessment : [UNCLEAR],
        medicationsPrescribed: prescriptions.length ? prescriptions : [UNCLEAR],
        testsOrdered: testsOrdered.length ? testsOrdered : [UNCLEAR],
        followUpInstructions: followUpInstructions.length ? followUpInstructions : [UNCLEAR],
        doctorsNotes: doctorNotes.length ? doctorNotes : [UNCLEAR],
        allergies: allergies.length ? allergies : [UNCLEAR],
        criticalHistory: criticalHistory.length ? criticalHistory : [UNCLEAR],
        worseningSymptoms: worseningComparedToPrior.length ? worseningComparedToPrior : (worsening.length ? worsening : [UNCLEAR]),
        riskLevel: evaluateRiskLevel(redFlags),
        confidenceScore,
        recurringSymptoms,
        redFlags,
        executiveSummary: chiefComplaint,
        followUpStatus: getFollowUpWarningText(priorSessions),
        speakerTranscript: allLines.map(item => `[${item.speaker}] ${item.text}`).join('\n'),
        subjective: subjective.length ? subjective : [UNCLEAR],
        objective: objective.length ? objective : [UNCLEAR],
        assessment: assessment.length ? assessment : [UNCLEAR],
        plan: plan.length ? plan : [UNCLEAR],
        symptomHighlights: medicalTerms.symptoms,
        medicineHighlights: medicalTerms.medicines,
        dosageHighlights: medicalTerms.dosages,
        conditionHighlights: medicalTerms.conditions,
        medicalHistoryHighlights: medicalTerms.medicalHistory,
        source: 'local-heuristic'
    };
}

function buildAiConsultationNote(transcript, aiSummary) {
    const cleanTranscript = transcript.trim();
    const summary = aiSummary || generateAiConsultationSummary(cleanTranscript);
    const patientName = summary.patientName || `${MOCK_PATIENT_RECORD.name}`;
    const patientAge = summary.patientAge || `${MOCK_PATIENT_RECORD.age}`;
    const patientGender = summary.patientGender || MOCK_PATIENT_RECORD.gender || UNCLEAR;
    const subjective = uniqueKeepOrder(normalizeList(summary.subjective).length
        ? normalizeList(summary.subjective)
        : normalizeList(summary.symptomsDescribed).length
            ? normalizeList(summary.symptomsDescribed)
            : [UNCLEAR]);
    const objective = uniqueKeepOrder(normalizeList(summary.objective).length
        ? normalizeList(summary.objective)
        : normalizeList(summary.vitalsRecorded).length
            ? normalizeList(summary.vitalsRecorded)
            : [UNCLEAR]);
    const assessment = uniqueKeepOrder(normalizeList(summary.assessment).length
        ? normalizeList(summary.assessment)
        : normalizeList(summary.doctorAssessment).length
            ? normalizeList(summary.doctorAssessment)
            : [UNCLEAR]);
    const plan = uniqueKeepOrder(normalizeList(summary.plan).length
        ? normalizeList(summary.plan)
        : (
            normalizeList(summary.medicationsPrescribed)
                .concat(normalizeList(summary.testsOrdered))
                .concat(normalizeList(summary.followUpInstructions))
        ).length
            ? normalizeList(
                normalizeList(summary.medicationsPrescribed).concat(normalizeList(summary.testsOrdered)).concat(normalizeList(summary.followUpInstructions))
            )
            : [UNCLEAR]);

    const familyHistory = normalizeList(summary.familyHistory);
    const vitals = normalizeList(summary.vitalsRecorded);
    const allergies = normalizeList(summary.allergies);
    const criticalHistory = normalizeList(summary.criticalHistory);
    const worseningSymptoms = normalizeList(summary.worseningSymptoms);
    const medicationHighlights = uniqueKeepOrder(
        normalizeList(summary.medicationHighlights).concat(normalizeList(summary.medicationsPrescribed))
    );
    const dosageHighlights = uniqueKeepOrder(normalizeList(summary.dosageHighlights));
    const conditionHighlights = uniqueKeepOrder(
        normalizeList(summary.conditionHighlights).concat(normalizeList(summary.symptomHighlights))
    );
    const symptomHighlights = uniqueKeepOrder(
        normalizeList(summary.symptomHighlights).concat(normalizeList(summary.symptomsDescribed || summary.subjective))
    );
    const followUpStatus = summary.followUpStatus || UNCLEAR;
    const transcriptDate = new Date().toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });

    const patientLine = `${patientName}, ${patientAge}y`;
    const doctor = summary.doctorName || CLINICIAN_DEFAULT;
    const sessionId = summary.sessionId || `MS-${Date.now()}`;
    const dateOfSession = summary.sessionDate || transcriptDate;
    const risk = summary.riskLevel || evaluateRiskLevel(summary.redFlags || []);
    const recurring = (summary.recurringSymptoms || []).length
        ? summary.recurringSymptoms.join(', ')
        : 'None identified';

    return `AI CONSULTATION NOTE (SOAP)
======================================
Session ID      : ${sessionId}
Patient         : ${patientLine}
Doctor          : ${doctor}
Patient Gender  : ${patientGender}
Date of Session : ${dateOfSession}

Chief Complaint: ${summary.chiefComplaint || UNCLEAR}

Subjective:
${toBulletListWithLimit(subjective, KEY_POINT_LIMIT)}

Objective:
${toBulletListWithLimit(objective, KEY_POINT_LIMIT)}

Assessment:
${toBulletListWithLimit(assessment, KEY_POINT_LIMIT)}

Plan:
${toBulletListWithLimit(plan, KEY_POINT_LIMIT)}

Medical Highlights:
- Family & Medical History: ${toBulletListWithLimit(familyHistory, KEY_POINT_LIMIT)}
- Allergies / Critical History: ${toBulletListWithLimit(criticalHistory, KEY_POINT_LIMIT)}
- Vitals / Findings: ${toBulletListWithLimit(vitals, KEY_POINT_LIMIT)}
- Worsening Conditions: ${toBulletListWithLimit(worseningSymptoms, KEY_POINT_LIMIT)}

Detected Medical Terms:
- Symptoms / Complaints: ${toBulletListWithLimit(symptomHighlights, KEY_POINT_LIMIT)}
- Conditions: ${toBulletListWithLimit(conditionHighlights, KEY_POINT_LIMIT)}
- Medicines: ${toBulletListWithLimit(medicationHighlights, KEY_POINT_LIMIT)}
- Dosages: ${toBulletListWithLimit(dosageHighlights, KEY_POINT_LIMIT)}

Safety & Follow-up:
- Recurring Symptoms: ${recurring}
- Risk Level: ${risk}
- Follow-up Status: ${followUpStatus}
- Confidence: ${summary.confidenceScore || '0%'}

Note: Requires doctor verification before clinical action.`;
}

function normalizePythonSummary(summary, context = {}) {
    if (!summary || typeof summary !== 'object') {
        return generateAiConsultationSummary('', {
            sessionId: context.sessionId,
            sessionDate: context.sessionDate,
            patient: context.patient
        });
    }

    const patient = context.patient || MOCK_PATIENT_RECORD;
    const normalizedSummary = {
        ...summary,
        patientName: summary.patientName || patient.name || MOCK_PATIENT_RECORD.name || UNCLEAR,
        patientAge: summary.patientAge || summary.age || patient.age || MOCK_PATIENT_RECORD.age || UNCLEAR,
        patientGender: summary.patientGender || patient.gender || MOCK_PATIENT_RECORD.gender || UNCLEAR,
        sessionId: summary.sessionId || context.sessionId || `MS-${Date.now()}`,
        sessionDate: summary.sessionDate || context.sessionDate || new Date().toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        }),
        doctorName: summary.doctorName || context.doctorName || CLINICIAN_DEFAULT,
        chiefComplaint: summary.chiefComplaint || UNCLEAR,
        symptomsDescribed: normalizeList(summary.symptomsDescribed || summary.subjective),
        familyHistory: normalizeList(summary.familyHistory || summary.history || summary.pastHistory),
        vitalsRecorded: normalizeList(summary.vitalsRecorded || summary.vitals),
        doctorAssessment: normalizeList(summary.doctorAssessment || summary.assessment),
        medicationsPrescribed: normalizeList(summary.medicationsPrescribed || summary.medications),
        testsOrdered: normalizeList(summary.testsOrdered || summary.tests),
        followUpInstructions: normalizeList(summary.followUpInstructions || summary.nextSteps || summary.followUp),
        doctorsNotes: normalizeList(summary.doctorsNotes || summary.notes),
        subjective: normalizeList(summary.subjective || summary.symptomsDescribed || []),
        objective: normalizeList(summary.objective || summary.vitalsRecorded || summary.observations),
        assessment: normalizeList(summary.assessment || summary.doctorAssessment || []),
        plan: normalizeList(summary.plan || ([
            ...normalizeList(summary.medicationsPrescribed),
            ...normalizeList(summary.testsOrdered),
            ...normalizeList(summary.followUpInstructions)
        ])),
        symptomHighlights: normalizeList(summary.symptomHighlights || summary.symptomsDescribed || summary.subjective),
        medicationHighlights: normalizeList(summary.medicationHighlights || summary.medicineHighlights || summary.medications || summary.medicationList),
        dosageHighlights: normalizeList(summary.dosageHighlights || summary.dosages),
        conditionHighlights: normalizeList(summary.conditionHighlights || summary.assessment || summary.conditions),
        summaryHighlights: normalizeList(summary.summaryHighlights || summary.keyPoints),
        followUpStatus: summary.followUpStatus || '',
        allergies: normalizeList(summary.allergies),
        criticalHistory: normalizeList(summary.criticalHistory || summary.criticalHistoryNotes),
        worseningSymptoms: normalizeList(summary.worseningSymptoms || summary.worsening),
        recurringSymptoms: normalizeList(summary.recurringSymptoms),
        redFlags: normalizeList(summary.redFlags),
        speakerTranscript: summary.speakerTranscript || '',
        source: summary.source || 'python-nlp',
        model: summary.model || AI_SIMULATED_MODEL,
        generatedAt: summary.generatedAt || new Date().toLocaleString('en-US'),
        confidenceScore: summary.confidenceScore || '0%',
        transcriptLength: Number(summary.transcriptLength || 0),
        sentenceCount: Number(summary.sentenceCount || 0),
        riskLevel: summary.riskLevel || evaluateRiskLevel(normalizeList(summary.redFlags))
    };

    return normalizedSummary;
}

function isLowSignalSummary(summary) {
    if (!summary || typeof summary !== 'object') {
        return true;
    }
    const buckets = [
        summary.subjective,
        summary.objective,
        summary.assessment,
        summary.plan,
        summary.summaryHighlights,
        summary.symptomHighlights,
        summary.medicineHighlights,
        summary.dosageHighlights,
        summary.conditionHighlights
    ];
    const tokens = buckets
        .flatMap(item => normalizeList(item))
        .map(item => item.trim())
        .filter(item => item && !item.includes(UNCLEAR));
    return tokens.length < 2;
}

async function generateAiSummaryFromPython(transcript, options = {}) {
    const payload = {
        transcript,
        sessionTime: options.sessionTime || null,
        sessionId: options.sessionId || `MS-${Date.now()}`,
        sessionDate: options.sessionDate || new Date().toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        }),
        doctorName: options.doctorName || CLINICIAN_DEFAULT,
        patient: MOCK_PATIENT_RECORD,
        savedAt: options.savedAt || new Date().toLocaleString('en-US'),
        priorSessionContext: options.priorSessionContext || null
    };

    const controller = new AbortController();
    const timer = setTimeout(() => {
        controller.abort(new Error('AI request timeout'));
    }, AI_REQUEST_TIMEOUT_MS);

    try {
        const response = await fetch(AI_SUMMARY_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload),
            signal: controller.signal
        });

        if (!response.ok) {
            throw new Error(`Python service returned ${response.status}`);
        }

        const payloadData = await response.json();
        const summary = payloadData.summary || payloadData.aiSummary || null;
        const normalizedSummary = normalizePythonSummary(summary, {
            sessionId: payload.sessionId,
            sessionDate: payload.sessionDate,
            patient: options.patient,
            doctorName: payload.doctorName
        });
        const fallbackSummary = isLowSignalSummary(normalizedSummary)
            ? generateAiConsultationSummary(transcript, {
                sessionId: payload.sessionId,
                sessionDate: payload.sessionDate,
                patient: options.patient,
                doctorName: payload.doctorName,
                priorSessionContext: options.priorSessionContext,
                savedAt: options.savedAt
            })
            : null;
        const finalSummary = fallbackSummary || normalizedSummary;
        const note = buildAiConsultationNote(transcript, finalSummary);

        if (!finalSummary || !note) {
            throw new Error('Python response was missing note or summary fields.');
        }

        return {
            note,
            aiSummary: finalSummary,
            source: fallbackSummary ? 'browser-heuristic-fallback' : (payloadData.source || 'python-nlp')
        };
    } catch (error) {
        console.error('Python AI summary request failed:', error);
        if (recordingStatus) {
            recordingStatus.textContent = formatErrorSummaryText(error.message || 'Python AI service unavailable');
        }

        const fallbackSummary = generateAiConsultationSummary(transcript, {
            sessionId: options.sessionId || `MS-${Date.now()}`,
            sessionDate: options.sessionDate,
            patient: MOCK_PATIENT_RECORD,
            doctorName: options.doctorName || CLINICIAN_DEFAULT,
            priorSessionContext: options.priorSessionContext
        });
        return {
            note: buildAiConsultationNote(transcript, fallbackSummary),
            aiSummary: fallbackSummary,
            source: 'browser-heuristic-fallback'
        };
    } finally {
        clearTimeout(timer);
    }
}

function resetRecordingState() {
    clearRecordingTimer();
    recognition = null;
    finalTranscript = '';
    interimTranscript = '';
    finalSegments = [];
    if (interimFlushTimer) {
        clearTimeout(interimFlushTimer);
        interimFlushTimer = null;
    }
    resetRecordingTime();
    savePending = false;
    shouldFinalizeOnStop = false;
    isPaused = false;
    if (transcriptionBox) {
        transcriptionBox.textContent = '';
    }
    if (notesBox) {
        notesBox.textContent = '';
    }
    if (recordingMetadata) {
        recordingMetadata.textContent = '';
    }
}

async function finalizeConsultation() {
    if (savePending) {
        return;
    }

    savePending = true;
    isFinalizing = true;
    recognition = null;
    const capturedDuration = Math.max(0, Number(recordingDurationAtStop || recordingTime || 0));
    const transcript = getFullTranscript();
    const sessionTime = formatRecordingTime(capturedDuration);
    const savedAt = new Date().toLocaleString('en-US');
    const sessionId = nextSessionId(getSavedConsultations());
    const sessionDate = new Date().toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    });
    const priorSessions = getSessionsForCurrentPatient();

    if (!transcript) {
        if (transcriptionBox) {
            transcriptionBox.textContent = 'No intelligible speech captured. Please try recording again.';
        }
        if (notesBox) {
            notesBox.textContent = 'SOAP note could not be generated without transcript text.';
        }
        if (recordingStatus) {
            recordingStatus.textContent = '⚠ No speech detected. Please try again.';
        }
        if (recordingMetadata) {
            recordingMetadata.textContent = 'No consultation record was saved.';
        }
        shouldFinalizeOnStop = false;
        isFinalizing = false;
        recordingDurationAtStop = 0;
        savePending = false;
        return;
    }

    if (recordingStatus) {
        recordingStatus.textContent = '📄 Generating initial SOAP preview from transcript...';
    }

    const immediateSummary = generateAiConsultationSummary(transcript, {
        sessionId,
        sessionDate,
        patient: MOCK_PATIENT_RECORD,
        doctorName: CLINICIAN_DEFAULT,
        savedAt,
        priorSessionContext: priorSessions
    });
    const labeledTranscript = immediateSummary.speakerTranscript || buildSpeakerTranscript(transcript);
    if (notesBox) {
        notesBox.textContent = buildAiConsultationNote(transcript, immediateSummary);
    }
    if (transcriptionBox) {
        transcriptionBox.textContent = labeledTranscript;
    }
    if (recordingStatus) {
        recordingStatus.textContent = '🧠 Sending transcript to Python AI summary service...';
    }

    try {
        const analysis = await generateAiSummaryFromPython(transcript, {
            sessionTime,
            sessionId,
            sessionDate,
            doctorName: CLINICIAN_DEFAULT,
            patient: MOCK_PATIENT_RECORD,
            priorSessionContext: priorSessions,
            savedAt
        });
        const { note, aiSummary, source } = analysis;

        transcriptionBox.textContent = aiSummary.speakerTranscript || labeledTranscript || buildSpeakerTranscript(transcript);
        notesBox.textContent = note;

        try {
            const savedRecord = saveConsultationRecord({
                id: `CONS-${Date.now()}`,
                savedAt,
                duration: sessionTime,
                transcript,
                note,
                patient: MOCK_PATIENT_RECORD,
                sessionId,
                aiSummary,
                source
            });

            if (recordingMetadata) {
                recordingMetadata.textContent = `Saved consultation record ${savedRecord.id} at ${savedRecord.savedAt} (${savedRecord.duration})`;
            }
        } catch (error) {
            console.error('Failed to save consultation locally:', error);
            if (recordingMetadata) {
                recordingMetadata.textContent = 'Consultation generated, but local save was blocked by browser storage settings.';
            }
            if (recordingStatus) {
                recordingStatus.textContent = `⚠ Consultation processed with ${source.toUpperCase()}; preview is shown, but save was not possible.`;
            }
        }

        if (recordingStatus) {
            if (!recordingMetadata || !recordingMetadata.textContent.includes('not possible')) {
                recordingStatus.textContent = `✅ Consultation saved with ${source.toUpperCase()}; SOAP preview and notes are now available.`;
            }
        }
    } catch (error) {
        console.error('Failed to finalize consultation:', error);
        if (recordingStatus) {
            recordingStatus.textContent = `⚠ Could not generate preview: ${error.message || 'Unknown error'}`;
        }
        if (notesBox) {
            const fallbackSummary = generateAiConsultationSummary(transcript, {
                sessionId,
                sessionDate,
                patient: MOCK_PATIENT_RECORD,
                doctorName: CLINICIAN_DEFAULT,
                priorSessionContext: priorSessions
            });
            const fallbackTranscript = fallbackSummary.speakerTranscript || buildSpeakerTranscript(transcript);
            notesBox.textContent = buildAiConsultationNote(transcript, fallbackSummary);
            if (transcriptionBox) {
                transcriptionBox.textContent = fallbackTranscript;
            }
        }
        if (recordingMetadata) {
            recordingMetadata.textContent = `Fallback consultation generated for ${sessionId}.`;
        }
    }

    if (recordingTimeDisplay) {
        recordingTimeDisplay.textContent = formatRecordingTime(0);
    }
    recordingTime = 0;
    shouldFinalizeOnStop = false;
    isFinalizing = false;
    recordingDurationAtStop = 0;
    savePending = false;
}

function startSpeechRecognition() {
    if (!isRecording || isPaused) {
        return;
    }

    const recognitionClass = getSpeechRecognizer();
    if (!recognitionClass) {
        if (recordingStatus) {
            recordingStatus.textContent = 'SpeechRecognition is not supported in this browser. Try Chrome or Edge.';
        }
        isRecording = false;
        updateRecordingControls();
        return;
    }

    recognition = new recognitionClass();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = MAX_SPEECH_ALTERNATIVES;

    recognition.onresult = event => {
        const results = Array.from(event.results).slice(event.resultIndex);
        const interimPieces = [];
        let sawFinalSegment = false;

        results.forEach(result => {
            const text = result[0] && result[0].transcript ? result[0].transcript.trim() : '';
            if (!text) {
                return;
            }

            if (result.isFinal) {
                appendFinalSegment(text);
                sawFinalSegment = true;
            } else {
                interimPieces.push(text);
            }
        });

        interimTranscript = interimPieces.join(' ').trim();

        if (sawFinalSegment) {
            interimTranscript = '';
        }

        updateLiveTranscriptDisplay();
        scheduleInterimFlush();
    };

    recognition.onerror = () => {
        flushInterimToFinal('error');
        if (recordingStatus) {
            recordingStatus.textContent = '❌ Error while transcribing. Please try again.';
        }
    };

    recognition.onend = () => {
        flushInterimToFinal('onend');
        if (!isRecording || isPaused) {
            if (!isRecording && shouldFinalizeOnStop) {
                void finalizeConsultation();
                shouldFinalizeOnStop = false;
            }
            return;
        }

        startSpeechRecognition();
    };

    try {
        recognition.start();
    } catch (error) {
        console.error('Speech recognition failed to start:', error);
        if (recordingStatus) {
            recordingStatus.textContent = '❌ Could not start speech recognition.';
        }
        isRecording = false;
        updateRecordingControls();
    }
}

function startRecording() {
    if (!recordBtn || !recordingStatus || !recordingTimeDisplay || !transcriptionBox || !notesBox) {
        return;
    }

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        recordingStatus.textContent = 'Microphone is not supported in this browser.';
        return;
    }

    if (isRecording) {
        return;
    }

    if (isFinalizing || savePending) {
        if (recordingStatus) {
            recordingStatus.textContent = 'Please wait for the current consultation to finish before starting again.';
        }
        return;
    }

    resetRecordingState();

    if (!getSpeechRecognizer()) {
        recordingStatus.textContent = 'SpeechRecognition is not supported in this browser. Try Chrome or Edge.';
        return;
    }

    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());

            isRecording = true;
            isPaused = false;
            shouldFinalizeOnStop = false;
            isFinalizing = false;
            recordingStatus.textContent = '🎙️ Permission granted. Starting consultant listening session...';
            if (recordingIndicator) {
                recordingIndicator.textContent = 'Microphone active';
                recordingIndicator.classList.add('active');
            }
            updateRecordingControls();
            startSpeechRecognition();
            resetRecordingTime();

            recordingStatus.textContent = 'Recording in progress...';

            if (transcriptionBox) {
                transcriptionBox.textContent = 'Listening...';
            }
            if (notesBox) {
                const priorSessions = getSessionsForCurrentPatient();
                const brief = formatHistoricalBrief();
                const priorText = priorSessions.length
                    ? `Previous Session Brief:
${brief}`
                    : 'No previous session for this patient found. This will be the first session.';
                notesBox.textContent = priorText;
            }
            if (recordingMetadata) {
                recordingMetadata.textContent = formatHistoricalBrief();
            }

            recordingTimer = setInterval(() => {
                recordingTime++;
                recordingTimeDisplay.textContent = formatRecordingTime(recordingTime);
            }, 1000);
        })
        .catch(error => {
            console.error('Error accessing microphone:', error);
            recordingStatus.textContent = 'Microphone access denied. Please allow microphone access to record.';
            isRecording = false;
            updateRecordingControls();
        });
}

function stopRecording() {
    if (!isRecording || !recordBtn) {
        return;
    }

    recordingDurationAtStop = recordingTime;
    isRecording = false;
    isPaused = false;
    shouldFinalizeOnStop = true;
    clearRecordingTimer();
    flushInterimToFinal('stop');
    resetRecordingTime();
    isFinalizing = true;
    if (recognition) {
        try {
            recognition.stop();
        } catch (error) {
            void finalizeConsultation();
            shouldFinalizeOnStop = false;
        }
    } else {
        void finalizeConsultation();
        shouldFinalizeOnStop = false;
    }

    if (recordingIndicator) {
        recordingIndicator.classList.remove('active');
        recordingIndicator.textContent = '';
    }
    if (recordingStatus) {
        recordingStatus.textContent = '🟢 Processing speech and generating notes...';
    }

    updateRecordingControls();
}

function pauseRecording() {
    if (!isRecording || !pauseBtn) {
        return;
    }

    if (isPaused) {
    isPaused = false;
    shouldFinalizeOnStop = false;

    clearRecordingTimer();

        if (recordingIndicator) {
            recordingIndicator.textContent = 'Microphone active';
            recordingIndicator.classList.add('active');
        }
        if (recordingStatus) {
            recordingStatus.textContent = '🎙️ Resuming consultation capture...';
        }

        startSpeechRecognition();
        recordingTimer = setInterval(() => {
            recordingTime++;
            recordingTimeDisplay.textContent = formatRecordingTime(recordingTime);
        }, 1000);
        updateRecordingControls();
        return;
    }

    isPaused = true;
    shouldFinalizeOnStop = false;
    clearRecordingTimer();
    flushInterimToFinal('pause');

    if (recordingIndicator) {
        recordingIndicator.classList.remove('active');
        recordingIndicator.textContent = '';
    }
    if (recordingStatus) {
        recordingStatus.textContent = '⏸ Recording paused. Click Resume to continue.';
    }

    if (recognition) {
        recognition.stop();
    }

    updateRecordingControls();
}

function restartRecording() {
    clearRecordingTimer();

    if (recordingIndicator) {
        recordingIndicator.classList.remove('active');
        recordingIndicator.textContent = '';
    }

    if (recognition) {
        recognition.onend = null;
        recognition.onerror = null;
        recognition.onresult = null;
        try {
            recognition.stop();
        } catch (error) {
            console.error('Error stopping recognizer before restart:', error);
        }
        recognition = null;
    }

    isRecording = false;
    isPaused = false;
    shouldFinalizeOnStop = false;
    isFinalizing = false;
    savePending = false;
    if (recordingStatus) {
        recordingStatus.textContent = '🔄 Restarting consultation recording.';
    }
    flushInterimToFinal('reset');
    resetRecordingState();
    updateRecordingControls();
    clearRecordingTimer();
    if (recordingTimeDisplay) {
        recordingTimeDisplay.textContent = formatRecordingTime(0);
    }
    if (transcriptionBox) {
        transcriptionBox.textContent = 'Ready for a fresh consultation recording.';
    }
    if (notesBox) {
        notesBox.textContent = 'AI summary + SOAP note will be generated after you end recording.';
    }

    recordingTimeDisplay.textContent = formatRecordingTime(recordingTime);
    startRecording();
}

// ==========================================
// Scroll to Section
// ==========================================

function scrollToSection(sectionId) {
    const element = document.getElementById(sectionId);
    if (element) {
        element.scrollIntoView({ behavior: 'smooth' });
    }
}

// ==========================================
// Chart Rendering (Health Analytics)
// ==========================================

function drawHealthChart() {
    const canvas = document.getElementById('healthChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    // Set canvas size
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    
    const width = rect.width;
    const height = rect.height;
    const padding = 40;
    const chartWidth = width - 2 * padding;
    const chartHeight = height - 2 * padding;
    
    // Sample health data (visits over time)
    const data = [
        { visit: 'Visit 1', healthScore: 65 },
        { visit: 'Visit 2', healthScore: 70 },
        { visit: 'Visit 3', healthScore: 75 },
        { visit: 'Visit 4', healthScore: 80 },
        { visit: 'Visit 5', healthScore: 85 },
        { visit: 'Visit 6', healthScore: 90 }
    ];
    
    const maxScore = 100;
    const minScore = 0;
    
    // Draw background
    ctx.fillStyle = '#f5f5f5';
    ctx.fillRect(padding, padding, chartWidth, chartHeight);
    
    // Draw grid lines
    ctx.strokeStyle = '#e0e0e0';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
        const y = padding + (chartHeight / 5) * i;
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(padding + chartWidth, y);
        ctx.stroke();
    }
    
    // Draw axes
    ctx.strokeStyle = '#333333';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, padding + chartHeight);
    ctx.lineTo(padding + chartWidth, padding + chartHeight);
    ctx.stroke();
    
    // Draw data points and line
    ctx.strokeStyle = '#0066cc';
    ctx.lineWidth = 3;
    ctx.fillStyle = '#0066cc';
    
    const xStep = chartWidth / (data.length - 1);
    
    // Draw line
    ctx.beginPath();
    data.forEach((point, index) => {
        const x = padding + index * xStep;
        const y = padding + chartHeight - ((point.healthScore - minScore) / (maxScore - minScore)) * chartHeight;
        
        if (index === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    });
    ctx.stroke();
    
    // Draw points
    data.forEach((point, index) => {
        const x = padding + index * xStep;
        const y = padding + chartHeight - ((point.healthScore - minScore) / (maxScore - minScore)) * chartHeight;
        
        ctx.fillRect(x - 5, y - 5, 10, 10);
    });
    
    // Draw labels
    ctx.fillStyle = '#333333';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    
    data.forEach((point, index) => {
        const x = padding + index * xStep;
        ctx.fillText(point.visit, x, padding + chartHeight + 25);
    });
    
    // Draw Y-axis labels
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
        const value = minScore + (maxScore - minScore) * (i / 5);
        const y = padding + chartHeight - (chartHeight / 5) * i;
        ctx.fillText(Math.round(value), padding - 10, y + 5);
    }
}

// Initialize chart when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    updateRecordingControls();

    drawHealthChart();
    
    // Redraw chart on window resize
    window.addEventListener('resize', drawHealthChart);
});

// ==========================================
// Smooth interactions for demo cards
// ==========================================

document.querySelectorAll('.demo-card').forEach(card => {
    card.addEventListener('mouseenter', function() {
        this.style.transform = 'translateY(-5px)';
    });
    card.addEventListener('mouseleave', function() {
        this.style.transform = 'translateY(0)';
    });
});

// ==========================================
// Add smooth scroll behavior for buttons
// ==========================================

document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        const href = this.getAttribute('href');
        if (href !== '#') {
            e.preventDefault();
            const element = document.querySelector(href);
            if (element) {
                element.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        }
    });
});

// ==========================================
// Intersection Observer for fade-in animations
// ==========================================

const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
};

const observer = new IntersectionObserver(function(entries) {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.animation = 'fadeIn 0.6s ease-in-out';
        }
    });
}, observerOptions);

// Observe all major section elements
document.querySelectorAll('.problem-card, .solution-card, .feature-card, .benefit-item, .dashboard-card, .analytics-card').forEach(el => {
    el.style.opacity = '0';
    observer.observe(el);
});

// Add fade-in animation
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
`;
document.head.appendChild(style);

