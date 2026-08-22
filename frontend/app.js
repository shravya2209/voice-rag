/* Voice-RAG Frontend Logic */

const API_BASE = window.location.origin;

// ── State ─────────────────────────────────────────────────────
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];

// ── Tab Switching ─────────────────────────────────────────────
function switchTab(mode) {
    document.getElementById('tab-text').classList.toggle('active', mode === 'text');
    document.getElementById('tab-voice').classList.toggle('active', mode === 'voice');
    document.getElementById('mode-text').classList.toggle('hidden', mode !== 'text');
    document.getElementById('mode-voice').classList.toggle('hidden', mode !== 'voice');
}

// ── Text Query ────────────────────────────────────────────────
async function sendTextQuery() {
    const input = document.getElementById('query-input');
    const query = input.value.trim();
    if (!query) return;

    showLoading('Processing text query...');
    hideResults();

    try {
        const res = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        displayResults(data);
    } catch (err) {
        showError(err.message);
    } finally {
        hideLoading();
    }
}

// Enter key handler
document.addEventListener('DOMContentLoaded', () => {
    const input = document.getElementById('query-input');
    if (input) {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') sendTextQuery();
        });
    }
});

// ── Voice Recording ───────────────────────────────────────────
async function toggleRecording() {
    if (isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
        audioChunks = [];

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) audioChunks.push(e.data);
        };

        mediaRecorder.onstop = async () => {
            stream.getTracks().forEach(t => t.stop());
            const blob = new Blob(audioChunks, { type: 'audio/webm' });
            await sendVoiceQuery(blob);
        };

        mediaRecorder.start();
        isRecording = true;

        const btn = document.getElementById('mic-btn');
        btn.classList.add('recording');
        document.getElementById('mic-label').textContent = 'Recording... Click to stop';
    } catch (err) {
        showError('Microphone access denied or not available');
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    isRecording = false;

    const btn = document.getElementById('mic-btn');
    btn.classList.remove('recording');
    document.getElementById('mic-label').textContent = 'Click to start recording';
}

async function sendVoiceQuery(blob) {
    showLoading('Transcribing and processing...');
    hideResults();

    const langSelect = document.getElementById('lang-select');
    const selectedLang = langSelect ? langSelect.value : 'auto';

    const formData = new FormData();
    formData.append('file', blob, 'audio.webm');
    formData.append('language', selectedLang);

    try {
        const res = await fetch(`${API_BASE}/api/voice-query`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        displayResults(data);
    } catch (err) {
        showError(err.message);
    } finally {
        hideLoading();
    }
}

// ── Display Results ───────────────────────────────────────────
function displayResults(data) {
    // Transcript
    if (data.transcript) {
        const section = document.getElementById('transcript-section');
        section.classList.remove('hidden');
        section.classList.add('fade-in');
        document.getElementById('transcript-text').textContent = `"${data.transcript}"`;
    }

    // Answer
    if (data.answer) {
        const section = document.getElementById('answer-section');
        section.classList.remove('hidden');
        section.classList.add('fade-in');
        document.getElementById('answer-text').textContent = data.answer;

        // Grounding bar
        const score = data.grounding_score || 0;
        const pct = Math.round(score * 100);
        document.getElementById('grounding-fill').style.width = pct + '%';
        document.getElementById('grounding-value').textContent = pct + '%';
    }

    // Sources
    if (data.sources && data.sources.length > 0) {
        const section = document.getElementById('sources-section');
        section.classList.remove('hidden');
        section.classList.add('fade-in');

        const list = document.getElementById('sources-list');
        list.innerHTML = '';

        data.sources.forEach((src, i) => {
            const scoreClass = src.score > 0.5 ? 'score-high' :
                              src.score > 0.3 ? 'score-medium' : 'score-low';

            const item = document.createElement('div');
            item.className = 'source-item';
            item.innerHTML = `
                <div class="source-header">
                    <span class="source-id">#${i + 1} ${src.chunk_id}</span>
                    <span class="source-score ${scoreClass}">${src.score.toFixed(4)}</span>
                </div>
                <p class="source-text">${escapeHtml(src.text)}</p>
            `;
            list.appendChild(item);
        });
    }

    // Latency
    if (data.latency_ms) {
        const section = document.getElementById('latency-section');
        section.classList.remove('hidden');
        section.classList.add('fade-in');

        const grid = document.getElementById('latency-grid');
        grid.innerHTML = '';

        const items = data.latency_ms;
        const order = ['stt', 'embedding', 'retrieval', 'reranking', 'generation', 'total'];

        for (const key of order) {
            if (items[key] !== null && items[key] !== undefined) {
                const div = document.createElement('div');
                div.className = 'latency-item';
                div.innerHTML = `
                    <div class="latency-label">${key}</div>
                    <div class="latency-value">${items[key].toFixed(1)}<span style="font-size:0.7rem;color:var(--text-muted)">ms</span></div>
                `;
                grid.appendChild(div);
            }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────
function showLoading(text) {
    document.getElementById('loading-text').textContent = text;
    document.getElementById('loading-section').classList.remove('hidden');
    document.getElementById('status-badge').textContent = '● Processing';
    document.getElementById('status-badge').style.color = 'var(--warning)';
    document.getElementById('status-badge').style.background = 'rgba(251,191,36,0.12)';
}

function hideLoading() {
    document.getElementById('loading-section').classList.add('hidden');
    document.getElementById('status-badge').textContent = '● Ready';
    document.getElementById('status-badge').style.color = 'var(--success)';
    document.getElementById('status-badge').style.background = 'rgba(52,211,153,0.12)';
}

function hideResults() {
    ['transcript-section', 'answer-section', 'sources-section', 'latency-section']
        .forEach(id => document.getElementById(id).classList.add('hidden'));
}

function showError(message) {
    hideLoading();
    const section = document.getElementById('answer-section');
    section.classList.remove('hidden');
    section.classList.add('fade-in');
    document.getElementById('answer-text').textContent = `Error: ${message}`;
    document.getElementById('grounding-fill').style.width = '0%';
    document.getElementById('grounding-value').textContent = '—';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
