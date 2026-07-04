let ws;
let isRecording = false;

// Check initial status on load
window.addEventListener('load', async () => {
    try {
        const res = await fetch('/api/status');
        const data = await res.json();
        
        document.getElementById('hw-status').textContent = `Device: ${data.device} | Model: ${data.model}`;
        
        if (data.recording) {
            setRecordingUI(true);
            connectWebSocket();
        }
    } catch (e) {
        console.error('Failed to connect to API:', e);
        document.getElementById('hw-status').textContent = 'Backend Connection Offline';
    }
});

const btnAction = document.getElementById('btn-action');

btnAction.addEventListener('click', toggleRecording);

async function toggleRecording() {
    if (!isRecording) {
        // Start recording
        btnAction.disabled = true;
        btnAction.textContent = "Starting...";
        try {
            const res = await fetch('/api/start', { method: 'POST' });
            const data = await res.json();
            if (data.status === 'started' || data.status === 'already_recording') {
                setRecordingUI(true);
                connectWebSocket();
            }
        } catch (e) {
            console.error('Start failed:', e);
            alert('Failed to start transcription session.');
        } finally {
            btnAction.disabled = false;
        }
    } else {
        // Stop recording
        btnAction.disabled = true;
        btnAction.textContent = "Stopping...";
        try {
            const res = await fetch('/api/stop', { method: 'POST' });
            const data = await res.json();
            if (data.status === 'stopped') {
                setRecordingUI(false);
                if (ws) ws.close();
                showStats(data.stats);
            }
        } catch (e) {
            console.error('Stop failed:', e);
            alert('Failed to stop transcription session.');
        } finally {
            btnAction.disabled = false;
        }
    }
}

function setRecordingUI(recording) {
    isRecording = recording;
    const body = document.body;
    
    if (recording) {
        body.classList.add('recording');
        btnAction.classList.add('active');
        btnAction.textContent = "Stop Session";
        document.querySelector('.status-text').textContent = "Live";
        
        // Clear viewport
        const viewport = document.getElementById('transcript-viewport');
        viewport.innerHTML = '';
    } else {
        body.classList.remove('recording');
        btnAction.classList.remove('active');
        btnAction.textContent = "Start Session";
        document.querySelector('.status-text').textContent = "Idle";
    }
}

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        addTranscriptBubble(data.speaker, data.start_ts, data.text);
    };
    
    ws.onclose = () => {
        console.log('WebSocket closed.');
    };
}

function addTranscriptBubble(speaker, timestamp, text) {
    const viewport = document.getElementById('transcript-viewport');
    
    // Remove empty state if present
    const emptyState = viewport.querySelector('.empty-state');
    if (emptyState) {
        viewport.removeChild(emptyState);
    }
    
    const bubble = document.createElement('div');
    const isMe = speaker === 'Me';
    bubble.className = `msg-bubble ${isMe ? 'me' : 'other'}`;
    
    const formattedTime = `${timestamp.toFixed(1)}s`;
    
    bubble.innerHTML = `
        <div class="msg-header">
            <span class="msg-speaker">${isMe ? 'Me' : 'Speaker 1'}</span>
            <span class="msg-time">${formattedTime}</span>
        </div>
        <div class="msg-text">${text}</div>
    `;
    
    viewport.appendChild(bubble);
    
    // Smooth auto scroll to the bottom
    viewport.scrollTo({
        top: viewport.scrollHeight,
        behavior: 'smooth'
    });
}

// Stats modal controls
function showStats(stats) {
    document.getElementById('stat-duration').textContent = `${stats.duration.toFixed(1)}s`;
    
    const maxRtf = Math.max(stats.mic.rtf, stats.sys.rtf);
    document.getElementById('stat-rtf').textContent = maxRtf.toFixed(3);
    
    document.getElementById('stat-mic-segments').textContent = `Segments Transcribed: ${stats.mic.segments}`;
    document.getElementById('stat-mic-time').textContent = `Inference Time: ${stats.mic.inference_time.toFixed(1)}s`;
    
    document.getElementById('stat-sys-segments').textContent = `Segments Transcribed: ${stats.sys.segments}`;
    document.getElementById('stat-sys-time').textContent = `Inference Time: ${stats.sys.inference_time.toFixed(1)}s`;
    
    document.getElementById('stats-modal').style.display = 'flex';
}

function closeModal() {
    document.getElementById('stats-modal').style.display = 'none';
}

function getTranscriptText() {
    const bubbles = document.querySelectorAll('.msg-bubble');
    let text = '';
    bubbles.forEach(b => {
        const speaker = b.querySelector('.msg-speaker').textContent;
        const time = b.querySelector('.msg-time').textContent;
        const content = b.querySelector('.msg-text').textContent;
        text += `[${time}] ${speaker}: ${content}\n`;
    });
    return text;
}

function copyTranscript() {
    const text = getTranscriptText();
    if (!text.trim()) {
        alert('Transcript is empty.');
        return;
    }
    
    navigator.clipboard.writeText(text).then(() => {
        alert('Transcript copied to clipboard!');
    }).catch(err => {
        console.error('Copy failed:', err);
    });
}

function exportTranscript() {
    const text = getTranscriptText();
    if (!text.trim()) {
        alert('Transcript is empty.');
        return;
    }
    
    const blob = new Blob([text], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transcript_${new Date().toISOString().slice(0,10)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}
