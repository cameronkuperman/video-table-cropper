// State
let uploadedVideo = null;
let uploadedJsons = [];

// DOM Elements
const videoZone = document.getElementById('video-zone');
const jsonZone = document.getElementById('json-zone');
const videoInput = document.getElementById('video-input');
const jsonInput = document.getElementById('json-input');
const videoInfo = document.getElementById('video-info');
const jsonList = document.getElementById('json-list');
const processBtn = document.getElementById('process-btn');
const clearBtn = document.getElementById('clear-btn');
const status = document.getElementById('status');
const results = document.getElementById('results');
const resultsList = document.getElementById('results-list');

// Drag and drop handlers
function setupDropZone(zone, input, type) {
    zone.addEventListener('click', () => input.click());

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('dragover');
    });

    zone.addEventListener('dragleave', () => {
        zone.classList.remove('dragover');
    });

    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('dragover');
        const files = Array.from(e.dataTransfer.files);
        handleFiles(files, type);
    });

    input.addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        handleFiles(files, type);
    });
}

// Handle file uploads
async function handleFiles(files, type) {
    for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('type', type);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                if (type === 'video') {
                    uploadedVideo = data.filename;
                    videoInfo.textContent = `✓ ${data.original}`;
                    videoZone.classList.add('has-file');
                } else {
                    uploadedJsons.push(data.filename);
                    const item = document.createElement('div');
                    item.className = 'file-item';
                    item.innerHTML = `
                        <span class="name">${data.original}</span>
                        <span class="tables">${data.tables} tables</span>
                    `;
                    jsonList.appendChild(item);
                    jsonZone.classList.add('has-file');
                }
                updateProcessButton();
            } else {
                showStatus(`Error: ${data.error}`, 'error');
            }
        } catch (err) {
            showStatus(`Upload failed: ${err.message}`, 'error');
        }
    }
}

// Update process button state
function updateProcessButton() {
    processBtn.disabled = !(uploadedVideo && uploadedJsons.length > 0);
}

// Show status message
function showStatus(message, type = 'processing') {
    status.textContent = message;
    status.className = `status ${type}`;
    status.hidden = false;
}

// Hide status
function hideStatus() {
    status.hidden = true;
}

// Process videos
async function processVideos() {
    const btnText = processBtn.querySelector('.btn-text');
    const btnLoading = processBtn.querySelector('.btn-loading');

    btnText.hidden = true;
    btnLoading.hidden = false;
    processBtn.disabled = true;

    showStatus('Processing videos... This may take a while.', 'processing');

    try {
        const response = await fetch('/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                video: uploadedVideo,
                jsons: uploadedJsons
            })
        });

        const data = await response.json();

        if (data.success) {
            showStatus(`Successfully created ${data.count} cropped videos!`, 'success');
            displayResults(data.videos);
        } else {
            showStatus(`Error: ${data.error}`, 'error');
        }
    } catch (err) {
        showStatus(`Processing failed: ${err.message}`, 'error');
    } finally {
        btnText.hidden = false;
        btnLoading.hidden = true;
        updateProcessButton();
    }
}

// Display results
function displayResults(videos) {
    resultsList.innerHTML = '';

    for (const video of videos) {
        const item = document.createElement('div');
        item.className = 'result-item';
        item.innerHTML = `
            <div class="result-info">
                <span class="result-name">${video.filename}</span>
                <span class="result-bbox">Table ${video.table_id} • (${video.bbox.x1}, ${video.bbox.y1}) to (${video.bbox.x2}, ${video.bbox.y2})</span>
            </div>
            <a href="${video.download_url}" class="download-btn">Download</a>
        `;
        resultsList.appendChild(item);
    }

    results.hidden = false;
}

// Clear all
async function clearAll() {
    try {
        await fetch('/cleanup', { method: 'POST' });
    } catch (err) {
        console.error('Cleanup failed:', err);
    }

    uploadedVideo = null;
    uploadedJsons = [];

    videoInfo.textContent = '';
    jsonList.innerHTML = '';
    resultsList.innerHTML = '';

    videoZone.classList.remove('has-file');
    jsonZone.classList.remove('has-file');

    hideStatus();
    results.hidden = true;
    updateProcessButton();
}

// Initialize
setupDropZone(videoZone, videoInput, 'video');
setupDropZone(jsonZone, jsonInput, 'json');
processBtn.addEventListener('click', processVideos);
clearBtn.addEventListener('click', clearAll);
