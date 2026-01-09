// State
let uploadedVideo = null;
let uploadedJsons = [];
let currentJobId = null;
let currentVideos = [];

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
const jsonPaste = document.getElementById('json-paste');
const addJsonBtn = document.getElementById('add-json-btn');
const downloadAllBtn = document.getElementById('download-all-btn');

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

// Handle pasted JSON
async function handlePastedJson() {
    const text = jsonPaste.value.trim();
    if (!text) {
        showStatus('Please paste JSON first', 'error');
        return;
    }

    // Validate JSON
    try {
        JSON.parse(text);
    } catch (e) {
        showStatus('Invalid JSON format', 'error');
        return;
    }

    // Create a blob and upload it
    const blob = new Blob([text], { type: 'application/json' });
    const file = new File([blob], 'pasted.json', { type: 'application/json' });

    const formData = new FormData();
    formData.append('file', file);
    formData.append('type', 'json');

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            uploadedJsons.push(data.filename);
            const item = document.createElement('div');
            item.className = 'file-item';
            item.innerHTML = `
                <span class="name">Pasted JSON</span>
                <span class="tables">${data.tables} tables</span>
            `;
            jsonList.appendChild(item);
            jsonZone.classList.add('has-file');
            jsonPaste.value = '';
            showStatus('JSON added successfully!', 'success');
            updateProcessButton();
        } else {
            showStatus(`Error: ${data.error}`, 'error');
        }
    } catch (err) {
        showStatus(`Upload failed: ${err.message}`, 'error');
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

// Trigger browser download
function triggerDownload(url, filename) {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
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
            currentJobId = data.job_id;
            currentVideos = data.videos;
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
            <a href="${video.download_url}" download="${video.filename}" class="download-btn">Download</a>
        `;
        resultsList.appendChild(item);
    }

    results.hidden = false;
}

// Download all videos
async function downloadAll() {
    if (currentVideos.length === 0) return;

    showStatus('Downloading all videos...', 'processing');

    try {
        const response = await fetch(`/download-zip/${currentJobId}`);
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            triggerDownload(url, `cropped_videos_${currentJobId}.zip`);
            window.URL.revokeObjectURL(url);
            showStatus('Download complete!', 'success');
        } else {
            // Fallback: download individually
            for (const video of currentVideos) {
                triggerDownload(video.download_url, video.filename);
                await new Promise(r => setTimeout(r, 500)); // Small delay between downloads
            }
            showStatus('All downloads started!', 'success');
        }
    } catch (err) {
        // Fallback: download individually
        for (const video of currentVideos) {
            triggerDownload(video.download_url, video.filename);
            await new Promise(r => setTimeout(r, 500));
        }
        showStatus('All downloads started!', 'success');
    }
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
    currentJobId = null;
    currentVideos = [];

    videoInfo.textContent = '';
    jsonList.innerHTML = '';
    resultsList.innerHTML = '';
    jsonPaste.value = '';

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
addJsonBtn.addEventListener('click', handlePastedJson);
downloadAllBtn.addEventListener('click', downloadAll);
