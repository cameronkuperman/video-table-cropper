// State
let uploadedVideos = [];  // Array of {filename, original, camera, autoDetected}
let availableCameras = [];  // Loaded from /cameras API
let currentJobId = null;
let currentVideos = [];

// DOM Elements
const videoZone = document.getElementById('video-zone');
const videoInput = document.getElementById('video-input');
const videoList = document.getElementById('video-list');
const processBtn = document.getElementById('process-btn');
const clearBtn = document.getElementById('clear-btn');
const status = document.getElementById('status');
const results = document.getElementById('results');
const resultsList = document.getElementById('results-list');
const downloadAllBtn = document.getElementById('download-all-btn');

// Drag and drop handlers for video zone
videoZone.addEventListener('click', () => videoInput.click());

videoZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    videoZone.classList.add('dragover');
});

videoZone.addEventListener('dragleave', () => {
    videoZone.classList.remove('dragover');
});

videoZone.addEventListener('drop', (e) => {
    e.preventDefault();
    videoZone.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files);
    handleVideoUpload(files);
});

videoInput.addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    handleVideoUpload(files);
});

// Load available cameras from API
async function loadCameras() {
    if (availableCameras.length > 0) return;  // Already loaded

    try {
        const response = await fetch('/cameras');
        const data = await response.json();
        availableCameras = data.cameras;
    } catch (error) {
        console.error('Failed to load cameras:', error);
        showStatus('Failed to load camera list', 'error');
    }
}

// Handle video file uploads
async function handleVideoUpload(files) {
    // Load cameras first if not loaded
    await loadCameras();

    for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('type', 'video');

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.success) {
                // Auto-detect camera from filename
                const detectedCamera = detectCameraFromFilename(data.original);

                uploadedVideos.push({
                    filename: data.filename,
                    original: data.original,
                    camera: detectedCamera || "",
                    autoDetected: !!detectedCamera
                });

                renderVideoList();
                updateProcessButton();
            } else {
                showStatus(`Error: ${data.error}`, 'error');
            }
        } catch (error) {
            showStatus(`Upload failed: ${error.message}`, 'error');
        }
    }
}

// Detect camera ID from filename
function detectCameraFromFilename(filename) {
    const match = filename.match(/IPC(\d+)/i);
    if (match) {
        return `IPC${match[1]}`;
    }
    return null;
}

// Render the video list with camera selectors
function renderVideoList() {
    videoList.innerHTML = '';

    if (uploadedVideos.length === 0) {
        return;
    }

    uploadedVideos.forEach((video, index) => {
        const item = document.createElement('div');
        item.className = 'video-item';
        if (!video.camera) {
            item.classList.add('missing-camera');
        }
        item.dataset.index = index;

        const cameraOptions = availableCameras.map(cam =>
            `<option value="${cam.id}" ${video.camera === cam.id ? 'selected' : ''}>
                ${cam.name} (${cam.tables} tables)
             </option>`
        ).join('');

        item.innerHTML = `
            <div class="video-info">
                <span class="video-name">${video.original}</span>
                <button class="remove-btn" onclick="removeVideo(${index})">✕</button>
            </div>
            <div class="camera-selector">
                <label>Camera:</label>
                <select class="camera-dropdown" onchange="updateVideoCamera(${index}, this.value)">
                    <option value="">Select camera...</option>
                    ${cameraOptions}
                </select>
                ${video.autoDetected ? '<span class="detection-badge">Auto-detected</span>' : ''}
            </div>
        `;

        videoList.appendChild(item);
    });
}

// Update camera selection for a video
function updateVideoCamera(index, cameraId) {
    uploadedVideos[index].camera = cameraId;
    uploadedVideos[index].autoDetected = false;
    renderVideoList();
    updateProcessButton();
}

// Remove a video from the list
function removeVideo(index) {
    uploadedVideos.splice(index, 1);
    renderVideoList();
    updateProcessButton();
}

// Update process button state
function updateProcessButton() {
    // Enable if all videos have cameras assigned
    const allHaveCameras = uploadedVideos.length > 0 &&
                          uploadedVideos.every(v => v.camera);

    processBtn.disabled = !allHaveCameras;

    if (uploadedVideos.length > 0 && !allHaveCameras) {
        showStatus('Please select a camera for each video', 'warning');
    } else if (allHaveCameras) {
        hideStatus();
    }
}

// Process videos
async function processVideos() {
    const btnText = processBtn.querySelector('.btn-text');
    const btnLoading = processBtn.querySelector('.btn-loading');

    btnText.hidden = true;
    btnLoading.hidden = false;
    processBtn.disabled = true;

    const cameraCount = new Set(uploadedVideos.map(v => v.camera)).size;
    showStatus(`Cropping tables from ${uploadedVideos.length} video(s) across ${cameraCount} camera(s)...`, 'processing');

    try {
        const response = await fetch('/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                videos: uploadedVideos.map(v => ({
                    filename: v.filename,
                    camera: v.camera
                }))
            })
        });

        const data = await response.json();

        if (data.success) {
            currentJobId = data.job_id;
            currentVideos = data.videos;
            showStatus(
                `Success! Cropped ${data.count} tables from ${data.cameras_processed.length} camera(s). ` +
                `Extracted ${data.frame_count} frames.`,
                'success'
            );
            displayResults(data.videos, data.labeling_job_id, data.frame_count);
        } else {
            showStatus(`Error: ${data.error}`, 'error');
        }
    } catch (error) {
        showStatus(`Processing failed: ${error.message}`, 'error');
    } finally {
        btnText.hidden = false;
        btnLoading.hidden = true;
        processBtn.disabled = false;
    }
}

// Display results
function displayResults(videos, labelingJobId, frameCount) {
    resultsList.innerHTML = '';

    // Group videos by camera for organized display
    const videosByCamera = {};
    for (const video of videos) {
        const camera = video.camera || 'Unknown';
        if (!videosByCamera[camera]) {
            videosByCamera[camera] = [];
        }
        videosByCamera[camera].push(video);
    }

    // Display videos grouped by camera
    for (const [camera, cameraVideos] of Object.entries(videosByCamera)) {
        const cameraSection = document.createElement('div');
        cameraSection.style.marginBottom = '20px';

        const cameraHeader = document.createElement('h3');
        cameraHeader.textContent = `Camera ${camera.replace('IPC', '')} (${cameraVideos.length} tables)`;
        cameraHeader.style.marginBottom = '10px';
        cameraHeader.style.color = '#555';
        cameraSection.appendChild(cameraHeader);

        for (const video of cameraVideos) {
            const item = document.createElement('div');
            item.className = 'result-item';

            // Format bbox info
            let bboxInfo = '';
            if (video.bbox.center) {
                bboxInfo = `Table ${video.table_id} • Rotated bbox`;
            } else if (video.bbox.x1 !== undefined) {
                bboxInfo = `Table ${video.table_id} • (${video.bbox.x1}, ${video.bbox.y1}) to (${video.bbox.x2}, ${video.bbox.y2})`;
            } else {
                bboxInfo = `Table ${video.table_id}`;
            }

            item.innerHTML = `
                <div class="result-info">
                    <span class="result-name">${video.filename}</span>
                    <span class="result-bbox">${bboxInfo}</span>
                </div>
                <a href="${video.download_url}" download="${video.filename}" class="download-btn">Download</a>
            `;
            cameraSection.appendChild(item);
        }

        resultsList.appendChild(cameraSection);
    }

    // Add "Label Frames" button if frames were extracted
    if (labelingJobId && frameCount > 0) {
        const labelSection = document.createElement('div');
        labelSection.style.marginTop = '30px';
        labelSection.style.padding = '20px';
        labelSection.style.background = '#f0f8ff';
        labelSection.style.borderRadius = '12px';
        labelSection.style.border = '2px solid #007bff';
        labelSection.innerHTML = `
            <h3 style="margin-top: 0; color: #007bff;">Ready for Labeling</h3>
            <p style="margin: 10px 0;">
                Extracted <strong>${frameCount} frames</strong> from all cropped videos.
                Label them as Clean, Occupied, or Dirty to create your training dataset.
            </p>
            <button id="label-frames-btn" class="btn primary" style="margin-top: 15px; padding: 15px 30px; font-size: 16px;">
                Label ${frameCount} Frames →
            </button>
        `;
        resultsList.appendChild(labelSection);

        // Add click handler
        document.getElementById('label-frames-btn').addEventListener('click', () => {
            window.location.href = `/legacy/label/${labelingJobId}`;
        });
    }

    results.hidden = false;
}

// Clear all
function clearAll() {
    uploadedVideos = [];
    currentJobId = null;
    currentVideos = [];
    videoList.innerHTML = '';
    resultsList.innerHTML = '';
    results.hidden = true;
    videoInput.value = '';
    updateProcessButton();
    hideStatus();
}

// Show status message
function showStatus(message, type) {
    status.textContent = message;
    status.className = `status ${type}`;
    status.hidden = false;
}

// Hide status
function hideStatus() {
    status.hidden = true;
}

// Download all as ZIP
async function downloadAllAsZip() {
    if (!currentJobId) return;

    downloadAllBtn.disabled = true;
    downloadAllBtn.textContent = 'Preparing ZIP...';

    try {
        window.location.href = `/download-zip/${currentJobId}`;

        setTimeout(() => {
            downloadAllBtn.disabled = false;
            downloadAllBtn.textContent = 'Download All as ZIP';
        }, 2000);
    } catch (error) {
        showStatus(`Download failed: ${error.message}`, 'error');
        downloadAllBtn.disabled = false;
        downloadAllBtn.textContent = 'Download All as ZIP';
    }
}

// Event listeners
processBtn.addEventListener('click', processVideos);
clearBtn.addEventListener('click', clearAll);
downloadAllBtn.addEventListener('click', downloadAllAsZip);

// Initialize
loadCameras();
