// AutoLabeler label UI

let folders = [];
let currentIndex = 0;
const frameCache = {};  // folder_id -> {frame_0, frame_1, frame_2}
const imagePreload = {}; // folder_id -> [Image, Image, Image] (browser-level preload)
let labeling = false;  // guard against double-tap
let pendingMoves = 0;  // count of in-flight Drive moves

const PREFETCH_AHEAD = 3;

const cardEl = document.getElementById('card');
const progressEl = document.getElementById('progress');
const statsEl = document.getElementById('stats');
const doneEl = document.getElementById('done');
const errorEl = document.getElementById('error');

async function init() {
    showError(null);
    cardEl.innerHTML = '<p class="loading">Loading from Drive...</p>';
    try {
        const res = await fetch('/api/folders');
        const data = await res.json();
        if (data.error) { showError(data.error); return; }
        folders = data.folders;
        currentIndex = 0;
        loadStats();
        if (folders.length === 0) {
            cardEl.innerHTML = '';
            doneEl.style.display = 'block';
        } else {
            doneEl.style.display = 'none';
            await renderCard();
        }
    } catch (e) {
        showError('Failed to connect: ' + e.message);
    }
}

async function loadStats() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        if (!data.error) {
            statsEl.textContent =
                `unlabeled: ${data.unlabeled}  ·  clean: ${data.clean}  ·  dirty: ${data.dirty}  ·  occupied: ${data.occupied}`;
        }
    } catch (_) {}
}

async function fetchFrames(folder) {
    if (frameCache[folder.folder_id]) return frameCache[folder.folder_id];
    const res = await fetch(`/api/folder/${folder.folder_id}/frames`);
    const data = await res.json();
    frameCache[folder.folder_id] = data;
    return data;
}

// Preload actual image bytes into browser cache so rendering is instant
function preloadImages(frames) {
    const keys = ['frame_0', 'frame_1', 'frame_2'];
    for (const k of keys) {
        if (frames[k]) {
            const img = new Image();
            img.src = `/api/preview/${frames[k]}`;
        }
    }
}

// Prefetch frames + images for the next N folders
function prefetchAhead() {
    for (let i = 1; i <= PREFETCH_AHEAD; i++) {
        const idx = currentIndex + i;
        if (idx < folders.length) {
            const folder = folders[idx];
            fetchFrames(folder).then(frames => {
                if (frames.frame_0 && frames.frame_1 && frames.frame_2) {
                    preloadImages(frames);
                }
            }).catch(() => {});
        }
    }
}

async function renderCard() {
    if (currentIndex >= folders.length) {
        cardEl.innerHTML = '';
        doneEl.style.display = 'block';
        loadStats();
        return;
    }
    doneEl.style.display = 'none';
    const folder = folders[currentIndex];
    progressEl.textContent = `${currentIndex + 1} / ${folders.length}`;

    cardEl.innerHTML = '<p class="loading">Loading images...</p>';

    let frames;
    try {
        frames = await fetchFrames(folder);
    } catch (e) {
        showError('Failed to load frames: ' + e.message);
        return;
    }

    // Skip folders that don't have all 3 frames yet (still uploading)
    if (!frames.frame_0 || !frames.frame_1 || !frames.frame_2) {
        folders.splice(currentIndex, 1);
        if (currentIndex >= folders.length && currentIndex > 0) currentIndex--;
        await renderCard();
        return;
    }

    const frameKeys = ['frame_0', 'frame_1', 'frame_2'];
    const imgHtml = frameKeys.map(f => {
        const fileId = frames[f];
        if (!fileId) return `<div class="img-placeholder">no image</div>`;
        return `<img src="/api/preview/${fileId}" alt="${f}" />`;
    }).join('');

    cardEl.innerHTML = `
        <div class="folder-name">${escapeHtml(folder.folder_name)}</div>
        <div class="images">${imgHtml}</div>
        <div class="buttons">
            <button class="btn occupied" onclick="labelCurrent('occupied')">Occupied [1]</button>
            <button class="btn dirty"    onclick="labelCurrent('dirty')">Dirty [2]</button>
            <button class="btn clean"    onclick="labelCurrent('clean')">Clean [3]</button>
            <button class="btn skip"     onclick="skipCurrent()">Skip &rarr; [&rarr;]</button>
        </div>
    `;

    labeling = false;
    prefetchAhead();
}

async function labelCurrent(label) {
    if (labeling) return;  // prevent double-tap
    labeling = true;

    const folder = folders[currentIndex];

    // Optimistic: remove from list and advance immediately
    delete frameCache[folder.folder_id];
    folders.splice(currentIndex, 1);
    if (currentIndex >= folders.length && currentIndex > 0) currentIndex--;

    // Show next card without waiting for Drive
    renderCard();

    // Fire Drive move in background
    pendingMoves++;
    try {
        const res = await fetch('/api/label', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                folder_id: folder.folder_id,
                parent_id: folder.parent_id,
                label: label,
            }),
        });
        const data = await res.json();
        if (data.error) {
            showError(`Move failed for ${folder.folder_name}: ${data.error}`);
        }
    } catch (e) {
        showError(`Move failed for ${folder.folder_name}: ${e.message}`);
    } finally {
        pendingMoves--;
        if (pendingMoves === 0) loadStats();
    }
}

async function skipCurrent() {
    currentIndex = (currentIndex + 1) % Math.max(folders.length, 1);
    await renderCard();
}

function showError(msg) {
    errorEl.textContent = msg || '';
    errorEl.style.display = msg ? 'block' : 'none';
}

function escapeHtml(s) {
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// keyboard shortcuts
document.addEventListener('keydown', e => {
    if (folders.length === 0) return;
    if (e.key === '1') labelCurrent('occupied');
    else if (e.key === '2') labelCurrent('dirty');
    else if (e.key === '3') labelCurrent('clean');
    else if (e.key === 'ArrowRight' || e.key === ' ') { e.preventDefault(); skipCurrent(); }
});

init();
