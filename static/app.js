// AutoLabeler label UI

let folders = [];
let currentIndex = 0;
const frameCache = {};  // folder_id -> {frame_0, frame_1, frame_2}

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
        await loadStats();
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
            <button class="btn skip"     onclick="skipCurrent()">Skip → [→]</button>
        </div>
    `;

    // Prefetch next folder's frames in the background
    if (currentIndex + 1 < folders.length) {
        fetchFrames(folders[currentIndex + 1]).catch(() => {});
    }
}

async function labelCurrent(label) {
    const folder = folders[currentIndex];
    disableButtons();
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
            showError('Label error: ' + data.error);
            enableButtons();
            return;
        }
        showError(null);
        delete frameCache[folder.folder_id];
        folders.splice(currentIndex, 1);
        if (currentIndex >= folders.length && currentIndex > 0) currentIndex--;
        await renderCard();
        loadStats();
    } catch (e) {
        showError('Network error: ' + e.message);
        enableButtons();
    }
}

async function skipCurrent() {
    currentIndex = (currentIndex + 1) % Math.max(folders.length, 1);
    await renderCard();
}

function disableButtons() {
    cardEl.querySelectorAll('button').forEach(b => b.disabled = true);
}

function enableButtons() {
    cardEl.querySelectorAll('button').forEach(b => b.disabled = false);
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
