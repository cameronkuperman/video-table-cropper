// AutoLabeler label UI

let folders = [];
let currentIndex = 0;
let labeling = false;
let hasMore = true;
let queueRequest = null;
let renderToken = 0;
let initialTotalUnlabeled = 0;
let readyBufferCount = 0;
let warmingCount = 0;
let queueRetryMs = 1000;
let nextQueueFetchAt = 0;
let stats = {
    unlabeled: 0,
    clean: 0,
    dirty: 0,
    occupied: 0,
    label_later: 0,
};

const INITIAL_QUEUE_BATCH_SIZE = 60;
const REFILL_QUEUE_BATCH_SIZE = 120;
const WARM_BUFFER_SIZE = 72;
const LOW_WATERMARK = 80;
const TIMING_LOGS_ENABLED = true;

const cardEl = document.getElementById('card');
const progressEl = document.getElementById('progress');
const statsEl = document.getElementById('stats');
const bufferStatusEl = document.getElementById('buffer-status');
const doneEl = document.getElementById('done');
const errorEl = document.getElementById('error');

function logTiming(event, fields = {}) {
    if (!TIMING_LOGS_ENABLED) {
        return;
    }
    const details = Object.entries(fields)
        .map(([key, value]) => `${key}=${value}`)
        .join(' ');
    console.debug(`[timing] ${event}${details ? ` ${details}` : ''}`);
}

function setStats(nextStats) {
    stats = {
        ...stats,
        ...nextStats,
    };
    statsEl.textContent =
        `unlabeled: ${stats.unlabeled}  |  clean: ${stats.clean}  |  dirty: ${stats.dirty}  |  occupied: ${stats.occupied}  |  later: ${stats.label_later}`;
}

function localReadyCount() {
    return Math.max(0, folders.length - currentIndex);
}

function updateBufferStatus() {
    if (!bufferStatusEl) {
        return;
    }

    const localReady = localReadyCount();
    if (localReady > 0) {
        const warmingSuffix = warmingCount > 0 ? ` · warming ${warmingCount}` : '';
        bufferStatusEl.textContent = `Ready buffer: ${localReady} triplets${warmingSuffix}`;
        return;
    }

    if (stats.unlabeled <= 0 && !hasMore) {
        bufferStatusEl.textContent = '';
        return;
    }

    if (warmingCount > 0 || hasMore) {
        bufferStatusEl.textContent = 'Warming next triplets from Drive...';
        return;
    }

    bufferStatusEl.textContent = '';
}

function updateProgress() {
    if (initialTotalUnlabeled > 0) {
        if (stats.unlabeled <= 0) {
            progressEl.textContent = `${initialTotalUnlabeled} / ${initialTotalUnlabeled}`;
            updateBufferStatus();
            return;
        }
        const completed = Math.max(0, initialTotalUnlabeled - stats.unlabeled);
        const current = Math.min(initialTotalUnlabeled, completed + 1);
        progressEl.textContent = `${current} / ${initialTotalUnlabeled}`;
        updateBufferStatus();
        return;
    }

    if (folders.length === 0) {
        progressEl.textContent = '0 / 0';
    } else {
        progressEl.textContent = `${Math.min(currentIndex + 1, folders.length)} / ${folders.length}`;
    }
    updateBufferStatus();
}

async function fetchQueue({
    reset = false,
    includeStats = false,
    limit = REFILL_QUEUE_BATCH_SIZE,
    forceRefresh = false,
} = {}) {
    const startedAt = performance.now();
    if (reset) {
        folders = [];
        currentIndex = 0;
        hasMore = true;
        queueRequest = null;
        initialTotalUnlabeled = 0;
        readyBufferCount = 0;
        warmingCount = 0;
        queueRetryMs = 1000;
        nextQueueFetchAt = 0;
    }

    if (!hasMore && !reset) {
        return { folders: [] };
    }

    if (queueRequest) {
        return queueRequest;
    }

    const now = performance.now();
    if (!forceRefresh && !reset && now < nextQueueFetchAt) {
        return {
            folders: [],
            has_more: hasMore,
            retry_ms: Math.max(0, Math.ceil(nextQueueFetchAt - now)),
            total_unlabeled: initialTotalUnlabeled,
            ready_buffer_count: readyBufferCount,
            warming_count: warmingCount,
        };
    }

    const params = new URLSearchParams({ limit: String(limit) });
    if (includeStats) {
        params.set('include_stats', '1');
    }
    if (forceRefresh) {
        params.set('refresh', '1');
    }

    queueRequest = fetch(`/api/queue?${params.toString()}`)
        .then(async res => {
            const data = await res.json();
            if (!res.ok || data.error) {
                throw new Error(data.error || `Queue request failed (${res.status})`);
            }
            return data;
        })
        .then(data => {
            hasMore = Boolean(data.has_more);
            readyBufferCount = data.ready_buffer_count || 0;
            warmingCount = data.warming_count || 0;
            queueRetryMs = data.retry_ms || queueRetryMs;

            if (includeStats && data.stats) {
                setStats(data.stats);
            }
            if (reset || initialTotalUnlabeled === 0) {
                initialTotalUnlabeled = data.total_unlabeled || 0;
            }

            const existingIds = new Set(folders.map(folder => folder.folder_id));
            let added = 0;
            for (const folder of data.folders || []) {
                if (!existingIds.has(folder.folder_id)) {
                    folders.push(folder);
                    existingIds.add(folder.folder_id);
                    added += 1;
                }
            }

            nextQueueFetchAt = added === 0 && hasMore
                ? performance.now() + queueRetryMs
                : 0;

            logTiming('fetchQueue', {
                ms: (performance.now() - startedAt).toFixed(1),
                reset: Number(reset),
                forceRefresh: Number(forceRefresh),
                includeStats: Number(includeStats),
                requestedLimit: limit,
                returned: (data.folders || []).length,
                added,
                buffered: folders.length,
                hasMore: Number(hasMore),
                readyBufferCount,
                warmingCount,
                retryMs: queueRetryMs,
                fetchCooldownMs: Math.max(0, Math.ceil(nextQueueFetchAt - performance.now())),
                totalUnlabeled: data.total_unlabeled || 0,
            });
            updateProgress();
            return data;
        })
        .finally(() => {
            queueRequest = null;
        });

    return queueRequest;
}

function preloadImage(url) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.decoding = 'async';
        img.loading = 'eager';
        let settled = false;

        const finish = () => {
            if (settled) return;
            settled = true;
            if (typeof img.decode === 'function') {
                img.decode().catch(() => {}).finally(() => resolve(img));
            } else {
                resolve(img);
            }
        };

        img.onload = finish;
        img.onerror = () => {
            if (settled) return;
            settled = true;
            reject(new Error(`Failed to load ${url}`));
        };
        img.src = url;

        if (img.complete) {
            finish();
        }
    });
}

function preloadFolder(folder) {
    if (folder.preloadPromise) {
        return folder.preloadPromise;
    }

    const startedAt = performance.now();
    const urls = ['frame_0', 'frame_1', 'frame_2']
        .map(key => folder.preview_urls?.[key])
        .filter(Boolean);

    folder.preloadPromise = Promise.all(urls.map(preloadImage))
        .then(images => {
            folder.preloadedImages = images;
            folder.preloaded = true;
            logTiming('preloadFolder', {
                ms: (performance.now() - startedAt).toFixed(1),
                folderId: folder.folder_id,
                folderName: folder.folder_name,
                images: images.length,
            });
            return folder;
        })
        .catch(error => {
            folder.preloadPromise = null;
            throw error;
        });

    return folder.preloadPromise;
}

function sleep(ms) {
    return new Promise(resolve => window.setTimeout(resolve, ms));
}

async function ensureCurrentFolder() {
    while (currentIndex >= folders.length) {
        if (!hasMore) {
            return null;
        }

        const data = await fetchQueue();
        if (currentIndex < folders.length) {
            return folders[currentIndex];
        }
        if (!data.has_more) {
            return null;
        }

        await sleep(data.retry_ms || queueRetryMs);
    }
    return folders[currentIndex] || null;
}

function warmBuffer() {
    for (let idx = currentIndex; idx < Math.min(folders.length, currentIndex + WARM_BUFFER_SIZE); idx++) {
        preloadFolder(folders[idx]).catch(() => {});
    }

    if (hasMore && localReadyCount() < LOW_WATERMARK) {
        fetchQueue({ limit: REFILL_QUEUE_BATCH_SIZE })
            .then(() => {
                for (let idx = currentIndex; idx < Math.min(folders.length, currentIndex + WARM_BUFFER_SIZE); idx++) {
                    preloadFolder(folders[idx]).catch(() => {});
                }
            })
            .catch(() => {});
    }
}

async function refreshStats() {
    const startedAt = performance.now();
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();
        if (!res.ok || data.error) {
            throw new Error(data.error || `Stats request failed (${res.status})`);
        }
        setStats(data);
        updateProgress();
        logTiming('refreshStats', {
            ms: (performance.now() - startedAt).toFixed(1),
            unlabeled: data.unlabeled,
            clean: data.clean,
            dirty: data.dirty,
            occupied: data.occupied,
            labelLater: data.label_later,
        });
    } catch (_) {}
}

async function renderCard() {
    const startedAt = performance.now();
    const token = ++renderToken;
    showError(null);
    cardEl.innerHTML = '<p class="loading">Warming next images from Drive...</p>';

    let folder;
    try {
        folder = await ensureCurrentFolder();
    } catch (e) {
        showError('Failed to load queue: ' + e.message);
        return;
    }

    if (token !== renderToken) return;

    if (!folder) {
        cardEl.innerHTML = '';
        doneEl.style.display = 'block';
        updateProgress();
        return;
    }

    doneEl.style.display = 'none';
    updateProgress();

    try {
        await preloadFolder(folder);
    } catch (e) {
        showError('Failed to load images: ' + e.message);
        return;
    }

    if (token !== renderToken) return;

    const frameKeys = ['frame_0', 'frame_1', 'frame_2'];
    const imgHtml = frameKeys.map(key => {
        const url = folder.preview_urls?.[key];
        if (!url) return '<div class="img-placeholder">no image</div>';
        return `<img src="${url}" alt="${key}" />`;
    }).join('');

    cardEl.innerHTML = `
        <div class="folder-name">${escapeHtml(folder.folder_name)}</div>
        <div class="images">${imgHtml}</div>
        <div class="buttons">
            <button class="btn occupied" onclick="labelCurrent('occupied')">Occupied [1]</button>
            <button class="btn dirty"    onclick="labelCurrent('dirty')">Dirty [2]</button>
            <button class="btn clean"    onclick="labelCurrent('clean')">Clean [3]</button>
            <button class="btn later"    onclick="labelCurrent('label_later')">Label Later [4]</button>
            <button class="btn skip"     onclick="skipCurrent()">Skip &rarr; [&rarr;]</button>
        </div>
    `;

    labeling = false;
    logTiming('renderCard', {
        ms: (performance.now() - startedAt).toFixed(1),
        folderId: folder.folder_id,
        folderName: folder.folder_name,
        buffered: localReadyCount(),
    });
    warmBuffer();
}

function applyOptimisticLabel(label) {
    if (typeof stats.unlabeled === 'number' && stats.unlabeled > 0) {
        stats.unlabeled -= 1;
    }
    if (typeof stats[label] === 'number') {
        stats[label] += 1;
    }
    updateProgress();
    setStats(stats);
}

async function init(forceRefresh = false) {
    labeling = false;
    showError(null);
    doneEl.style.display = 'none';
    cardEl.innerHTML = '<p class="loading">Building ready buffer from Drive...</p>';

    try {
        await fetchQueue({
            reset: true,
            includeStats: false,
            limit: INITIAL_QUEUE_BATCH_SIZE,
            forceRefresh,
        });
        if (folders.length === 0 && !hasMore) {
            cardEl.innerHTML = '';
            doneEl.style.display = 'block';
            updateProgress();
            refreshStats();
            return;
        }
        await renderCard();
        refreshStats();
    } catch (e) {
        showError('Failed to connect: ' + e.message);
    }
}

async function labelCurrent(label) {
    if (labeling) return;

    const folder = folders[currentIndex];
    if (!folder) return;

    const startedAt = performance.now();
    labeling = true;
    folders.splice(currentIndex, 1);
    if (currentIndex >= folders.length && currentIndex > 0) {
        currentIndex--;
    }

    applyOptimisticLabel(label);
    renderCard();

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
        if (!res.ok || data.error) {
            throw new Error(data.error || `Move failed (${res.status})`);
        }
        logTiming('labelCurrent', {
            ms: (performance.now() - startedAt).toFixed(1),
            label,
            folderId: folder.folder_id,
            folderName: folder.folder_name,
        });
    } catch (e) {
        showError(`Move failed for ${folder.folder_name}: ${e.message}`);
    } finally {
        labeling = false;
        warmBuffer();
    }
}

async function skipCurrent() {
    if (folders.length === 0) return;

    if (currentIndex + 1 < folders.length) {
        currentIndex += 1;
    } else if (hasMore) {
        currentIndex += 1;
    } else {
        currentIndex = 0;
    }

    await renderCard();
}

function showError(msg) {
    errorEl.textContent = msg || '';
    errorEl.style.display = msg ? 'block' : 'none';
}

function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

document.addEventListener('keydown', e => {
    if (folders.length === 0) return;
    if (e.key === '1') labelCurrent('occupied');
    else if (e.key === '2') labelCurrent('dirty');
    else if (e.key === '3') labelCurrent('clean');
    else if (e.key === '4') labelCurrent('label_later');
    else if (e.key === 'ArrowRight' || e.key === ' ') {
        e.preventDefault();
        skipCurrent();
    }
});

init();
