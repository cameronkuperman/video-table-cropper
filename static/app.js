// AutoLabeler label UI

let folders = [];
let currentIndex = 0;
let labeling = false;
let suppressedFolderIds = new Set();
let suppressedFrameSignatures = new Set();
let suppressedContentSignatures = new Set();
let hasMore = true;
let queueRequest = null;
let renderToken = 0;
let initialTotalUnlabeled = 0;
let readyBufferCount = 0;
let warmingCount = 0;
let queueRetryMs = 1000;
let nextQueueFetchAt = 0;
let currentImagesReady = false;
let availableSources = [];
let reolinkSites = [];
let activeSource = 'video';
let activeSiteKey = null;
let activePendingLabel = 'unlabeled';
let activeDisplayName = 'Video';
let stats = {
    unlabeled: 0,
    clean: 0,
    dirty: 0,
    occupied: 0,
    label_later: 0,
    discarded: 0,
};

const INITIAL_QUEUE_BATCH_SIZE = 12;
const REFILL_QUEUE_BATCH_SIZE = 120;
const WARM_BUFFER_SIZE = 144;
const LOW_WATERMARK = 160;
const TIMING_LOGS_ENABLED = true;
const QUEUE_SNAPSHOT_PREFIX = 'autolabeler.queueSnapshot.';
const QUEUE_SNAPSHOT_SCHEMA_VERSION = 3;
const QUEUE_SNAPSHOT_TTL_MS = 6 * 60 * 60 * 1000;
const QUEUE_SNAPSHOT_MAX_FOLDERS = 300;
const SUPPRESSED_QUEUE_MAX = 1000;
const PENDING_LABELS_KEY = 'autolabeler.pendingLabels.v1';
const PENDING_LABEL_KEEPALIVE_LIMIT = 20;

const cardEl = document.getElementById('card');
const progressEl = document.getElementById('progress');
const statsEl = document.getElementById('stats');
const bufferStatusEl = document.getElementById('buffer-status');
const doneEl = document.getElementById('done');
const errorEl = document.getElementById('error');
const sourceModeEl = document.getElementById('source-mode');
const sourceSiteEl = document.getElementById('source-site');
const siteControlEl = document.getElementById('site-control');
const sourceSummaryEl = document.getElementById('source-summary');
const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content || '';

function buildApiError(data, res, fallbackLabel) {
    const error = new Error(data.error || `${fallbackLabel} failed (${res.status})`);
    error.setupRequired = Boolean(data.setup_required);
    error.setupUrl = data.setup_url || null;
    error.missingChannels = data.missing_channels || [];
    error.siteKey = data.site_key || null;
    return error;
}

async function readApiJson(res, fallbackLabel) {
    const text = await res.text();
    if (!text) {
        return {};
    }
    try {
        return JSON.parse(text);
    } catch (_error) {
        throw new Error(`${fallbackLabel} returned non-JSON response: ${text.slice(0, 160)}`);
    }
}

function formatErrorContent(prefix, error = null) {
    const message = `${prefix}${error?.message || ''}`;
    if (error?.setupRequired && error?.setupUrl) {
        return {
            body: `${escapeHtml(message)} <a href="${escapeHtml(error.setupUrl)}">Open crop editor</a>`,
            isHtml: true,
        };
    }
    return {
        body: message,
        isHtml: false,
    };
}

function logTiming(event, fields = {}) {
    if (!TIMING_LOGS_ENABLED) {
        return;
    }
    const details = Object.entries(fields)
        .map(([key, value]) => `${key}=${value}`)
        .join(' ');
    console.debug(`[timing] ${event}${details ? ` ${details}` : ''}`);
}

function activeQueueKey() {
    return activeSource === 'reolink'
        ? `reolink:${activeSiteKey || ''}`
        : 'video';
}

function frameSignature(folder) {
    if (folder?.frame_signature) {
        return String(folder.frame_signature);
    }
    const frames = folder?.frames || {};
    return ['frame_0', 'frame_1', 'frame_2']
        .map(key => frames[key] || '')
        .join('|');
}

function contentSignature(folder) {
    return String(folder?.content_signature || '');
}

function isSuppressedFolder(folder) {
    const folderId = String(folder?.folder_id || '');
    const signature = frameSignature(folder);
    const content = contentSignature(folder);
    return (folderId && suppressedFolderIds.has(folderId))
        || (signature && suppressedFrameSignatures.has(signature))
        || (content && suppressedContentSignatures.has(content));
}

function rememberSuppressedFolder(folder) {
    const folderId = String(folder?.folder_id || '');
    const signature = frameSignature(folder);
    const content = contentSignature(folder);
    if (folderId) {
        suppressedFolderIds.add(folderId);
    }
    if (signature) {
        suppressedFrameSignatures.add(signature);
    }
    if (content) {
        suppressedContentSignatures.add(content);
    }
    while (suppressedFolderIds.size > SUPPRESSED_QUEUE_MAX) {
        suppressedFolderIds.delete(suppressedFolderIds.values().next().value);
    }
    while (suppressedFrameSignatures.size > SUPPRESSED_QUEUE_MAX) {
        suppressedFrameSignatures.delete(suppressedFrameSignatures.values().next().value);
    }
    while (suppressedContentSignatures.size > SUPPRESSED_QUEUE_MAX) {
        suppressedContentSignatures.delete(suppressedContentSignatures.values().next().value);
    }
}

function queueSnapshotKey(queueKey = activeQueueKey()) {
    return `${QUEUE_SNAPSHOT_PREFIX}${queueKey}`;
}

function saveQueueSnapshot() {
    try {
        const remaining = folders
            .slice(currentIndex, currentIndex + QUEUE_SNAPSHOT_MAX_FOLDERS)
            .map(folder => {
                const clone = { ...folder };
                delete clone.preloadPromise;
                delete clone.preloadedImages;
                delete clone.preloaded;
                return clone;
            });
        window.localStorage.setItem(
            queueSnapshotKey(),
            JSON.stringify({
                schemaVersion: QUEUE_SNAPSHOT_SCHEMA_VERSION,
                savedAt: Date.now(),
                queueKey: activeQueueKey(),
                source: activeSource,
                siteKey: activeSiteKey,
                displayName: activeDisplayName,
                pendingLabel: activePendingLabel,
                folders: remaining,
                hasMore,
                initialTotalUnlabeled,
                readyBufferCount,
                warmingCount,
                stats,
                suppressedFolderIds: Array.from(suppressedFolderIds).slice(-SUPPRESSED_QUEUE_MAX),
                suppressedFrameSignatures: Array.from(suppressedFrameSignatures).slice(-SUPPRESSED_QUEUE_MAX),
                suppressedContentSignatures: Array.from(suppressedContentSignatures).slice(-SUPPRESSED_QUEUE_MAX),
            }),
        );
    } catch (_error) {}
}

function clearQueueSnapshot(queueKey = activeQueueKey()) {
    try {
        window.localStorage.removeItem(queueSnapshotKey(queueKey));
    } catch (_error) {}
}

function restoreQueueSnapshot() {
    try {
        const raw = window.localStorage.getItem(queueSnapshotKey());
        if (!raw) {
            return false;
        }
        const snapshot = JSON.parse(raw);
        if (!snapshot || snapshot.queueKey !== activeQueueKey()) {
            return false;
        }
        if (Number(snapshot.schemaVersion || 0) !== QUEUE_SNAPSHOT_SCHEMA_VERSION) {
            clearQueueSnapshot();
            return false;
        }
        if ((Date.now() - Number(snapshot.savedAt || 0)) > QUEUE_SNAPSHOT_TTL_MS) {
            clearQueueSnapshot();
            return false;
        }
        const restoredFolders = Array.isArray(snapshot.folders) ? snapshot.folders : [];
        if (restoredFolders.length === 0) {
            return false;
        }
        folders = restoredFolders;
        currentIndex = 0;
        hasMore = Boolean(snapshot.hasMore);
        initialTotalUnlabeled = Number(snapshot.initialTotalUnlabeled || restoredFolders.length);
        readyBufferCount = Number(snapshot.readyBufferCount || 0);
        warmingCount = Number(snapshot.warmingCount || 0);
        suppressedFolderIds = new Set((snapshot.suppressedFolderIds || []).map(String));
        suppressedFrameSignatures = new Set((snapshot.suppressedFrameSignatures || []).map(String));
        suppressedContentSignatures = new Set((snapshot.suppressedContentSignatures || []).map(String));
        activeDisplayName = snapshot.displayName || activeDisplayName;
        activePendingLabel = snapshot.pendingLabel || activePendingLabel;
        if (snapshot.stats) {
            setStats(extractStats(snapshot.stats));
        }
        updateSourceSummary();
        updateProgress();
        logTiming('restoreQueueSnapshot', {
            folders: restoredFolders.length,
            queueKey: activeQueueKey(),
        });
        return true;
    } catch (_error) {
        clearQueueSnapshot();
        return false;
    }
}

function getActiveSiteLabel() {
    return reolinkSites.find(site => site.site_key === activeSiteKey)?.label || 'Select site';
}

function updateSourceSummary() {
    if (!sourceSummaryEl) {
        return;
    }

    if (activeSource === 'reolink') {
        sourceSummaryEl.textContent = `Reolink · ${getActiveSiteLabel()}`;
        return;
    }

    sourceSummaryEl.textContent = activeDisplayName || 'Video';
}

function normalizeSourceSelection(defaultSource = {}) {
    if (!availableSources.some(source => source.source === activeSource)) {
        activeSource = defaultSource.source || availableSources[0]?.source || 'video';
    }

    if (!activeSiteKey) {
        activeSiteKey = defaultSource.site_key || reolinkSites[0]?.site_key || null;
    }

    if (activeSource === 'reolink' && !reolinkSites.some(site => site.site_key === activeSiteKey)) {
        activeSiteKey = defaultSource.site_key || reolinkSites[0]?.site_key || null;
    }

    activePendingLabel = activeSource === 'reolink' ? 'unassociated' : 'unlabeled';
    activeDisplayName = activeSource === 'reolink' ? getActiveSiteLabel() : 'Video';
}

function renderSourceControls() {
    if (!sourceModeEl || !sourceSiteEl) {
        return;
    }

    sourceModeEl.innerHTML = availableSources
        .map(source => `<option value="${source.source}">${escapeHtml(source.label)}</option>`)
        .join('');
    sourceModeEl.value = activeSource;

    sourceSiteEl.innerHTML = reolinkSites
        .map(site => `<option value="${site.site_key}">${escapeHtml(site.label)}</option>`)
        .join('');
    if (activeSiteKey) {
        sourceSiteEl.value = activeSiteKey;
    }

    const showSiteControl = activeSource === 'reolink';
    siteControlEl.hidden = !showSiteControl;
    sourceSiteEl.disabled = !showSiteControl || reolinkSites.length === 0;
    sourceModeEl.disabled = availableSources.length === 0;
    updateSourceSummary();
}

function applySourceContext(sourceContext) {
    if (!sourceContext) {
        return;
    }

    activeSource = sourceContext.source || activeSource;
    activeSiteKey = sourceContext.site_key ?? activeSiteKey;
    activePendingLabel = sourceContext.pending_label || activePendingLabel;
    activeDisplayName = sourceContext.display_name || activeDisplayName;
    renderSourceControls();
}

function extractStats(payload) {
    return {
        unlabeled: payload.unlabeled || 0,
        clean: payload.clean || 0,
        dirty: payload.dirty || 0,
        occupied: payload.occupied || 0,
        label_later: payload.label_later || 0,
        discarded: payload.discarded || 0,
    };
}

function setStats(nextStats) {
    stats = {
        ...stats,
        ...nextStats,
    };

    statsEl.textContent =
        `${activePendingLabel}: ${stats.unlabeled}  |  clean: ${stats.clean}  |  dirty: ${stats.dirty}  |  occupied: ${stats.occupied}  |  later: ${stats.label_later}  |  discarded: ${stats.discarded}`;
}

function localReadyCount() {
    return Math.max(0, folders.length - currentIndex);
}

function updateBufferStatus() {
    if (!bufferStatusEl) {
        return;
    }

    const pendingCount = readPendingLabels().length;
    const pendingSuffix = pendingCount > 0 ? ` · ${pendingCount} Drive move${pendingCount === 1 ? '' : 's'} pending` : '';
    const localReady = localReadyCount();
    if (localReady > 0) {
        const warmingSuffix = warmingCount > 0 ? ` · warming ${warmingCount}` : '';
        bufferStatusEl.textContent = `Ready buffer: ${localReady} triplets${warmingSuffix}${pendingSuffix}`;
        return;
    }

    if (stats.unlabeled <= 0 && !hasMore) {
        bufferStatusEl.textContent = '';
        return;
    }

    if (warmingCount > 0 || hasMore) {
        bufferStatusEl.textContent = `Warming next triplets from Drive...${pendingSuffix}`;
        return;
    }

    bufferStatusEl.textContent = pendingSuffix ? pendingSuffix.slice(3) : '';
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

function appendSourceParams(params, source = activeSource, siteKey = activeSiteKey) {
    params.set('source', source);
    if (source === 'reolink' && siteKey) {
        params.set('site', siteKey);
    }
}

function jsonPostHeaders() {
    const headers = { 'Content-Type': 'application/json' };
    if (csrfToken) {
        headers['X-CSRF-Token'] = csrfToken;
    }
    return headers;
}

function readPendingLabels() {
    try {
        const raw = window.localStorage.getItem(PENDING_LABELS_KEY);
        const parsed = raw ? JSON.parse(raw) : [];
        return Array.isArray(parsed) ? parsed : [];
    } catch (_error) {
        return [];
    }
}

function writePendingLabels(pending) {
    try {
        window.localStorage.setItem(PENDING_LABELS_KEY, JSON.stringify(pending));
    } catch (_error) {}
}

function pendingLabelKey(operation) {
    return [
        operation.source || '',
        operation.site_key || '',
        operation.folder_id || '',
        operation.label || '',
    ].join('|');
}

function enqueuePendingLabel(operation) {
    const pending = readPendingLabels();
    const key = pendingLabelKey(operation);
    const next = pending.filter(item => pendingLabelKey(item) !== key);
    next.push({
        ...operation,
        pending_key: key,
        queued_at: Date.now(),
        attempts: Number(operation.attempts || 0),
    });
    writePendingLabels(next);
}

function removePendingLabel(operation) {
    const key = pendingLabelKey(operation);
    writePendingLabels(readPendingLabels().filter(item => pendingLabelKey(item) !== key));
}

async function sendLabelOperation(operation, { keepalive = true } = {}) {
    const res = await fetch('/api/label', {
        method: 'POST',
        headers: jsonPostHeaders(),
        keepalive,
        body: JSON.stringify({
            folder_id: operation.folder_id,
            parent_id: operation.parent_id,
            label: operation.label,
            source: operation.source,
            site_key: operation.site_key,
            folder_name: operation.folder_name,
            frames: operation.frames,
            frame_signature: operation.frame_signature,
            content_signature: operation.content_signature,
        }),
    });
    const data = await readApiJson(res, 'Label request');
    if (res.ok && !data.error) {
        removePendingLabel(operation);
        updateBufferStatus();
        return data;
    }
    if (res.status === 409 && data.code === 'already_labeled') {
        removePendingLabel(operation);
        updateBufferStatus();
        return data;
    }
    throw new Error(data.error || `Move failed (${res.status})`);
}

function drainPendingLabels({ keepalive = false, limit = Infinity } = {}) {
    const pending = readPendingLabels().slice(0, limit);
    for (const operation of pending) {
        const nextOperation = {
            ...operation,
            attempts: Number(operation.attempts || 0) + 1,
            last_attempt_at: Date.now(),
        };
        enqueuePendingLabel(nextOperation);
        sendLabelOperation(nextOperation, { keepalive }).catch(error => {
            const stillPending = readPendingLabels();
            const key = pendingLabelKey(nextOperation);
            writePendingLabels(stillPending.map(item => (
                pendingLabelKey(item) === key
                    ? { ...item, last_error: error.message, attempts: nextOperation.attempts }
                    : item
            )));
        });
    }
}

async function fetchSources() {
    const res = await fetch('/api/sources');
    const data = await readApiJson(res, 'Sources request');
    if (!res.ok || data.error) {
        throw buildApiError(data, res, 'Sources request');
    }

    availableSources = data.sources || [];
    reolinkSites = data.reolink_sites || [];
    normalizeSourceSelection(data.default_source || {});
    renderSourceControls();
}

async function fetchQueue({
    reset = false,
    includeStats = false,
    limit = REFILL_QUEUE_BATCH_SIZE,
    forceRefresh = false,
} = {}) {
    const startedAt = performance.now();
    const requestSource = activeSource;
    const requestSiteKey = activeSiteKey;
    const requestQueueKey = requestSource === 'reolink'
        ? `reolink:${requestSiteKey || ''}`
        : 'video';

            if (reset) {
                folders = [];
                suppressedFolderIds = new Set();
                suppressedFrameSignatures = new Set();
                suppressedContentSignatures = new Set();
                currentIndex = 0;
                hasMore = true;
                queueRequest = null;
        initialTotalUnlabeled = 0;
        readyBufferCount = 0;
        warmingCount = 0;
        queueRetryMs = 1000;
        nextQueueFetchAt = 0;
    }

    if (requestSource === 'reolink' && !requestSiteKey) {
        return { folders: [] };
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
    appendSourceParams(params, requestSource, requestSiteKey);
    if (includeStats) {
        params.set('include_stats', '1');
    }
    if (forceRefresh) {
        params.set('refresh', '1');
    }

    queueRequest = fetch(`/api/queue?${params.toString()}`)
        .then(async res => {
            const data = await readApiJson(res, 'Queue request');
            if (!res.ok || data.error) {
                throw buildApiError(data, res, 'Queue request');
            }
            return data;
        })
        .then(data => {
            if (data.source_context?.queue_key && data.source_context.queue_key !== requestQueueKey) {
                return data;
            }
            if (requestQueueKey !== activeQueueKey()) {
                return data;
            }

            applySourceContext(data.source_context);
            hasMore = Boolean(data.has_more);
            readyBufferCount = data.ready_buffer_count || 0;
            warmingCount = data.warming_count || 0;
            queueRetryMs = data.retry_ms || queueRetryMs;

            if (includeStats && data.stats) {
                setStats(extractStats(data.stats));
            }
            if (reset || initialTotalUnlabeled === 0) {
                initialTotalUnlabeled = data.total_unlabeled || 0;
            }

            const existingIds = new Set(folders.map(folder => folder.folder_id));
            const existingSignatures = new Set(folders.map(frameSignature).filter(Boolean));
            const existingContentSignatures = new Set(folders.map(contentSignature).filter(Boolean));
            let added = 0;
            for (const folder of data.folders || []) {
                const signature = frameSignature(folder);
                const content = contentSignature(folder);
                if (
                    !isSuppressedFolder(folder)
                    && !existingIds.has(folder.folder_id)
                    && (!signature || !existingSignatures.has(signature))
                    && (!content || !existingContentSignatures.has(content))
                ) {
                    folders.push(folder);
                    existingIds.add(folder.folder_id);
                    if (signature) {
                        existingSignatures.add(signature);
                    }
                    if (content) {
                        existingContentSignatures.add(content);
                    }
                    added += 1;
                }
            }

            nextQueueFetchAt = added === 0 && hasMore
                ? performance.now() + queueRetryMs
                : 0;
            saveQueueSnapshot();

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
                queueKey: requestQueueKey,
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
        .map(key => folder.thumb_urls?.[key] ?? folder.preview_urls?.[key])
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

function renderFolderCard(folder) {
    currentImagesReady = false;
    const frameKeys = ['frame_0', 'frame_1', 'frame_2'];
    const imgHtml = frameKeys.map((key, idx) => {
        const url = folder.thumb_urls?.[key] ?? folder.preview_urls?.[key];
        const fullUrl = folder.preview_urls?.[key] || '';
        if (!url) {
            return `
                <div class="image-slot error" data-frame="${key}">
                    <div class="img-placeholder">no image</div>
                </div>
            `;
        }
        return `
            <div class="image-slot" data-frame="${key}">
                <div class="img-placeholder">loading</div>
                <img src="${url}" data-full-src="${fullUrl}" alt="${key}" loading="eager" decoding="async" data-frame-index="${idx}" />
            </div>
        `;
    }).join('');

    cardEl.innerHTML = `
        <div class="folder-name">${escapeHtml(folder.folder_name)}</div>
        <div class="images">${imgHtml}</div>
        <div class="image-status" id="image-status">Loading frames...</div>
        <div class="buttons">
            <button class="btn occupied label-action" onclick="labelCurrent('occupied')" disabled>Occupied [1]</button>
            <button class="btn dirty label-action"    onclick="labelCurrent('dirty')" disabled>Dirty [2]</button>
            <button class="btn clean label-action"    onclick="labelCurrent('clean')" disabled>Clean [3]</button>
            <button class="btn later label-action"    onclick="labelCurrent('label_later')" disabled>Label Later [4]</button>
            <button class="btn skip label-action"     onclick="skipCurrent()" disabled>Discard [S / &rarr;]</button>
        </div>
    `;

    const statusEl = document.getElementById('image-status');
    const buttons = Array.from(cardEl.querySelectorAll('.label-action'));
    const images = Array.from(cardEl.querySelectorAll('.image-slot img'));
    const requiredCount = frameKeys.length;
    let loadedCount = 0;
    let failed = images.length !== requiredCount;

    function updateReadyState() {
        currentImagesReady = !failed && loadedCount === requiredCount;
        buttons.forEach(button => {
            button.disabled = !currentImagesReady;
        });
        if (!statusEl) return;
        if (failed) {
            statusEl.textContent = 'Frame failed to load. Refresh before labeling this triplet.';
        } else if (currentImagesReady) {
            statusEl.textContent = '';
        } else {
            statusEl.textContent = `Loading frames ${loadedCount}/${requiredCount}`;
        }
    }

    images.forEach(img => {
        const slot = img.closest('.image-slot');
        const markLoaded = () => {
            if (slot?.classList.contains('loaded')) return;
            slot?.classList.add('loaded');
            loadedCount += 1;
            updateReadyState();
        };
        img.addEventListener('load', markLoaded, { once: true });
        img.addEventListener('error', () => {
            failed = true;
            slot?.classList.add('error');
            const placeholder = slot?.querySelector('.img-placeholder');
            if (placeholder) {
                placeholder.textContent = 'failed';
            }
            updateReadyState();
        }, { once: true });
        if (img.complete && img.naturalWidth > 0) {
            markLoaded();
        }
    });
    updateReadyState();
}

async function refreshStats() {
    const startedAt = performance.now();
    const requestSource = activeSource;
    const requestSiteKey = activeSiteKey;
    const requestQueueKey = requestSource === 'reolink'
        ? `reolink:${requestSiteKey || ''}`
        : 'video';

    if (requestSource === 'reolink' && !requestSiteKey) {
        return;
    }

    try {
        const params = new URLSearchParams();
        appendSourceParams(params, requestSource, requestSiteKey);
        const res = await fetch(`/api/stats?${params.toString()}`);
        const data = await readApiJson(res, 'Stats request');
        if (!res.ok || data.error) {
            throw buildApiError(data, res, 'Stats request');
        }
        if (data.source_context?.queue_key && data.source_context.queue_key !== requestQueueKey) {
            return;
        }
        if (requestQueueKey !== activeQueueKey()) {
            return;
        }
        applySourceContext(data.source_context);
        setStats(extractStats(data));
        updateProgress();
        logTiming('refreshStats', {
            ms: (performance.now() - startedAt).toFixed(1),
            unlabeled: data.unlabeled,
            clean: data.clean,
            dirty: data.dirty,
            occupied: data.occupied,
            labelLater: data.label_later,
            queueKey: requestQueueKey,
        });
    } catch (_) {}
}

async function renderCard() {
    const startedAt = performance.now();
    const token = ++renderToken;
    showError(null);

    const immediateFolder = folders[currentIndex] || null;
    if (immediateFolder) {
        doneEl.style.display = 'none';
        updateProgress();
        renderFolderCard(immediateFolder);
        labeling = false;
        logTiming('renderCard', {
            ms: (performance.now() - startedAt).toFixed(1),
            folderId: immediateFolder.folder_id,
            folderName: immediateFolder.folder_name,
            buffered: localReadyCount(),
            queueKey: activeQueueKey(),
            immediate: 1,
        });
        warmBuffer();
        return;
    }

    cardEl.innerHTML = '<p class="loading">Warming next images from Drive...</p>';

    let folder;
    try {
        folder = await ensureCurrentFolder();
    } catch (e) {
        const errorContent = formatErrorContent('Failed to load queue: ', e);
        showError(errorContent.body, errorContent.isHtml);
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

    if (token !== renderToken) return;

    renderFolderCard(folder);

    labeling = false;
    logTiming('renderCard', {
        ms: (performance.now() - startedAt).toFixed(1),
        folderId: folder.folder_id,
        folderName: folder.folder_name,
        buffered: localReadyCount(),
        queueKey: activeQueueKey(),
        immediate: 0,
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
    cardEl.innerHTML = '<p class="loading">Loading first triplet...</p>';

    if (activeSource === 'reolink' && !activeSiteKey) {
        cardEl.innerHTML = '';
        showError('Select a Reolink site to load its queue.');
        return;
    }

    try {
        drainPendingLabels({ keepalive: false });
        const restored = !forceRefresh && restoreQueueSnapshot();
        if (restored) {
            await renderCard();
            refreshStats();
            fetchQueue({
                reset: false,
                includeStats: false,
                limit: REFILL_QUEUE_BATCH_SIZE,
            }).then(() => {
                updateProgress();
                warmBuffer();
            }).catch(() => {});
            return;
        }

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
            clearQueueSnapshot();
            return;
        }
        await renderCard();
        refreshStats();
    } catch (e) {
        const errorContent = formatErrorContent('Failed to connect: ', e);
        showError(errorContent.body, errorContent.isHtml);
    }
}

async function labelCurrent(label) {
    if (!currentImagesReady) return;
    if (labeling) return;

    const folder = folders[currentIndex];
    if (!folder) return;

    const startedAt = performance.now();
    labeling = true;
    rememberSuppressedFolder(folder);
    const operation = {
        folder_id: folder.folder_id,
        parent_id: folder.parent_id,
        label: label,
        source: folder.source,
        site_key: folder.site_key,
        queue_key: folder.queue_key,
        folder_name: folder.folder_name,
        frames: folder.frames || {},
        frame_signature: frameSignature(folder),
        content_signature: contentSignature(folder),
    };
    enqueuePendingLabel(operation);
    folders.splice(currentIndex, 1);
    saveQueueSnapshot();

    applyOptimisticLabel(label);
    renderCard();

    try {
        await sendLabelOperation(operation, { keepalive: true });
        logTiming('labelCurrent', {
            ms: (performance.now() - startedAt).toFixed(1),
            label,
            folderId: folder.folder_id,
            folderName: folder.folder_name,
            queueKey: folder.queue_key,
        });
    } catch (e) {
        showError(`Move queued for retry: ${folder.folder_name}: ${e.message}`);
    } finally {
        labeling = false;
        warmBuffer();
    }
}

function skipCurrent() {
    if (!currentImagesReady) return;
    labelCurrent('discarded');
}

function showError(msg, isHtml = false) {
    errorEl.textContent = '';
    errorEl.innerHTML = '';
    if (isHtml) {
        errorEl.innerHTML = msg || '';
    } else {
        errorEl.textContent = msg || '';
    }
    errorEl.style.display = msg ? 'block' : 'none';
}

function escapeHtml(s) {
    return String(s ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

async function handleSourceChange() {
    normalizeSourceSelection({ source: activeSource, site_key: activeSiteKey });
    renderSourceControls();
    clearQueueSnapshot();
    await init(true);
}

sourceModeEl?.addEventListener('change', () => {
    activeSource = sourceModeEl.value;
    if (activeSource === 'reolink' && !activeSiteKey) {
        activeSiteKey = reolinkSites[0]?.site_key || null;
    }
    handleSourceChange();
});

sourceSiteEl?.addEventListener('change', () => {
    activeSiteKey = sourceSiteEl.value || null;
    handleSourceChange();
});

window.addEventListener('pagehide', () => {
    drainPendingLabels({
        keepalive: true,
        limit: PENDING_LABEL_KEEPALIVE_LIMIT,
    });
});

document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') {
        drainPendingLabels({
            keepalive: true,
            limit: PENDING_LABEL_KEEPALIVE_LIMIT,
        });
    } else {
        drainPendingLabels({ keepalive: false });
    }
});

document.addEventListener('keydown', e => {
    if (folders.length === 0) return;
    if (e.key === '1') labelCurrent('occupied');
    else if (e.key === '2') labelCurrent('dirty');
    else if (e.key === '3') labelCurrent('clean');
    else if (e.key === '4') labelCurrent('label_later');
    else if (e.key === 's' || e.key === 'S' || e.key === 'ArrowRight' || e.key === ' ') {
        e.preventDefault();
        skipCurrent();
    }
});

async function bootstrap() {
    try {
        await fetchSources();
        await init(true);
    } catch (e) {
        const errorContent = formatErrorContent('Failed to load sources: ', e);
        showError(errorContent.body, errorContent.isHtml);
    }
}

bootstrap();
