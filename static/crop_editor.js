const defaults = window.CROP_EDITOR_DEFAULTS || {};
const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content || '';

const state = {
    siteKey: defaults.siteKey || 'reolink-matthews-01',
    siteLabel: 'Reolink Matthews 01',
    channels: [],
    activeChannelCode: defaults.channelCode || null,
    reference: null,
    referenceSize: { width: 0, height: 0 },
    crops: [],
    selectedCropId: null,
    dragging: null,
};

const COLORS = ['#4f9cff', '#00c389', '#ff8f3f', '#ff5c7a', '#b081ff', '#f0c646', '#2fd3e6'];

const channelListEl = document.getElementById('channel-list');
const siteLabelEl = document.getElementById('site-label');
const editorTitleEl = document.getElementById('editor-title');
const editorSubtitleEl = document.getElementById('editor-subtitle');
const addCropBtn = document.getElementById('add-crop-btn');
const resetPointsBtn = document.getElementById('reset-points-btn');
const saveBtn = document.getElementById('save-btn');
const imageStageEl = document.getElementById('image-stage');
const referenceImageEl = document.getElementById('reference-image');
const overlayEl = document.getElementById('overlay');
const cropListEl = document.getElementById('crop-list');
const editorEmptyEl = document.getElementById('editor-empty');
const editorMessageEl = document.getElementById('editor-message');

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function showMessage(text, type = 'success') {
    if (!text) {
        editorMessageEl.hidden = true;
        editorMessageEl.textContent = '';
        editorMessageEl.className = 'message';
        return;
    }
    editorMessageEl.hidden = false;
    editorMessageEl.textContent = text;
    editorMessageEl.className = `message ${type}`;
}

function jsonPostHeaders() {
    const headers = { 'Content-Type': 'application/json' };
    if (csrfToken) {
        headers['X-CSRF-Token'] = csrfToken;
    }
    return headers;
}

function colorForIndex(index) {
    return COLORS[index % COLORS.length];
}

function nextCropName() {
    const existing = new Set(
        state.crops.map(crop => String(crop.name || '').trim().toLowerCase())
    );
    let index = 1;
    while (existing.has(`table_${index}`)) {
        index += 1;
    }
    return `table_${index}`;
}

function orderQuadrilateralPoints(points) {
    if (!Array.isArray(points) || points.length !== 4) {
        return points;
    }

    const center = points.reduce(
        (accumulator, point) => ({
            x: accumulator.x + Number(point.x),
            y: accumulator.y + Number(point.y),
        }),
        { x: 0, y: 0 }
    );
    center.x /= 4;
    center.y /= 4;

    const ordered = points
        .map(point => ({
            x: Number(point.x),
            y: Number(point.y),
        }))
        .sort((left, right) => {
            const leftAngle = Math.atan2(left.y - center.y, left.x - center.x);
            const rightAngle = Math.atan2(right.y - center.y, right.x - center.x);
            return leftAngle - rightAngle;
        });

    const topLeftIndex = ordered.reduce((bestIndex, point, index, items) => {
        const bestPoint = items[bestIndex];
        if ((point.x + point.y) < (bestPoint.x + bestPoint.y)) {
            return index;
        }
        return bestIndex;
    }, 0);

    return [
        ordered[topLeftIndex],
        ordered[(topLeftIndex + 1) % 4],
        ordered[(topLeftIndex + 2) % 4],
        ordered[(topLeftIndex + 3) % 4],
    ];
}

function normalizeCropPointOrder(crop) {
    if (!crop || crop.points.length !== 4) {
        return;
    }
    crop.points = orderQuadrilateralPoints(crop.points);
}

function selectedCrop() {
    return state.crops.find(crop => crop.id === state.selectedCropId) || null;
}

function setActiveChannel(channelCode) {
    if (!channelCode) {
        return;
    }
    state.activeChannelCode = channelCode;
    loadChannel(channelCode);
}

function renderChannels() {
    if (state.channels.length === 0) {
        channelListEl.innerHTML = '<div class="muted">No Matthews channels found in <code>unassociated/</code> yet.</div>';
        return;
    }

    channelListEl.innerHTML = state.channels.map(channel => {
        const status = channel.has_config
            ? `${channel.crop_count} crop${channel.crop_count === 1 ? '' : 's'} saved`
            : 'setup required';
        return `
            <button
                type="button"
                class="channel-btn ${channel.channel_code === state.activeChannelCode ? 'active' : ''}"
                data-channel="${escapeHtml(channel.channel_code)}"
            >
                <div>${escapeHtml(channel.channel_code)}</div>
                <div class="channel-meta">
                    <span>${escapeHtml(status)}</span>
                    <span>${channel.reference_available ? 'reference ready' : 'no frame'}</span>
                </div>
            </button>
        `;
    }).join('');

    channelListEl.querySelectorAll('[data-channel]').forEach(button => {
        button.addEventListener('click', () => setActiveChannel(button.dataset.channel));
    });
}

function mergeFallbackChannels(channelCodes) {
    const existingByCode = new Map(
        state.channels.map(channel => [channel.channel_code, channel])
    );

    for (const channelCode of channelCodes) {
        if (!channelCode || existingByCode.has(channelCode)) {
            continue;
        }
        existingByCode.set(channelCode, {
            channel_code: channelCode,
            has_config: false,
            crop_count: 0,
            reference_available: true,
            setup_url: `/crop-editor?site=${encodeURIComponent(state.siteKey)}&channel=${encodeURIComponent(channelCode)}`,
        });
    }

    state.channels = Array.from(existingByCode.values()).sort((left, right) =>
        String(left.channel_code).localeCompare(String(right.channel_code), undefined, { numeric: true })
    );
}

function renderCropList() {
    if (!state.activeChannelCode) {
        cropListEl.innerHTML = '<div class="muted">Select a channel to start drawing crops.</div>';
        return;
    }

    if (state.crops.length === 0) {
        cropListEl.innerHTML = '<div class="muted">No crops yet. Add one, then click four points on the image.</div>';
        return;
    }

    cropListEl.innerHTML = state.crops.map((crop, index) => `
        <div class="crop-row ${crop.id === state.selectedCropId ? 'selected' : ''}" data-crop-id="${escapeHtml(crop.id)}">
            <div class="crop-row-top">
                <input type="text" value="${escapeHtml(crop.name)}" data-crop-name="${escapeHtml(crop.id)}" />
                <button type="button" class="danger" data-remove-crop="${escapeHtml(crop.id)}">Remove</button>
            </div>
            <small style="color:${colorForIndex(index)}">${escapeHtml(crop.points.length)} / 4 points</small>
        </div>
    `).join('');

    cropListEl.querySelectorAll('[data-crop-id]').forEach(row => {
        row.addEventListener('click', event => {
            if (event.target.matches('[data-crop-name], [data-remove-crop]')) {
                return;
            }
            state.selectedCropId = row.dataset.cropId;
            render();
        });
    });

    cropListEl.querySelectorAll('[data-crop-name]').forEach(input => {
        input.addEventListener('input', () => {
            const crop = state.crops.find(entry => entry.id === input.dataset.cropName);
            if (!crop) {
                return;
            }
            crop.name = input.value;
        });
        input.addEventListener('focus', () => {
            state.selectedCropId = input.dataset.cropName;
            render();
        });
    });

    cropListEl.querySelectorAll('[data-remove-crop]').forEach(button => {
        button.addEventListener('click', event => {
            event.stopPropagation();
            state.crops = state.crops.filter(crop => crop.id !== button.dataset.removeCrop);
            if (state.selectedCropId === button.dataset.removeCrop) {
                state.selectedCropId = state.crops[0]?.id || null;
            }
            render();
        });
    });
}

function toSvgPoints(points) {
    return points.map(point => `${point.x},${point.y}`).join(' ');
}

function renderOverlay() {
    overlayEl.innerHTML = '';

    const width = state.referenceSize.width || referenceImageEl.naturalWidth || 1;
    const height = state.referenceSize.height || referenceImageEl.naturalHeight || 1;
    overlayEl.setAttribute('viewBox', `0 0 ${width} ${height}`);

    const hit = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    hit.setAttribute('class', 'overlay-hit');
    hit.setAttribute('x', '0');
    hit.setAttribute('y', '0');
    hit.setAttribute('width', String(width));
    hit.setAttribute('height', String(height));
    hit.addEventListener('click', onOverlayClick);
    overlayEl.appendChild(hit);

    state.crops.forEach((crop, index) => {
        if (!crop.points.length) {
            return;
        }

        const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        polygon.setAttribute('points', toSvgPoints(crop.points));
        polygon.setAttribute('class', `crop-polygon ${crop.id === state.selectedCropId ? 'selected' : ''}`);
        polygon.setAttribute('fill', colorForIndex(index));
        polygon.setAttribute('stroke', colorForIndex(index));
        polygon.addEventListener('click', event => {
            event.stopPropagation();
            state.selectedCropId = crop.id;
            render();
        });
        overlayEl.appendChild(polygon);

        crop.points.forEach((point, pointIndex) => {
            const handle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            handle.setAttribute('class', `crop-handle ${state.dragging?.cropId === crop.id && state.dragging?.pointIndex === pointIndex ? 'dragging' : ''}`);
            handle.setAttribute('cx', String(point.x));
            handle.setAttribute('cy', String(point.y));
            handle.setAttribute('r', '8');
            handle.setAttribute('fill', colorForIndex(index));
            handle.dataset.cropId = crop.id;
            handle.dataset.pointIndex = String(pointIndex);
            handle.addEventListener('pointerdown', startDrag);
            overlayEl.appendChild(handle);
        });
    });
}

function render() {
    siteLabelEl.textContent = state.siteLabel;
    renderChannels();
    renderCropList();

    const hasChannel = Boolean(state.activeChannelCode);
    addCropBtn.disabled = !hasChannel || !state.reference;
    resetPointsBtn.disabled = !selectedCrop();
    saveBtn.disabled = !hasChannel || !state.reference || !state.crops.length;

    editorTitleEl.textContent = hasChannel ? `${state.activeChannelCode}` : 'Select a channel';
    editorSubtitleEl.textContent = hasChannel
        ? 'Load or define permanent 4-point table polygons for this channel.'
        : 'The editor starts blank for channels without saved configs.';

    if (!state.reference) {
        editorEmptyEl.hidden = false;
        imageStageEl.hidden = true;
        return;
    }

    editorEmptyEl.hidden = true;
    imageStageEl.hidden = false;
    renderOverlay();
}

function imagePointFromEvent(event) {
    const rect = referenceImageEl.getBoundingClientRect();
    const width = state.referenceSize.width || referenceImageEl.naturalWidth;
    const height = state.referenceSize.height || referenceImageEl.naturalHeight;
    if (!rect.width || !rect.height || !width || !height) {
        return null;
    }

    const x = ((event.clientX - rect.left) / rect.width) * width;
    const y = ((event.clientY - rect.top) / rect.height) * height;
    return {
        x: Math.max(0, Math.min(width, Number(x.toFixed(2)))),
        y: Math.max(0, Math.min(height, Number(y.toFixed(2)))),
    };
}

function onOverlayClick(event) {
    const crop = selectedCrop();
    if (!crop) {
        showMessage('Add a crop first, then click four points on the image.', 'error');
        return;
    }
    if (crop.points.length >= 4) {
        showMessage('This crop already has four points. Drag a handle or reset the points to redraw it.', 'error');
        return;
    }

    const point = imagePointFromEvent(event);
    if (!point) {
        return;
    }
    crop.points.push(point);
    normalizeCropPointOrder(crop);
    showMessage(null);
    render();
}

function startDrag(event) {
    event.preventDefault();
    event.stopPropagation();
    state.selectedCropId = event.target.dataset.cropId;
    state.dragging = {
        cropId: event.target.dataset.cropId,
        pointIndex: Number(event.target.dataset.pointIndex),
    };
    overlayEl.setPointerCapture(event.pointerId);
    render();
}

function onPointerMove(event) {
    if (!state.dragging) {
        return;
    }
    const crop = state.crops.find(entry => entry.id === state.dragging.cropId);
    if (!crop) {
        return;
    }
    const point = imagePointFromEvent(event);
    if (!point) {
        return;
    }
    crop.points[state.dragging.pointIndex] = point;
    renderOverlay();
}

function stopDrag(event) {
    if (!state.dragging) {
        return;
    }
    const crop = state.crops.find(entry => entry.id === state.dragging.cropId);
    if (event?.pointerId != null) {
        try {
            overlayEl.releasePointerCapture(event.pointerId);
        } catch (_) {}
    }
    normalizeCropPointOrder(crop);
    state.dragging = null;
    render();
}

function addCrop() {
    const crop = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        name: nextCropName(),
        points: [],
    };
    state.crops.push(crop);
    state.selectedCropId = crop.id;
    showMessage('Click four corners on the image to place this crop.', 'success');
    render();
}

function resetSelectedCropPoints() {
    const crop = selectedCrop();
    if (!crop) {
        return;
    }
    crop.points = [];
    showMessage('Points cleared. Click four new corners for the selected crop.', 'success');
    render();
}

function normalizedCropsForSave() {
    return state.crops.map(crop => ({
        name: String(crop.name || '').trim() || nextCropName(),
        polygon: orderQuadrilateralPoints(crop.points).map(point => [point.x, point.y]),
    }));
}

function validateBeforeSave() {
    if (!state.activeChannelCode) {
        return 'Select a channel first.';
    }
    if (!state.reference) {
        return 'A reference frame is required before saving.';
    }
    if (!state.crops.length) {
        return 'Add at least one crop before saving.';
    }

    const seenNames = new Set();
    for (const crop of state.crops) {
        const name = String(crop.name || '').trim();
        if (!name) {
            return 'Every crop needs a name.';
        }
        const normalized = name.toLowerCase();
        if (seenNames.has(normalized)) {
            return `Duplicate crop name: ${name}`;
        }
        seenNames.add(normalized);
        if (crop.points.length !== 4) {
            return `${name} must have exactly four points before saving.`;
        }
    }
    return null;
}

async function saveConfig() {
    const validationError = validateBeforeSave();
    if (validationError) {
        showMessage(validationError, 'error');
        return;
    }

    saveBtn.disabled = true;
    showMessage('Saving crop config to Drive...', 'success');
    try {
        const response = await fetch('/api/reolink/crop-config', {
            method: 'POST',
            headers: jsonPostHeaders(),
            body: JSON.stringify({
                site_key: state.siteKey,
                channel_code: state.activeChannelCode,
                reference: {
                    raw_folder_id: state.reference.raw_folder_id,
                    raw_folder_name: state.reference.raw_folder_name,
                    frame_file_id: state.reference.frame_file_id,
                    width: state.referenceSize.width || referenceImageEl.naturalWidth,
                    height: state.referenceSize.height || referenceImageEl.naturalHeight,
                },
                crops: normalizedCropsForSave(),
            }),
        });
        const data = await response.json();
        if (!response.ok || data.error) {
            throw new Error(data.error || `Save failed (${response.status})`);
        }
        showMessage(`Saved ${state.activeChannelCode} with ${state.crops.length} crop${state.crops.length === 1 ? '' : 's'}.`, 'success');
        await refreshStatus();
        await loadChannel(state.activeChannelCode, { preserveMessage: true });
    } catch (error) {
        showMessage(error.message, 'error');
    } finally {
        saveBtn.disabled = false;
    }
}

async function refreshStatus() {
    const response = await fetch(`/api/reolink/crop-configs/status?site=${encodeURIComponent(state.siteKey)}`);
    const data = await response.json();
    if (!response.ok || data.error) {
        throw new Error(data.error || `Status request failed (${response.status})`);
    }
    state.siteLabel = data.site_label || state.siteLabel;
    state.channels = data.channels || [];
    if (state.channels.length === 0 && defaults.channelCode) {
        mergeFallbackChannels([defaults.channelCode]);
    }
    if (state.channels.length === 0) {
        await refreshChannelsFromQueueFallback();
    }
    if (!state.activeChannelCode) {
        state.activeChannelCode = defaults.channelCode || state.channels[0]?.channel_code || null;
    }
    renderChannels();
}

async function refreshChannelsFromQueueFallback() {
    const response = await fetch(
        `/api/queue?limit=1&source=reolink&site=${encodeURIComponent(state.siteKey)}&refresh=1`
    );
    const data = await response.json();

    if (response.status === 409 && data.setup_required && Array.isArray(data.missing_channels)) {
        mergeFallbackChannels(data.missing_channels);
        if (!state.activeChannelCode) {
            state.activeChannelCode = data.missing_channels[0] || null;
        }
        return;
    }

    if (!response.ok || data.error) {
        throw new Error(data.error || `Queue fallback failed (${response.status})`);
    }
}

function mapConfigCrops(config) {
    return (config?.crops || []).map((crop, index) => ({
        id: `${Date.now()}-${index}-${Math.random().toString(36).slice(2, 8)}`,
        name: crop.name || `table_${index + 1}`,
        points: orderQuadrilateralPoints(
            (crop.polygon || []).map(point => ({
                x: Number(point[0]),
                y: Number(point[1]),
            }))
        ),
    }));
}

async function loadChannel(channelCode, { preserveMessage = false } = {}) {
    if (!channelCode) {
        return;
    }
    if (!preserveMessage) {
        showMessage(null);
    }

    editorEmptyEl.hidden = false;
    editorEmptyEl.textContent = 'Loading reference frame...';
    imageStageEl.hidden = true;

    try {
        const response = await fetch(
            `/api/reolink/crop-config?site=${encodeURIComponent(state.siteKey)}&channel=${encodeURIComponent(channelCode)}`
        );
        const data = await response.json();
        if (!response.ok || data.error) {
            throw new Error(data.error || `Channel request failed (${response.status})`);
        }

        state.activeChannelCode = channelCode;
        state.reference = data.reference || null;
        state.referenceSize = {
            width: Number(data.reference?.width || data.config?.reference?.width || 0),
            height: Number(data.reference?.height || data.config?.reference?.height || 0),
        };
        state.crops = mapConfigCrops(data.config);
        state.selectedCropId = state.crops[0]?.id || null;

        referenceImageEl.onload = () => {
            state.referenceSize = {
                width: referenceImageEl.naturalWidth,
                height: referenceImageEl.naturalHeight,
            };
            render();
        };
        referenceImageEl.src = data.reference.preview_url;
        render();
        if (!data.has_config) {
            showMessage(`No saved config for ${channelCode} yet. Add crops manually and save when ready.`, 'success');
        }
    } catch (error) {
        state.reference = null;
        state.crops = [];
        state.selectedCropId = null;
        showMessage(error.message, 'error');
        render();
    }
}

addCropBtn.addEventListener('click', addCrop);
resetPointsBtn.addEventListener('click', resetSelectedCropPoints);
saveBtn.addEventListener('click', saveConfig);
overlayEl.addEventListener('pointermove', onPointerMove);
overlayEl.addEventListener('pointerup', stopDrag);
overlayEl.addEventListener('pointercancel', stopDrag);

async function bootstrap() {
    try {
        await refreshStatus();
        render();
        if (state.activeChannelCode) {
            await loadChannel(state.activeChannelCode);
        }
    } catch (error) {
        showMessage(error.message, 'error');
        render();
    }
}

bootstrap();
