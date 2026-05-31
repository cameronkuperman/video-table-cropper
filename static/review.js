const reviewMode = document.querySelector('meta[name="review-mode"]')?.content || 'labeled';
const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content || '';

const state = {
    sources: [],
    sites: [],
    folders: [],
    selected: new Set(),
    nextCursor: null,
    loading: false,
};

const el = {
    subtitle: document.getElementById('page-subtitle'),
    source: document.getElementById('source'),
    site: document.getElementById('site'),
    bucket: document.getElementById('bucket'),
    folderSourceType: document.getElementById('folder-source-type'),
    cropSourceKind: document.getElementById('crop-source-kind'),
    channel: document.getElementById('channel'),
    table: document.getElementById('table'),
    query: document.getElementById('query'),
    frameCount: document.getElementById('frame-count'),
    targetLabel: document.getElementById('target-label'),
    load: document.getElementById('load'),
    loadMore: document.getElementById('load-more'),
    grid: document.getElementById('grid'),
    status: document.getElementById('status'),
    selectionStatus: document.getElementById('selection-status'),
    selectVisible: document.getElementById('select-visible'),
    clearSelection: document.getElementById('clear-selection'),
    relabelSelected: document.getElementById('relabel-selected'),
    trashSelected: document.getElementById('trash-selected'),
};

function escapeHtml(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

function selectedBuckets() {
    return Array.from(el.bucket.selectedOptions).map(option => option.value);
}

function activeSourcePayload() {
    return {
        source: el.source.value || 'video',
        site_key: el.source.value === 'reolink' ? (el.site.value || null) : null,
    };
}

function reviewParams(cursor = 0) {
    const params = new URLSearchParams();
    const sourcePayload = activeSourcePayload();
    params.set('mode', reviewMode);
    params.set('source', sourcePayload.source);
    if (sourcePayload.site_key) {
        params.set('site', sourcePayload.site_key);
    }
    for (const bucket of selectedBuckets()) {
        params.append('bucket', bucket);
    }
    params.set('limit', '30');
    params.set('cursor', String(cursor));
    for (const [key, input] of [
        ['folder_source_type', el.folderSourceType],
        ['channel', el.channel],
        ['table', el.table],
        ['q', el.query],
        ['frame_count', el.frameCount],
        ['crop_source_kind', el.cropSourceKind],
    ]) {
        const value = input.value.trim();
        if (value) {
            params.set(key, value);
        }
    }
    return params;
}

async function readJson(response) {
    const text = await response.text();
    const data = text ? JSON.parse(text) : {};
    if (!response.ok) {
        throw new Error(data.error || `Request failed (${response.status})`);
    }
    return data;
}

function updateSelectionStatus() {
    el.selectionStatus.textContent = `${state.selected.size} selected`;
    document.querySelectorAll('.card').forEach(card => {
        card.classList.toggle('selected', state.selected.has(card.dataset.folderId));
        const checkbox = card.querySelector('input[type="checkbox"]');
        if (checkbox) {
            checkbox.checked = state.selected.has(card.dataset.folderId);
        }
    });
}

function frameEntries(folder) {
    const frames = folder.frames || {};
    return Object.keys(frames)
        .filter(key => /^frame_\d+$/.test(key))
        .sort((a, b) => Number(a.slice(6)) - Number(b.slice(6)))
        .map(key => [key, frames[key]])
        .filter(([_key, fileId]) => fileId)
        .slice(0, 3);
}

function renderFolder(folder) {
    const frames = frameEntries(folder);
    const frameHtml = frames.map(([key, fileId]) => `
        <a class="frame" href="/api/preview/${encodeURIComponent(fileId)}" target="_blank" title="${escapeHtml(key)}">
          <img loading="lazy" src="/api/thumb/${encodeURIComponent(fileId)}" alt="${escapeHtml(key)}" />
        </a>
    `).join('');
    const chips = [
        folder.bucket,
        folder.review_source_type,
        folder.crop_provenance?.label || folder.crop_source_kind,
        folder.channel_hint,
        folder.table_hint,
        `${folder.frame_count || 0} frames`,
    ].filter(Boolean).map(value => `<span class="chip">${escapeHtml(value)}</span>`).join('');
    const sidecars = [
        folder.metadata_file_id ? ['metadata', folder.metadata_file_id] : null,
        folder.perception_file_id ? ['perception', folder.perception_file_id] : null,
    ].filter(Boolean).map(([label, fileId]) => `
        <a class="chip" href="https://drive.google.com/file/d/${encodeURIComponent(fileId)}/view" target="_blank">${label}</a>
    `).join('');
    return `
      <article class="card" data-folder-id="${escapeHtml(folder.folder_id)}">
        <div class="card-head">
          <input type="checkbox" aria-label="Select folder" />
          <div>
            <div class="folder-name" title="${escapeHtml(folder.folder_name)}">${escapeHtml(folder.folder_name)}</div>
            <div class="chips">${chips}</div>
            ${sidecars ? `<div class="chips">${sidecars}</div>` : ''}
            <div class="muted" style="margin-top:6px;">${escapeHtml(folder.folder_id)}</div>
          </div>
        </div>
        <div class="frames">${frameHtml}</div>
        <div class="card-actions">
          <button class="secondary" data-action="compare">Compare</button>
          <button class="secondary" data-action="relabel-one">Relabel</button>
          <button class="danger" data-action="trash-one">Trash</button>
        </div>
        <div class="compare"></div>
      </article>
    `;
}

function renderFolders(append = false) {
    const html = state.folders.map(renderFolder).join('');
    if (append) {
        el.grid.insertAdjacentHTML('beforeend', html);
    } else {
        el.grid.innerHTML = html;
    }
    updateSelectionStatus();
}

async function loadFolders({ append = false } = {}) {
    if (state.loading) {
        return;
    }
    state.loading = true;
    const cursor = append && state.nextCursor != null ? state.nextCursor : 0;
    el.status.textContent = 'Loading Drive folders...';
    el.load.disabled = true;
    el.loadMore.disabled = true;
    try {
        const data = await fetch(`/api/review/folders?${reviewParams(cursor)}`).then(readJson);
        state.nextCursor = data.next_cursor;
        const incoming = data.folders || [];
        if (append) {
            state.folders = incoming;
        } else {
            state.folders = incoming;
            state.selected.clear();
        }
        renderFolders(append);
        const totalLabel = data.total_is_candidate_count ? 'candidate folders' : 'matching folders';
        el.status.textContent = `${data.total || 0} ${totalLabel}`;
        el.loadMore.disabled = state.nextCursor == null;
    } catch (error) {
        el.status.textContent = error.message;
    } finally {
        state.loading = false;
        el.load.disabled = false;
    }
}

async function mutateSelected(action, folderIds, extra = {}) {
    if (!folderIds.length) {
        el.status.textContent = 'Select at least one folder';
        return;
    }
    const sourcePayload = activeSourcePayload();
    const body = {
        ...sourcePayload,
        folder_ids: folderIds,
        ...extra,
    };
    const data = await fetch(`/api/review/${action}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRF-Token': csrfToken,
        },
        body: JSON.stringify(body),
    }).then(readJson);
    for (const folderId of folderIds) {
        state.selected.delete(folderId);
    }
    await loadFolders();
    return data;
}

async function relabel(folderIds) {
    const targetLabel = el.targetLabel.value;
    el.status.textContent = `Moving ${folderIds.length} folder(s) to ${targetLabel}...`;
    try {
        await mutateSelected('relabel', folderIds, { target_label: targetLabel });
        el.status.textContent = `Moved ${folderIds.length} folder(s) to ${targetLabel}`;
    } catch (error) {
        el.status.textContent = error.message;
    }
}

async function trash(folderIds) {
    const typed = window.prompt(`Type TRASH to soft-trash ${folderIds.length} folder(s).`);
    if (typed !== 'TRASH') {
        return;
    }
    el.status.textContent = `Trashing ${folderIds.length} folder(s)...`;
    try {
        await mutateSelected('trash', folderIds, { confirm: 'TRASH' });
        el.status.textContent = `Trashed ${folderIds.length} folder(s)`;
    } catch (error) {
        el.status.textContent = error.message;
    }
}

async function compareFolder(card) {
    const folderId = card.dataset.folderId;
    const panel = card.querySelector('.compare');
    if (panel.classList.contains('open')) {
        panel.classList.remove('open');
        return;
    }
    panel.classList.add('open');
    panel.innerHTML = '<div class="muted">Loading likely matches...</div>';
    try {
        const sourcePayload = activeSourcePayload();
        const params = new URLSearchParams({ source: sourcePayload.source, folder_id: folderId });
        if (sourcePayload.site_key) {
            params.set('site', sourcePayload.site_key);
        }
        const data = await fetch(`/api/review/compare?${params}`).then(readJson);
        const matches = data.matches || [];
        if (!matches.length) {
            panel.innerHTML = '<div class="muted">No likely matches found.</div>';
            return;
        }
        panel.innerHTML = matches.slice(0, 6).map(match => {
            const firstFrame = frameEntries(match)[0];
            const image = firstFrame ? `<img src="/api/thumb/${encodeURIComponent(firstFrame[1])}" alt="" />` : '<div></div>';
            return `
              <div class="compare-row">
                ${image}
                <div>
                  <div class="folder-name" title="${escapeHtml(match.folder_name)}">${escapeHtml(match.folder_name)}</div>
                  <div class="chips">
                    <span class="chip">${escapeHtml(match.bucket)}</span>
                    <span class="chip">${escapeHtml(match.review_source_type)}</span>
                    <span class="chip">${escapeHtml(match.crop_provenance?.label || match.crop_source_kind || 'unknown crop source')}</span>
                  </div>
                </div>
              </div>
            `;
        }).join('');
    } catch (error) {
        panel.innerHTML = `<div class="muted">${escapeHtml(error.message)}</div>`;
    }
}

function setupBuckets() {
    const buckets = reviewMode === 'legacy'
        ? ['unlabeled', 'screenrecord_3frame_unlabeled', 'clean', 'dirty', 'occupied', 'label_later', 'discarded']
        : ['clean', 'dirty', 'occupied', 'label_later', 'discarded'];
    const defaults = reviewMode === 'legacy'
        ? new Set(['unlabeled', 'screenrecord_3frame_unlabeled', 'clean', 'dirty', 'occupied'])
        : new Set(['clean', 'dirty', 'occupied']);
    el.bucket.innerHTML = buckets.map(bucket => `
      <option value="${bucket}" ${defaults.has(bucket) ? 'selected' : ''}>${bucket}</option>
    `).join('');
}

async function setupSources() {
    const data = await fetch('/api/sources').then(readJson);
    state.sources = data.sources || [];
    state.sites = data.reolink_sites || [];
    el.source.innerHTML = state.sources.map(source => `
      <option value="${source.source}">${escapeHtml(source.label)}</option>
    `).join('');
    const defaultSource = data.default_source || {};
    el.source.value = defaultSource.source || 'video';
    el.site.innerHTML = state.sites.map(site => `
      <option value="${site.site_key}">${escapeHtml(site.label)}</option>
    `).join('');
    el.site.value = defaultSource.site_key || state.sites[0]?.site_key || '';
    el.site.disabled = el.source.value !== 'reolink';
}

function bindEvents() {
    el.source.addEventListener('change', () => {
        el.site.disabled = el.source.value !== 'reolink';
    });
    el.load.addEventListener('click', () => loadFolders());
    el.loadMore.addEventListener('click', () => loadFolders({ append: true }));
    el.selectVisible.addEventListener('click', () => {
        document.querySelectorAll('.card').forEach(card => state.selected.add(card.dataset.folderId));
        updateSelectionStatus();
    });
    el.clearSelection.addEventListener('click', () => {
        state.selected.clear();
        updateSelectionStatus();
    });
    el.relabelSelected.addEventListener('click', () => relabel(Array.from(state.selected)));
    el.trashSelected.addEventListener('click', () => trash(Array.from(state.selected)));
    el.grid.addEventListener('change', event => {
        const checkbox = event.target.closest('input[type="checkbox"]');
        if (!checkbox) {
            return;
        }
        const card = checkbox.closest('.card');
        if (checkbox.checked) {
            state.selected.add(card.dataset.folderId);
        } else {
            state.selected.delete(card.dataset.folderId);
        }
        updateSelectionStatus();
    });
    el.grid.addEventListener('click', event => {
        const button = event.target.closest('button[data-action]');
        if (!button) {
            return;
        }
        const card = button.closest('.card');
        const folderId = card.dataset.folderId;
        if (button.dataset.action === 'compare') {
            compareFolder(card);
        } else if (button.dataset.action === 'relabel-one') {
            relabel([folderId]);
        } else if (button.dataset.action === 'trash-one') {
            trash([folderId]);
        }
    });
}

async function init() {
    el.subtitle.textContent = reviewMode === 'legacy'
        ? 'On-demand cleanup for old fallback crops, bad angles, and duplicate training artifacts.'
        : 'On-demand browser for already-labeled folders with relabel and soft-trash controls.';
    setupBuckets();
    bindEvents();
    try {
        await setupSources();
        await loadFolders();
    } catch (error) {
        el.status.textContent = error.message;
    }
}

init();
