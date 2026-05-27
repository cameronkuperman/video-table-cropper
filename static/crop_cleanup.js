const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content || '';

const state = {
    sources: [],
    sites: [],
    hiddenGroups: new Set(),
};

const el = {
	source: document.getElementById('source'),
	site: document.getElementById('site'),
	channel: document.getElementById('channel'),
	table: document.getElementById('table'),
	query: document.getElementById('query'),
	load: document.getElementById('load'),
	status: document.getElementById('status'),
	supabaseGrid: document.getElementById('supabase-grid'),
	fallbackGrid: document.getElementById('fallback-grid'),
};

let reloadTimer = null;

function escapeHtml(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

function activeSourcePayload() {
	return {
		source: el.source.value || 'reolink',
        site_key: el.source.value === 'reolink' ? (el.site.value || null) : null,
    };
}

function requestParams() {
    const params = new URLSearchParams();
    const sourcePayload = activeSourcePayload();
    params.set('source', sourcePayload.source);
	if (sourcePayload.site_key) {
		params.set('site', sourcePayload.site_key);
	}
	for (const [key, input] of [
		['channel', el.channel],
		['table', el.table],
        ['q', el.query],
    ]) {
        const value = input.value.trim();
        if (value) {
            params.set(key, value);
        }
    }
    return params;
}

function requestUrl(includeFallback) {
    const params = requestParams();
    params.set('include_fallback', includeFallback ? '1' : '0');
    return `/api/cleanup/crops/inventory?${params}`;
}

async function readJson(response) {
    const text = await response.text();
    const data = text ? JSON.parse(text) : {};
    if (!response.ok) {
        throw new Error(data.error || `Request failed (${response.status})`);
    }
    return data;
}

async function setupSources() {
    const data = await fetch('/api/sources').then(readJson);
    state.sources = data.sources || [];
    state.sites = data.reolink_sites || [];
    el.source.innerHTML = state.sources.map(source => `
      <option value="${source.source}">${escapeHtml(source.label)}</option>
    `).join('');
    const defaultSource = data.default_source || {};
    el.source.value = defaultSource.source || 'reolink';
    el.site.innerHTML = state.sites.map(site => `
      <option value="${site.site_key}">${escapeHtml(site.label)}</option>
    `).join('');
    el.site.value = defaultSource.site_key || state.sites[0]?.site_key || '';
    el.site.disabled = el.source.value !== 'reolink';
}

function frameEntries(folder) {
    const frames = folder?.frames || {};
    return Object.keys(frames)
        .filter(key => /^frame_\d+$/.test(key))
        .sort((a, b) => Number(a.slice(6)) - Number(b.slice(6)))
        .map(key => [key, frames[key]])
        .filter(([_key, fileId]) => fileId)
        .slice(0, 3);
}

function chipHtml(values) {
    return values.filter(Boolean).map(value => `<span class="chip">${escapeHtml(value)}</span>`).join('');
}

function polygonPoints(points) {
    return (points || [])
        .map(point => `${Number(point[0] || 0)},${Number(point[1] || 0)}`)
        .join(' ');
}

function renderReferenceVisual(reference, polygon) {
	const width = Number(reference.width || 0);
	const height = Number(reference.height || 0);
	const hasOverlay = reference.preview_url && width > 0 && height > 0 && (polygon || []).length >= 3;
	if (!reference.preview_url) {
		return '';
	}
	return `
	  <div class="reference">
	    <img loading="lazy" src="${escapeHtml(reference.preview_url)}" alt="" />
	    ${hasOverlay ? `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet"><polygon points="${escapeHtml(polygonPoints(polygon))}"></polygon></svg>` : ''}
	  </div>
	`;
}

function renderSupabaseCard(crop) {
	const reference = crop.reference || {};
    const width = Number(crop.frame_width || reference.width || 0);
    const height = Number(crop.frame_height || reference.height || 0);
    const hasOverlay = reference.preview_url && width > 0 && height > 0 && (crop.polygon || []).length >= 3;
    const visual = reference.preview_url ? `
      <div class="reference">
        <img loading="lazy" src="${escapeHtml(reference.preview_url)}" alt="" />
        ${hasOverlay ? `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet"><polygon points="${escapeHtml(polygonPoints(crop.polygon))}"></polygon></svg>` : ''}
      </div>
    ` : '<div class="reference"><div class="muted" style="padding:10px;">No reference frame</div></div>';
    return `
      <article class="card">
        ${visual}
        <div class="card-head">
          <div class="card-title" title="${escapeHtml(crop.label)}">${escapeHtml(crop.label)}</div>
          <div class="chips">${chipHtml([
              'Supabase',
              crop.channel_hint,
              crop.table_hint,
              crop.crop_version ? `v${crop.crop_version}` : '',
              crop.camera_name,
          ])}</div>
        </div>
        <div class="crop-meta">
          <div>crop: ${escapeHtml(crop.table_camera_crops_id || crop.id)}</div>
          <div>camera: ${escapeHtml(crop.camera_source_id || '')}</div>
          <div>table: ${escapeHtml(crop.table_id || '')}</div>
        </div>
      </article>
    `;
}

function renderFallbackGroup(group) {
	const representative = group.representative || {};
	const referenceVisual = renderReferenceVisual(group.reference || {}, group.polygon || []);
	const groupLabel = group.table_hint || group.group_id || 'unknown crop';
	const frames = frameEntries(representative);
	const frameHtml = frames.map(([key, fileId]) => `
	  <a class="frame" href="/api/preview/${encodeURIComponent(fileId)}" target="_blank" title="${escapeHtml(key)}">
	    <img loading="lazy" src="/api/thumb/${encodeURIComponent(fileId)}" alt="${escapeHtml(key)}" />
	  </a>
	`).join('') || '<div class="frame"><div class="muted" style="padding:10px;">No frame</div></div>';
    const bucketSummary = Object.entries(group.bucket_counts || {})
        .map(([bucket, count]) => `${bucket}:${count}`)
        .join(' ');
    const folders = (group.folders || []).map(folder => `
      <li>${escapeHtml(folder.bucket)} · ${escapeHtml(folder.folder_name)} · ${escapeHtml(folder.folder_id)}</li>
    `).join('');
	return `
	  <article class="card" data-group-id="${escapeHtml(group.group_id)}">
	    ${referenceVisual || `<div class="frames">${frameHtml}</div>`}
	    <div class="card-head">
	      <div class="card-title" title="${escapeHtml(group.table_hint || group.group_id)}">${escapeHtml(group.table_hint || 'unknown table')}</div>
	      <div class="chips">${chipHtml([
	          group.channel_hint,
	          `${group.folder_count || 0} folders`,
	          bucketSummary,
          ])}</div>
	    </div>
	    <div class="card-actions">
	      <button class="danger" data-action="trash" data-group-label="${escapeHtml(groupLabel)}" data-folder-ids="${escapeHtml((group.folder_ids || []).join(','))}">Trash all Drive folders for this crop</button>
	      <button class="secondary" data-action="hide">Keep / Hide</button>
        </div>
        <details>
          <summary>Associated Drive folders</summary>
          <ul class="folder-list">${folders}</ul>
        </details>
      </article>
    `;
}

function render(data) {
	const supabaseCrops = data.supabase_crops || [];
	const fallbackGroups = (data.fallback_groups || []).filter(group => !state.hiddenGroups.has(group.group_id));
	el.supabaseGrid.innerHTML = supabaseCrops.length
	    ? supabaseCrops.map(renderSupabaseCard).join('')
	    : '<div class="muted">No active Supabase crops found for this restaurant.</div>';
	if (el.fallbackGrid) {
		el.fallbackGrid.innerHTML = fallbackGroups.length
		    ? fallbackGroups.map(renderFallbackGroup).join('')
		    : '<div class="muted">No generated Drive artifact folders currently match legacy/fallback crop provenance for this restaurant.</div>';
	}
	const counts = data.counts || {};
	el.status.textContent = `${counts.supabase_crops || 0} active crops · ${counts.fallback_groups || 0} fallback groups · ${counts.fallback_folders || 0} Drive folders`;
}

async function loadInventory() {
    el.status.textContent = 'Loading active crops...';
    el.load.disabled = true;
    try {
        const activeData = await fetch(requestUrl(false)).then(readJson);
        render(activeData);
        el.status.textContent = `${activeData.counts?.supabase_crops || 0} active crops · scanning legacy Drive crop artifacts...`;
        const cleanupData = await fetch(requestUrl(true)).then(readJson);
        render(cleanupData);
    } catch (error) {
        el.status.textContent = error.message;
    } finally {
        el.load.disabled = false;
    }
}

async function trashGroup(folderIds, groupLabel) {
    if (!folderIds.length) {
        return;
    }
    const typed = window.prompt(
        `This will move ${folderIds.length} Drive artifact folder(s) for "${groupLabel}" to Drive trash.\n\nType TRASH to confirm.`
    );
    if (typed !== 'TRASH') {
        return;
    }
    const sourcePayload = activeSourcePayload();
    el.status.textContent = `Trashing ${folderIds.length} folder(s)...`;
    await fetch('/api/cleanup/crops/trash', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRF-Token': csrfToken,
        },
        body: JSON.stringify({
            ...sourcePayload,
            folder_ids: folderIds,
            confirm: 'TRASH',
        }),
    }).then(readJson);
    await loadInventory();
}

function bindEvents() {
    el.source.addEventListener('change', () => {
        el.site.disabled = el.source.value !== 'reolink';
        state.hiddenGroups.clear();
        loadInventory();
    });
    el.site.addEventListener('change', () => {
        state.hiddenGroups.clear();
        loadInventory();
    });
    for (const input of [el.channel, el.table, el.query]) {
        input.addEventListener('input', scheduleReload);
    }
    el.load.addEventListener('click', loadInventory);
    el.fallbackGrid?.addEventListener('click', event => {
        const button = event.target.closest('button[data-action]');
        if (!button) {
            return;
        }
        const card = button.closest('.card');
        if (button.dataset.action === 'hide') {
            state.hiddenGroups.add(card.dataset.groupId);
            card.remove();
            return;
        }
        if (button.dataset.action === 'trash') {
            const folderIds = String(button.dataset.folderIds || '').split(',').filter(Boolean);
            trashGroup(folderIds, button.dataset.groupLabel || 'this crop').catch(error => {
                el.status.textContent = error.message;
            });
        }
    });
}

function scheduleReload() {
    window.clearTimeout(reloadTimer);
    reloadTimer = window.setTimeout(loadInventory, 250);
}

async function init() {
	bindEvents();
    try {
        await setupSources();
        await loadInventory();
    } catch (error) {
        el.status.textContent = error.message;
    }
}

init();
