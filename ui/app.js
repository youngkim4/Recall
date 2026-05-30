const $ = (selector) => document.querySelector(selector);

const state = {
  contacts: [],
  selectedContact: null,
  preview: null,
  defaults: null,
  activeJobId: null,
  pollTimer: null,
  scope: "all",
  previewRequestId: 0,
  reportReady: false,
  analysis: null,
  resultTab: "story",
  workspaceView: "analysis",
  reports: [],
  analysisCache: new Map(),
  analysisRequests: new Map(),
  contactNameCount: 0,
  contactNamesLoading: false,
  contactNamesSummary: null,
};

const icons = {
  layers: `<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"><path d="m12 3 8 4.5-8 4.5-8-4.5L12 3Z"/><path d="M4 12l8 4.5 8-4.5"/><path d="M4 16.5l8 4.5 8-4.5"/></svg>`,
  file: `<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5Z"/><path d="M14 3v5h5M8 13h8M8 17h6"/></svg>`,
  settings: `<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"><path d="M12 15.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Z"/><path d="M19.4 15a1.8 1.8 0 0 0 .36 1.98l.05.05a2.1 2.1 0 0 1-2.97 2.97l-.05-.05a1.8 1.8 0 0 0-1.98-.36 1.8 1.8 0 0 0-1.1 1.66V21a2.1 2.1 0 0 1-4.2 0v-.08A1.8 1.8 0 0 0 8.4 19.3a1.8 1.8 0 0 0-1.98.36l-.05.05a2.1 2.1 0 0 1-2.97-2.97l.05-.05A1.8 1.8 0 0 0 3.8 14.7 1.8 1.8 0 0 0 2.15 13H2a2.1 2.1 0 0 1 0-4.2h.08A1.8 1.8 0 0 0 3.7 7.7a1.8 1.8 0 0 0-.36-1.98l-.05-.05A2.1 2.1 0 0 1 6.26 2.7l.05.05a1.8 1.8 0 0 0 1.98.36H8.4A1.8 1.8 0 0 0 9.5 1.45V1a2.1 2.1 0 0 1 4.2 0v.08a1.8 1.8 0 0 0 1.1 1.66 1.8 1.8 0 0 0 1.98-.36l.05-.05a2.1 2.1 0 0 1 2.97 2.97l-.05.05a1.8 1.8 0 0 0-.36 1.98v.1A1.8 1.8 0 0 0 21.05 8H21a2.1 2.1 0 0 1 0 4.2h-.08A1.8 1.8 0 0 0 19.4 15Z"/></svg>`,
  lock: `<svg viewBox="0 0 24 24" fill="none" stroke-width="1.9"><rect x="4.5" y="10" width="15" height="10" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/></svg>`,
  database: `<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"><ellipse cx="12" cy="5" rx="7" ry="3"/><path d="M5 5v6c0 1.66 3.13 3 7 3s7-1.34 7-3V5"/><path d="M5 11v6c0 1.66 3.13 3 7 3s7-1.34 7-3v-6"/></svg>`,
  refresh: `<svg viewBox="0 0 24 24" fill="none" stroke-width="1.9"><path d="M20 6v5h-5"/><path d="M4 18v-5h5"/><path d="M18.5 9A7 7 0 0 0 6.2 6.7L4 9"/><path d="M5.5 15a7 7 0 0 0 12.3 2.3L20 15"/></svg>`,
  download: `<svg viewBox="0 0 24 24" fill="none" stroke-width="1.9"><path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 20h14"/></svg>`,
  search: `<svg viewBox="0 0 24 24" fill="none" stroke-width="1.9"><circle cx="11" cy="11" r="7"/><path d="m16 16 4 4"/></svg>`,
  spark: `<svg viewBox="0 0 24 24" fill="none" stroke-width="1.8"><path d="M13 2 9.8 9.2 3 12l6.8 2.8L13 22l3.2-7.2L23 12l-6.8-2.8L13 2Z"/><path d="M5 3v4M3 5h4"/></svg>`,
};

function installIcons() {
  document.querySelectorAll("[data-icon]").forEach((el) => {
    el.innerHTML = icons[el.dataset.icon] || "";
  });
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("visible");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => toast.classList.remove("visible"), 4200);
}

function hideToast() {
  const toast = $("#toast");
  toast.classList.remove("visible");
  window.clearTimeout(showToast.timer);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data;
}

function reportAnalysisCacheKey(report, messagesPath) {
  return [
    report?.path || "",
    report?.updatedAt || "",
    report?.size || "",
    report?.contact || "",
    messagesPath || "",
    $("#modelSelect")?.value || "",
  ].join("::");
}

function rememberAnalysis(key, analysis) {
  if (!key || !analysis) return;
  state.analysisCache.set(key, analysis);
  while (state.analysisCache.size > 24) {
    state.analysisCache.delete(state.analysisCache.keys().next().value);
  }
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return Number(value).toLocaleString();
}

function formatMoney(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
  return `$${Number(value).toFixed(2)}`;
}

function shortDate(value) {
  if (!value) return "--";
  return String(value).slice(0, 10);
}

function cleanInsightValue(value) {
  const text = String(value ?? "");
  const midnightDate = text.match(/^(\d{4}-\d{2}-\d{2})(?:[ T]00:00:00)?$/);
  return midnightDate ? midnightDate[1] : text;
}

function isoDate(date) {
  return date.toISOString().slice(0, 10);
}

function initials(value) {
  const clean = String(value || "R").replace(/[^a-zA-Z0-9]/g, "");
  return clean.slice(0, 2).toUpperCase() || "R";
}

function displayContact(value) {
  const text = String(value || "");
  if (text.startsWith("chat")) return `Group ${text.slice(-4)}`;
  if (text.startsWith("+") && text.length > 5) return "Unnamed contact";
  if (text.includes("@")) return text.split("@")[0];
  return text || "Conversation";
}

function contactTitle(contact) {
  if (!contact || typeof contact !== "object") return displayContact(contact);
  return contact.display_name || contact.displayName || displayContact(contact.chat_id);
}

function reportTitle(report) {
  return report?.displayName || report?.display_name || displayContact(report?.contact) || report?.name || "Report";
}

function contactKind(value) {
  const text = String(value || "");
  if (text.startsWith("chat")) return "Group thread";
  if (text.includes("@")) return "Email thread";
  return "Direct thread";
}

function latestMarkdownReport() {
  return state.reports.find((report) => report.kind === "md");
}

function uniqueMarkdownReports() {
  const seen = new Set();
  return state.reports
    .map((report, index) => ({ report, index }))
    .filter(({ report }) => {
      if (report.kind !== "md") return false;
      const key = report.contact || report.path || report.name;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
}

function updateSidebarSummary() {
  const conversationCount = $("#sidebarConversationCount");
  const namedCount = $("#sidebarNamedCount");
  const reportCount = $("#sidebarReportCount");
  const recentReports = $("#sidebarRecentReports");
  if (!conversationCount || !namedCount || !reportCount || !recentReports) return;

  const cachedNames = Number(state.contactNamesSummary?.count || 0);
  const uniqueReports = uniqueMarkdownReports();
  const namedValue = state.contacts.length ? state.contactNameCount : cachedNames;
  conversationCount.textContent = state.contacts.length ? formatNumber(state.contacts.length) : "--";
  namedCount.textContent = namedValue ? formatNumber(namedValue) : "--";
  reportCount.textContent = uniqueReports.length ? formatNumber(uniqueReports.length) : "--";

  recentReports.innerHTML = uniqueReports.length
    ? uniqueReports.slice(0, 6).map(({ report, index }) => `
      <button class="sidebar-report-row" type="button" data-sidebar-report-index="${index}">
        <span>${escapeHtml(reportTitle(report))}</span>
        <small>${escapeHtml(shortDate(report.updatedAt))}</small>
      </button>
    `).join("")
    : `<span class="sidebar-empty">Reports appear here after generation.</span>`;
}

async function loadDefaults() {
  const data = await api("/api/defaults");
  const reports = data.reports || [];
  state.defaults = data;
  state.reports = reports;
  state.contactNamesSummary = data.contactNames || null;
  $("#dbPath").value = data.dbPath || "";
  $("#messagesPath").value = data.messagesPath || "";
  renderModels(data.models || [], data.defaultModel);
  renderReports(reports);
  syncReportsVisibility();
  resetPreview();
  renderContacts([]);
  updateContactNameControls();
  updateSidebarSummary();
  if (data.hasMessages) {
    await loadContacts({ silent: true });
  }
  updateSteps();
}

function renderModels(models, selected) {
  const select = $("#modelSelect");
  select.innerHTML = models.map((model) => `<option value="${escapeHtml(model)}">${escapeHtml(model)}</option>`).join("");
  select.value = selected;
}

async function loadContacts(options = {}) {
  const silent = Boolean(options.silent);
  const dbPath = $("#dbPath").value.trim();
  const messagesPath = $("#messagesPath").value.trim();
  if (!messagesPath && !dbPath) {
    if (!silent) showToast("Open Advanced and enter a Messages CSV or chat.db path first.");
    return;
  }
  if (!silent) setButtonLoading("#loadContactsBtn", true, "Loading");
  try {
    const params = new URLSearchParams({ limit: "80" });
    if (messagesPath) params.set("messagesPath", messagesPath);
    if (dbPath) params.set("dbPath", dbPath);
    const data = await api(`/api/contacts?${params.toString()}`);
    state.contacts = data.contacts || [];
    state.selectedContact = null;
    state.preview = null;
    state.reportReady = false;
    state.analysis = null;
    state.resultTab = "story";
    resetPreview();
    renderResults(null);
    renderContacts(state.contacts);
    $("#connectionText").textContent = state.contacts.length
      ? `${state.contacts.length} conversations loaded.`
      : "No conversations found.";
    const source = data.source === "messages" ? "message export" : "database";
    const names = Number(data.contactNameCount || 0);
    state.contactNameCount = names;
    updateContactNameControls();
    updateSidebarSummary();
    const nameText = names ? ` (${names} named)` : "";
    if (!silent) showToast(`Loaded ${state.contacts.length} conversations from ${source}${nameText}.`);
  } catch (error) {
    if (!silent) showToast(error.message);
  } finally {
    if (!silent) setButtonLoading("#loadContactsBtn", false);
    updateSteps();
  }
}

async function resolveContactNames() {
  if (state.contactNamesLoading) return;
  state.contactNamesLoading = true;
  updateContactNameControls();
  try {
    const data = await api("/api/contact-names", {
      method: "POST",
      body: JSON.stringify({}),
    });
    state.contactNamesSummary = data.contactNames || null;
    state.analysisCache.clear();
    await loadContacts({ silent: true });
    const names = state.contactNameCount;
    showToast(names ? `Matched names for ${names} conversations.` : (data.message || "Contacts cache updated."));
  } catch (error) {
    showToast(error.message);
  } finally {
    state.contactNamesLoading = false;
    updateContactNameControls();
  }
}

function updateContactNameControls() {
  const button = $("#resolveContactsBtn");
  if (!button) return;
  const cached = Number(state.contactNamesSummary?.count || 0);
  button.disabled = state.contactNamesLoading;
  button.classList.toggle("active", state.contactNameCount > 0 || cached > 0);
  if (state.contactNamesLoading) {
    button.textContent = "Opening Contacts";
  } else if (state.contactNameCount > 0) {
    button.textContent = `${state.contactNameCount} named`;
  } else if (cached > 0) {
    button.textContent = "Names saved";
  } else {
    button.textContent = "Use Contacts";
  }
  updateSidebarSummary();
}

function renderContacts(contacts) {
  const query = $("#contactSearch").value.trim().toLowerCase();
  const filtered = contacts.filter((contact) => {
    const searchText = [contact.chat_id, contact.display_name, contact.displayName]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return searchText.includes(query);
  });
  $("#contactCount").textContent = contacts.length
    ? `${filtered.length} of ${contacts.length} conversations`
    : "Reload conversations to begin.";

  $("#contactList").innerHTML = filtered.length
    ? filtered.map((contact, index) => contactRow(contact, index)).join("")
    : `<div class="empty-state">No conversations loaded.</div>`;

  document.querySelectorAll(".contact-row").forEach((row) => {
    row.addEventListener("click", () => {
      const contact = filtered[Number(row.dataset.index)];
      selectContact(contact);
    });
  });
  updateSidebarSummary();
}

function contactRow(contact, index) {
  const active = state.selectedContact?.chat_id === contact.chat_id ? "active" : "";
  const meta = [
    Number(contact.is_group) ? "Group" : "",
    `${shortDate(contact.first_msg)} to ${shortDate(contact.last_msg)}`,
  ].filter(Boolean).join(" / ");
  return `
    <button class="contact-row ${active}" type="button" data-index="${index}" aria-pressed="${active ? "true" : "false"}">
      <span class="avatar">${escapeHtml(initials(contactTitle(contact)))}</span>
      <span class="contact-main">
        <span class="contact-name">${escapeHtml(contactTitle(contact))}</span>
        <span class="contact-meta">${escapeHtml(meta)}</span>
      </span>
      <span class="count-pill">${formatNumber(contact.message_count)}</span>
    </button>
  `;
}

function resetPreview() {
  state.previewRequestId += 1;
  $("#selectedTitle").textContent = "Choose a conversation";
  $("#threadRange").textContent = "Waiting";
  $("#previewSummary").textContent = "Volume, cost, and recent messages.";
  $("#metricTotal").textContent = "--";
  $("#metricSent").textContent = "--";
  $("#metricReceived").textContent = "--";
  $("#metricDays").textContent = "--";
  $("#costEstimate").textContent = "--";
  $("#tokenEstimate").textContent = "Select a conversation to calculate tokens.";
  $("#chartLabel").textContent = "No preview";
  renderChart([]);
  renderMessages([]);
  $("#analyzeBtn").disabled = true;
  updateSteps();
}

function selectContact(contact) {
  state.selectedContact = contact;
  state.preview = null;
  state.reportReady = false;
  state.analysis = null;
  state.resultTab = "story";
  applyScopeToDateInputs();
  $("#selectedTitle").textContent = contactTitle(contact);
  $("#threadRange").textContent = `${shortDate(contact.first_msg)} to ${shortDate(contact.last_msg)}`;
  $("#connectionText").textContent = `${contactKind(contact.chat_id)} selected.`;
  renderContacts(state.contacts);
  renderResults(null);
  renderPreviewLoading();
  updateSteps();
  loadPreview();
}

function setScope(scope) {
  state.scope = scope;
  document.querySelectorAll(".segment").forEach((button) => {
    button.classList.toggle("active", button.dataset.scope === scope);
  });
  $("#customDates").hidden = scope !== "custom";
  applyScopeToDateInputs();
  if (state.selectedContact) {
    renderPreviewLoading();
    loadPreview();
  }
}

function applyScopeToDateInputs() {
  const sinceInput = $("#sinceInput");
  const untilInput = $("#untilInput");
  if (state.scope === "all") {
    sinceInput.value = "";
    untilInput.value = "";
    return;
  }
  if (state.scope === "last-year" && state.selectedContact?.last_msg) {
    const until = new Date(shortDate(state.selectedContact.last_msg));
    const since = new Date(until);
    since.setFullYear(since.getFullYear() - 1);
    sinceInput.value = isoDate(since);
    untilInput.value = isoDate(until);
  }
}

function renderPreviewLoading() {
  $("#metricTotal").textContent = "--";
  $("#metricSent").textContent = "--";
  $("#metricReceived").textContent = "--";
  $("#metricDays").textContent = "--";
  $("#costEstimate").textContent = "--";
  $("#tokenEstimate").textContent = "Calculating preview and token estimate...";
  $("#chartLabel").textContent = "Loading";
  $("#previewSummary").textContent = "Reading the selected export and calculating model usage.";
  renderChart([]);
  renderMessages([]);
  $("#analyzeBtn").disabled = true;
}

async function loadPreview() {
  const contact = state.selectedContact?.chat_id;
  const messagesPath = $("#messagesPath").value.trim();
  if (!contact || !messagesPath) return;
  const requestId = ++state.previewRequestId;
  try {
    const params = new URLSearchParams({
      messagesPath,
      contact,
      model: $("#modelSelect").value,
      since: $("#sinceInput").value,
      until: $("#untilInput").value,
    });
    const data = await api(`/api/preview?${params.toString()}`);
    if (requestId !== state.previewRequestId || state.selectedContact?.chat_id !== contact) return;
    state.preview = data;
    renderPreview(data);
    $("#analyzeBtn").disabled = false;
    updateSteps();
  } catch (error) {
    if (requestId !== state.previewRequestId) return;
    showToast(error.message);
    $("#analyzeBtn").disabled = true;
    updateSteps();
  }
}

function renderPreview(data) {
  const stats = data.stats || {};
  $("#metricTotal").textContent = formatNumber(stats.totalMessages);
  $("#metricSent").textContent = formatNumber(stats.sentCount);
  $("#metricReceived").textContent = formatNumber(stats.receivedCount);
  $("#metricDays").textContent = formatNumber(stats.activeDays);
  $("#threadRange").textContent = `${shortDate(stats.firstTimestamp)} to ${shortDate(stats.lastTimestamp)}`;
  $("#costEstimate").textContent = formatMoney(data.estimate?.estimated_cost);
  $("#tokenEstimate").textContent = `${formatNumber(data.estimate?.input_tokens)} input tokens / ${formatNumber(data.estimate?.output_tokens)} output tokens${data.estimate?.needs_chunking ? " / chunked" : ""}`;
  $("#chartLabel").textContent = `${data.monthly?.length || 0} months`;
  $("#previewSummary").textContent = `${formatNumber(stats.totalMessages)} messages across ${formatNumber(stats.activeDays)} active days.`;
  renderChart(data.monthly || []);
  renderMessages(data.recentMessages || []);
}

function renderMessages(messages) {
  $("#messageList").innerHTML = messages.length
    ? messages.map((message) => `
      <div class="message-row">
        <div class="message-time">${escapeHtml(shortDate(message.timestamp))}</div>
        <div class="message-text ${Number(message.isFromMe) === 1 ? "me" : ""}">${escapeHtml(message.text)}</div>
      </div>
    `).join("")
    : `<div class="empty-state">Choose a conversation to preview recent messages.</div>`;
}

function renderChart(monthly) {
  const svg = $("#volumeChart");
  svg.innerHTML = chartMarkup(monthly);
}

function chartMarkup(monthly, width = 720, height = 180) {
  const pad = { left: 24, right: 24, top: 18, bottom: 30 };
  const innerWidth = width - pad.left - pad.right;
  const innerHeight = height - pad.top - pad.bottom;
  const max = Math.max(1, ...monthly.map((row) => row.total));

  if (!monthly.length) {
    return `<text x="${width / 2}" y="${height / 2 + 2}" text-anchor="middle" fill="#8a95a8" font-size="13">No activity preview yet</text>`;
  }

  const step = innerWidth / Math.max(1, monthly.length - 1);
  const points = monthly.map((row, index) => {
    const x = pad.left + index * step;
    const y = pad.top + innerHeight - (row.total / max) * innerHeight;
    return [x, y];
  });
  const linePath = points.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const areaPath = `${pad.left},${pad.top + innerHeight} ${linePath} ${width - pad.right},${pad.top + innerHeight}`;
  const guideLines = [0.25, 0.5, 0.75].map((ratio) => {
    const y = pad.top + innerHeight * ratio;
    return `<line x1="${pad.left}" y1="${y.toFixed(1)}" x2="${width - pad.right}" y2="${y.toFixed(1)}" stroke="rgba(226,234,255,0.08)" />`;
  }).join("");
  const peak = monthly.reduce((best, row) => row.total > best.total ? row : best, monthly[0]);

  return `
    ${guideLines}
    <polygon points="${areaPath}" fill="rgba(124,140,255,0.16)" />
    <polyline points="${linePath}" fill="none" stroke="#9aa8ff" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
    <text x="${pad.left}" y="${height - 8}" fill="#8a95a8" font-size="11">${escapeHtml(monthly[0].month)}</text>
    <text x="${width - pad.right}" y="${height - 8}" text-anchor="end" fill="#8a95a8" font-size="11">${escapeHtml(monthly[monthly.length - 1].month)}</text>
    <text x="${width / 2}" y="${height - 8}" text-anchor="middle" fill="#8a95a8" font-size="11">Peak ${escapeHtml(peak.month)}: ${formatNumber(peak.total)} messages</text>
  `;
}

async function startJob(action) {
  if (action === "analyze" && !state.preview) {
    showToast("Choose a conversation and wait for the preview first.");
    return;
  }

  const payload = {
    action,
    dbPath: $("#dbPath").value.trim(),
    messagesPath: $("#messagesPath").value.trim(),
    outDir: state.defaults?.outDir,
    contact: state.selectedContact?.chat_id,
    model: $("#modelSelect").value,
    since: $("#sinceInput").value,
    until: $("#untilInput").value,
    html: $("#htmlToggle").checked,
    extractFirst: action === "export" ? true : $("#extractToggle").checked,
  };

  try {
    state.reportReady = false;
    state.analysis = null;
    state.resultTab = "story";
    if (action === "export" || payload.extractFirst) {
      state.analysisCache.clear();
      state.analysisRequests.clear();
    }
    renderResults(null);
    const data = await api("/api/jobs", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.activeJobId = data.job.id;
    setStatus("running");
    renderJob(data.job);
    pollJob();
    showToast(action === "export" ? "Refreshing message export." : "Generating memory report.");
  } catch (error) {
    showToast(error.message);
  }
}

async function pollJob() {
  if (!state.activeJobId) return;
  window.clearTimeout(state.pollTimer);
  try {
    const data = await api(`/api/jobs/${state.activeJobId}`);
    renderJob(data.job);
    if (data.job.status === "running" || data.job.status === "queued") {
      state.pollTimer = window.setTimeout(pollJob, 1600);
    } else {
      setStatus(data.job.status === "completed" ? "done" : "failed");
      if (data.job.result?.reports) {
        state.reportReady = true;
        renderReports(data.job.result.reports);
        syncReportsVisibility();
      }
      if (data.job.result?.analysis) {
        state.analysis = data.job.result.analysis;
        const report = data.job.result.reports?.find((item) => item.path === data.job.result.reportPath);
        if (report) {
          rememberAnalysis(reportAnalysisCacheKey(report, $("#messagesPath").value.trim()), state.analysis);
        }
        renderResults(state.analysis);
        setWorkspaceView("report", { scroll: false });
      }
      if (data.job.status === "completed") showToast("Report generated.");
      if (data.job.status === "failed") showToast(data.job.error || "Run failed.");
    }
  } catch (error) {
    showToast(error.message);
  }
}

function renderJob(job) {
  const lines = job.logs?.map((entry) => `[${entry.time.split("T").pop()}] ${entry.message}`) || [];
  $("#jobLog").textContent = lines.length ? lines.join("\n") : `${job.status}...`;
  $("#jobLog").scrollTop = $("#jobLog").scrollHeight;
}

function setStatus(status) {
  const statusText = $("#statusText");
  if (status === "running") statusText.textContent = "Generating report";
  else if (status === "done") statusText.textContent = "Report generated";
  else if (status === "failed") statusText.textContent = "Run failed";
  else statusText.textContent = "No active run";
  updateSteps();
}

async function refreshReports() {
  try {
    const data = await api("/api/reports");
    renderReports(data.reports || []);
    syncReportsVisibility();
  } catch (error) {
    showToast(error.message);
  }
}

function renderReports(reports) {
  state.reports = reports;
  renderLatestReportEntry();
  updateSidebarSummary();
  $("#reportsList").innerHTML = reports.length
    ? reports.map((report, index) => `
      <div class="report-row">
        ${report.kind === "md"
          ? `<button class="report-title-button report-load" type="button" data-report-index="${index}">${escapeHtml(reportTitle(report))}</button>`
          : `<a href="/api/report?path=${encodeURIComponent(report.path)}" target="_blank" rel="noreferrer">${escapeHtml(reportTitle(report))}</a>`}
        <span class="report-actions">
          ${escapeHtml(report.kind)}
          ${report.kind === "md" ? `<button class="text-button report-load" type="button" data-report-index="${index}">View</button>` : `<a class="text-button" href="/api/report?path=${encodeURIComponent(report.path)}" target="_blank" rel="noreferrer">File</a>`}
        </span>
      </div>
    `).join("")
    : `<div class="empty-state">Reports will appear here after generation.</div>`;

  document.querySelectorAll(".report-load").forEach((button) => {
    button.addEventListener("click", () => loadReportAnalysis(reports[Number(button.dataset.reportIndex)]));
  });
}

function renderLatestReportEntry() {
  const latest = latestMarkdownReport();
  const shortcut = $("#latestReportShortcut");
  const readerEntry = $("#reportsReaderEntry");
  const title = latest ? `${reportTitle(latest)} · ${shortDate(latest.updatedAt)}` : "No saved Markdown report yet.";

  $("#latestReportMeta").textContent = title;
  $("#reportsReaderTitle").textContent = title;
  shortcut.hidden = !latest;
  readerEntry.hidden = !latest;
}

function syncReportsVisibility() {
  $("#reportSuccess").hidden = state.workspaceView !== "reports";
}

function updateWorkspaceHeading() {
  $(".topbar h1").textContent = state.workspaceView === "reports" ? "Reports" : (state.workspaceView === "report" ? "Report" : "Analyze");
}

function setWorkspaceView(view, options = {}) {
  state.workspaceView = view === "reports" || view === "report" ? view : "analysis";
  if (state.workspaceView === "analysis" && state.analysis) {
    state.analysis = null;
    state.resultTab = "story";
    renderResults(null);
    resetPreview();
  }
  document.body.classList.toggle("view-reports", state.workspaceView === "reports");
  updateWorkspaceHeading();
  document.querySelectorAll(".side-nav-item").forEach((item) => {
    const ownsView = item.dataset.workspaceView === "reports"
      ? state.workspaceView === "reports" || state.workspaceView === "report"
      : item.dataset.workspaceView === state.workspaceView;
    item.classList.toggle("active", ownsView);
  });
  syncReportsVisibility();
  if (options.scroll === false) return;
  const target = state.workspaceView === "reports"
    ? $("#reportSuccess")
    : state.workspaceView === "report"
      ? $("#resultsPanel")
      : $(".review-panel");
  if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
}

async function loadReportAnalysis(report, options = {}) {
  if (!report) return;
  const silent = Boolean(options.silent);
  const messagesPath = $("#messagesPath").value.trim();
  if (!messagesPath) {
    if (!silent) showToast("Enter a Messages CSV path before loading report details.");
    return;
  }
  const cacheKey = reportAnalysisCacheKey(report, messagesPath);
  const cachedAnalysis = state.analysisCache.get(cacheKey);
  if (cachedAnalysis) {
    applyReportAnalysis(cachedAnalysis, { silent, fromCache: true });
    return;
  }
  let loadingTimer = null;
  try {
    const params = new URLSearchParams({
      reportPath: report.path,
      messagesPath,
      contact: report.contact || "",
      model: $("#modelSelect").value,
    });
    if (!silent) {
      loadingTimer = window.setTimeout(() => showToast("Opening report..."), 250);
    }
    let request = state.analysisRequests.get(cacheKey);
    if (!request) {
      request = api(`/api/analysis?${params.toString()}`);
      state.analysisRequests.set(cacheKey, request);
    }
    const data = await request;
    window.clearTimeout(loadingTimer);
    state.analysisRequests.delete(cacheKey);
    rememberAnalysis(cacheKey, data.analysis);
    applyReportAnalysis(data.analysis, { silent, fromCache: Boolean(data.cached) });
  } catch (error) {
    window.clearTimeout(loadingTimer);
    state.analysisRequests.delete(cacheKey);
    if (!silent) showToast(error.message);
  }
}

function applyReportAnalysis(analysis, options = {}) {
  if (!analysis) return;
  const silent = Boolean(options.silent);
  state.selectedContact = null;
  state.preview = null;
  state.analysis = analysis;
  state.resultTab = "story";
  state.reportReady = true;
  renderContacts(state.contacts);
  renderResults(state.analysis);
  setWorkspaceView("report", { scroll: false });
  if (!silent) {
    $("#resultsPanel").scrollIntoView({ behavior: "smooth", block: "start" });
    hideToast();
  }
}

function renderResults(analysis) {
  const panel = $("#resultsPanel");
  document.body.classList.toggle("has-analysis", Boolean(analysis));
  updateWorkspaceHeading();
  if (!analysis) {
    panel.hidden = true;
    $("#resultContent").innerHTML = "";
    return;
  }
  panel.hidden = false;
  renderAnalysisContext(analysis);
  $("#resultsSummary").textContent = `${formatNumber(analysis.events?.length || 0)} moments · ${formatNumber(analysis.stats?.totalMessages)} messages · ${formatNumber(analysis.files?.length || 0)} files`;
  $("#resultContent").innerHTML = renderCompleteReport(analysis);
  wireRenderedResult();
  renderResultTab(state.resultTab, { scroll: false });
}

function renderAnalysisContext(analysis) {
  const stats = analysis.stats || {};
  const title = analysis.contactDisplayName || displayContact(analysis.contact);
  $("#selectedTitle").textContent = title;
  $("#threadRange").textContent = `${shortDate(stats.firstTimestamp)} to ${shortDate(stats.lastTimestamp)}`;
  $("#connectionText").textContent = `Saved report: ${title}`;
}

function setResultTab(tab) {
  state.resultTab = tab;
  renderResultTab(tab);
}

function renderResultTab(tab, options = {}) {
  if (!state.analysis) return;
  if (!$("#resultContent").innerHTML.trim()) {
    $("#resultContent").innerHTML = renderCompleteReport(state.analysis);
    wireRenderedResult();
  }
  document.querySelectorAll(".result-tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.resultTab === tab);
  });
  if (options.scroll === false) return;
  const target = document.querySelector(`#report-${tab}`);
  if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderCompleteReport(analysis) {
  const events = analysis.events || [];
  const patterns = analysis.patterns || [];
  const files = analysis.files || [];
  const monthly = analysis.monthly || [];
  return `
    <div class="complete-report">
      <section id="report-story" class="report-section report-section-story">
        ${renderStoryResult(analysis)}
      </section>
      <section id="report-timeline" class="report-section">
        ${reportSectionHeading("Events", `${formatNumber(events.length)} moments`)}
        ${renderTimelineResult(analysis)}
      </section>
      <section id="report-patterns" class="report-section">
        ${reportSectionHeading("Patterns", `${formatNumber(patterns.length)} signals`)}
        ${renderPatternsResult(analysis)}
      </section>
      <section id="report-media" class="report-section">
        ${reportSectionHeading("Media", "Attachments and reactions")}
        ${renderMediaResult(analysis)}
      </section>
      <section id="report-charts" class="report-section">
        ${reportSectionHeading("Activity", `${formatNumber(monthly.length)} months`)}
        ${renderChartsResult(analysis)}
      </section>
      <section id="report-files" class="report-section">
        ${reportSectionHeading("Files", `${formatNumber(files.length)} generated`)}
        ${renderFilesResult(analysis)}
      </section>
    </div>
  `;
}

function reportSectionHeading(title, meta) {
  return `
    <div class="report-section-heading">
      <h4>${escapeHtml(title)}</h4>
      <span>${escapeHtml(meta)}</span>
    </div>
  `;
}

function renderStoryResult(analysis) {
  const events = analysis.events || [];
  const quoteEvents = events.filter((event) => event.quote).slice(0, 2);
  const patterns = (analysis.patterns || []).slice(0, 3);
  return `
    <div class="story-grid">
      <article class="story-panel">
        <h4>Overview</h4>
        <div class="story-copy">${analysis.summaryHtml || formatMarkdown(analysis.summary || "The narrative summary will appear here when the generated Markdown report includes a Summary section.")}</div>
      </article>
      <aside class="insight-rail">
        ${patterns.map((pattern) => `
          <div class="insight-card">
            <span>${escapeHtml(pattern.label)}</span>
            <strong>${escapeHtml(cleanInsightValue(pattern.value))}</strong>
            <p>${escapeHtml(pattern.detail)}</p>
          </div>
        `).join("")}
        ${quoteEvents.length ? `
          <div class="quote-stack">
            <h4>Quoted moments</h4>
            ${quoteEvents.map((event) => `
              <blockquote>
                ${escapeHtml(event.quote)}
                <span>${escapeHtml(shortDate(event.date))} / ${escapeHtml(event.title)}</span>
              </blockquote>
            `).join("")}
          </div>
        ` : ""}
      </aside>
    </div>
  `;
}

function wireRenderedResult() {
  const toggle = document.querySelector(".story-toggle");
  const panel = document.querySelector(".story-panel");
  if (!toggle || !panel) return;
  const copy = panel.querySelector(".story-copy");
  if (copy && copy.scrollHeight <= copy.clientHeight + 2) {
    toggle.hidden = true;
    return;
  }
  toggle.addEventListener("click", () => {
    const expanded = toggle.dataset.expanded === "true";
    toggle.dataset.expanded = String(!expanded);
    panel.classList.toggle("is-collapsed", expanded);
    toggle.textContent = expanded ? "Full story" : "Short view";
  });
}

function renderTimelineResult(analysis) {
  const events = analysis.events || [];
  return events.length
    ? `<div class="timeline-list">
        ${events.map((event) => `
          <article class="timeline-item">
            <div class="timeline-date">${escapeHtml(shortDate(event.date))}</div>
            <div>
              <div class="timeline-heading">
                <h4>${escapeHtml(event.title || "Untitled moment")}</h4>
                <span>${escapeHtml(event.category || "moment")}${event.score === null || event.score === undefined ? "" : ` / ${Number(event.score).toFixed(1)}`}</span>
              </div>
              <p>${escapeHtml(event.detail)}</p>
              ${event.quote ? `<blockquote>${escapeHtml(event.quote)}</blockquote>` : ""}
            </div>
          </article>
        `).join("")}
      </div>`
    : `<div class="empty-state">No event timeline was generated for this report.</div>`;
}

function renderPatternsResult(analysis) {
  const patterns = analysis.patterns || [];
  return `
    <div class="pattern-grid">
      ${patterns.map((pattern) => `
        <div class="pattern-card">
          <span>${escapeHtml(pattern.label)}</span>
          <strong>${escapeHtml(cleanInsightValue(pattern.value))}</strong>
          <p>${escapeHtml(pattern.detail)}</p>
        </div>
      `).join("")}
    </div>
  `;
}

function renderMediaResult(analysis) {
  const attachments = analysis.media?.attachments || {};
  const reactions = analysis.media?.reactions || {};
  return `
    <div class="media-grid">
      ${mediaMetric("Photos", attachments.photos)}
      ${mediaMetric("Videos", attachments.videos)}
      ${mediaMetric("Audio", attachments.audio)}
      ${mediaMetric("GIFs", attachments.gifs)}
      ${mediaMetric("Documents", attachments.documents)}
      ${mediaMetric("Other attachments", attachments.other)}
      ${mediaMetric("Loves", reactions.loves)}
      ${mediaMetric("Likes", reactions.likes)}
      ${mediaMetric("Laughs", reactions.laughs)}
      ${mediaMetric("Emphasis", reactions.emphasis)}
      ${mediaMetric("Questions", reactions.questions)}
      ${mediaMetric("Dislikes", reactions.dislikes)}
    </div>
  `;
}

function renderChartsResult(analysis) {
  const monthly = analysis.monthly || [];
  return `
    <div class="results-chart-card">
      <div class="chart-header">
        <div>
          <h4>Activity over time</h4>
          <p>Messages by month.</p>
        </div>
        <span>${formatNumber(monthly.length)} months</span>
      </div>
      <svg class="results-chart" viewBox="0 0 720 220" role="img" aria-label="Monthly message activity trend">${chartMarkup(monthly, 720, 220)}</svg>
    </div>
    <div class="monthly-table-wrap">
      <h4>Monthly data</h4>
      ${monthlyTable(monthly)}
    </div>
  `;
}

function renderFilesResult(analysis) {
  const files = analysis.files || [];
  return files.length
    ? `<div class="file-list">
        ${files.map((file) => `
          <a class="file-card" href="/api/report?path=${encodeURIComponent(file.path)}" target="_blank" rel="noreferrer">
            <span>${escapeHtml(file.label)}</span>
            <strong>${escapeHtml(file.name)}</strong>
            <small>${escapeHtml(file.kind)} / ${formatNumber(file.size)} bytes</small>
          </a>
        `).join("")}
      </div>`
    : `<div class="empty-state">No generated files were returned for this report.</div>`;
}

function mediaMetric(label, value) {
  return `
    <div class="media-card">
      <span>${escapeHtml(label)}</span>
      <strong>${formatNumber(value || 0)}</strong>
    </div>
  `;
}

function monthlyTable(monthly) {
  if (!monthly.length) return `<div class="empty-state">No monthly data available.</div>`;
  return `
    <table class="monthly-table">
      <thead>
        <tr>
          <th>Month</th>
          <th>Total</th>
          <th>Sent</th>
          <th>Received</th>
          <th>Sent ratio</th>
        </tr>
      </thead>
      <tbody>
        ${monthly.map((row) => `
          <tr>
            <td>${escapeHtml(row.month)}</td>
            <td>${formatNumber(row.total)}</td>
            <td>${formatNumber(row.sent)}</td>
            <td>${formatNumber(row.received)}</td>
            <td>${Math.round(Number(row.sentRatio || 0) * 100)}%</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function formatMarkdown(text) {
  const escaped = escapeHtml(text);
  return escaped
    .split(/\n{2,}/)
    .map((block) => {
      const trimmed = block.trim();
      if (!trimmed) return "";
      if (trimmed === "---") return `<hr>`;
      const heading = trimmed.match(/^#{2,3}\s+([^\n]+)(?:\n([\s\S]+))?$/);
      if (heading) {
        const body = heading[2]?.trim();
        return `<h4>${formatInlineMarkdown(heading[1])}</h4>${body ? `<p>${formatInlineMarkdown(body.replaceAll("\n", "<br>"))}</p>` : ""}`;
      }
      if (trimmed.startsWith("&gt; ")) return `<blockquote>${formatInlineMarkdown(trimmed.replace(/^&gt;\s?/, ""))}</blockquote>`;
      const lines = trimmed.split("\n");
      if (lines.every((line) => line.startsWith("- "))) {
        return `<ul>${lines.map((line) => `<li>${formatInlineMarkdown(line.replace(/^- /, ""))}</li>`).join("")}</ul>`;
      }
      if (lines.every((line) => /^\d+\.\s+/.test(line))) {
        return `<ol>${lines.map((line) => `<li>${formatInlineMarkdown(line.replace(/^\d+\.\s+/, ""))}</li>`).join("")}</ol>`;
      }
      return `<p>${formatInlineMarkdown(trimmed.replaceAll("\n", "<br>"))}</p>`;
    })
    .join("");
}

function formatInlineMarkdown(text) {
  return String(text)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>");
}

function updateSteps() {
  const connected = state.contacts.length > 0;
  const chosen = Boolean(state.selectedContact);
  const readyToGenerate = Boolean(state.preview);
  setStep("#stepConnect", connected, !connected);
  setStep("#stepChoose", chosen, connected && !chosen);
  setStep("#stepGenerate", state.reportReady, chosen && (readyToGenerate || !state.reportReady));
}

function setStep(selector, done, active) {
  const step = $(selector);
  if (!step) return;
  step.classList.toggle("done", done);
  step.classList.toggle("active", active);
}

function setButtonLoading(selector, loading, label = "Working") {
  const button = $(selector);
  if (!button) return;
  if (loading) {
    button.dataset.original = button.innerHTML;
    button.innerHTML = label;
    button.disabled = true;
  } else {
    button.innerHTML = button.dataset.original || button.innerHTML;
    button.disabled = false;
    installIcons();
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function wireEvents() {
  $("#loadContactsBtn").addEventListener("click", loadContacts);
  $("#resolveContactsBtn").addEventListener("click", resolveContactNames);
  $("#exportBtn").addEventListener("click", () => startJob("export"));
  $("#analyzeBtn").addEventListener("click", () => startJob("analyze"));
  $("#clearLogBtn").addEventListener("click", () => {
    $("#jobLog").textContent = "No active run.";
    setStatus("idle");
  });
  $("#refreshReportsBtn").addEventListener("click", refreshReports);
  $("#refreshReportsListBtn").addEventListener("click", refreshReports);
  $("#openLatestReportBtn").addEventListener("click", () => loadReportAnalysis(latestMarkdownReport()));
  $("#openLatestReportFromReportsBtn").addEventListener("click", () => loadReportAnalysis(latestMarkdownReport()));
  $("#showReportsShortcutBtn").addEventListener("click", () => setWorkspaceView("reports"));
  $("#sidebarRecentReports").addEventListener("click", (event) => {
    const row = event.target.closest(".sidebar-report-row");
    if (!row) return;
    loadReportAnalysis(state.reports[Number(row.dataset.sidebarReportIndex)]);
  });
  $("#contactSearch").addEventListener("input", () => renderContacts(state.contacts));
  document.querySelectorAll(".segment").forEach((button) => {
    button.addEventListener("click", () => setScope(button.dataset.scope));
  });
  document.querySelectorAll(".result-tab").forEach((button) => {
    button.addEventListener("click", () => setResultTab(button.dataset.resultTab));
  });
  document.querySelectorAll(".side-nav-item").forEach((button) => {
    button.addEventListener("click", () => setWorkspaceView(button.dataset.workspaceView));
  });
  $("#modelSelect").addEventListener("change", () => {
    if (state.selectedContact) loadPreview();
  });
  ["messagesPath", "sinceInput", "untilInput"].forEach((id) => {
    $(`#${id}`).addEventListener("change", () => {
      if (state.scope === "custom" && state.selectedContact) loadPreview();
    });
  });
}

installIcons();
wireEvents();
loadDefaults().catch((error) => {
  showToast(error.message);
  renderContacts([]);
});
