import { useCallback, useEffect, useMemo, useState, type CSSProperties } from 'react'
import './App.css'
import { AnalyzePanel } from './components/AnalyzePanel'
import { AskView } from './components/AskView'
import { ConversationList } from './components/ConversationList'
import { ExplorerView } from './components/ExplorerView'
import { HomeView } from './components/HomeView'
import { JobsView } from './components/JobsView'
import { ReportReader } from './components/ReportReader'
import { ResizeHandle } from './components/ResizeHandle'
import { SettingsView } from './components/SettingsView'
import { Sidebar } from './components/Sidebar'
import { recallApi } from './lib/api'
import {
  clearPreviewCache,
  makePreviewCacheKey,
  previewCacheStats,
  readPreviewCache,
  writePreviewCache,
} from './lib/cache'
import type {
  AnalysisPayload,
  Contact,
  Defaults,
  Job,
  PreviewPayload,
  ReportFile,
  RuntimePaths,
  ViewKey,
} from './types'

const MODEL_STORAGE_KEY = 'recall.selectedModel'
const PATHS_STORAGE_KEY = 'recall.runtimePaths.v1'
const LAYOUT_STORAGE_KEY = 'recall.layout.v2'

type LayoutState = {
  sidebarWidth: number
  listWidth: number
  reportListWidth: number
  sidebarCollapsed: boolean
}

const emptyPaths: RuntimePaths = {
  dbPath: '',
  messagesPath: '',
  outDir: '',
}

const defaultLayout: LayoutState = {
  sidebarWidth: 236,
  listWidth: 430,
  reportListWidth: 318,
  sidebarCollapsed: false,
}

function preferredModel(models: string[], fallback: string) {
  try {
    const stored = window.localStorage.getItem(MODEL_STORAGE_KEY)
    return stored && models.includes(stored) ? stored : fallback
  } catch {
    return fallback
  }
}

function storePreferredModel(nextModel: string) {
  try {
    window.localStorage.setItem(MODEL_STORAGE_KEY, nextModel)
  } catch {
    // Model persistence is optional; the app still runs without browser storage.
  }
}

function pathsFromDefaults(defaults: Defaults): RuntimePaths {
  return {
    dbPath: defaults.dbPath,
    messagesPath: defaults.messagesPath,
    outDir: defaults.outDir,
  }
}

function hasRuntimePath(paths: RuntimePaths) {
  return Boolean(paths.dbPath || paths.messagesPath || paths.outDir)
}

function readStoredPaths(fallback: RuntimePaths) {
  try {
    const stored = window.localStorage.getItem(PATHS_STORAGE_KEY)
    if (!stored) return fallback
    const parsed = JSON.parse(stored) as Partial<RuntimePaths>
    return {
      dbPath: parsed.dbPath || fallback.dbPath,
      messagesPath: parsed.messagesPath || fallback.messagesPath,
      outDir: parsed.outDir || fallback.outDir,
    }
  } catch {
    return fallback
  }
}

function storeRuntimePaths(paths: RuntimePaths) {
  try {
    window.localStorage.setItem(PATHS_STORAGE_KEY, JSON.stringify(paths))
  } catch {
    // Path persistence is optional; the app still runs with defaults.
  }
}

function readStoredLayout() {
  try {
    const stored = window.localStorage.getItem(LAYOUT_STORAGE_KEY)
    if (!stored) return defaultLayout
    const parsed = JSON.parse(stored) as Partial<LayoutState>
    return {
      sidebarWidth: clampNumber(parsed.sidebarWidth, 220, 340, defaultLayout.sidebarWidth),
      listWidth: clampNumber(parsed.listWidth, 360, 720, defaultLayout.listWidth),
      reportListWidth: clampNumber(parsed.reportListWidth, 280, 560, defaultLayout.reportListWidth),
      sidebarCollapsed: Boolean(parsed.sidebarCollapsed),
    }
  } catch {
    return defaultLayout
  }
}

function storeLayout(layout: LayoutState) {
  try {
    window.localStorage.setItem(LAYOUT_STORAGE_KEY, JSON.stringify(layout))
  } catch {
    // Layout persistence is optional.
  }
}

function clampNumber(value: unknown, min: number, max: number, fallback: number) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return fallback
  return Math.max(min, Math.min(max, numeric))
}

function validIsoDate(value: string) {
  return /^\d{4}-\d{2}-\d{2}$/.test(value.trim()) ? value.trim() : ''
}

function reportGroupKey(report: ReportFile) {
  return report.contact || report.displayName || report.path.replace(/\.(html|md)$/i, '')
}

function App() {
  const [activeView, setActiveView] = useState<ViewKey>('ask')
  const [defaults, setDefaults] = useState<Defaults | null>(null)
  const [paths, setPaths] = useState<RuntimePaths>(emptyPaths)
  const [layout, setLayout] = useState<LayoutState>(() => readStoredLayout())
  const [contacts, setContacts] = useState<Contact[]>([])
  const [reports, setReports] = useState<ReportFile[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [selectedContact, setSelectedContact] = useState<Contact | null>(null)
  const [selectedReport, setSelectedReport] = useState<ReportFile | null>(null)
  const [scope, setScope] = useState({ since: '', until: '' })
  const [previewState, setPreviewState] = useState<{
    key: string
    payload: PreviewPayload | null
    error?: string
  }>({ key: '', payload: null })
  const [analysis, setAnalysis] = useState<AnalysisPayload | null>(null)
  const [query, setQuery] = useState('')
  const [model, setModel] = useState('')
  const [contactNameCount, setContactNameCount] = useState(0)
  const [loadingContacts, setLoadingContacts] = useState(true)
  const [loadingReport, setLoadingReport] = useState(false)
  const [runningJob, setRunningJob] = useState<Job | null>(null)
  const [utilityJob, setUtilityJob] = useState<Job | null>(null)
  const [utilityBusy, setUtilityBusy] = useState(false)
  const [jobsLoading, setJobsLoading] = useState(false)
  const [settingsStatus, setSettingsStatus] = useState('')
  const [cacheStats, setCacheStats] = useState(() => previewCacheStats())
  const [error, setError] = useState<string | null>(null)

  const effectiveDefaults = useMemo(
    () =>
      defaults
        ? {
            ...defaults,
            dbPath: paths.dbPath || defaults.dbPath,
            messagesPath: paths.messagesPath || defaults.messagesPath,
            outDir: paths.outDir || defaults.outDir,
          }
        : null,
    [defaults, paths],
  )
  const apiScope = useMemo(
    () => ({ since: validIsoDate(scope.since), until: validIsoDate(scope.until) }),
    [scope.since, scope.until],
  )

  const updateLayoutValue = useCallback((key: keyof LayoutState, value: number) => {
    setLayout((current) => {
      const next = { ...current, [key]: value }
      storeLayout(next)
      return next
    })
  }, [])

  const toggleSidebar = useCallback(() => {
    setLayout((current) => {
      const next = { ...current, sidebarCollapsed: !current.sidebarCollapsed }
      storeLayout(next)
      return next
    })
  }, [])

  const loadContacts = useCallback(async (data: Defaults, runtimePaths: RuntimePaths) => {
    const response = await recallApi.contacts({
      messagesPath: runtimePaths.messagesPath || data.messagesPath,
      dbPath: runtimePaths.dbPath || data.dbPath,
      limit: 80,
    })
    const nextContacts = response.contacts || []
    setContacts(nextContacts)
    setContactNameCount(response.contactNameCount || data.contactNames?.count || 0)
    setSelectedContact((current) => {
      if (!current) return nextContacts[0] || null
      return nextContacts.find((contact) => contact.chat_id === current.chat_id) || nextContacts[0] || null
    })
  }, [])

  const loadJobs = useCallback(async () => {
    setJobsLoading(true)
    try {
      const response = await recallApi.jobs()
      setJobs(response.jobs || [])
    } catch (apiError) {
      setError(apiError instanceof Error ? apiError.message : 'Unable to load jobs.')
    } finally {
      setJobsLoading(false)
    }
  }, [])

  const reloadRuntimeData = useCallback(async (pathOverride?: RuntimePaths) => {
    const data = await recallApi.defaults()
    const nextPaths =
      pathOverride || (hasRuntimePath(paths) ? paths : readStoredPaths(pathsFromDefaults(data)))
    setDefaults(data)
    setPaths(nextPaths)
    setReports(data.reports || [])
    setSelectedReport((current) => {
      if (!current) return data.reports?.[0] || null
      return data.reports?.find((report) => report.path === current.path) || data.reports?.[0] || null
    })
    setContactNameCount(data.contactNames?.count || 0)
    await loadContacts(data, nextPaths)
    await loadJobs()
    return data
  }, [loadContacts, loadJobs, paths])

  useEffect(() => {
    let cancelled = false
    async function boot() {
      try {
        const data = await recallApi.defaults()
        if (cancelled) return
        setDefaults(data)
        const nextPaths = readStoredPaths(pathsFromDefaults(data))
        setPaths(nextPaths)
        setReports(data.reports || [])
        setModel(preferredModel(data.models || [], data.defaultModel))
        setContactNameCount(data.contactNames?.count || 0)
        setSelectedReport(data.reports?.[0] || null)
        void loadJobs()

        const response = await recallApi.contacts({
          messagesPath: nextPaths.messagesPath || data.messagesPath,
          dbPath: nextPaths.dbPath || data.dbPath,
          limit: 80,
        })
        if (cancelled) return
        setContacts(response.contacts || [])
        setContactNameCount(response.contactNameCount || data.contactNames?.count || 0)
        setSelectedContact(response.contacts?.[0] || null)
      } catch (apiError) {
        if (!cancelled) setError(apiError instanceof Error ? apiError.message : 'Unable to load Recall.')
      } finally {
        if (!cancelled) setLoadingContacts(false)
      }
    }
    void boot()
    return () => {
      cancelled = true
    }
  }, [loadJobs])

  const previewKey =
    effectiveDefaults && selectedContact && model
      ? makePreviewCacheKey({
          messagesPath: effectiveDefaults.messagesPath,
          contact: selectedContact.chat_id,
          model,
          since: apiScope.since,
          until: apiScope.until,
        })
      : ''

  useEffect(() => {
    if (!effectiveDefaults || !selectedContact || !model || !previewKey) return
    let cancelled = false
    const cached = readPreviewCache(previewKey)
    if (cached) {
      Promise.resolve().then(() => {
        if (!cancelled) setPreviewState({ key: previewKey, payload: cached })
      })
    }
    recallApi
      .preview({
        messagesPath: effectiveDefaults.messagesPath,
        contact: selectedContact.chat_id,
        model,
        since: apiScope.since,
        until: apiScope.until,
      })
      .then((payload) => {
        writePreviewCache(previewKey, payload)
        if (!cancelled) {
          setCacheStats(previewCacheStats())
          setPreviewState({ key: previewKey, payload })
        }
      })
      .catch((apiError) => {
        if (!cancelled && !cached) {
          setPreviewState({
            key: previewKey,
            payload: null,
            error: apiError instanceof Error ? apiError.message : 'Unable to load preview.',
          })
        }
      })
    return () => {
      cancelled = true
    }
  }, [apiScope.since, apiScope.until, effectiveDefaults, model, previewKey, selectedContact])

  const openReport = useCallback(
    async (report: ReportFile) => {
      if (!effectiveDefaults || !model) return
      setSelectedReport(report)
      setActiveView('reports')
      setLoadingReport(true)
      setError(null)
      try {
        const data = await recallApi.analysis({
          messagesPath: effectiveDefaults.messagesPath,
          model,
          reportPath: report.path,
          contact: report.contact,
        })
        setAnalysis(data.analysis)
      } catch (apiError) {
        setAnalysis(null)
        setError(apiError instanceof Error ? apiError.message : 'Unable to open report.')
      } finally {
        setLoadingReport(false)
      }
    },
    [effectiveDefaults, model],
  )

  const runReport = useCallback(async () => {
    if (!effectiveDefaults || !selectedContact || !model) return
    setError(null)
    try {
      const created = await recallApi.createAnalyzeJob({
        dbPath: effectiveDefaults.dbPath,
        messagesPath: effectiveDefaults.messagesPath,
        outDir: effectiveDefaults.outDir,
        contact: selectedContact.chat_id,
        model,
        html: true,
        extractFirst: false,
        since: apiScope.since,
        until: apiScope.until,
      })
      setRunningJob(created.job)
      await loadJobs()
    } catch (apiError) {
      setError(apiError instanceof Error ? apiError.message : 'Unable to start report.')
    }
  }, [apiScope.since, apiScope.until, effectiveDefaults, loadJobs, model, selectedContact])

  useEffect(() => {
    if (!runningJob || runningJob.status === 'completed' || runningJob.status === 'failed') return
    const timer = window.setInterval(() => {
      recallApi
        .job(runningJob.id)
        .then(async ({ job }) => {
          setRunningJob(job)
          if (job.status === 'completed') {
            const nextReports = job.result?.reports || (await recallApi.reports()).reports
            setReports(nextReports)
            await loadJobs()
            if (job.result?.analysis) {
              setAnalysis(job.result.analysis)
              const reportPath = job.result.reportPath
              const report = nextReports.find((item) => item.path === reportPath) || nextReports[0]
              setSelectedReport(report || null)
              // only auto-open the report if the user is still where they
              // started it -- never yank them out of Chat/Search mid-thought
              setActiveView((current) =>
                current === 'analyze' || current === 'reports' ? 'reports' : current,
              )
            }
          }
        })
        .catch((apiError) => {
          setError(apiError instanceof Error ? apiError.message : 'Unable to poll report job.')
        })
    }, 1200)
    return () => window.clearInterval(timer)
  }, [loadJobs, runningJob])

  useEffect(() => {
    if (!utilityJob || utilityJob.status === 'completed' || utilityJob.status === 'failed') return
    const timer = window.setInterval(() => {
      recallApi
        .job(utilityJob.id)
        .then(async ({ job }) => {
          setUtilityJob(job)
          if (job.status === 'completed') {
            setUtilityBusy(false)
            setSettingsStatus('Message export refreshed.')
            await reloadRuntimeData()
          } else if (job.status === 'failed') {
            setUtilityBusy(false)
          }
        })
        .catch((apiError) => {
          setUtilityBusy(false)
          setError(apiError instanceof Error ? apiError.message : 'Unable to poll maintenance job.')
        })
    }, 1200)
    return () => window.clearInterval(timer)
  }, [reloadRuntimeData, utilityJob])

  const handleModelChange = useCallback((nextModel: string) => {
    storePreferredModel(nextModel)
    setModel(nextModel)
  }, [])

  const refreshContacts = useCallback(async () => {
    setUtilityBusy(true)
    setUtilityJob(null)
    setSettingsStatus('Refreshing contacts...')
    setError(null)
    try {
      const response = await recallApi.refreshContactNames()
      setContactNameCount(response.contactNames.count || 0)
      setSettingsStatus(response.message || 'Contacts refreshed.')
      await reloadRuntimeData()
    } catch (apiError) {
      setSettingsStatus('')
      setError(apiError instanceof Error ? apiError.message : 'Unable to refresh contacts.')
    } finally {
      setUtilityBusy(false)
    }
  }, [reloadRuntimeData])

  const applyRuntimePaths = useCallback(
    async (nextPaths: RuntimePaths) => {
      const normalized = {
        dbPath: nextPaths.dbPath.trim(),
        messagesPath: nextPaths.messagesPath.trim(),
        outDir: nextPaths.outDir.trim(),
      }
      setUtilityBusy(true)
      setUtilityJob(null)
      setSettingsStatus('Applying paths...')
      setError(null)
      try {
        storeRuntimePaths(normalized)
        setPaths(normalized)
        setPreviewState({ key: '', payload: null })
        if (defaults) {
          await loadContacts(defaults, normalized)
        }
        setSettingsStatus('Paths updated.')
      } catch (apiError) {
        setSettingsStatus('')
        setError(apiError instanceof Error ? apiError.message : 'Unable to apply paths.')
      } finally {
        setUtilityBusy(false)
      }
    },
    [defaults, loadContacts],
  )

  const refreshExport = useCallback(async () => {
    if (!effectiveDefaults) return
    setUtilityBusy(true)
    setSettingsStatus('Refreshing message export...')
    setError(null)
    try {
      const created = await recallApi.createExportJob({
        dbPath: effectiveDefaults.dbPath,
        messagesPath: effectiveDefaults.messagesPath,
      })
      setUtilityJob(created.job)
    } catch (apiError) {
      setUtilityBusy(false)
      setSettingsStatus('')
      setError(apiError instanceof Error ? apiError.message : 'Unable to refresh message export.')
    }
  }, [effectiveDefaults])

  const clearPreviewCaches = useCallback(async () => {
    clearPreviewCache()
    setCacheStats(previewCacheStats())
    setSettingsStatus('Preview cache cleared.')
    try {
      await recallApi.clearPreviewCache()
    } catch {
      // The local browser cache is cleared even if the server cache is unavailable.
    }
  }, [])

  const shellTitle = useMemo(() => {
    if (activeView === 'home') return 'Overview'
    if (activeView === 'explore') return 'Search'
    if (activeView === 'ask') return 'Chat'
    if (activeView === 'reports') return 'Reports'
    if (activeView === 'jobs') return 'Jobs'
    if (activeView === 'settings') return 'Settings'
    return 'Analyze'
  }, [activeView])

  const preview = previewState.key === previewKey ? previewState.payload : null
  const loadingPreview = Boolean(previewKey && previewState.key !== previewKey)
  const visibleError = error || (previewState.key === previewKey ? previewState.error || null : null)
  const reportGroupCount = useMemo(() => new Set(reports.map(reportGroupKey)).size, [reports])
  const shellStyle = { '--sidebar-width': `${layout.sidebarWidth}px` } as CSSProperties
  const analyzeStyle = { '--list-width': `${layout.listWidth}px` } as CSSProperties
  const defaultSettingsPaths = useMemo(() => (defaults ? pathsFromDefaults(defaults) : emptyPaths), [defaults])
  const appliedSettingsPaths = useMemo(
    () => (effectiveDefaults ? pathsFromDefaults(effectiveDefaults) : paths),
    [effectiveDefaults, paths],
  )

  const handleViewChange = useCallback(
    (view: ViewKey) => {
      setActiveView(view)
      if (view === 'reports' && selectedReport && !analysis && !loadingReport) {
        void openReport(selectedReport)
      }
    },
    [analysis, loadingReport, openReport, selectedReport],
  )

  const selectContactForAnalyze = useCallback((contact: Contact) => {
    setSelectedContact(contact)
    setActiveView('analyze')
  }, [])

  return (
    <main className={`app-shell ${layout.sidebarCollapsed ? 'sidebar-collapsed' : ''}`} style={shellStyle}>
      <Sidebar
        activeView={activeView}
        defaults={effectiveDefaults}
        namedCount={contactNameCount}
        conversationCount={contacts.length}
        reportCount={reportGroupCount}
        collapsed={layout.sidebarCollapsed}
        onToggleCollapse={toggleSidebar}
        onViewChange={handleViewChange}
      />
      <ResizeHandle
        label="Resize navigation"
        value={layout.sidebarWidth}
        min={220}
        max={340}
        onChange={(value) => updateLayoutValue('sidebarWidth', value)}
      />

      <section className="workspace">
        <header className="topbar">
          <div>
            <h1>{shellTitle}</h1>
          </div>
        </header>

        {visibleError ? <div className="error-banner">{visibleError}</div> : null}

        {activeView === 'home' ? (
          <HomeView
            defaults={effectiveDefaults}
            contacts={contacts}
            reports={reports}
            reportCount={reportGroupCount}
            namedCount={contactNameCount}
            jobs={jobs}
            onViewChange={handleViewChange}
            onSelectContact={selectContactForAnalyze}
            onOpenReport={openReport}
          />
        ) : activeView === 'reports' ? (
          <ReportReader
            reports={reports}
            selectedReport={selectedReport}
            analysis={analysis}
            loading={loadingReport}
            listWidth={layout.reportListWidth}
            onListWidthChange={(value) => updateLayoutValue('reportListWidth', value)}
            onSelectReport={openReport}
          />
        ) : activeView === 'explore' ? (
          <ExplorerView
            defaults={effectiveDefaults}
            contacts={contacts}
            selectedContact={selectedContact}
            onSelectContact={selectContactForAnalyze}
          />
        ) : activeView === 'ask' ? (
          <AskView
            defaults={effectiveDefaults}
            model={model}
            onModelChange={handleModelChange}
            contacts={contacts}
            onSelectContact={selectContactForAnalyze}
          />
        ) : activeView === 'jobs' ? (
          <JobsView
            jobs={jobs}
            runningJob={runningJob}
            utilityJob={utilityJob}
            loading={jobsLoading}
            onRefresh={loadJobs}
          />
        ) : activeView === 'settings' ? (
          <SettingsView
            key={`${appliedSettingsPaths.dbPath}\n${appliedSettingsPaths.messagesPath}\n${appliedSettingsPaths.outDir}`}
            defaults={effectiveDefaults}
            defaultPaths={defaultSettingsPaths}
            paths={appliedSettingsPaths}
            model={model}
            contactNameCount={contactNameCount}
            cacheStats={cacheStats}
            utilityJob={utilityJob}
            utilityBusy={utilityBusy}
            statusMessage={settingsStatus}
            onModelChange={handleModelChange}
            onApplyPaths={applyRuntimePaths}
            onRefreshContacts={refreshContacts}
            onRefreshExport={refreshExport}
            onClearPreviewCache={clearPreviewCaches}
          />
        ) : (
          <div className="analyze-layout" style={analyzeStyle}>
            <ConversationList
              contacts={contacts}
              selectedContact={selectedContact}
              loading={loadingContacts}
              query={query}
              contactNameCount={contactNameCount}
              onQueryChange={setQuery}
              onSelect={setSelectedContact}
            />
            <ResizeHandle
              label="Resize conversation list"
              value={layout.listWidth}
              min={360}
              max={720}
              onChange={(value) => updateLayoutValue('listWidth', value)}
            />
            <AnalyzePanel
              contact={selectedContact}
              defaults={effectiveDefaults}
              model={model}
              scope={scope}
              preview={preview}
              reports={reports}
              loadingPreview={loadingPreview}
              runningJob={runningJob}
              onRunReport={runReport}
              onOpenReport={openReport}
              onViewReports={() => setActiveView('reports')}
              onModelChange={handleModelChange}
              onScopeChange={setScope}
            />
          </div>
        )}
      </section>
    </main>
  )
}

export default App
