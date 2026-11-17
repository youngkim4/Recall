import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { AnalyzeView } from './components/analyze/AnalyzeView'
import { ChatView } from './components/chat/ChatView'
import { FirstWords } from './components/firstwords/FirstWords'
import { JobsView } from './components/jobs/JobsView'
import { OverviewView } from './components/overview/OverviewView'
import { ArchiveWait } from './components/onboarding/ArchiveWait'
import { OnboardingWizard } from './components/onboarding/OnboardingWizard'
import { ReportsView } from './components/reports/ReportsView'
import { SearchView } from './components/search/SearchView'
import { ResizeHandle } from './components/ResizeHandle'
import { SettingsView } from './components/settings/SettingsView'
import { Sidebar } from './components/Sidebar'
import { CloseIcon } from './components/Icons'
import { recallApi } from './lib/api'
import { orBuilding } from './lib/archiveBuild'
import { isActiveJob, mergeJobs } from './lib/jobs'
import {
  emptyPaths,
  hasRuntimePath,
  pathsFromDefaults,
  preferredModel,
  RAIL_WIDTH,
  readStoredPaths,
  storePreferredModel,
  storeRuntimePaths,
} from './lib/appStorage'
import { NARROW_WINDOW } from './lib/layout'
import { offerModels } from './lib/models'
import { useLayout } from './lib/useLayout'
import { nameReports } from './lib/reports'
import { isUpdate, revealsFirstWords } from './lib/setupGuide'
import { useMediaQuery } from './lib/useMediaQuery'
import { clearPreviewCache, previewCacheStats } from './lib/cache'
import { useConversationPreview } from './lib/useConversationPreview'
import { useJobPoll } from './lib/useJobPoll'
import type {
  AnalysisPayload,
  ArchiveSummary,
  Contact,
  Defaults,
  FirstWordsPayload,
  Job,
  ReportFile,
  RuntimePaths,
  SetupImportResult,
  SetupStatus,
  ViewKey,
} from './types'

// every conversation is listed; the server builds the list once per archive, so the size costs nothing
const CONVERSATION_LIMIT = 5000

const NO_DATES = { since: '', until: '' }

// A marker that setup got somewhere. One that fails to save changes nothing
// now: the same thing is found out again at the next launch.
function markSetup(fields: Parameters<typeof recallApi.setupComplete>[0]) {
  recallApi.setupComplete(fields).catch(() => undefined)
}

function App() {
  // the app opens where first-run setup leaves you: on the first item in the rail
  const [activeView, setActiveView] = useState<ViewKey>('home')
  const [defaults, setDefaults] = useState<Defaults | null>(null)
  const [paths, setPaths] = useState<RuntimePaths>(emptyPaths)
  const { layout, change: changeLayout, toggleSidebar } = useLayout()
  const [contacts, setContacts] = useState<Contact[]>([])
  const [archive, setArchive] = useState<ArchiveSummary | null>(null)
  const [reportFiles, setReports] = useState<ReportFile[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [selectedContact, setSelectedContact] = useState<Contact | null>(null)
  const [selectedReport, setSelectedReport] = useState<ReportFile | null>(null)
  // the dates that were chosen, and the conversation they were chosen for
  const [scoped, setScoped] = useState({ subject: '', ...NO_DATES })
  // the report that was opened, with the file it was read from
  const [opened, setOpened] = useState<{ path: string; analysis: AnalysisPayload } | null>(null)
  // the file of a report that could not be opened
  const [failedReport, setFailedReport] = useState('')
  const [query, setQuery] = useState('')
  const [model, setModel] = useState('')
  const [contactNameCount, setContactNameCount] = useState(0)
  const [loadingContacts, setLoadingContacts] = useState(true)
  const [loadingReport, setLoadingReport] = useState(false)
  const [runningJob, setRunningJob] = useState<Job | null>(null)
  const [utilityBusy, setUtilityBusy] = useState(false)
  const [settingsStatus, setSettingsStatus] = useState('')
  const [cacheStats, setCacheStats] = useState(() => previewCacheStats())
  const [error, setError] = useState<string | null>(null)
  // the job that loads the archive from an earlier export; the window waits while it runs
  const [building, setBuilding] = useState<Job | null>(null)
  const [setupStatus, setSetupStatus] = useState<SetupStatus | null>(null)
  const [showOnboarding, setShowOnboarding] = useState(false)
  const [rerunSetup, setRerunSetup] = useState(false)
  const [firstWords, setFirstWords] = useState<FirstWordsPayload | null>(null)
  const [showFirstWords, setShowFirstWords] = useState(false)

  const effectiveDefaults = useMemo(
    () =>
      defaults
        ? {
            ...defaults,
            messagesPath: paths.messagesPath || defaults.messagesPath,
            outDir: paths.outDir || defaults.outDir,
          }
        : null,
    [defaults, paths],
  )
  // a report goes by the name its conversation has in the list
  const reports = useMemo(() => nameReports(reportFiles, contacts), [reportFiles, contacts])
  // Dates belong to one conversation in one archive. Chosen for another, they
  // are no dates at all, however the conversation came to change.
  const scopeSubject = `${effectiveDefaults?.messagesPath ?? ''}\n${selectedContact?.chat_id ?? ''}`
  const scope = useMemo(
    () => (scoped.subject === scopeSubject ? { since: scoped.since, until: scoped.until } : NO_DATES),
    [scoped, scopeSubject],
  )
  const setScope = useCallback(
    (next: { since: string; until: string }) => setScoped({ subject: scopeSubject, ...next }),
    [scopeSubject],
  )
  // the report on show is the one that was opened, never one left over from another choice
  const analysis = opened && selectedReport && opened.path === selectedReport.path ? opened.analysis : null

  // the newest request for the list wins; an older answer that comes late is dropped
  const contactsRequest = useRef(0)
  const loadContacts = useCallback(async (data: Defaults, runtimePaths: RuntimePaths) => {
    const request = ++contactsRequest.current
    const response = await recallApi
      .contacts({ messagesPath: runtimePaths.messagesPath || data.messagesPath, limit: CONVERSATION_LIMIT })
      .catch(orBuilding)
    if (request !== contactsRequest.current) return
    // an archive that is still being loaded is waited for, and that is no error
    setBuilding('building' in response ? response.building : null)
    if ('building' in response) return
    const nextContacts = response.contacts || []
    setContacts(nextContacts)
    setArchive(response.archive ?? null)
    setContactNameCount(response.contactNameCount || data.contactNames?.count || 0)
    setSelectedContact((current) => {
      if (!current) return nextContacts[0] || null
      return nextContacts.find((contact) => contact.chat_id === current.chat_id) || nextContacts[0] || null
    })
  }, [])

  const loadJobs = useCallback(async () => {
    try {
      const response = await recallApi.jobs()
      setJobs(response.jobs || [])
    } catch (apiError) {
      setError(apiError instanceof Error ? apiError.message : 'Unable to load jobs.')
    }
  }, [])

  const reloadRuntimeData = useCallback(async (pathOverride?: RuntimePaths) => {
    const data = await recallApi.defaults()
    const nextPaths =
      pathOverride || (hasRuntimePath(paths) ? paths : readStoredPaths(pathsFromDefaults(data)))
    offerModels(data.modelChoices)
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
        offerModels(data.modelChoices)
        setDefaults(data)
        const nextPaths = readStoredPaths(pathsFromDefaults(data))
        setPaths(nextPaths)
        setReports(data.reports || [])
        setModel(preferredModel(data.models || [], data.defaultModel))
        setContactNameCount(data.contactNames?.count || 0)
        setSelectedReport(data.reports?.[0] || null)
        void loadJobs()

        // first run: no archive and no completed marker -> the wizard decides
        if (!data.setupCompleted && !data.hasMessages) {
          try {
            const status = await recallApi.setupStatus({ messagesPath: nextPaths.messagesPath })
            if (!cancelled) {
              if (status.state === 'ready') {
                markSetup({ completed: true })
              } else {
                setSetupStatus(status)
                setShowOnboarding(true)
              }
            }
          } catch {
            // never trap the user behind a failed probe
          }
        } else if (data.hasMessages && !data.setupCompleted) {
          // dev mode / data already exists: short-circuit future boots
          markSetup({ completed: true })
        }

        await loadContacts(data, nextPaths)
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
  }, [loadContacts, loadJobs])

  const refreshCacheStats = useCallback(() => setCacheStats(previewCacheStats()), [])
  const preview = useConversationPreview(
    {
      messagesPath: effectiveDefaults?.messagesPath ?? '',
      // the figures are worked out only where they are shown
      contact: activeView === 'analyze' ? (selectedContact?.chat_id ?? '') : '',
      model,
      since: scope.since,
      until: scope.until,
    },
    refreshCacheStats,
  )

  // the newest report asked for wins; one that finishes loading late is dropped
  const reportRequest = useRef(0)
  const openReport = useCallback(
    async (report: ReportFile) => {
      if (!effectiveDefaults || !model) return
      const request = ++reportRequest.current
      setSelectedReport(report)
      setActiveView('reports')
      setLoadingReport(true)
      setFailedReport('')
      setError(null)
      try {
        const data = await recallApi.analysis({
          messagesPath: effectiveDefaults.messagesPath,
          model,
          reportPath: report.path,
          contact: report.contact,
        })
        // an answer with no report in it would be asked for again and again
        if (!data.analysis) throw new Error('This report could not be read.')
        if (request === reportRequest.current) setOpened({ path: report.path, analysis: data.analysis })
      } catch (apiError) {
        if (request !== reportRequest.current) return
        setFailedReport(report.path)
        setError(apiError instanceof Error ? apiError.message : 'Unable to open report.')
      } finally {
        if (request === reportRequest.current) setLoadingReport(false)
      }
    },
    [effectiveDefaults, model],
  )

  // On Reports, the chosen report is opened if it is not yet: when the view
  // is first come to, or after the list changed under the choice.
  useEffect(() => {
    if (activeView !== 'reports' || !selectedReport || analysis || loadingReport) return
    if (failedReport === selectedReport.path) return
    let cancelled = false
    queueMicrotask(() => {
      if (!cancelled) void openReport(selectedReport)
    })
    return () => {
      cancelled = true
    }
  }, [activeView, analysis, failedReport, loadingReport, openReport, selectedReport])

  // a second press before the first has its answer starts nothing
  const startingReport = useRef(false)
  const runReport = useCallback(async () => {
    if (!effectiveDefaults || !selectedContact || !model || startingReport.current) return
    startingReport.current = true
    setError(null)
    try {
      const created = await recallApi.createAnalyzeJob({
        messagesPath: effectiveDefaults.messagesPath,
        outDir: effectiveDefaults.outDir,
        contact: selectedContact.chat_id,
        model,
        since: scope.since,
        until: scope.until,
      })
      setRunningJob(created.job)
      await loadJobs()
    } catch (apiError) {
      setError(apiError instanceof Error ? apiError.message : 'Unable to start report.')
    } finally {
      startingReport.current = false
    }
  }, [effectiveDefaults, loadJobs, model, scope.since, scope.until, selectedContact])

  // What a finished report job leaves: the list of reports, and the report
  // itself opened. `watched` says whether the job is still the one being
  // followed. When it is not, the lists are brought up to date and nothing is
  // opened over what the reader has gone on to.
  const showFinished = useCallback(
    async (job: Job, watched: () => boolean) => {
      try {
        const nextReports = job.result?.reports || (await recallApi.reports()).reports
        setReports(nextReports)
        await loadJobs()
        if (!watched()) return
        const written = job.result?.analysis
        const report = nextReports.find((item) => item.path === job.result?.reportPath)
        if (!written || !report) {
          // the job ended well and has nothing to show: say so, and leave the
          // report chosen, if it is among the reports, for Reports to read
          if (report) setSelectedReport(report)
          setError('The report finished, but it could not be opened. Look for it in Reports.')
          return
        }
        setOpened({ path: report.path, analysis: written })
        setSelectedReport(report)
        // only auto-open the report if the user is still where they
        // started it -- never yank them out of Chat/Search mid-thought
        setActiveView((current) => (current === 'analyze' || current === 'reports' ? 'reports' : current))
      } catch (apiError) {
        setError(apiError instanceof Error ? apiError.message : 'The report was written, but the list of reports could not be read.')
      }
    },
    [loadJobs],
  )

  // A report job is asked about until it ends: one question at a time. An
  // answer about a job that is no longer the one watched changes nothing, and
  // a job the server no longer knows stops being asked about. Once the watch
  // is over, an answer still on its way opens no report and changes no job.
  const watchedJob = runningJob && isActiveJob(runningJob) ? runningJob.id : ''
  const followReport = useCallback(
    async (job: Job, watched: () => boolean) => {
      if (job.status === 'completed') {
        await showFinished(job, watched)
        if (!watched()) return
      }
      setRunningJob((current) => (current?.id === job.id ? job : current))
    },
    [showFinished],
  )
  const reportLost = useCallback(() => {
    const lost = { status: 'failed' as const, error: 'The report stopped answering. Run it again.' }
    setRunningJob((current) => (current?.id === watchedJob ? { ...current, ...lost } : current))
  }, [watchedJob])
  useJobPoll(watchedJob, followReport, reportLost)

  const handleModelChange = useCallback((nextModel: string) => {
    storePreferredModel(nextModel)
    setModel(nextModel)
  }, [])

  const refreshContacts = useCallback(async () => {
    setUtilityBusy(true)
    setSettingsStatus('Reading contact names')
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
        messagesPath: nextPaths.messagesPath.trim(),
        outDir: nextPaths.outDir.trim(),
      }
      setUtilityBusy(true)
      setSettingsStatus('Applying file locations')
      setError(null)
      try {
        storeRuntimePaths(normalized)
        setPaths(normalized)
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

  const activeJobCount = useMemo(
    () => mergeJobs(jobs, runningJob).filter(isActiveJob).length,
    [jobs, runningJob],
  )
  const shellStyle = { '--rail-w': `${layout.sidebarWidth}px` } as CSSProperties
  // a narrow window folds the rail whatever was chosen; the choice comes back with the room
  const narrowWindow = useMediaQuery(NARROW_WINDOW)
  const railCollapsed = layout.sidebarCollapsed || narrowWindow
  const defaultSettingsPaths = useMemo(() => (defaults ? pathsFromDefaults(defaults) : emptyPaths), [defaults])
  const appliedSettingsPaths = useMemo(
    () => (effectiveDefaults ? pathsFromDefaults(effectiveDefaults) : paths),
    [effectiveDefaults, paths],
  )

  const selectContactForAnalyze = useCallback((contact: Contact) => {
    setSelectedContact(contact)
    setActiveView('analyze')
  }, [])

  const finishOnboarding = useCallback(
    async (result: SetupImportResult | null) => {
      // the first words follow the first setup, not each update
      const reveal = revealsFirstWords(rerunSetup, result)
      // no preview of the archive before the import is shown again, and no error from before it
      clearPreviewCache()
      setError(null)
      setShowOnboarding(false)
      setRerunSetup(false)
      setActiveView('home')
      // the archive the import just wrote; the paths in hand are from before it
      let written = paths.messagesPath
      try {
        const data = await reloadRuntimeData()
        written = written || data.messagesPath
      } catch {
        // the shell still renders; data loads on the next interaction
      }
      if (!reveal) return
      try {
        const { firstWords: payload } = await recallApi.firstWords({ messagesPath: written || undefined })
        if (payload.entries.length) {
          setFirstWords(payload)
          setShowFirstWords(true)
          markSetup({ firstWordsShown: true })
        }
      } catch {
        // the reveal is a nicety, never a blocker
      }
    },
    [paths.messagesPath, reloadRuntimeData, rerunSetup],
  )

  const skipOnboarding = useCallback(() => {
    setShowOnboarding(false)
    setRerunSetup(false)
  }, [])

  const replayFirstWords = useCallback(async () => {
    try {
      const { firstWords: payload } = await recallApi.firstWords({
        messagesPath: effectiveDefaults?.messagesPath || undefined,
      })
      if (payload.entries.length) {
        setFirstWords(payload)
        setShowFirstWords(true)
      }
    } catch (apiError) {
      setError(apiError instanceof Error ? apiError.message : 'Unable to load first words.')
    }
  }, [effectiveDefaults])

  // The setup guide builds the archive the first time and updates it after
  // that. Opening it only reads the current state; nothing changes until the
  // guide is told to start.
  const openSetup = useCallback(async () => {
    try {
      const status = await recallApi.setupStatus({ deep: true, messagesPath: effectiveDefaults?.messagesPath })
      setSetupStatus(status)
      setRerunSetup(isUpdate(status))
      setShowOnboarding(true)
    } catch (apiError) {
      setError(apiError instanceof Error ? apiError.message : 'Unable to start setup.')
    }
  }, [effectiveDefaults])

  if (building) return <ArchiveWait job={building} onReload={reloadRuntimeData} />

  if (showOnboarding && setupStatus) {
    return (
      <OnboardingWizard
        initialStatus={setupStatus}
        messagesPath={effectiveDefaults?.messagesPath ?? ''}
        rerun={rerunSetup}
        onFinished={(result) => void finishOnboarding(result)}
        onSkip={skipOnboarding}
      />
    )
  }

  return (
    <div className={`app${railCollapsed ? ' rail-collapsed' : ''}`} style={shellStyle}>
      <Sidebar
        activeView={activeView}
        activeJobCount={activeJobCount}
        collapsed={railCollapsed}
        canExpand={!narrowWindow}
        onToggleCollapse={toggleSidebar}
        onViewChange={setActiveView}
      />
      <ResizeHandle
        label="Resize sidebar"
        value={layout.sidebarWidth}
        min={RAIL_WIDTH.min}
        max={RAIL_WIDTH.max}
        disabled={railCollapsed}
        onChange={(sidebarWidth) => changeLayout({ sidebarWidth })}
      />

      <main className="workspace">
        <header className="titlebar">
          <h1 key={shellTitle} className="swap">
            {shellTitle}
          </h1>
        </header>

        {error ? (
          <div className="banner appear" role="alert">
            <p>{error}</p>
            <button type="button" className="icon-btn" aria-label="Dismiss" onClick={() => setError(null)}>
              <CloseIcon />
            </button>
          </div>
        ) : null}

        {activeView === 'home' ? (
          <OverviewView
            defaults={effectiveDefaults}
            archive={archive}
            contacts={contacts}
            reports={reports}
            loading={loadingContacts}
            model={model}
            onViewChange={setActiveView}
            onSelectContact={selectContactForAnalyze}
            onOpenReport={openReport}
            onReplayFirstWords={() => void replayFirstWords()}
            onOpenSetup={() => void openSetup()}
          />
        ) : activeView === 'reports' ? (
          <ReportsView
            reports={reports}
            selectedReport={selectedReport}
            analysis={analysis}
            loading={loadingReport}
            failed={Boolean(selectedReport) && failedReport === selectedReport?.path}
            listWidth={layout.reportListWidth}
            onListWidthChange={(reportListWidth) => changeLayout({ reportListWidth })}
            onSelectReport={openReport}
            onOpenConversation={(chatId, name) =>
              selectContactForAnalyze(
                contacts.find((contact) => contact.chat_id === chatId) ?? { chat_id: chatId, display_name: name },
              )
            }
            onChooseConversation={() => setActiveView('analyze')}
          />
        ) : activeView === 'explore' ? (
          <SearchView defaults={effectiveDefaults} contacts={contacts} onSelectContact={selectContactForAnalyze} />
        ) : activeView === 'ask' ? (
          <ChatView
            defaults={effectiveDefaults}
            model={model}
            contacts={contacts}
            onModelChange={handleModelChange}
            onSelectContact={selectContactForAnalyze}
            onOpenSetup={() => void openSetup()}
          />
        ) : activeView === 'jobs' ? (
          <JobsView
            jobs={jobs}
            runningJob={runningJob}
            contacts={contacts}
            reports={reports}
            onOpenReport={openReport}
            onRefresh={loadJobs}
          />
        ) : activeView === 'settings' ? (
          <SettingsView
            key={`${appliedSettingsPaths.messagesPath}\n${appliedSettingsPaths.outDir}`}
            defaults={effectiveDefaults}
            archive={archive}
            defaultPaths={defaultSettingsPaths}
            paths={appliedSettingsPaths}
            model={model}
            contactNameCount={contactNameCount}
            cacheStats={cacheStats}
            busy={utilityBusy}
            statusMessage={settingsStatus}
            onModelChange={handleModelChange}
            onApplyPaths={(next) => void applyRuntimePaths(next)}
            onRefreshContacts={() => void refreshContacts()}
            onClearPreviewCache={() => void clearPreviewCaches()}
            onOpenSetup={() => void openSetup()}
          />
        ) : (
          <AnalyzeView
            defaults={effectiveDefaults}
            contacts={contacts}
            selectedContact={selectedContact}
            loadingContacts={loadingContacts}
            query={query}
            onQueryChange={setQuery}
            onSelect={setSelectedContact}
            preview={preview.payload}
            previewSettled={preview.settled}
            previewError={preview.error}
            onRetryPreview={preview.retry}
            reports={reports}
            model={model}
            onModelChange={handleModelChange}
            scope={scope}
            onScopeChange={setScope}
            runningJob={runningJob}
            onRunReport={runReport}
            onOpenReport={openReport}
            listWidth={layout.listWidth}
            onListWidthChange={(listWidth) => changeLayout({ listWidth })}
          />
        )}
      </main>
      {showFirstWords && firstWords ? (
        <FirstWords payload={firstWords} onClose={() => setShowFirstWords(false)} />
      ) : null}
    </div>
  )
}

export default App
