import { useCallback, useEffect, useState, type ReactNode } from 'react'
import type { Defaults, Job, PreviewCacheSummary, RuntimePaths, SemanticStatus } from '../types'
import { formatNumber, shortDateTime } from '../lib/format'
import { modelLabel } from '../lib/models'
import { recallApi } from '../lib/api'

const LIVE_DB_PATH = '~/Library/Messages/chat.db'

type SettingsViewProps = {
  defaults: Defaults | null
  defaultPaths: RuntimePaths
  paths: RuntimePaths
  model: string
  contactNameCount: number
  cacheStats: PreviewCacheSummary
  utilityJob: Job | null
  utilityBusy: boolean
  statusMessage: string
  onModelChange: (model: string) => void
  onApplyPaths: (paths: RuntimePaths) => Promise<void>
  onRefreshContacts: () => Promise<void>
  onRefreshExport: () => Promise<void>
  onClearPreviewCache: () => Promise<void>
}

export function SettingsView({
  defaults,
  defaultPaths,
  paths,
  model,
  contactNameCount,
  cacheStats,
  utilityJob,
  utilityBusy,
  statusMessage,
  onModelChange,
  onApplyPaths,
  onRefreshContacts,
  onRefreshExport,
  onClearPreviewCache,
}: SettingsViewProps) {
  const [draftPaths, setDraftPaths] = useState(paths)
  const normalizedDraftPaths = normalizePaths(draftPaths)
  const contactUpdated = defaults?.contactNames?.exportedAt || defaults?.contactNames?.updatedAt || ''
  const cacheUpdated = cacheStats.lastSavedAt
  const utilityFailed = utilityJob?.status === 'failed'
  const utilityRunning = utilityJob?.status === 'queued' || utilityJob?.status === 'running'
  const utilityMessage = utilityRunning
    ? lastLog(utilityJob?.logs) || statusMessage
    : statusMessage || lastLog(utilityJob?.logs)
  const pathsChanged =
    normalizedDraftPaths.dbPath !== paths.dbPath ||
    normalizedDraftPaths.messagesPath !== paths.messagesPath ||
    normalizedDraftPaths.outDir !== paths.outDir
  const canApplyPaths = Boolean(
    normalizedDraftPaths.dbPath && normalizedDraftPaths.messagesPath && normalizedDraftPaths.outDir,
  )

  return (
    <section className="settings-view" aria-label="Settings">
      <div className="settings-header">
        <div>
          <h2>Local setup</h2>
        </div>
      </div>

      <div className="settings-table">
        <PathRow
          label="Messages database"
          description="chat.db rebuilt into the export. Point at your live database for current messages."
          value={draftPaths.dbPath}
          placeholder={defaultPaths.dbPath || '/Users/.../Library/Messages/chat.db'}
          onChange={(value) => setDraftPaths((current) => ({ ...current, dbPath: value }))}
          hint={
            draftPaths.dbPath.trim() === LIVE_DB_PATH ? (
              <span className="settings-hint-note">Live database — needs Full Disk Access for this app, then Apply paths + Refresh.</span>
            ) : (
              <button
                type="button"
                className="settings-hint-button"
                onClick={() => setDraftPaths((current) => ({ ...current, dbPath: LIVE_DB_PATH }))}
              >
                Use live Messages database
              </button>
            )
          }
        />
        <PathRow
          label="Message export"
          description="CSV used for previews, estimates, and reports."
          value={draftPaths.messagesPath}
          placeholder={defaultPaths.messagesPath || '/path/to/messages.csv'}
          onChange={(value) => setDraftPaths((current) => ({ ...current, messagesPath: value }))}
        />
        <PathRow
          label="Reports folder"
          description="Where generated Markdown and HTML reports are saved."
          value={draftPaths.outDir}
          placeholder={defaultPaths.outDir || '/path/to/out'}
          onChange={(value) => setDraftPaths((current) => ({ ...current, outDir: value }))}
        />
        <SettingsRow
          label="Contacts"
          value={`${formatNumber(contactNameCount || defaults?.contactNames?.count || 0)} names${
            contactUpdated ? ` / ${shortDateTime(contactUpdated)}` : ''
          }`}
          state={contactNameCount || defaults?.contactNames?.count ? 'Ready' : 'Optional'}
          good={Boolean(contactNameCount || defaults?.contactNames?.count)}
        />
        <SettingsRow
          label="Preview cache"
          value={`${formatNumber(cacheStats.count)} conversations${
            cacheUpdated ? ` / ${shortDateTime(cacheUpdated)}` : ''
          }`}
          state={cacheStats.count ? 'Warm' : 'Empty'}
          good={Boolean(cacheStats.count)}
        />
        <div className="settings-row">
          <div>
            <strong>Model</strong>
            <span>Used for estimates and new reports.</span>
          </div>
          <div className="settings-control">
            <select value={model} onChange={(event) => onModelChange(event.target.value)}>
              {(defaults?.models || [model]).map((option) => (
                <option key={option} value={option}>
                  {modelLabel(option)}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="settings-actions">
        <button
          type="button"
          className="button secondary"
          disabled={utilityBusy || !pathsChanged || !canApplyPaths}
          onClick={() => {
            setDraftPaths(normalizedDraftPaths)
            void onApplyPaths(normalizedDraftPaths)
          }}
        >
          Apply paths
        </button>
        <button
          type="button"
          className="button ghost"
          disabled={utilityBusy}
          onClick={() => setDraftPaths(defaultPaths)}
        >
          Reset defaults
        </button>
        <button
          type="button"
          className="button ghost"
          disabled={utilityBusy}
          onClick={() => {
            void onRefreshContacts()
          }}
        >
          Refresh contacts
        </button>
        <button
          type="button"
          className="button ghost"
          disabled={utilityBusy || !paths.dbPath || !paths.messagesPath}
          onClick={() => {
            void onRefreshExport()
          }}
        >
          Refresh message export
        </button>
        <button
          type="button"
          className="button ghost"
          onClick={() => {
            void onClearPreviewCache()
          }}
        >
          Clear preview cache
        </button>
      </div>

      {statusMessage || utilityJob ? (
        <div className={`job-banner ${utilityFailed ? 'failed' : utilityRunning ? 'running' : ''}`}>
          <strong>{utilityFailed ? 'Maintenance failed' : utilityRunning ? 'Working' : 'Status'}</strong>
          <span>{utilityJob?.error || utilityMessage}</span>
        </div>
      ) : null}

      <SemanticIndexCard
        messagesPath={paths.messagesPath || defaults?.messagesPath || ''}
        refreshKey={utilityJob?.status === 'completed' ? utilityJob.id : ''}
      />
    </section>
  )
}

function SemanticIndexCard({ messagesPath, refreshKey }: { messagesPath: string; refreshKey: string }) {
  const [status, setStatus] = useState<SemanticStatus | null>(null)
  const [building, setBuilding] = useState(false)
  const [buildJobId, setBuildJobId] = useState('')
  const [note, setNote] = useState('')

  const refresh = useCallback(async () => {
    if (!messagesPath) return
    try {
      const response = await recallApi.semanticStatus({ messagesPath })
      setStatus(response.semantic)
      if (response.semantic.state === 'fresh') setBuilding(false)
    } catch {
      // status is best-effort; the card just shows unknown
    }
  }, [messagesPath])

  // refetch on mount AND whenever a maintenance job (refresh export) finishes
  // -- that's the moment the index silently goes stale
  useEffect(() => {
    if (!messagesPath) return
    let cancelled = false
    recallApi
      .semanticStatus({ messagesPath })
      .then((response) => {
        if (!cancelled) setStatus(response.semantic)
      })
      .catch(() => {
        // status is best-effort; the card just shows unknown
      })
    return () => {
      cancelled = true
    }
  }, [messagesPath, refreshKey])

  useEffect(() => {
    if (!building) return
    const timer = window.setInterval(() => {
      void refresh()
      if (buildJobId) {
        recallApi
          .jobs()
          .then(({ jobs }) => {
            const job = jobs.find((item) => item.id === buildJobId)
            if (job?.status === 'failed') {
              setBuilding(false)
              setNote(job.error || 'The index build failed — see Jobs for details.')
            }
          })
          .catch(() => {
            // polling is best-effort
          })
      }
    }, 4000)
    return () => window.clearInterval(timer)
  }, [building, buildJobId, refresh])

  async function build() {
    if (!messagesPath || building) return
    setBuilding(true)
    setNote('Building — progress in Jobs.')
    try {
      const response = await recallApi.createSemanticJob({ messagesPath })
      setBuildJobId(response.job.id)
    } catch (error) {
      setBuilding(false)
      setNote(error instanceof Error ? error.message : 'Could not start the build.')
    }
  }

  const state = status?.state ?? 'none'
  const cost = status?.estimate?.estimatedCost
  const statusLine =
    state === 'fresh'
      ? `Up to date — ${formatNumber(status?.windows || 0)} moments indexed${status?.builtAt ? `, built ${shortDateTime(status.builtAt)}` : ''}.`
      : state === 'stale'
        ? 'Out of date — the message export changed since the last build. Chat falls back to keyword search until rebuilt.'
        : 'Not built yet — chat retrieves by keywords only. Build it so questions match by meaning, not exact words.'

  return (
    <div className="semantic-card">
      <div className="semantic-card-main">
        <h3>Semantic memory index</h3>
        <p>{statusLine}</p>
        {note ? <p className="semantic-note">{note}</p> : null}
      </div>
      {state === 'fresh' && !building ? (
        <span className="status-pill good">Indexed</span>
      ) : (
        <button
          type="button"
          className="button secondary"
          disabled={!messagesPath || building}
          onClick={() => void build()}
        >
          {building
            ? 'Building…'
            : `${state === 'stale' ? 'Rebuild' : 'Build'} index${typeof cost === 'number' ? ` (~$${cost.toFixed(2)})` : ''}`}
        </button>
      )}
    </div>
  )
}

function normalizePaths(paths: RuntimePaths) {
  return {
    dbPath: paths.dbPath.trim(),
    messagesPath: paths.messagesPath.trim(),
    outDir: paths.outDir.trim(),
  }
}

function PathRow({
  label,
  description,
  value,
  placeholder,
  onChange,
  hint,
}: {
  label: string
  description: string
  value: string
  placeholder: string
  onChange: (value: string) => void
  hint?: ReactNode
}) {
  return (
    <label className="settings-row path-row">
      <div>
        <strong>{label}</strong>
        <span>{description}</span>
        {hint ? <div className="settings-hint">{hint}</div> : null}
      </div>
      <input
        type="text"
        spellCheck={false}
        value={value}
        placeholder={placeholder}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  )
}

function SettingsRow({
  label,
  value,
  state,
  good,
}: {
  label: string
  value: string
  state: string
  good: boolean
}) {
  return (
    <div className="settings-row">
      <div>
        <strong>{label}</strong>
        <span>{value}</span>
      </div>
      <span className={`status-pill ${good ? 'good' : ''}`}>{state}</span>
    </div>
  )
}

function lastLog(logs: Array<string | { message?: string }> = []) {
  for (let index = logs.length - 1; index >= 0; index -= 1) {
    const entry = logs[index]
    const message = typeof entry === 'string' ? entry : entry?.message || ''
    if (message && !message.startsWith('127.0.0.1 - -')) return message
  }
  return ''
}
