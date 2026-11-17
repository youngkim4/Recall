import { useCallback, useEffect, useRef, useState } from 'react'
import { recallApi } from '../../lib/api'
import { recallBridge } from '../../lib/bridge'
import { importRequest } from '../../lib/setupGuide'
import type { Job, SetupImportResult, SetupStatus } from '../../types'
import { ImportStep } from './ImportStep'
import { SetupFrame } from './SetupFrame'
import { DbProblemStep, DoneStep, NoMessagesStep, PermissionStep, WelcomeStep } from './SetupSteps'

// The wizard is a pure function of server state: no stored step index, every
// entry (cold boot, relaunch after FDA, re-run from Settings) re-derives its
// phase from a fresh /api/setup/status. That is what makes relaunch-resume free.

type Phase = 'welcome' | 'permission' | 'no_messages' | 'db_problem' | 'import' | 'done'

// this many unanswered questions in a row, and the import is taken to be lost
const LOST_AFTER_MISSES = 4

// A marker of how far setup got. One that fails to save changes nothing now:
// the same thing is worked out again from the files at the next launch.
function markSetup(fields: Parameters<typeof recallApi.setupComplete>[0]) {
  recallApi.setupComplete(fields).catch(() => undefined)
}

type Props = {
  initialStatus: SetupStatus
  // the archive file that the app reads: an import writes this one
  messagesPath: string
  rerun?: boolean
  onFinished: (result: SetupImportResult | null) => void
  onSkip: () => void
}

function phaseForStatus(status: SetupStatus): Phase {
  switch (status.state) {
    case 'needs_permission':
      return 'permission'
    case 'no_messages':
      return 'no_messages'
    case 'needs_export':
    case 'ready':
      return 'import'
    default:
      return 'db_problem'
  }
}

// An update is of an archive that is already there, so the state of setup
// says nothing about it. What there is to do is read from the database:
// where Messages cannot be read, the update says so before it is started.
function phaseForDatabase(status: SetupStatus): Phase {
  switch (status.db.status) {
    case 'readable':
      return 'import'
    case 'fda_blocked':
      return 'permission'
    case 'missing':
    case 'empty':
      return 'no_messages'
    default:
      return 'db_problem'
  }
}

function frameStep(phase: Phase): number {
  if (phase === 'welcome') return 0
  if (phase === 'permission' || phase === 'no_messages' || phase === 'db_problem') return 1
  if (phase === 'import') return 2
  return 3
}

export function OnboardingWizard({ initialStatus, messagesPath, rerun = false, onFinished, onSkip }: Props) {
  // an update has no welcome to read: it opens on what there is to do
  const [phase, setPhase] = useState<Phase>(() => (rerun ? phaseForDatabase(initialStatus) : 'welcome'))
  const [status, setStatus] = useState<SetupStatus>(initialStatus)
  // A copy chosen in this guide, for the import it starts. It is not kept:
  // the next time the guide opens, it reads Messages itself again.
  const [pickedDbPath, setPickedDbPath] = useState('')
  const [pickBusy, setPickBusy] = useState(false)
  const [pickError, setPickError] = useState('')
  const [job, setJob] = useState<Job | null>(null)
  const [result, setResult] = useState<SetupImportResult | null>(null)
  const [returnedFromSettings, setReturnedFromSettings] = useState(false)
  const [recheckCount, setRecheckCount] = useState(0)
  // System Settings was opened from here; coming back to the window means something only then
  const wentToSettings = useRef(false)

  const probe = useCallback(
    async (deep = false): Promise<SetupStatus | null> => {
      try {
        const next = await recallApi.setupStatus({
          dbPath: pickedDbPath || undefined,
          messagesPath: messagesPath || undefined,
          deep,
        })
        setStatus(next)
        return next
      } catch {
        // the wizard must never trap the user behind a failed probe
        return null
      }
    },
    [messagesPath, pickedDbPath],
  )

  // permission phase: poll quietly; the shell pings recall:becameActive the
  // moment the user returns from System Settings
  useEffect(() => {
    if (phase !== 'permission') return
    let stopped = false
    const tick = async () => {
      const next = await probe()
      if (stopped || !next) return
      if (next.db.status === 'readable') {
        void probe(true)
        setPhase('import')
      } else if (next.db.status === 'empty') {
        setPhase('no_messages')
      }
    }
    const interval = window.setInterval(tick, 1500)
    const onActive = () => {
      if (wentToSettings.current) setReturnedFromSettings(true)
      void tick()
    }
    window.addEventListener('recall:becameActive', onActive)
    window.addEventListener('focus', onActive)
    return () => {
      stopped = true
      window.clearInterval(interval)
      window.removeEventListener('recall:becameActive', onActive)
      window.removeEventListener('focus', onActive)
    }
  }, [phase, probe])

  // The import is asked about until it ends. One unanswered question is not
  // an ending; several in a row mean the server was restarted, and jobs are
  // kept in memory, so the files are looked at instead.
  const watched = job?.id && (job.status === 'queued' || job.status === 'running') ? job.id : ''
  useEffect(() => {
    if (!watched) return
    let stopped = false
    let asking = false
    let misses = 0
    const interrupted = () =>
      setJob((current) => (current ? { ...current, status: 'failed', error: 'The import was interrupted.' } : current))
    const ask = async () => {
      if (asking) return
      asking = true
      try {
        const { job: next } = await recallApi.job(watched)
        if (stopped) return
        misses = 0
        setJob(next)
        if (next.status === 'completed') {
          setResult((next.result || null) as SetupImportResult | null)
          markSetup({ completed: true })
          setPhase('done')
        }
      } catch {
        misses += 1
        if (stopped || misses < LOST_AFTER_MISSES) return
        // An update starts with an archive already there, so the file being
        // there says nothing about this run; only a first import can be read that way.
        const next = rerun ? null : await probe()
        if (stopped) return
        if (next?.export.exists) {
          markSetup({ completed: true })
          setPhase('done')
        } else {
          interrupted()
        }
      } finally {
        asking = false
      }
    }
    const interval = window.setInterval(() => void ask(), 800)
    return () => {
      stopped = true
      window.clearInterval(interval)
    }
  }, [watched, probe, rerun])

  const start = useCallback(() => {
    const next = phaseForStatus(status)
    if (next === 'import') void probe(true)
    setPhase(next)
  }, [probe, status])

  const validatePath = useCallback(async (path: string) => {
    setPickError('')
    setPickBusy(true)
    try {
      const next = await recallApi.setupStatus({ dbPath: path, messagesPath: messagesPath || undefined, deep: true })
      if (next.db.status === 'readable') {
        setPickedDbPath(path)
        setStatus(next)
        setPhase('import')
      } else if (next.db.status === 'empty') {
        setPickError('That database has nothing in it.')
      } else if (next.db.status === 'fda_blocked') {
        setPickError(
          'macOS blocked that folder. Allow it in the prompt macOS showed, or move the file to your home folder and choose it again.',
        )
      } else {
        setPickError('That file is not a Messages database. Choose chat.db, not chat.db-wal or chat.db-shm.')
      }
    } catch {
      setPickError('That file could not be checked.')
    } finally {
      setPickBusy(false)
    }
  }, [messagesPath])

  const pickFile = useCallback(async () => {
    const path = await recallBridge.pickDatabaseFile()
    if (path) await validatePath(path)
  }, [validatePath])

  const startImport = useCallback(async () => {
    setPickError('')
    try {
      const request = importRequest({ archivePath: messagesPath, copyPath: pickedDbPath })
      const { job: created } = await recallApi.createSetupImportJob(request)
      setJob(created)
    } catch (error) {
      setJob({
        id: '',
        action: 'setup_import',
        status: 'failed',
        createdAt: '',
        updatedAt: '',
        logs: [],
        error: error instanceof Error ? error.message : 'The import could not be started.',
      })
    }
  }, [messagesPath, pickedDbPath])

  const retryImport = useCallback(() => {
    setJob(null)
    void probe(true)
  }, [probe])

  const recheck = useCallback(async () => {
    setRecheckCount((count) => count + 1)
    const next = await probe(true)
    if (next && next.db.status === 'readable') setPhase('import')
  }, [probe])

  const skip = useCallback(() => {
    markSetup({ skipped: true })
    onSkip()
  }, [onSkip])

  // Leaving an update goes back to the app and records nothing: setup was
  // finished before. Leaving the first setup is recorded as skipped.
  const leave = rerun ? onSkip : skip

  const finish = useCallback(() => {
    onFinished(result)
  }, [onFinished, result])

  const importBlocked = Boolean(job && (job.status === 'queued' || job.status === 'running'))

  return (
    <SetupFrame step={rerun ? null : frameStep(phase)} phase={phase}>
      {phase === 'welcome' ? (
        <WelcomeStep onStart={start} onSkip={skip} />
      ) : phase === 'permission' ? (
        <PermissionStep
          returnedFromSettings={returnedFromSettings}
          pickBusy={pickBusy}
          pickError={pickError}
          onOpenSettings={() => {
            wentToSettings.current = true
            recallBridge.openFullDiskAccess()
          }}
          onRelaunch={() => {
            if (!importBlocked) recallBridge.relaunch()
          }}
          onPick={pickFile}
          onManualPath={validatePath}
          onSkip={leave}
        />
      ) : phase === 'no_messages' ? (
        <NoMessagesStep
          variant={status.db.status === 'empty' ? 'empty' : 'missing'}
          pickBusy={pickBusy}
          pickError={pickError}
          onRecheck={recheck}
          onPick={pickFile}
          onManualPath={validatePath}
          onSkip={leave}
          stillNothing={recheckCount > 0}
        />
      ) : phase === 'db_problem' ? (
        <DbProblemStep
          variant={status.db.status === 'locked' ? 'locked' : 'invalid'}
          detail={status.db.detail || ''}
          pickBusy={pickBusy}
          pickError={pickError}
          onRetry={() => void recheck()}
          onPick={pickFile}
          onManualPath={validatePath}
          // an update was opened from the app, and leaving it goes back there
          onBack={rerun ? onSkip : () => setPhase('welcome')}
        />
      ) : phase === 'import' ? (
        <ImportStep
          status={status}
          job={job}
          rerun={rerun}
          onStart={() => void startImport()}
          onRetry={retryImport}
          onBack={rerun ? onSkip : () => setPhase('welcome')}
        />
      ) : (
        <DoneStep result={result} rerun={rerun} onFinish={finish} />
      )}
    </SetupFrame>
  )
}
