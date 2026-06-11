import { useCallback, useEffect, useState } from 'react'
import { recallApi } from '../../lib/api'
import { recallBridge } from '../../lib/bridge'
import type { Job, SetupImportResult, SetupStatus } from '../../types'
import {
  DbProblemStep,
  DoneStep,
  ImportStep,
  NoMessagesStep,
  OnboardingFrame,
  PermissionStep,
  WelcomeStep,
} from './OnboardingSteps'

// The wizard is a pure function of server state: no stored step index, every
// entry (cold boot, relaunch after FDA, re-run from Settings) re-derives its
// phase from a fresh /api/setup/status. That is what makes relaunch-resume free.

type Phase = 'welcome' | 'permission' | 'no_messages' | 'db_problem' | 'import' | 'done'

type Props = {
  initialStatus: SetupStatus
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

function frameStep(phase: Phase): number {
  if (phase === 'welcome') return 0
  if (phase === 'permission' || phase === 'no_messages' || phase === 'db_problem') return 1
  if (phase === 'import') return 2
  return 3
}

export function OnboardingWizard({ initialStatus, rerun = false, onFinished, onSkip }: Props) {
  const [phase, setPhase] = useState<Phase>('welcome')
  const [status, setStatus] = useState<SetupStatus>(initialStatus)
  const [pickedDbPath, setPickedDbPath] = useState(initialStatus.setup?.pickedDbPath || '')
  const [pickBusy, setPickBusy] = useState(false)
  const [pickError, setPickError] = useState('')
  const [job, setJob] = useState<Job | null>(null)
  const [result, setResult] = useState<SetupImportResult | null>(null)
  const [returnedFromSettings, setReturnedFromSettings] = useState(false)
  const [recheckCount, setRecheckCount] = useState(0)

  const probe = useCallback(
    async (deep = false): Promise<SetupStatus | null> => {
      try {
        const next = await recallApi.setupStatus({
          dbPath: pickedDbPath || undefined,
          deep,
        })
        setStatus(next)
        return next
      } catch {
        // the wizard must never trap the user behind a failed probe
        return null
      }
    },
    [pickedDbPath],
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
      setReturnedFromSettings(true)
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

  // import job polling
  useEffect(() => {
    if (!job?.id || (job.status !== 'queued' && job.status !== 'running')) return
    let stopped = false
    const interval = window.setInterval(async () => {
      try {
        const { job: next } = await recallApi.job(job.id)
        if (stopped) return
        setJob(next)
        if (next.status === 'completed') {
          setResult((next.result || null) as SetupImportResult | null)
          void recallApi.setupComplete({ completed: true })
          setPhase('done')
        }
      } catch {
        // jobs are in-memory; a restarted server means re-deriving from disk
        const next = await probe()
        if (stopped) return
        if (next?.export.exists) {
          void recallApi.setupComplete({ completed: true })
          setPhase('done')
        } else {
          setJob((current) =>
            current
              ? {
                  ...current,
                  status: 'failed',
                  error: 'The import was interrupted. Nothing was changed — try again.',
                }
              : current,
          )
        }
      }
    }, 800)
    return () => {
      stopped = true
      window.clearInterval(interval)
    }
  }, [job, probe])

  const start = useCallback(() => {
    const next = phaseForStatus(status)
    if (next === 'import') void probe(true)
    setPhase(next)
  }, [probe, status])

  const validatePath = useCallback(async (path: string) => {
    setPickError('')
    setPickBusy(true)
    try {
      const next = await recallApi.setupStatus({ dbPath: path, deep: true })
      if (next.db.status === 'readable') {
        setPickedDbPath(path)
        setStatus(next)
        void recallApi.setupComplete({ pickedDbPath: path })
        setPhase('import')
      } else if (next.db.status === 'empty') {
        setPickError('That database is there, but empty.')
      } else if (next.db.status === 'fda_blocked') {
        setPickError(
          'macOS blocked that folder. Approve the prompt it showed, or move the file to your home folder and choose it again.',
        )
      } else {
        setPickError(
          'That file isn’t a Messages database. Choose chat.db itself — not chat.db-wal or chat.db-shm.',
        )
      }
    } catch {
      setPickError('Could not check that file. Try again.')
    } finally {
      setPickBusy(false)
    }
  }, [])

  const pickFile = useCallback(async () => {
    const path = await recallBridge.pickDatabaseFile()
    if (path) await validatePath(path)
  }, [validatePath])

  const startImport = useCallback(async () => {
    setPickError('')
    try {
      const { job: created } = await recallApi.createSetupImportJob({
        dbPath: pickedDbPath || undefined,
        includeContacts: true,
      })
      setJob(created)
    } catch (error) {
      setJob({
        id: '',
        action: 'setup_import',
        status: 'failed',
        createdAt: '',
        updatedAt: '',
        logs: [],
        error: error instanceof Error ? error.message : 'Could not start the import.',
      })
    }
  }, [pickedDbPath])

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
    void recallApi.setupComplete({ skipped: true })
    onSkip()
  }, [onSkip])

  const finish = useCallback(() => {
    onFinished(result)
  }, [onFinished, result])

  const importBlocked = Boolean(job && (job.status === 'queued' || job.status === 'running'))

  return (
    <OnboardingFrame step={frameStep(phase)}>
      {phase === 'welcome' ? (
        <WelcomeStep rerun={rerun} onStart={start} onSkip={skip} />
      ) : phase === 'permission' ? (
        <PermissionStep
          returnedFromSettings={returnedFromSettings}
          pickBusy={pickBusy}
          pickError={pickError}
          onOpenSettings={() => recallBridge.openFullDiskAccess()}
          onRelaunch={() => {
            if (!importBlocked) recallBridge.relaunch()
          }}
          onPick={pickFile}
          onManualPath={validatePath}
          onSkip={skip}
        />
      ) : phase === 'no_messages' ? (
        <NoMessagesStep
          variant={status.db.status === 'empty' ? 'empty' : 'missing'}
          pickBusy={pickBusy}
          pickError={pickError}
          onRecheck={recheck}
          onPick={pickFile}
          onManualPath={validatePath}
          onSkip={skip}
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
          onBack={() => setPhase('welcome')}
        />
      ) : phase === 'import' ? (
        <ImportStep
          status={status}
          job={job}
          rerun={rerun}
          onStart={() => void startImport()}
          onRetry={retryImport}
          onBack={() => setPhase('welcome')}
        />
      ) : (
        <DoneStep result={result} rerun={rerun} onFinish={finish} />
      )}
    </OnboardingFrame>
  )
}
