import { useCallback, useEffect, useRef, useState } from 'react'
import { recallApi } from '../../lib/api'
import { recallBridge } from '../../lib/bridge'
import { isActiveJob } from '../../lib/jobs'
import {
  canImport,
  foundTheSame,
  frameStep,
  guideStep,
  importRequest,
  importView,
  interruptedReason,
  noNotes,
  noted,
  notStartedJob,
  whyNotThisCopy,
  type GuideStage,
  type StepNotes,
} from '../../lib/setupGuide'
import type { Job, SetupImportResult, SetupStatus } from '../../types'
import { ImportStep } from './ImportStep'
import { SetupFrame } from './SetupFrame'
import { DbProblemStep, DoneStep, NoMessagesStep, PermissionStep, WelcomeStep } from './SetupSteps'

// The guide keeps no step. What shows is worked out each time from what is
// known (guideStep): how far the reader has gone, the last check of Messages,
// a copy if one was chosen, and the import if one was started. Every entry
// (cold boot, relaunch after access was given, an update from the app) starts
// from a fresh /api/setup/status, and that is what makes carrying on after a
// relaunch free.

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

export function OnboardingWizard({ initialStatus, messagesPath, rerun = false, onFinished, onSkip }: Props) {
  // an update has no welcome to read: it opens on what there is to do
  const [stage, setStage] = useState<GuideStage>(rerun ? 'source' : 'welcome')
  // the last check of Messages itself
  const [live, setLive] = useState<SetupStatus>(initialStatus)
  // A copy chosen in this guide: the path the reader gave, and what its check
  // found. It is for the import that the next step offers, and it goes when
  // that step is left: the guide then reads Messages itself again.
  const [picked, setPicked] = useState<{ path: string; found: SetupStatus } | null>(null)
  const [pickBusy, setPickBusy] = useState(false)
  const [job, setJob] = useState<Job | null>(null)
  const [result, setResult] = useState<SetupImportResult | null>(null)
  // The request for an import is on its way. Until it is answered there is
  // no job to show, and the step cannot be left: the import may have begun.
  const [starting, setStarting] = useState(false)
  // System Settings was opened from here; coming back to the window means something only then
  const wentToSettings = useRef(false)
  // the newest check wins; an older answer that comes late is dropped
  const checks = useRef(0)
  // a second press before the first has its answer starts nothing
  const asked = useRef(false)

  const step = guideStep({ stage, job, picked: picked?.found ?? null, live })
  const archivePath = messagesPath.trim() || undefined

  // What the reader was told at the step that shows. When the step changes,
  // it is gone: a note of one step is never read at another.
  const [notes, setNotes] = useState<StepNotes>(() => noNotes(step))
  if (notes.step !== step) setNotes(noNotes(step))
  const told = notes.step === step ? notes : noNotes(step)
  const tell = useCallback(
    (change: Partial<Omit<StepNotes, 'step'>>) => setNotes((current) => noted(current, step, change)),
    [step],
  )

  const check = useCallback(
    async (deep = false): Promise<SetupStatus | null> => {
      const mine = ++checks.current
      try {
        const next = await recallApi.setupStatus({ messagesPath: archivePath, deep })
        if (mine === checks.current) setLive(next)
        return next
      } catch {
        // the guide must never trap the reader behind a check that failed
        return null
      }
    },
    [archivePath],
  )

  // The step follows what a check finds. Where that is a database to import,
  // its size is asked for too.
  const checkThenSize = useCallback(async () => {
    const next = await check()
    if (next && canImport(next.db.status)) void check(true)
  }, [check])

  // Where access is asked for, Messages is looked at again and again; the
  // shell says recall:becameActive the moment the reader is back from System
  // Settings. One look at a time, so that each answer is the newest.
  useEffect(() => {
    if (step !== 'permission') return
    let looking = false
    const look = async () => {
      if (looking) return
      looking = true
      try {
        await checkThenSize()
      } finally {
        looking = false
      }
    }
    const interval = window.setInterval(() => void look(), 1500)
    const onActive = () => {
      if (wentToSettings.current) tell({ returnedFromSettings: true })
      void look()
    }
    window.addEventListener('recall:becameActive', onActive)
    window.addEventListener('focus', onActive)
    return () => {
      window.clearInterval(interval)
      window.removeEventListener('recall:becameActive', onActive)
      window.removeEventListener('focus', onActive)
      // the trip to System Settings belongs to this visit of the step
      wentToSettings.current = false
    }
  }, [step, checkThenSize, tell])

  // the archive is there: setup is recorded as done, and the copy has served
  const ended = useCallback(() => {
    markSetup({ completed: true })
    setPicked(null)
    setStage('done')
  }, [])

  // What the server says of the import, taken as it is. An import that has
  // ended well ends the guide, whichever answer says so first.
  const follow = useCallback(
    (next: Job) => {
      setJob(next)
      if (next.status !== 'completed') return
      setResult((next.result || null) as SetupImportResult | null)
      ended()
    },
    [ended],
  )

  // The import is asked about until it ends. One unanswered question is not
  // an ending; several in a row mean the server was restarted, and jobs are
  // kept in memory, so the files are looked at instead.
  const watched = job?.id && isActiveJob(job) ? job.id : ''
  useEffect(() => {
    if (!watched) return
    let stopped = false
    let asking = false
    let misses = 0
    const ask = async () => {
      if (asking) return
      asking = true
      try {
        const { job: next } = await recallApi.job(watched)
        if (stopped) return
        misses = 0
        follow(next)
      } catch {
        misses += 1
        if (stopped || misses < LOST_AFTER_MISSES) return
        // An update starts with an archive already there, so the file being
        // there says nothing about this run; only a first import can be read that way.
        const next = rerun ? null : await check()
        if (stopped) return
        if (next?.export.holdsMessages) {
          ended()
        } else {
          setJob((current) => (current ? { ...current, status: 'failed', error: interruptedReason(rerun) } : current))
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
  }, [watched, check, ended, follow, rerun])

  // the welcome was read at launch; Messages is looked at as it is now
  const start = useCallback(() => {
    setStage('source')
    void checkThenSize()
  }, [checkThenSize])

  const validatePath = useCallback(
    async (path: string) => {
      tell({ pickError: '' })
      setPickBusy(true)
      try {
        const found = await recallApi.setupStatus({ dbPath: path, messagesPath: archivePath, deep: true })
        const against = whyNotThisCopy(found.db.status)
        if (against) tell({ pickError: against })
        else setPicked({ path, found })
      } catch {
        tell({ pickError: 'That file could not be checked.' })
      } finally {
        setPickBusy(false)
      }
    },
    [archivePath, tell],
  )

  const pickFile = useCallback(async () => {
    const path = await recallBridge.pickDatabaseFile()
    if (path) await validatePath(path)
  }, [validatePath])

  // allowRemoval: the reader read how many messages the update removes, and said to go on
  const startImport = useCallback(async (allowRemoval = false) => {
    if (asked.current) return
    asked.current = true
    setStarting(true)
    try {
      const request = importRequest({ archivePath: messagesPath, copyPath: picked?.path ?? '', rerun, allowRemoval })
      const { job: created } = await recallApi.createSetupImportJob(request)
      follow(created)
    } catch (error) {
      setJob(notStartedJob(error, rerun))
    } finally {
      asked.current = false
      setStarting(false)
    }
  }, [follow, messagesPath, picked, rerun])

  // Messages is looked at again before the next try. A copy is tried as it was chosen.
  const retryImport = useCallback(() => {
    setJob(null)
    if (!picked) void check(true)
  }, [check, picked])

  // "Still nothing" is said only of an answer, and only when it is the same answer
  const recheck = useCallback(async () => {
    const before = live.db.status
    const next = await check(true)
    tell({ sameAgain: foundTheSame(before, next) })
  }, [check, live.db.status, tell])

  const skip = useCallback(() => {
    markSetup({ skipped: true })
    onSkip()
  }, [onSkip])

  // Leaving an update goes back to the app and records nothing: setup was
  // finished before. Leaving the first setup is recorded as skipped.
  const leave = rerun ? onSkip : skip

  // Back from the import. A copy is given up, and the guide reads Messages
  // itself again. With no copy, an update goes back to the app and a first
  // setup to its welcome.
  const back = useCallback(() => {
    setJob(null)
    if (picked) {
      setPicked(null)
      void checkThenSize()
    } else if (!rerun) {
      setStage('welcome')
    } else if (job) {
      // An update was started, and how it ended is not known for sure. The
      // app reads the archive again, with no first words to show.
      onFinished(null)
    } else {
      onSkip()
    }
  }, [checkThenSize, job, onFinished, onSkip, picked, rerun])

  const finish = useCallback(() => {
    onFinished(result)
  }, [onFinished, result])

  return (
    <SetupFrame step={rerun ? null : frameStep(step)} phase={step}>
      {step === 'welcome' ? (
        <WelcomeStep onStart={start} onSkip={skip} />
      ) : step === 'permission' ? (
        <PermissionStep
          rerun={rerun}
          returnedFromSettings={told.returnedFromSettings}
          pickBusy={pickBusy}
          pickError={told.pickError}
          onOpenSettings={() => {
            wentToSettings.current = true
            recallBridge.openFullDiskAccess()
          }}
          onRelaunch={() => recallBridge.relaunch()}
          onPick={pickFile}
          onManualPath={validatePath}
          onSkip={leave}
        />
      ) : step === 'no_messages' ? (
        <NoMessagesStep
          rerun={rerun}
          variant={live.db.status === 'empty' ? 'empty' : 'missing'}
          pickBusy={pickBusy}
          pickError={told.pickError}
          onRecheck={() => void recheck()}
          onPick={pickFile}
          onManualPath={validatePath}
          onSkip={leave}
          stillNothing={told.sameAgain}
        />
      ) : step === 'db_problem' ? (
        <DbProblemStep
          rerun={rerun}
          variant={live.db.status === 'invalid' ? 'invalid' : 'unread'}
          detail={live.db.detail || ''}
          pickBusy={pickBusy}
          pickError={told.pickError}
          onRetry={() => void recheck()}
          onPick={pickFile}
          onManualPath={validatePath}
          // an update was opened from the app, and leaving it goes back there
          onBack={rerun ? onSkip : () => setStage('welcome')}
        />
      ) : step === 'done' ? (
        <DoneStep result={result} rerun={rerun} onFinish={finish} />
      ) : (
        <ImportStep
          view={importView(step) ?? 'ready'}
          status={picked?.found ?? live}
          job={job}
          rerun={rerun}
          starting={starting}
          onStart={() => void startImport()}
          onRetry={retryImport}
          onAllowRemoval={() => void startImport(true)}
          onBack={back}
        />
      )}
    </SetupFrame>
  )
}
