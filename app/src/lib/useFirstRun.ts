import { useCallback, useEffect, useState } from 'react'
import type { FirstRunStatus, Job } from '../types'
import { recallApi } from './api'
import { chatPhase } from './firstRun.ts'
import { isActiveJob } from './jobs.ts'
import { useJobPoll } from './useJobPoll'

// Whether chat is ready, and the one job that gets it ready. The status is
// asked when this mounts and again when a job ends; while the job runs, the
// job is asked instead (the estimate plans again with every note it writes).
// A job already running when this mounts, from elsewhere, is followed too.
export function useFirstRun(messagesPath: string) {
  const [status, setStatus] = useState<FirstRunStatus | null>(null)
  // the status could not be read: an older server, or none
  const [failed, setFailed] = useState(false)
  const [job, setJob] = useState<Job | null>(null)
  const [starting, setStarting] = useState(false)
  // why a start was refused, in the server's one sentence
  const [refusal, setRefusal] = useState('')

  // the status, and the job that runs now if there is one
  const read = useCallback(async () => {
    const found = await recallApi.firstRun({ messagesPath: messagesPath || undefined })
    const running = found.jobId ? (await recallApi.job(found.jobId)).job : null
    return { found, running }
  }, [messagesPath])

  const take = useCallback((answer: { found: FirstRunStatus; running: Job | null }) => {
    setStatus(answer.found)
    setFailed(false)
    if (answer.running) setJob(answer.running)
  }, [])

  const load = useCallback(
    () =>
      read().then(take, () => {
        setFailed(true)
      }),
    [read, take],
  )

  useEffect(() => {
    let current = true
    read().then(
      (answer) => current && take(answer),
      () => current && setFailed(true),
    )
    return () => {
      current = false
    }
  }, [read, take])

  const onJob = useCallback(
    (next: Job) => {
      setJob(next)
      if (!isActiveJob(next)) void load()
    },
    [load],
  )
  // the server lost the job (it keeps jobs in memory): what is ready is read again
  const onLost = useCallback(() => {
    setJob(null)
    void load()
  }, [load])
  useJobPoll(job && isActiveJob(job) ? job.id : '', onJob, onLost)

  const start = useCallback(async () => {
    setRefusal('')
    setStarting(true)
    try {
      setJob((await recallApi.createFirstRunJob({ messagesPath: messagesPath || undefined })).job)
    } catch (error) {
      setRefusal(error instanceof Error ? error.message : 'Chat could not be got ready.')
    } finally {
      setStarting(false)
    }
  }, [messagesPath])

  return { status, failed, job, starting, refusal, phase: chatPhase(status, job), start, reload: load }
}
