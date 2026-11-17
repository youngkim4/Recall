// A job is asked about until it ends: one question at a time. This is the
// rule with no timer and no server in it; useJobPoll.ts gives it both.
import type { Job } from '../types'
import { isActiveJob } from './jobs.ts'

export type PollHandlers = {
  // asks the server about the job
  ask: () => Promise<Job>
  // What the server said. `watched` is false once the watch is over: work
  // that takes time checks it before it changes anything.
  onJob: (job: Job, watched: () => boolean) => void | Promise<void>
  // this many questions in a row went unanswered: the job is taken to be lost
  onLost: () => void
  lostAfter: number
}

export type Poller = {
  // one question, unless one is still out or the job has ended
  tick: () => Promise<void>
  // ends the watch: an answer still on its way changes nothing
  stop: () => void
}

export function makePoller({ ask, onJob, onLost, lostAfter }: PollHandlers): Poller {
  let asking = false
  let ended = false
  let stopped = false
  let misses = 0

  // the answer, or null for a question that went unanswered
  const answer = () => ask().catch(() => null)

  const tick = async () => {
    if (asking || ended || stopped) return
    asking = true
    try {
      const job = await answer()
      if (stopped) return
      if (!job) {
        misses += 1
        if (misses < lostAfter) return
        ended = true
        onLost()
        return
      }
      misses = 0
      ended = !isActiveJob(job)
      await onJob(job, () => !stopped)
    } finally {
      asking = false
    }
  }

  return {
    tick,
    stop: () => {
      stopped = true
    },
  }
}
