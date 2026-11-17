import { useEffect } from 'react'
import { recallApi } from './api'
import { makePoller, type PollHandlers } from './jobPoll'

const POLL_MS = 1200
// this many unanswered questions in a row, and a job is taken to be lost
const LOST_AFTER_MISSES = 5

// Asks the server about a job until it ends, one question at a time. '' for
// the id watches nothing. Give callbacks that keep their identity
// (useCallback): the watch starts again when one of them changes.
export function useJobPoll(id: string, onJob: PollHandlers['onJob'], onLost: PollHandlers['onLost']) {
  useEffect(() => {
    if (!id) return
    const poller = makePoller({
      ask: () => recallApi.job(id).then((answer) => answer.job),
      onJob,
      onLost,
      lostAfter: LOST_AFTER_MISSES,
    })
    const timer = window.setInterval(() => void poller.tick(), POLL_MS)
    return () => {
      poller.stop()
      window.clearInterval(timer)
    }
  }, [id, onJob, onLost])
}
