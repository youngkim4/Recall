import { useCallback, useState } from 'react'
import type { Job } from '../types'
import { errorSentence } from './apiError'
import { BUILD_LOST, buildEnd, buildFailure } from './archiveBuild'
import { useJobPoll } from './useJobPoll'

const NO_ANSWER = 'Recall did not answer.'

export type ArchiveBuild = {
  // the load, as the server last described it
  job: Job
  // why the wait ended with no archive; '' while it goes on
  failure: string
  // the data of the app is being loaded again
  loading: boolean
  // loads the data again, after a failure
  again: () => void
}

// Follows the job that loads the archive. `first` is the job as the server
// named it; `reload` loads the data of the app again and fails with the
// reason when it cannot. When the job completes, the data is loaded again.
// The wait itself ends outside: the caller stops showing it once the data
// is there.
export function useArchiveBuild(first: Job, reload: () => Promise<unknown>): ArchiveBuild {
  const [job, setJob] = useState(first)
  const [failure, setFailure] = useState(() => (buildEnd(first) === 'failed' ? buildFailure(first) : ''))
  const [loading, setLoading] = useState(false)

  // the server began another load: that one is followed, from its start
  const [followed, setFollowed] = useState(first.id)
  if (followed !== first.id) {
    setFollowed(first.id)
    setJob(first)
    setFailure('')
  }

  const load = useCallback(async () => {
    setLoading(true)
    try {
      await reload()
    } catch (error) {
      setFailure(errorSentence(error, NO_ANSWER))
    } finally {
      setLoading(false)
    }
  }, [reload])

  const onJob = useCallback(
    (next: Job) => {
      setJob(next)
      const end = buildEnd(next)
      if (end === 'ready') void load()
      if (end === 'failed') setFailure(buildFailure(next))
    },
    [load],
  )
  const onLost = useCallback(() => setFailure(BUILD_LOST), [])
  useJobPoll(failure || loading ? '' : job.id, onJob, onLost)

  const again = useCallback(() => {
    setFailure('')
    void load()
  }, [load])

  return { job, failure, loading, again }
}
