import { useCallback, useEffect, useState } from 'react'
import { recallApi } from '../../lib/api'
import { formatDate, formatMoney, formatNumber } from '../../lib/format'
import type { SemanticStatus } from '../../types'

type MeaningIndexProps = {
  messagesPath: string
  // changes when the archive has been rebuilt, which puts the index out of date
  refreshKey: string
}

// while the index is being built its state is read again this often
const POLL_MS = 4000

// The index that lets a question find messages by what they mean and not
// only by their exact words. Building it sends the archive's text to the AI
// service and costs money, so the button carries the price.
export function MeaningIndex({ messagesPath, refreshKey }: MeaningIndexProps) {
  const [status, setStatus] = useState<SemanticStatus | null>(null)
  const [building, setBuilding] = useState(false)
  const [jobId, setJobId] = useState('')
  const [failure, setFailure] = useState('')

  const read = useCallback(async () => {
    if (!messagesPath) return
    try {
      const response = await recallApi.semanticStatus({ messagesPath })
      setStatus(response.semantic)
      if (response.semantic.state === 'fresh') setBuilding(false)
    } catch {
      // the row then shows no state, which is the truth
    }
  }, [messagesPath])

  // read when the row appears, and again whenever the archive has changed under the index
  useEffect(() => {
    if (!messagesPath) return
    let cancelled = false
    recallApi
      .semanticStatus({ messagesPath })
      .then((response) => {
        if (!cancelled) setStatus(response.semantic)
      })
      .catch(() => {
        // the row then shows no state, which is the truth
      })
    return () => {
      cancelled = true
    }
  }, [messagesPath, refreshKey])

  useEffect(() => {
    if (!building) return
    const timer = window.setInterval(() => {
      void read()
      if (!jobId) return
      recallApi
        .jobs()
        .then(({ jobs }) => {
          const job = jobs.find((item) => item.id === jobId)
          if (job?.status !== 'failed') return
          setBuilding(false)
          setFailure(job.error || 'The index could not be built')
        })
        .catch(() => {
          // the next round asks again
        })
    }, POLL_MS)
    return () => window.clearInterval(timer)
  }, [building, jobId, read])

  async function build() {
    if (!messagesPath || building) return
    setBuilding(true)
    setFailure('')
    try {
      const response = await recallApi.createSemanticJob({ messagesPath })
      setJobId(response.job.id)
    } catch (error) {
      setBuilding(false)
      setFailure(error instanceof Error ? error.message : 'The build could not be started')
    }
  }

  const state = status?.state
  const cost = status?.estimate?.estimatedCost
  const value = building
    ? 'Building'
    : state === 'fresh'
      ? `Up to date, ${formatNumber(status?.windows ?? 0)} passages${status?.builtAt ? `, built ${formatDate(status.builtAt)}` : ''}`
      : state === 'stale'
        ? 'Out of date since the archive changed'
        : state === 'none'
          ? 'Not built'
          : ''

  return (
    <div className="setting">
      <span className="setting-label">Meaning index</span>
      <span className="setting-value">
        {value}
        {failure ? <span className="setting-note bad">{failure}</span> : null}
      </span>
      {state && state !== 'fresh' ? (
        <button type="button" className="btn" disabled={!messagesPath || building} onClick={() => void build()}>
          {building ? <span className="spinner" /> : null}
          {building
            ? 'Building'
            : `${state === 'stale' ? 'Rebuild' : 'Build'}${typeof cost === 'number' ? `, about ${formatMoney(cost)}` : ''}`}
        </button>
      ) : null}
    </div>
  )
}
