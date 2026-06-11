import { useEffect, useState } from 'react'
import type { FirstWordsPayload } from '../types'

// Full-screen gentle reveal: the first text ever exchanged with your top
// people. Auto-advances; click anywhere (or the dots) to move at your pace.

type Props = {
  payload: FirstWordsPayload
  onClose: () => void
}

const INTRO_MS = 2600
const PERSON_MS = 5200

function prefersReducedMotion(): boolean {
  try {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches
  } catch {
    return false
  }
}

function formatLongDate(timestamp: string): string {
  const date = new Date(timestamp.replace(' ', 'T'))
  if (Number.isNaN(date.getTime())) return timestamp.slice(0, 10)
  return date.toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' })
}

// remounted per entry via the parent page's key, so initial state is enough
function Typewriter({ text, instant }: { text: string; instant: boolean }) {
  const [shown, setShown] = useState(instant ? text.length : 0)
  useEffect(() => {
    if (instant) return
    // ~28ms per character, but never longer than 2s for the whole line
    const interval = Math.min(28, Math.max(8, 2000 / Math.max(1, text.length)))
    const timer = window.setInterval(() => {
      setShown((current) => {
        if (current >= text.length) {
          window.clearInterval(timer)
          return current
        }
        return current + 1
      })
    }, interval)
    return () => window.clearInterval(timer)
  }, [text, instant])
  return <span>{text.slice(0, shown)}</span>
}

export function FirstWordsOverlay({ payload, onClose }: Props) {
  const entries = payload.entries
  const pages = entries.length + 2 // intro + people + closing
  const [page, setPage] = useState(0)
  const [paused, setPaused] = useState(false)
  const [instant] = useState(prefersReducedMotion)

  useEffect(() => {
    if (paused || page >= pages - 1) return
    const wait = page === 0 ? INTRO_MS : PERSON_MS
    const timer = window.setTimeout(() => setPage((current) => Math.min(current + 1, pages - 1)), wait)
    return () => window.clearTimeout(timer)
  }, [page, pages, paused])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
      if (event.key === 'ArrowRight' || event.key === ' ') {
        setPaused(true)
        setPage((current) => Math.min(current + 1, pages - 1))
      }
      if (event.key === 'ArrowLeft') {
        setPaused(true)
        setPage((current) => Math.max(current - 1, 0))
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, pages])

  const advance = () => {
    setPaused(true)
    setPage((current) => Math.min(current + 1, pages - 1))
  }

  const yearsSpan =
    payload.totals.firstYear && payload.totals.lastYear
      ? payload.totals.lastYear - payload.totals.firstYear
      : null

  const closing = page === pages - 1
  const entry = page >= 1 && page <= entries.length ? entries[page - 1] : null

  return (
    <div className="fw-overlay" role="dialog" aria-label="First words">
      <button type="button" className="fw-skip" onClick={onClose}>
        Skip
      </button>
      <div className="fw-stage" onClick={closing ? undefined : advance}>
        {page === 0 ? (
          <div className="fw-page fw-intro" key="intro">
            {yearsSpan ? (
              <div className="fw-kicker">Your archive goes back {yearsSpan} years</div>
            ) : (
              <div className="fw-kicker">From the very beginning</div>
            )}
            <h1 className="fw-display">These were the first words.</h1>
          </div>
        ) : entry ? (
          <div className="fw-page fw-person" key={entry.chatId}>
            <div className="fw-ghost-number" aria-hidden="true">
              {entry.yearsAgo || ''}
            </div>
            <div className="fw-kicker">
              {entry.person} · {formatLongDate(entry.timestamp)}
              {entry.yearsAgo > 0
                ? ` · ${entry.yearsAgo} ${entry.yearsAgo === 1 ? 'year' : 'years'} ago`
                : ''}
            </div>
            <div className="fw-direction">{entry.direction === 'outgoing' ? 'you wrote' : 'they wrote'}</div>
            <blockquote className="fw-quote">
              <Typewriter text={entry.text} instant={instant} />
            </blockquote>
            {entry.reply ? (
              <div className={`fw-reply${instant ? '' : ' fw-reply-fade'}`}>
                <span className="fw-reply-direction">
                  {entry.reply.direction === 'outgoing' ? 'you replied' : 'they replied'}
                </span>
                <span className="fw-reply-text">{entry.reply.text}</span>
              </div>
            ) : null}
          </div>
        ) : (
          <div className="fw-page fw-closing" key="closing">
            <h1 className="fw-display">
              {yearsSpan ? `${yearsSpan} years. ` : ''}
              {entries.length} {entries.length === 1 ? 'person' : 'people'}.{' '}
              {payload.totals.messages.toLocaleString()} messages.
            </h1>
            <p className="fw-closing-line">They are all here.</p>
            <div className="fw-actions">
              <button type="button" className="button primary" onClick={onClose}>
                Explore your archive
              </button>
              <button
                type="button"
                className="button ghost"
                onClick={() => {
                  setPaused(false)
                  setPage(0)
                }}
              >
                Watch again
              </button>
            </div>
          </div>
        )}
      </div>
      <div className="fw-dots">
        {Array.from({ length: pages }, (_, index) => (
          <button
            key={index}
            type="button"
            className={`fw-dot${index === page ? ' is-active' : ''}`}
            aria-label={`Page ${index + 1}`}
            onClick={() => {
              setPaused(true)
              setPage(index)
            }}
          />
        ))}
      </div>
    </div>
  )
}
