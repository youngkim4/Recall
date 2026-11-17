import { useEffect, useRef, useState, type CSSProperties } from 'react'
import { conversationName } from '../../lib/conversations'
import { formatLongDate, formatNumber, plural } from '../../lib/format'
import type { FirstWordsEntry, FirstWordsPayload } from '../../types'
import { CloseIcon, PauseIcon, PlayIcon } from '../Icons'

type FirstWordsProps = {
  payload: FirstWordsPayload
  onClose: () => void
}

// how long each part stays before the next one follows on its own
const INTRO_MS = 2800
const EXCHANGE_MS = 6500

const prefersStill = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches

// The first message ever exchanged with the people written to most, one
// person at a time, each arriving the way a message does. It moves on by
// itself until the reader takes over: a click or the arrow keys step through
// it, and it then waits for them; the control beside Close holds it still or
// sets it going again. With reduced motion it never starts by itself.
export function FirstWords({ payload, onClose }: FirstWordsProps) {
  const dialog = useRef<HTMLDialogElement>(null)
  const { entries, totals } = payload
  // an opening, one part a person, and a closing
  const parts = entries.length + 2
  const [part, setPart] = useState(0)
  const [auto, setAuto] = useState(() => !prefersStill())
  const [still] = useState(prefersStill)
  const last = parts - 1
  const duration = part === 0 ? INTRO_MS : EXCHANGE_MS

  useEffect(() => {
    const element = dialog.current
    if (element && !element.open) element.showModal()
  }, [])

  useEffect(() => {
    if (!auto || part >= last) return
    const timer = window.setTimeout(() => setPart((current) => Math.min(current + 1, last)), duration)
    return () => window.clearTimeout(timer)
  }, [auto, part, last, duration])

  function go(to: number) {
    setAuto(false)
    setPart(Math.max(0, Math.min(last, to)))
  }

  const years = totals.firstYear && totals.lastYear ? totals.lastYear - totals.firstYear : 0
  const entry = part >= 1 && part <= entries.length ? entries[part - 1] : null

  return (
    <dialog
      ref={dialog}
      className="story"
      aria-label="First words"
      // takes the focus when the button that had it goes away
      tabIndex={-1}
      onClose={onClose}
      onKeyDown={(event) => {
        // a shortcut of the system's is not a step
        if (event.metaKey || event.ctrlKey || event.altKey) return
        // Space belongs to the button the keyboard is on
        const onButton = event.target instanceof Element && event.target.closest('button') !== null
        if (event.key === 'ArrowRight' || (event.key === ' ' && !onButton)) {
          event.preventDefault()
          go(part + 1)
        } else if (event.key === 'ArrowLeft') {
          event.preventDefault()
          go(part - 1)
        }
      }}
    >
      <header className="story-bar">
        <ol className="story-progress" role="list">
          {Array.from({ length: parts }, (_, index) => (
            <li key={index} data-state={index < part ? 'done' : index === part ? 'now' : undefined}>
              <button
                type="button"
                aria-label={`Part ${index + 1} of ${parts}`}
                aria-current={index === part ? 'step' : undefined}
                onClick={() => go(index)}
              >
                {/* the part on show fills as its time runs, while it is moving on by itself */}
                <i
                  key={`${index === part} ${auto}`}
                  data-running={(index === part && auto && part < last) || undefined}
                  style={{ '--ms': `${duration}ms` } as CSSProperties}
                />
              </button>
            </li>
          ))}
        </ol>
        {part < last ? (
          <button
            type="button"
            className="icon-btn"
            aria-label={auto ? 'Pause' : 'Play'}
            onClick={() => setAuto((moving) => !moving)}
          >
            <span key={String(auto)} className="icon-swap">
              {auto ? <PauseIcon /> : <PlayIcon />}
            </span>
          </button>
        ) : null}
        <button type="button" className="icon-btn" aria-label="Close" onClick={() => dialog.current?.close()}>
          <CloseIcon />
        </button>
      </header>

      {/* a press anywhere on the page moves on, until the last part */}
      <div
        className="story-stage"
        aria-live="polite"
        aria-atomic="true"
        onClick={part < last ? () => go(part + 1) : undefined}
      >
        {part === 0 ? (
          <div key="opening" className="story-page">
            <p className="story-kicker">
              {years > 0 ? `Your archive goes back ${plural(years, 'year')}` : 'From the very beginning'}
            </p>
            <h1 className="story-display">The first words</h1>
          </div>
        ) : entry ? (
          <Exchange key={entry.chatId} entry={entry} still={still} />
        ) : (
          <div key="closing" className="story-page">
            <h1 className="story-display">
              {[
                years > 0 ? plural(years, 'year') : '',
                plural(entries.length, 'person', 'people'),
                `${formatNumber(totals.messages)} messages`,
              ]
                .filter(Boolean)
                .join(', ')}
            </h1>
            <div className="story-actions">
              <button type="button" className="btn primary lg" autoFocus onClick={() => dialog.current?.close()}>
                Open the archive
              </button>
              <button
                type="button"
                className="btn quiet lg"
                onClick={() => {
                  setPart(0)
                  setAuto(!still)
                  // this button goes away with the last part; the focus stays in the story
                  dialog.current?.focus()
                }}
              >
                Watch again
              </button>
            </div>
          </div>
        )}
      </div>
    </dialog>
  )
}

function Exchange({ entry, still }: { entry: FirstWordsEntry; still: boolean }) {
  const ago = entry.yearsAgo > 0 ? `${plural(entry.yearsAgo, 'year')} ago` : ''
  return (
    <div className="story-page exchange" data-still={still || undefined}>
      <p className="exchange-who">{conversationName(entry.person, entry.chatId)}</p>
      <div className="thread">
        <p className="stamp">{[formatLongDate(entry.timestamp), ago].filter(Boolean).join(', ')}</p>
        <Arrival mine={entry.direction === 'outgoing'} text={entry.text} order={0} />
        {entry.reply ? <Arrival mine={entry.reply.direction === 'outgoing'} text={entry.reply.text} order={1} /> : null}
      </div>
    </div>
  )
}

// One message arriving. Theirs is announced by the three dots, as it was when
// it was first sent; yours rises into place.
function Arrival({ mine, text, order }: { mine: boolean; text: string; order: number }) {
  return (
    <div className={`msg turn arrival${mine ? ' you' : ''}`} style={{ '--order': order } as CSSProperties}>
      {mine ? null : (
        <span className="typing arrival-typing" aria-hidden="true">
          <i />
          <i />
          <i />
        </span>
      )}
      <span className={`bubble lg tail arrival-bubble${mine ? ' you' : ''}`}>
        {mine ? <span className="sr-only">You: </span> : null}
        {text}
      </span>
    </div>
  )
}
