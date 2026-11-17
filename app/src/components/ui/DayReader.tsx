import { useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent } from 'react'
import { recallApi } from '../../lib/api'
import { conversationName, isGroupChat, isOutbound } from '../../lib/conversations'
import { PART, joinPart, partToAsk, quotedIndex } from '../../lib/dayParts'
import { plural } from '../../lib/format'
import type { DayMessages } from '../../types'
import { ChevronLeftIcon, ChevronRightIcon } from '../Icons'
import { Step } from './Step'
import { Thread, type ThreadMessage } from './Thread'

type DayReaderProps = {
  messagesPath: string
  chatId: string
  // the day to open on, "YYYY-MM-DD"
  day: string
  // when the message this was opened for was sent; null to open at the day's start
  at: string | null
  // that message's id, where the server gives one
  pickedId?: string
  // what the messages are called to assistive technology
  label: string
  // an id for the words that name the day in view, so a sheet can be named by them
  dayId?: string
  // the messages take the focus when the reader is put on the page
  focusOnOpen?: boolean
}

type DayLoad = { state: 'ready'; value: DayMessages } | { state: 'failed' }
type Quoted = { id: string | undefined; at: string | null }

const dayFormat = new Intl.DateTimeFormat(undefined, { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' })
const nameOf = (day: string) =>
  dayFormat.format(new Date(Number(day.slice(0, 4)), Number(day.slice(5, 7)) - 1, Number(day.slice(8, 10))))

function threadOf(value: DayMessages, group: boolean): ThreadMessage[] {
  return value.messages.map((message, index) => ({
    key: message.messageId || `${message.timestamp}-${value.offset + index}`,
    // a message of a day has a time: the day was found by it
    timestamp: message.timestamp ?? '',
    text: message.text,
    attachment: message.attachment || undefined,
    fromMe: message.isFromMe === null || message.isFromMe === undefined ? null : isOutbound(message),
    sender: group ? message.senderName || (message.sender ? conversationName('', message.sender) : undefined) : undefined,
  }))
}

// how tall the day in view is; the day on its way out is not counted
function shownHeight(box: HTMLElement | null) {
  return box?.querySelector<HTMLElement>(":scope .step > :not([data-step='out'])")?.offsetHeight ?? 0
}

// One day of one conversation: the messages around the one that was asked
// for, with the time written after each pause. The arrows, and the arrow
// keys, go to the nearest days that hold messages. A very long day opens
// around its message and can be read further in both directions. It stands
// in a sheet or in a pane; to open it on another message, give it a new key.
export function DayReader({ messagesPath, chatId, day, at, pickedId, label, dayId, focusOnOpen = false }: DayReaderProps) {
  const scroller = useRef<HTMLDivElement>(null)
  // the days asked for, answered or not, so that a day is read once
  const asked = useRef(new Set<string>())
  // the parts of long days being read: one at a time for each side of a day
  const asking = useRef(new Set<string>())
  // the day in view, for an answer that arrives after the reader has moved on
  const inView = useRef(day)
  // the day the messages were last put in place for
  const drawn = useRef<string | null>(null)
  // where the reader left each day
  const places = useRef(new Map<string, number>())
  // how tall the day in view was when earlier messages were on their way to it
  const heightBefore = useRef<{ day: string; offset: number; height: number } | null>(null)
  // the control the keyboard was on when it was pressed; it can be gone once its answer is drawn
  const pressed = useRef<Element | null>(null)
  const [view, setView] = useState({ day, slide: 0 })
  const [days, setDays] = useState<Record<string, DayLoad>>({})
  const [attempt, setAttempt] = useState(0)
  const group = isGroupChat(chatId)
  const current = days[view.day]
  const value = current?.state === 'ready' ? current.value : null

  useEffect(() => {
    if (focusOnOpen) scroller.current?.focus()
  }, [focusOnOpen])

  // A day is read once, and a day being read is left to arrive when the
  // reader moves on. The day the reader opened on is read around its message.
  useEffect(() => {
    const wanted = view.day
    if (asked.current.has(wanted)) return
    asked.current.add(wanted)
    const answer = (load: DayLoad) => setDays((held) => ({ ...held, [wanted]: load }))
    recallApi
      .day({ messagesPath, contact: chatId, day: wanted, around: wanted === day ? (at ?? undefined) : undefined, limit: PART })
      .then(
        (loaded) => answer({ state: 'ready', value: loaded }),
        () => answer({ state: 'failed' }),
      )
  }, [view.day, messagesPath, chatId, day, at, attempt])

  // A day that has just been drawn starts at the message it was opened for,
  // or at its top. A day the reader comes back to is where they left it.
  // Earlier messages put above what is shown keep it where it was.
  useLayoutEffect(() => {
    const box = scroller.current
    if (!box || !value) return
    const before = heightBefore.current
    heightBefore.current = null
    if (drawn.current === view.day) {
      if (before?.day === view.day && value.offset < before.offset) box.scrollTop += shownHeight(box) - before.height
      return
    }
    drawn.current = view.day
    const left = places.current.get(view.day)
    if (left !== undefined) {
      box.scrollTop = left
      return
    }
    const picked = box.querySelector<HTMLElement>(":scope .step > :not([data-step='out']) [data-picked]")
    if (!picked) {
      box.scrollTop = 0
      return
    }
    // Only the list is moved. Asking the message to scroll itself into view
    // would move what the list stands in as well. A sheet still growing into
    // place is drawn smaller than it is laid out, so what it measures on the
    // screen is brought back to its own size first.
    const spot = picked.getBoundingClientRect()
    const frame = box.getBoundingClientRect()
    const drawnAt = frame.width / box.offsetWidth || 1
    box.scrollTop += (spot.top - frame.top) / drawnAt - (box.offsetHeight - spot.height / drawnAt) / 2
  }, [value, view.day])

  // A button that was pressed can leave the page with its answer: Earlier at
  // the start of the day, Later at its end, Try again once the day is asked
  // for. The focus it held goes to the messages, unless it has moved on.
  useLayoutEffect(() => {
    const button = pressed.current
    pressed.current = null
    if (!button || button.isConnected) return
    const active = document.activeElement
    if (!active || active === document.body || !active.isConnected) scroller.current?.focus({ preventScroll: true })
  }, [current])

  function notePressed() {
    const active = document.activeElement
    pressed.current = active && scroller.current?.contains(active) ? active : null
  }

  function step(by: number) {
    const to = by < 0 ? value?.previousDay : value?.nextDay
    if (!to) return
    const box = scroller.current
    // only a day that was put in place has a place to come back to
    if (box && drawn.current === view.day) places.current.set(view.day, box.scrollTop)
    inView.current = to
    setView({ day: to, slide: by })
  }

  function retry() {
    notePressed()
    asked.current.delete(view.day)
    setDays((held) => Object.fromEntries(Object.entries(held).filter(([key]) => key !== view.day)))
    setAttempt((count) => count + 1)
  }

  // more of a long day: the part before what is shown, or the part after it
  function more(earlier: boolean) {
    if (!value) return
    const shownDay = view.day
    const shownFrom = value.offset
    const ask = partToAsk(value, earlier)
    const side = `${shownDay} ${earlier ? 'earlier' : 'later'}`
    if (!ask) return
    notePressed()
    if (asking.current.has(side)) return
    asking.current.add(side)
    recallApi
      .day({ messagesPath, contact: chatId, day: shownDay, ...ask })
      .then(
        (part) => {
          // only the day in view is on the page to be kept in place
          if (earlier && inView.current === shownDay) {
            heightBefore.current = { day: shownDay, offset: shownFrom, height: shownHeight(scroller.current) }
          }
          setDays((held) => {
            const had = held[shownDay]
            if (had?.state !== 'ready') return held
            const joined = joinPart(had.value, part, earlier)
            return joined === had.value ? held : { ...held, [shownDay]: { state: 'ready', value: joined } }
          })
        },
        // the part already shown stays; the button is still there to press again
        () => undefined,
      )
      .finally(() => {
        asking.current.delete(side)
      })
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.metaKey || event.ctrlKey || event.altKey) return
    if (event.key === 'ArrowLeft') step(-1)
    else if (event.key === 'ArrowRight') step(1)
    else return
    event.preventDefault()
  }

  return (
    <div className="dayview-reader" onKeyDown={handleKeyDown}>
      <div className="dayview-bar">
        <span className="stepper">
          <button
            type="button"
            className="icon-btn sm"
            aria-label="The day with messages before this one"
            aria-disabled={!value?.previousDay}
            onClick={() => step(-1)}
          >
            <ChevronLeftIcon />
          </button>
          <span key={view.day} id={dayId} className="stepper-value swap">
            {nameOf(view.day)}
          </span>
          <button
            type="button"
            className="icon-btn sm"
            aria-label="The day with messages after this one"
            aria-disabled={!value?.nextDay}
            onClick={() => step(1)}
          >
            <ChevronRightIcon />
          </button>
        </span>
        <span key={`${view.day} ${value?.total ?? ''}`} className="dayview-count swap">
          {value ? plural(value.total, 'message') : ''}
        </span>
      </div>

      <div ref={scroller} className="dayview-scroll" tabIndex={0} role="region" aria-label={label}>
        <Step
          id={view.day}
          slide={view.slide}
          render={(shown) => (
            <DayThread
              load={days[shown]}
              group={group}
              quoted={shown === day ? { id: pickedId, at } : null}
              onMore={more}
              onRetry={retry}
            />
          )}
        />
      </div>
    </div>
  )
}

type DayThreadProps = {
  load: DayLoad | undefined
  group: boolean
  // the message the reader was opened for, on the day that holds it
  quoted: Quoted | null
  onMore: (earlier: boolean) => void
  onRetry: () => void
}

function DayThread({ load, group, quoted, onMore, onRetry }: DayThreadProps) {
  if (!load) {
    return (
      <div className="thread" aria-hidden="true">
        {[62, 38, 54].map((width, index) => (
          <div key={index} className={index === 1 ? 'msg you' : 'msg'}>
            <span className="skeleton dayview-wait" style={{ width: `${width}%` }} />
          </div>
        ))}
      </div>
    )
  }
  if (load.state === 'failed') {
    return (
      <p className="dayview-note" role="alert">
        This day could not be read.{' '}
        <button type="button" className="link in-text" onClick={onRetry}>
          Try again
        </button>
      </p>
    )
  }
  const { value } = load
  if (!value.total) return <p className="dayview-note">No messages on this day</p>
  const messages = threadOf(value, group)
  const picked = quoted ? messages[quotedIndex(value.messages, quoted.id, quoted.at)]?.key : undefined
  const shownUpTo = value.offset + value.messages.length
  return (
    <>
      {value.offset > 0 ? (
        <button type="button" className="btn quiet dayview-more" onClick={() => onMore(true)}>
          Earlier that day
        </button>
      ) : null}
      <Thread messages={messages} times pickedKey={picked} />
      {shownUpTo < value.total ? (
        <button type="button" className="btn quiet dayview-more" onClick={() => onMore(false)}>
          Later that day
        </button>
      ) : null}
    </>
  )
}
