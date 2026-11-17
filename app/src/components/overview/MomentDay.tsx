import { useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent } from 'react'
import { recallApi } from '../../lib/api'
import { conversationName, isGroupChat, isOutbound } from '../../lib/conversations'
import { plural } from '../../lib/format'
import type { DayMessages } from '../../types'
import { ChevronLeftIcon, ChevronRightIcon, CloseIcon } from '../Icons'
import { Step } from '../ui/Step'
import { Thread, type ThreadMessage } from '../ui/Thread'
import { Truncate } from '../ui/Truncate'

type MomentDayProps = {
  messagesPath: string
  chatId: string
  title: string
  // the day to open on, "YYYY-MM-DD"
  day: string
  // when the message this was opened for was sent; null to open at the day's start
  at: string | null
  onOpenConversation: () => void
  onClose: () => void
}

type DayLoad = { state: 'ready'; value: DayMessages } | { state: 'failed' }

// how many more messages of a long day one press brings
const PART = 200
const dayFormat = new Intl.DateTimeFormat(undefined, { weekday: 'short', month: 'short', day: 'numeric', year: 'numeric' })
const nameOf = (day: string) =>
  dayFormat.format(new Date(Number(day.slice(0, 4)), Number(day.slice(5, 7)) - 1, Number(day.slice(8, 10))))

function threadOf(value: DayMessages, group: boolean): ThreadMessage[] {
  return value.messages.map((message, index) => ({
    key: message.messageId || `${message.timestamp}-${value.offset + index}`,
    timestamp: message.timestamp,
    text: message.text,
    attachment: message.attachment || undefined,
    fromMe: message.isFromMe === null || message.isFromMe === undefined ? null : isOutbound(message),
    sender: group ? message.senderName || (message.sender ? conversationName('', message.sender) : undefined) : undefined,
  }))
}

// One day of one conversation, opened from a moment: the messages around the
// one that was quoted, with the time written after each pause. The arrows go
// to the nearest days that hold messages; a very long day opens around the
// quoted message and can be read further in both directions.
export function MomentDay({ messagesPath, chatId, title, day, at, onOpenConversation, onClose }: MomentDayProps) {
  const dialog = useRef<HTMLDialogElement>(null)
  const scroller = useRef<HTMLDivElement>(null)
  // the days already put in place, so a day is positioned once and then left to the reader
  const placed = useRef(new Set<string>())
  // how tall the day was before earlier messages were put above what is in view
  const heightBefore = useRef<number | null>(null)
  const [view, setView] = useState({ day, slide: 0 })
  const [days, setDays] = useState<Record<string, DayLoad>>({})
  const [attempt, setAttempt] = useState(0)
  const group = isGroupChat(chatId)
  const current = days[view.day]
  const value = current?.state === 'ready' ? current.value : null

  useEffect(() => {
    const element = dialog.current
    if (element && !element.open) element.showModal()
    scroller.current?.focus()
  }, [])

  // a day is read once; the day the sheet opened on is read around its message
  useEffect(() => {
    const wanted = view.day
    if (days[wanted]) return
    let cancelled = false
    recallApi
      .day({ messagesPath, contact: chatId, day: wanted, around: wanted === day ? (at ?? undefined) : undefined, limit: PART })
      .then(
        (loaded) => {
          if (!cancelled) setDays((held) => ({ ...held, [wanted]: { state: 'ready', value: loaded } }))
        },
        () => {
          if (!cancelled) setDays((held) => ({ ...held, [wanted]: { state: 'failed' } }))
        },
      )
    return () => {
      cancelled = true
    }
  }, [view.day, days, messagesPath, chatId, day, at, attempt])

  // A day that has just been drawn starts at the quoted message, or at its
  // top. Earlier messages added above keep what was in view where it was.
  useLayoutEffect(() => {
    const box = scroller.current
    if (!box || !value) return
    if (heightBefore.current !== null) {
      box.scrollTop += box.scrollHeight - heightBefore.current
      heightBefore.current = null
      return
    }
    if (placed.current.has(view.day)) return
    placed.current.add(view.day)
    const picked = box.querySelector<HTMLElement>(":scope .step > :not([data-step='out']) [data-picked]")
    if (!picked) {
      box.scrollTop = 0
      return
    }
    // Only the list is moved. Asking the message to scroll itself into view
    // would move the sheet as well, and take its head out of sight.
    const at = picked.getBoundingClientRect()
    const frame = box.getBoundingClientRect()
    box.scrollTop += at.top - frame.top - (frame.height - at.height) / 2
  }, [value, view.day])

  function step(by: number) {
    const to = by < 0 ? value?.previousDay : value?.nextDay
    if (to) setView({ day: to, slide: by })
  }

  function retry() {
    setDays((held) => Object.fromEntries(Object.entries(held).filter(([key]) => key !== view.day)))
    setAttempt((count) => count + 1)
  }

  // more of a long day: the part before what is shown, or the part after it
  function more(earlier: boolean) {
    if (!value) return
    const shownDay = view.day
    const from = earlier ? Math.max(0, value.offset - PART) : value.offset + value.messages.length
    const limit = earlier ? value.offset - from : PART
    recallApi.day({ messagesPath, contact: chatId, day: shownDay, offset: from, limit }).then(
      (part) => {
        if (earlier) heightBefore.current = scroller.current?.scrollHeight ?? null
        setDays((held) => {
          const had = held[shownDay]
          if (had?.state !== 'ready') return held
          const messages = earlier ? [...part.messages, ...had.value.messages] : [...had.value.messages, ...part.messages]
          return {
            ...held,
            [shownDay]: { state: 'ready', value: { ...had.value, offset: earlier ? part.offset : had.value.offset, messages } },
          }
        })
      },
      // the part already shown stays; the button is still there to press again
      () => undefined,
    )
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDialogElement>) {
    if (event.metaKey || event.ctrlKey || event.altKey) return
    if (event.key === 'ArrowLeft') step(-1)
    else if (event.key === 'ArrowRight') step(1)
    else return
    event.preventDefault()
  }

  return (
    <dialog
      ref={dialog}
      className="sheet dayview"
      aria-label={`${title}, ${nameOf(view.day)}`}
      onClose={onClose}
      onKeyDown={handleKeyDown}
    >
      <header className="sheet-head">
        <h2>
          <Truncate className="dayview-title">{title}</Truncate>
        </h2>
        <span className="sheet-actions">
          <button type="button" className="link" onClick={onOpenConversation}>
            Open conversation
          </button>
          <button type="button" className="icon-btn" aria-label="Close" onClick={() => dialog.current?.close()}>
            <CloseIcon />
          </button>
        </span>
      </header>

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
          <span key={view.day} className="stepper-value swap">
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

      <div ref={scroller} className="dayview-scroll" tabIndex={0} role="region" aria-label="Messages of this day">
        <Step
          id={view.day}
          slide={view.slide}
          render={(shown) => (
            <DayThread
              load={days[shown]}
              group={group}
              pickedAt={shown === day ? at : null}
              onMore={more}
              onRetry={retry}
            />
          )}
        />
      </div>
    </dialog>
  )
}

type DayThreadProps = {
  load: DayLoad | undefined
  group: boolean
  pickedAt: string | null
  onMore: (earlier: boolean) => void
  onRetry: () => void
}

function DayThread({ load, group, pickedAt, onMore, onRetry }: DayThreadProps) {
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
  const picked = pickedAt ? messages.find((message) => message.timestamp === pickedAt)?.key : undefined
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
