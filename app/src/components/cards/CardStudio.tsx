import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { recallApi } from '../../lib/api'
import { copyCardPng, exportCardPng, renderCardToCanvas, type CardData, type CardFormat } from '../../lib/cardCanvas'
import { rovingKeyDown } from '../../lib/roving'
import { useLoad } from '../../lib/useLoad'
import type { Contact, Defaults, MemoriesPayload, PreviewPayload } from '../../types'
import { CloseIcon } from '../Icons'
import { ChoiceMenu } from '../ui/ChoiceMenu'
import { Truncate } from '../ui/Truncate'
import { withFigures } from './figures'
import { cardMoments, quoteLabel, type CardMoment } from './moments'

type CardStudioProps = {
  defaults: Defaults | null
  model: string
  contacts: Contact[]
  // today's moments, already loaded by the page that opens this
  memories: MemoriesPayload | null
  onClose: () => void
}

const GROUPS: Array<CardMoment['group']> = ['Anniversaries', 'On this day', 'You and them']
const SHAPES = [
  { value: 'story', label: 'Tall', note: 'For a story, 9 by 16' },
  { value: 'square', label: 'Square', note: 'For a post or a message' },
]
// how long the button goes on saying that the picture was copied
const COPIED_MS = 2000

const loadFirstWords = (messagesPath: string) =>
  recallApi.firstWords({ messagesPath }).then((response) => response.firstWords)

// Pick a moment, see the card, copy it or save it as a picture. The card is
// drawn here, on this Mac, from the conversation's own figures. A quoted
// message is someone's words, so it is left off the card until the box is
// ticked.
export function CardStudio({ defaults, model, contacts, memories, onClose }: CardStudioProps) {
  const dialog = useRef<HTMLDialogElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  // the conversations whose figures have been asked for
  const asked = useRef(new Set<string>())
  const messagesPath = defaults?.messagesPath ?? ''
  const { load: firstWords } = useLoad(messagesPath, loadFirstWords)
  const [chosenKey, setChosenKey] = useState('')
  const [format, setFormat] = useState<CardFormat>('story')
  // the card whose message was asked onto it; a tick never carries over to another card
  const [quoteFor, setQuoteFor] = useState('')
  // each conversation's figures, once they are read; null when they could not be
  const [figures, setFigures] = useState<Record<string, PreviewPayload | null>>({})
  // the card on show: the chosen one, once what goes on it is known
  const [shown, setShown] = useState<{ key: string; data: CardData } | null>(null)
  const [copied, setCopied] = useState(false)
  const [failure, setFailure] = useState('')

  const moments = useMemo(
    () => cardMoments(memories, firstWords.state === 'ready' ? firstWords.value : null, contacts),
    [memories, firstWords, contacts],
  )
  const chosen = moments.find((moment) => moment.key === chosenKey) ?? moments[0] ?? null
  const chatId = chosen?.chatId ?? ''
  const canRead = Boolean(chatId && messagesPath && model)
  const known = chatId in figures ? figures[chatId] : undefined

  useEffect(() => {
    const element = dialog.current
    if (element && !element.open) element.showModal()
  }, [])

  // a conversation's figures are asked for once, however many of its cards are looked at
  useEffect(() => {
    if (!canRead || asked.current.has(chatId)) return
    asked.current.add(chatId)
    recallApi
      .preview({ messagesPath, contact: chatId, model })
      .then((preview) => setFigures((current) => ({ ...current, [chatId]: preview })))
      // the card is still true without them; it only says less
      .catch(() => setFigures((current) => ({ ...current, [chatId]: null })))
  }, [canRead, chatId, messagesPath, model])

  // The chosen card with its figures on it, once they are in. Until then the
  // card before it stays up, so a card is never seen half made.
  const ready = useMemo(
    () => (chosen && (!canRead || known !== undefined) ? withFigures(chosen.data, known ?? null) : null),
    [chosen, canRead, known],
  )
  if (chosen && ready && (shown?.key !== chosen.key || shown.data !== ready)) {
    setShown({ key: chosen.key, data: ready })
  }

  const waiting = Boolean(chosen) && shown?.key !== chosen?.key
  const shownMoment = moments.find((moment) => moment.key === shown?.key) ?? null
  const withQuote = Boolean(shown) && quoteFor === shown?.key
  const asksFor = shown ? quoteLabel(shown.data) : ''

  // drawn before the frame is shown, so a card that has just arrived is never seen empty
  useLayoutEffect(() => {
    if (canvas.current && shown) renderCardToCanvas(canvas.current, shown.data, { format, includeQuote: withQuote })
  }, [shown, format, withQuote])

  useEffect(() => {
    if (!copied) return
    const timer = window.setTimeout(() => setCopied(false), COPIED_MS)
    return () => window.clearTimeout(timer)
  }, [copied])

  function choose(moment: CardMoment) {
    setChosenKey(moment.key)
    // each card starts without the message on it
    setQuoteFor('')
    setCopied(false)
    setFailure('')
  }

  function save() {
    if (!shown || !shownMoment) return
    const name = shownMoment.label.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
    setFailure('')
    exportCardPng(shown.data, { format, includeQuote: withQuote }, `recall-${shown.data.kind}-${name || 'card'}.png`).catch(() =>
      setFailure('The picture could not be saved'),
    )
  }

  function copy() {
    if (!shown) return
    setFailure('')
    copyCardPng(shown.data, { format, includeQuote: withQuote })
      .then(() => setCopied(true))
      .catch(() => {
        setCopied(false)
        setFailure('The picture could not be copied. Save it instead.')
      })
  }

  return (
    <dialog ref={dialog} className="sheet cards" aria-label="Make a card" onClose={onClose}>
      <header className="sheet-head">
        <h2>Make a card</h2>
        <button type="button" className="icon-btn" aria-label="Close" onClick={() => dialog.current?.close()}>
          <CloseIcon />
        </button>
      </header>
      <div className="cards-body">
        <nav className="cards-list" aria-label="Moments" onKeyDown={rovingKeyDown}>
          {moments.length ? (
            GROUPS.map((group) => {
              const items = moments.filter((moment) => moment.group === group)
              if (!items.length) return null
              return (
                <section key={group}>
                  <h3>{group}</h3>
                  <ul className="rows" role="list">
                    {items.map((moment) => {
                      const active = moment.key === chosen?.key
                      return (
                        <li key={moment.key}>
                          <button
                            type="button"
                            className={active ? 'row is-active' : 'row'}
                            aria-current={active ? 'true' : undefined}
                            data-roving
                            tabIndex={active ? 0 : -1}
                            onClick={() => choose(moment)}
                          >
                            <span className="row-main">
                              <Truncate className="row-title">{moment.label}</Truncate>
                              <Truncate className="row-sub">{moment.detail}</Truncate>
                            </span>
                          </button>
                        </li>
                      )
                    })}
                  </ul>
                </section>
              )
            })
          ) : (
            <p className="pane-note">Nothing to make a card of yet</p>
          )}
        </nav>
        <div className="cards-preview">
          <div className="cards-canvas" data-format={format} aria-busy={waiting}>
            {shown ? (
              <canvas
                // another card, or another shape of this one, arrives as a new picture
                key={`${shown.key} ${format}`}
                ref={canvas}
                className="arrive"
                role="img"
                aria-label={shownMoment ? `Card for ${shownMoment.label}` : 'Card'}
              />
            ) : moments.length ? (
              // the place of the first card while its figures are read
              <div className="skeleton cards-waiting" />
            ) : null}
          </div>
          <div className="cards-controls">
            <ChoiceMenu
              label="Shape"
              value={format}
              choices={SHAPES}
              onChange={(value) => setFormat(value === 'square' ? 'square' : 'story')}
            />
            {asksFor ? (
              <label className="check">
                <input
                  type="checkbox"
                  checked={withQuote}
                  disabled={waiting}
                  onChange={(event) => setQuoteFor(event.target.checked && shown ? shown.key : '')}
                />
                {asksFor}
              </label>
            ) : null}
            <span className="cards-actions">
              <button type="button" className="btn" disabled={!shown || waiting} onClick={copy}>
                {/* the word changes in place when the picture is on the clipboard */}
                <span key={String(copied)} className="swap">
                  {copied ? 'Copied' : 'Copy'}
                </span>
              </button>
              <button type="button" className="btn primary" disabled={!shown || waiting} onClick={save}>
                Save picture
              </button>
            </span>
          </div>
          <p className="sr-only" role="status">
            {copied ? 'The picture is on the clipboard' : ''}
          </p>
          {failure ? (
            <p className="notice bad" role="alert">
              {failure}
            </p>
          ) : null}
        </div>
      </div>
    </dialog>
  )
}
