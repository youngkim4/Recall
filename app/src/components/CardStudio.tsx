import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { recallApi } from '../lib/api'
import {
  exportCardPng,
  renderCardToCanvas,
  type CardData,
  type CardFormat,
} from '../lib/cardCanvas'
import type { Contact, Defaults, FirstWordsPayload, MemoriesPayload } from '../types'
import { contactTitle } from '../lib/format'

// Pick a moment, see the card, save a PNG. Everything renders locally on a
// canvas -- nothing about the card ever leaves the machine.

type Moment = {
  key: string
  group: 'Anniversaries' | 'On this day' | 'You & them'
  label: string
  detail: string
  data: CardData
  quoteDefault: boolean
  fetchContact?: Contact
}

type Props = {
  defaults: Defaults | null
  model: string
  contacts: Contact[]
  onClose: () => void
}

function formatLongDate(value: string): string {
  const date = new Date(`${value}T00:00:00`)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' })
}

function formatShortDate(value: string): string {
  const date = new Date(value.includes('T') ? value : `${value}T00:00:00`)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' })
}

export function CardStudio({ defaults, model, contacts, onClose }: Props) {
  const [memories, setMemories] = useState<MemoriesPayload | null>(null)
  const [firstWords, setFirstWords] = useState<FirstWordsPayload | null>(null)
  const [selectedKey, setSelectedKey] = useState('')
  const [format, setFormat] = useState<CardFormat>('story')
  // null = follow the moment's privacy default; boolean = explicit user choice
  const [quoteOverride, setQuoteOverride] = useState<boolean | null>(null)
  const [duo, setDuo] = useState<{ key: string; data: CardData | null } | null>(null)
  const [note, setNote] = useState('')
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const messagesPath = defaults?.messagesPath || ''

  useEffect(() => {
    if (!messagesPath) return
    let cancelled = false
    void recallApi
      .memories({ messagesPath })
      .then((response) => {
        if (!cancelled) setMemories(response.memories)
      })
      .catch(() => {})
    void recallApi
      .firstWords({ messagesPath })
      .then((response) => {
        if (!cancelled) setFirstWords(response.firstWords)
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [messagesPath])

  const moments = useMemo<Moment[]>(() => {
    const list: Moment[] = []
    const firstByPerson = new Map(
      (firstWords?.entries || []).map((entry) => [entry.person.toLowerCase(), entry]),
    )
    for (const item of memories?.anniversaries || []) {
      const exchange = firstByPerson.get(item.name.toLowerCase())
      list.push({
        key: `anni-${item.chatId}`,
        group: 'Anniversaries',
        label: item.name,
        detail: `${item.years} ${item.years === 1 ? 'year' : 'years'} · ${formatLongDate(item.date)}`,
        quoteDefault: false,
        data: {
          kind: 'anniversary',
          name: item.name,
          years: item.years,
          sinceDate: formatLongDate(item.date),
          messageCount: item.count,
          quote: exchange ? { text: exchange.text, direction: exchange.direction } : null,
        },
      })
    }
    for (const item of memories?.onThisDay || []) {
      list.push({
        key: `otd-${item.chatId}-${item.year}`,
        group: 'On this day',
        label: `${item.name} · ${item.year}`,
        detail: item.preview ? `“${item.preview.slice(0, 64)}${item.preview.length > 64 ? '…' : ''}”` : `${item.count} messages`,
        quoteDefault: Boolean(item.preview),
        data: {
          kind: 'onthisday',
          name: item.name,
          year: item.year,
          yearsAgo: item.yearsAgo,
          count: item.count,
          quote: item.preview || '',
        },
      })
    }
    const direct = contacts
      .filter((contact) => !contact.is_group && Number(contact.message_count || 0) > 0)
      .slice(0, 8)
    for (const contact of direct) {
      list.push({
        key: `duo-${contact.chat_id}`,
        group: 'You & them',
        label: contactTitle(contact),
        detail: `${Number(contact.message_count || 0).toLocaleString()} messages · stats only`,
        quoteDefault: false,
        fetchContact: contact,
        data: {
          kind: 'duo',
          name: contactTitle(contact),
          totalMessages: Number(contact.message_count || 0),
        },
      })
    }
    return list
  }, [contacts, firstWords, memories])

  const selected = moments.find((moment) => moment.key === selectedKey) || moments[0] || null
  const includeQuote = quoteOverride ?? selected?.quoteDefault ?? false

  // duo cards fetch real dynamics for the chosen person; results are keyed so
  // a stale response for a previous selection is simply ignored
  useEffect(() => {
    if (!selected?.fetchContact || !messagesPath || !model) return
    if (duo?.key === selected.key) return
    const key = selected.key
    const label = selected.label
    const fallbackCount = Number(selected.fetchContact.message_count || 0)
    let cancelled = false
    void recallApi
      .preview({ messagesPath, contact: selected.fetchContact.chat_id, model })
      .then((payload) => {
        if (cancelled) return
        setDuo({
          key,
          data: {
            kind: 'duo',
            name: label,
            totalMessages: payload.stats.totalMessages || fallbackCount,
            initiation: payload.dynamics?.initiationLifetime,
            balance: payload.dynamics?.balanceLifetime,
            volumeTrendPct: payload.dynamics?.volumeTrendPct,
            busiestDay: payload.stats.busiestDay ? formatShortDate(payload.stats.busiestDay) : undefined,
            busiestDayCount: payload.stats.busiestDayCount,
          },
        })
      })
      .catch(() => {
        if (cancelled) return
        setNote('Could not load the full stats; showing the basics.')
        setDuo({ key, data: null })
      })
    return () => {
      cancelled = true
    }
  }, [duo, messagesPath, model, selected])

  const duoLoading = Boolean(selected?.fetchContact) && duo?.key !== selected?.key
  const activeData: CardData | null = selected
    ? selected.fetchContact && duo?.key === selected.key && duo.data
      ? duo.data
      : selected.data
    : null

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !activeData) return
    void renderCardToCanvas(canvas, activeData, { format, includeQuote })
  }, [activeData, format, includeQuote])

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const download = useCallback(() => {
    if (!activeData || !selected) return
    const slug = selected.label.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
    void exportCardPng(activeData, { format, includeQuote }, `recall-${activeData.kind}-${slug}.png`)
  }, [activeData, format, includeQuote, selected])

  const hasQuote =
    selected?.data.kind === 'anniversary'
      ? Boolean(selected.data.quote?.text)
      : selected?.data.kind === 'onthisday'
        ? Boolean(selected.data.quote)
        : false

  const groups: Array<Moment['group']> = ['Anniversaries', 'On this day', 'You & them']

  return (
    <div className="cardstudio-backdrop" role="dialog" aria-label="Make a card">
      <div className="cardstudio">
        <div className="cardstudio-head">
          <h3>Make a card</h3>
          <button type="button" className="button ghost" onClick={onClose}>
            Close
          </button>
        </div>
        <div className="cardstudio-body">
          <div className="cardstudio-moments">
            {moments.length === 0 ? (
              <p className="cardstudio-empty">
                No moments yet — anniversaries and on-this-day memories appear here as your
                archive fills in.
              </p>
            ) : (
              groups.map((group) => {
                const items = moments.filter((moment) => moment.group === group)
                if (!items.length) return null
                return (
                  <div key={group} className="cardstudio-group">
                    <div className="cardstudio-group-title">{group}</div>
                    {items.map((moment) => (
                      <button
                        key={moment.key}
                        type="button"
                        className={`cardstudio-moment${selected?.key === moment.key ? ' is-active' : ''}`}
                        onClick={() => {
                          setSelectedKey(moment.key)
                          setQuoteOverride(null)
                          setNote('')
                        }}
                      >
                        <strong>{moment.label}</strong>
                        <span>{moment.detail}</span>
                      </button>
                    ))}
                  </div>
                )
              })
            )}
          </div>
          <div className="cardstudio-preview">
            <div className={`cardstudio-canvas-wrap ${format}`}>
              {duoLoading ? <div className="cardstudio-loading">Reading the numbers…</div> : null}
              <canvas ref={canvasRef} className="cardstudio-canvas" />
            </div>
            <div className="cardstudio-controls">
              <div className="cardstudio-toggle-row">
                <button
                  type="button"
                  className={`button ghost${format === 'story' ? ' is-active' : ''}`}
                  onClick={() => setFormat('story')}
                >
                  Story 9:16
                </button>
                <button
                  type="button"
                  className={`button ghost${format === 'square' ? ' is-active' : ''}`}
                  onClick={() => setFormat('square')}
                >
                  Square
                </button>
              </div>
              {hasQuote ? (
                <label className="cardstudio-quote-toggle">
                  <input
                    type="checkbox"
                    checked={includeQuote}
                    onChange={(event) => setQuoteOverride(event.target.checked)}
                  />
                  Include the message text
                </label>
              ) : null}
              {note ? <div className="cardstudio-note">{note}</div> : null}
              <button
                type="button"
                className="button primary"
                disabled={!activeData || duoLoading}
                onClick={download}
              >
                Save PNG
              </button>
              <p className="cardstudio-footnote">Rendered on this Mac. Nothing is uploaded.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
