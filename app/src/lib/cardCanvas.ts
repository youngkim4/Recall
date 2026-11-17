// Share cards, drawn straight onto a canvas. One draw function feeds the
// preview, the saved picture and the copied one, so the three cannot differ.
//
// A card is who it is about, one figure set large, and under it what makes
// the figure worth sending: the conversation's own chart, the split of the
// two voices, the words that were said. What a card has no data for is left
// off, and the parts that are there share the height between them.

import { days, voices } from './cardCharts.ts'
import type { DaysChart, VoicesChart } from './cardData.ts'
import { place, type Block } from './cardLayout.ts'
import { balance, facts, heading, hero, messages, spoken, wordmark, type CardQuote, type Fact } from './cardParts.ts'
import { COLOR, WIDTH, clipped, count, named, numbers, percent, shareOf, type Pen } from './cardPen.ts'
import { shortName } from './conversations.ts'

export type { CardQuote } from './cardParts.ts'

export type CardFormat = 'story' | 'square'

export type AnniversaryCardData = {
  kind: 'anniversary'
  name: string
  // a group's name is not a first name and a last name; it is kept whole
  group?: boolean
  years: number
  sinceDate: string
  messageCount: number
  // the first message of the conversation and the answer to it, when they are known
  quote?: CardQuote | null
  reply?: CardQuote | null
  // from the conversation's own figures, once they have been read
  voices?: VoicesChart | null
  // your share of the messages, 0 to 1
  balance?: number | null
}

export type OnThisDayCardData = {
  kind: 'onthisday'
  name: string
  yearsAgo: number
  count: number
  // the day itself, as "YYYY-MM-DD" and written out
  day?: string
  date?: string
  quote?: CardQuote | null
  // the day among the days around it
  days?: DaysChart | null
  // the share of that year's other days that held less
  busier?: { share: number; year: string } | null
}

export type DuoCardData = {
  kind: 'duo'
  name: string
  totalMessages: number
  // the months the conversation runs between
  span?: string
  // your share of the messages, 0 to 1
  balance?: number | null
  // the share of days on which you wrote first, 0 to 1
  initiation?: number | null
  busiestDay?: string
  busiestDayCount?: number | null
  voices?: VoicesChart | null
}

export type CardData = AnniversaryCardData | OnThisDayCardData | DuoCardData

export type CardOptions = {
  format: CardFormat
  // a quoted message is left off unless the reader asks for it
  includeQuote: boolean
}

// Where things go in each shape: the heading, the room the parts share, the
// gap between two parts, and the line the mark stands on. A tall card keeps
// clear of the top and bottom, where a story is covered by its own controls.
type Frame = { height: number; head: number; top: number; bottom: number; gap: number; mark: number }
const FRAMES: Record<CardFormat, Frame> = {
  story: { height: 1920, head: 230, top: 420, bottom: 1730, gap: 72, mark: 1848 },
  square: { height: 1080, head: 104, top: 262, bottom: 952, gap: 44, mark: 1016 },
}

type Card = { name: string; under: string; blocks: Array<Block | null> }

function anniversaryCard(pen: Pen, data: AnniversaryCardData, opts: CardOptions): Card {
  const story = opts.format === 'story'
  const them = data.group ? data.name : shortName(data.name)
  const first = opts.includeQuote ? spoken(data.quote) : null
  // an answer is shown only under the message it answers
  const exchange = first ? [first, spoken(data.reply)] : []
  const said = messages(
    pen,
    exchange,
    story ? { size: 44, maxLines: 3 } : { size: 36, maxLines: 2 },
    exchange[1] ? 'The first messages' : 'The first message',
  )
  const yours = shareOf(data.balance)
  return {
    name: `You and ${them}`,
    under: `Since ${data.sinceDate}`,
    blocks: [
      hero(pen, String(data.years), story ? (said ? 260 : 340) : said ? 150 : 168, [
        { text: `${data.years === 1 ? 'year' : 'years'} of messages`, size: story ? (said ? 52 : 60) : said ? 44 : 48 },
        { text: count(data.messageCount, 'message'), size: story ? 34 : 30, weight: 400, color: COLOR.quiet },
      ]),
      said ? { ...said, rank: 1 } : null,
      data.voices
        ? voices(data.voices, them, { min: said ? 240 : story ? 300 : 240, max: story ? (said ? 420 : 620) : 400, rank: said ? 2 : 1 })
        : null,
      yours === null ? null : { ...balance(yours, them), rank: 3 },
    ],
  }
}

// how the day stands among the days of its year, when it stands above most of them
function standing(data: OnThisDayCardData): Fact[] {
  const ranked = data.busier
  if (!ranked || !Number.isFinite(ranked.share) || ranked.share < 0.5) return []
  if (ranked.share >= 1) return [{ label: 'The busiest day of', value: () => ranked.year }]
  // rounded down, so that "100%" is never said of a day that another day matched
  return [{ label: 'Busier than', value: () => `${Math.floor(ranked.share * 100)}% of days in ${ranked.year}` }]
}

function onThisDayCard(pen: Pen, data: OnThisDayCardData, opts: CardOptions): Card {
  const story = opts.format === 'story'
  const said = opts.includeQuote ? messages(pen, [data.quote], story ? { size: 52, maxLines: 6 } : { size: 40, maxLines: 3 }) : null
  const ago = data.yearsAgo === 1 ? 'A year ago today' : `${data.yearsAgo} years ago today`
  return {
    name: data.name,
    under: data.date || ago,
    blocks: [
      // the words come first, under the name, the way a thread reads
      said ? { ...said, rank: 1 } : null,
      hero(pen, numbers.format(data.count), story ? (said ? 200 : 300) : said ? 130 : 190, [
        { text: `${data.count === 1 ? 'message' : 'messages'} that day`, size: story ? (said ? 48 : 56) : said ? 40 : 48 },
        ...(data.date ? [{ text: ago, size: story ? 34 : 30, weight: 400, color: COLOR.quiet }] : []),
      ]),
      data.days ? days(data.days, { min: story ? 260 : 200, max: story ? 420 : 330, rank: said ? 2 : 1 }) : null,
      facts(standing(data), story ? 96 : 80, 3),
    ],
  }
}

function duoFacts(data: DuoCardData, them: string): Fact[] {
  const youFirst = shareOf(data.initiation)
  const day = data.busiestDay
    ? data.busiestDayCount
      ? `${data.busiestDay}, ${numbers.format(data.busiestDayCount)}`
      : data.busiestDay
    : ''
  return [
    ...(youFirst === null
      ? []
      : [
          {
            label: 'First to write',
            value: (pen: Pen, room: number) =>
              youFirst >= 0.5
                ? `You, on ${percent(youFirst)} of days`
                : named(pen, them, `, on ${percent(1 - youFirst)} of days`, room),
          },
        ]),
    ...(day ? [{ label: 'Busiest day', value: (pen: Pen, room: number) => clipped(pen, day, room) }] : []),
  ]
}

function duoCard(pen: Pen, data: DuoCardData, opts: CardOptions): Card {
  const story = opts.format === 'story'
  const them = shortName(data.name)
  const yours = shareOf(data.balance)
  return {
    name: `You and ${them}`,
    under: data.span || 'Every message',
    blocks: [
      hero(pen, numbers.format(data.totalMessages), story ? 220 : 150, [
        { text: data.totalMessages === 1 ? 'message' : 'messages', size: story ? 52 : 44 },
      ]),
      data.voices ? voices(data.voices, them, { min: story ? 300 : 240, max: story ? 560 : 400, rank: 1 }) : null,
      yours === null ? null : { ...balance(yours, them), rank: 2 },
      facts(duoFacts(data, them), story ? 96 : 80, 3),
    ],
  }
}

function drawCard(pen: Pen, data: CardData, opts: CardOptions) {
  const frame = FRAMES[opts.format]
  pen.fillStyle = COLOR.paper
  pen.fillRect(0, 0, WIDTH, frame.height)
  const card =
    data.kind === 'anniversary'
      ? anniversaryCard(pen, data, opts)
      : data.kind === 'onthisday'
        ? onThisDayCard(pen, data, opts)
        : duoCard(pen, data, opts)
  heading(pen, card.name, card.under, frame.head)
  const blocks = card.blocks.filter((block): block is Block => block !== null)
  place(blocks, frame.top, frame.bottom, frame.gap).forEach(({ block, top, height }) => block.draw(pen, top, height))
  wordmark(pen, frame.mark)
}

// Draws the card on the canvas at its full size. False when the canvas
// cannot be drawn on, and so holds no card.
export function renderCardToCanvas(canvas: HTMLCanvasElement, data: CardData, opts: CardOptions) {
  // setting the size clears the canvas
  canvas.width = WIDTH
  canvas.height = FRAMES[opts.format].height
  const pen = canvas.getContext('2d')
  if (!pen) return false
  drawCard(pen, data, opts)
  return true
}

// the card as a PNG; it fails rather than hand over a picture with no card on it
function cardPng(data: CardData, opts: CardOptions) {
  const failed = () => new Error('The image could not be made')
  const canvas = document.createElement('canvas')
  if (!renderCardToCanvas(canvas, data, opts)) return Promise.reject(failed())
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(failed())), 'image/png')
  })
}

// Saves the card as a PNG through the browser's own download.
export async function exportCardPng(data: CardData, opts: CardOptions, filename: string) {
  const url = URL.createObjectURL(await cardPng(data, opts))
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  window.setTimeout(() => URL.revokeObjectURL(url), 10_000)
}

// Puts the card on the clipboard as a picture, ready to paste into a message.
export async function copyCardPng(data: CardData, opts: CardOptions) {
  if (typeof ClipboardItem === 'undefined' || !navigator.clipboard?.write) {
    throw new Error('This window cannot copy a picture')
  }
  // the picture is handed over still being made, so the copy belongs to the press that asked for it
  await navigator.clipboard.write([new ClipboardItem({ 'image/png': cardPng(data, opts) })])
}
