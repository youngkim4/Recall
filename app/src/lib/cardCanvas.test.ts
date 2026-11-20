import assert from 'node:assert/strict'
import { test } from 'node:test'
import { copyCardPng, exportCardPng, renderCardToCanvas, type CardData, type CardFormat } from './cardCanvas.ts'
import type { DaysChart, VoicesChart } from './cardData.ts'
import { GROUND } from './cardScenes.ts'
import { fakePen } from './cardPenDouble.test.ts'

const WIDTH = 1080
const MARGIN = 96
const YOU = '#0071e3'
const THEM = '#e9e9eb'
const MARK = '#8c96cc'
const INK = '#ffffff'
const BAR = 24
// the scene takes the top of the card, rows of 12px cells: 40 on a tall card, 17 on a square one
const SCENE_BOTTOM: Record<CardFormat, number> = { story: 40 * 12, square: 17 * 12 }

type Drawing = ReturnType<typeof draw>

function draw(data: CardData, format: CardFormat = 'story', includeQuote = true, hasRoundRect = true) {
  const { pen, texts, boxes } = fakePen(hasRoundRect)
  const canvas = { width: 0, height: 0, getContext: () => pen }
  const drawn = renderCardToCanvas(canvas as unknown as HTMLCanvasElement, data, { format, includeQuote })
  // the night is the one box that fills the card, and the scene is drawn above SCENE_BOTTOM
  return { drawn, canvas, texts, boxes: boxes.filter((box) => box.right - box.left < WIDTH && box.top >= SCENE_BOTTOM[format]) }
}

const said = (drawing: Drawing) => drawing.texts.map((item) => item.text)
const filled = (drawing: Drawing, color: string) => drawing.boxes.filter((box) => box.color === color)
// the bar that is split between the two voices, apart from the bars of a chart
const shareBar = (drawing: Drawing, color: string) =>
  filled(drawing, color).filter((box) => Math.abs(box.bottom - box.top - BAR) < 0.01 && box.right - box.left >= BAR)
const baseOf = (drawing: Drawing, text: string) => drawing.texts.find((item) => item.text === text)?.base ?? Number.NaN

function assertInsideMargins(drawing: Drawing, about: string) {
  drawing.texts.forEach((item) => {
    assert.ok(
      item.left >= MARGIN - 0.01 && item.right <= WIDTH - MARGIN + 0.01,
      `${about}: "${item.text}" runs from ${item.left} to ${item.right}`,
    )
  })
  drawing.boxes.forEach((box) => {
    assert.ok(
      box.left >= MARGIN - 0.01 && box.right <= WIDTH - MARGIN + 0.01,
      `${about}: a ${box.color} shape runs from ${box.left} to ${box.right}`,
    )
  })
}

const LONG_ADDRESS = 'a.remarkably.long.address.for.one.person.to.have@mail.example.com'
const LONG_GROUP = 'Avery Stone, Blake Reyes, Casey Lindqvist, Devon Park, Emery Castellanos and 6 more'
const LONG_DAY = 'Wednesday the twenty-third of September, two thousand and twenty'
// lines of single letters fill a line to its last place, where an ellipsis has no room left
const LONG_QUOTE = Array.from({ length: 300 }, () => 'a').join(' ')

const VOICES: VoicesChart = {
  unit: 'month',
  points: Array.from({ length: 72 }, (_, index) => ({ sent: (index * 37) % 900, received: (index * 53) % 1400 })),
  first: 'Mar 2019',
  last: 'Feb 2025',
}
const DAYS: DaysChart = {
  counts: Array.from({ length: 31 }, (_, index) => (index === 15 ? 259 : (index * 17) % 90)),
  at: 15,
  first: 'Nov 1',
  last: 'Dec 1',
  day: 'Nov 16',
}

const anniversary = (fields: Partial<Extract<CardData, { kind: 'anniversary' }>> = {}): CardData => ({
  kind: 'anniversary',
  name: 'Avery Stone',
  years: 6,
  sinceDate: 'March 4, 2019',
  messageCount: 48_210,
  quote: { text: 'is this the right number', fromMe: true },
  ...fields,
})

const onThisDay = (fields: Partial<Extract<CardData, { kind: 'onthisday' }>> = {}): CardData => ({
  kind: 'onthisday',
  name: 'Avery Stone',
  yearsAgo: 3,
  count: 12,
  quote: { text: 'we made it', fromMe: false },
  ...fields,
})

const duo = (fields: Partial<Extract<CardData, { kind: 'duo' }>> = {}): CardData => ({
  kind: 'duo',
  name: 'Avery Stone',
  totalMessages: 48_210,
  balance: 0.52,
  initiation: 0.62,
  busiestDay: 'Mar 3, 2024',
  busiestDayCount: 412,
  ...fields,
})

test('a card is drawn at the size of its shape', () => {
  const tall = draw(anniversary(), 'story')
  const square = draw(anniversary(), 'square')

  assert.equal(tall.drawn, true)
  assert.deepEqual([tall.canvas.width, tall.canvas.height], [1080, 1920])
  assert.deepEqual([square.canvas.width, square.canvas.height], [1080, 1080])
})

test('a canvas that cannot be drawn on says so', () => {
  const canvas = { width: 0, height: 0, getContext: () => null }

  assert.equal(renderCardToCanvas(canvas as unknown as HTMLCanvasElement, duo(), { format: 'story', includeQuote: false }), false)
})

test('a saved person goes by their first name, and anyone else by their whole label', () => {
  assert.ok(said(draw(duo())).includes('You and Avery'))
  assert.ok(said(draw(anniversary())).includes('You and Avery'))
  assert.ok(said(draw(duo({ name: 'Number ending in 0199' }))).includes('You and Number ending in 0199'))
  assert.ok(said(draw(duo({ name: 'Number ending in 0199', balance: 0.4 }))).includes('Number ending in 0199 60%'))
  assert.ok(said(draw(anniversary({ name: 'Unnamed group' }))).includes('You and Unnamed group'))
  assert.ok(said(draw(anniversary({ name: 'avery@example.com' }))).includes('You and avery@example.com'))
})

test('a group keeps its whole name', () => {
  assert.ok(said(draw(anniversary({ name: 'Book Club', group: true }))).includes('You and Book Club'))
  assert.ok(said(draw(onThisDay({ name: 'Book Club' }))).includes('Book Club'))
})

test('nothing on a card leaves its margins, whatever the names and words are', () => {
  const formats: CardFormat[] = ['story', 'square']
  const cards: Array<[string, CardData]> = [
    ['an anniversary', anniversary()],
    ['an anniversary with everything', anniversary({ voices: VOICES, balance: 0.4, reply: { text: LONG_QUOTE, fromMe: false } })],
    ['an anniversary with a long address', anniversary({ name: LONG_ADDRESS, voices: VOICES, quote: { text: LONG_QUOTE, fromMe: false } })],
    ['an anniversary of a group', anniversary({ name: LONG_GROUP, group: true, balance: 0.03, quote: { text: LONG_ADDRESS, fromMe: true } })],
    ['a day', onThisDay()],
    ['a day with everything', onThisDay({ date: LONG_DAY, days: DAYS, busier: { share: 0.96, year: '2024' } })],
    ['a day in a group', onThisDay({ name: LONG_GROUP, days: DAYS, quote: { text: LONG_QUOTE, fromMe: true } })],
    ['a day with a long address', onThisDay({ name: LONG_ADDRESS, quote: null })],
    ['two people', duo()],
    ['two people with their chart', duo({ voices: VOICES, span: 'Mar 2019 – Feb 2025' })],
    ['two people, one with a long address', duo({ name: LONG_ADDRESS, initiation: 0.2, voices: VOICES })],
    ['two people, one with a long label', duo({ name: LONG_GROUP, balance: 0.03 })],
    ['two people and a long day', duo({ busiestDay: LONG_DAY, busiestDayCount: 1_234_567 })],
    ['two people and a great many messages', duo({ totalMessages: 123_456_789_012, voices: VOICES })],
  ]

  formats.forEach((format) => {
    cards.forEach(([about, data]) => {
      assertInsideMargins(draw(data, format, true), `${about}, ${format}`)
      assertInsideMargins(draw(data, format, false), `${about}, ${format}, no message`)
    })
  })
})

test('everything sits between the heading and the mark, in both shapes', () => {
  const frames = { story: { top: 680, mark: 1848 }, square: { top: 360, mark: 1016 } }
  const cards = [
    anniversary({ voices: VOICES, balance: 0.4, reply: { text: LONG_QUOTE, fromMe: false }, quote: { text: LONG_QUOTE, fromMe: true } }),
    onThisDay({ date: 'Saturday, November 16, 2024', days: DAYS, busier: { share: 0.96, year: '2024' }, quote: { text: LONG_QUOTE, fromMe: false } }),
    duo({ voices: VOICES, span: 'Mar 2019 – Feb 2025' }),
  ]

  ;(['story', 'square'] as const).forEach((format) => {
    cards.forEach((data) => {
      ;[true, false].forEach((withMessage) => {
        const drawing = draw(data, format, withMessage)
        const about = `${data.kind}, ${format}${withMessage ? '' : ', no message'}`
        const mark = baseOf(drawing, 'Recall')

        assert.equal(mark, frames[format].mark, about)
        drawing.boxes.forEach((box) => {
          assert.ok(box.top >= frames[format].top - 0.01, `${about}: a shape starts at ${box.top}`)
          assert.ok(box.bottom <= mark - 60, `${about}: a shape ends at ${box.bottom}, the mark stands on ${mark}`)
        })
        drawing.texts
          .filter((item) => item.text !== 'Recall')
          .forEach((item) => assert.ok(item.base <= mark - 60, `${about}: "${item.text}" stands on ${item.base}`))
      })
    })
  })
})

test('a long name is set smaller before any of it is cut', () => {
  // 43 letters with "You and": too wide at the full size, not at the smallest
  const fits = draw(duo({ name: 'Bartholomew-Maximilian-Featherstone' })).texts.find((item) => item.text.startsWith('You and'))
  const cut = draw(duo({ name: LONG_ADDRESS })).texts.find((item) => item.text.startsWith('You and'))

  assert.equal(fits?.text, 'You and Bartholomew-Maximilian-Featherstone')
  assert.equal(fits?.size, 40)
  assert.equal(cut?.size, 40)
  assert.ok(cut?.text.endsWith('…'), cut?.text)
  assert.ok(cut?.text.startsWith('You and a.remarkably.long'), cut?.text)
})

test('a figure too wide for the card is set smaller, and stays whole', () => {
  const numeral = draw(duo({ totalMessages: 123_456_789_012 })).texts.find((item) => item.text.startsWith('123,'))

  assert.equal(numeral?.text, '123,456,789,012')
  assert.ok((numeral?.size ?? 999) < 220, `set at ${numeral?.size}`)
})

test('a message too long for its bubble trails off inside it', () => {
  // a square anniversary card gives the first message two lines of 36px type
  const drawing = draw(anniversary({ quote: { text: LONG_QUOTE, fromMe: false } }), 'square')
  const lines = drawing.texts.filter((item) => item.size === 36)
  const [bubble] = filled(drawing, THEM)
  const pad = 36 * 0.75

  assert.equal(lines.length, 2)
  assert.ok(lines[1].text.endsWith('…'), lines[1].text)
  assert.equal(lines[0].text.endsWith('…'), false)
  assert.ok(bubble.right <= WIDTH - MARGIN + 0.01, `the bubble ends ${bubble.right - (WIDTH - MARGIN)} past the margin`)
  lines.forEach((line) => {
    const past = line.right - (bubble.right - pad)

    assert.ok(line.left >= bubble.left + pad - 0.01, `"${line.text}" starts before its bubble's padding`)
    assert.ok(past <= 0.01, `"${line.text}" ends ${past} past its bubble's padding`)
  })
})

test('a message that fits is drawn whole, yours in blue and theirs in gray', () => {
  const yours = draw(anniversary(), 'story')
  const theirs = draw(onThisDay(), 'story')

  assert.ok(said(yours).includes('is this the right number'))
  assert.equal(filled(yours, YOU).length, 1)
  assert.equal(filled(yours, THEM).length, 0)
  assert.ok(said(theirs).includes('we made it'))
  assert.equal(filled(theirs, THEM).length, 1)
})

test('the first message is shown with the answer to it, each on its own side', () => {
  const drawing = draw(anniversary({ reply: { text: 'who is this', fromMe: false } }), 'story')
  const [mine] = filled(drawing, YOU)
  const [theirs] = filled(drawing, THEM)

  assert.ok(said(drawing).includes('The first messages'))
  assert.ok(said(drawing).includes('is this the right number') && said(drawing).includes('who is this'))
  assert.equal(mine.right, WIDTH - MARGIN)
  assert.equal(theirs.left, MARGIN)
  assert.ok(theirs.top > mine.bottom, 'the answer comes under the message it answers')
})

test('an answer is not shown without the message it answers', () => {
  const drawing = draw(anniversary({ quote: { text: '  ', fromMe: true }, reply: { text: 'who is this', fromMe: false } }))

  assert.equal(said(drawing).includes('who is this'), false)
  assert.equal(said(drawing).some((text) => text.startsWith('The first message')), false)
})

test('a message is left off unless it is asked for', () => {
  const first = draw(anniversary({ reply: { text: 'who is this', fromMe: false } }), 'story', false)
  const day = draw(onThisDay(), 'square', false)

  assert.equal(said(first).some((text) => text.startsWith('The first message')), false)
  assert.equal(said(first).includes('is this the right number') || said(first).includes('who is this'), false)
  assert.equal(filled(first, YOU).length + filled(first, THEM).length, 0)
  assert.equal(said(day).includes('we made it'), false)
  assert.equal(filled(day, THEM).length, 0)
})

test('a message with no words in it is no message', () => {
  const blank = { text: ' \n  ', fromMe: true }
  const first = draw(anniversary({ quote: blank }))
  const day = draw(onThisDay({ quote: blank }))

  assert.equal(said(first).includes('The first message'), false)
  assert.equal(filled(first, YOU).length, 0)
  // the day still leads with its count
  assert.equal(filled(day, YOU).length, 0)
  assert.ok(said(day).includes('12'))
  assert.ok(said(day).includes('messages that day'))
})

test('a figure that is not a number is left off the card', () => {
  const drawing = draw(duo({ balance: Number.NaN, initiation: Number.NaN, busiestDay: undefined }))

  assert.deepEqual(said(drawing).filter((text) => /NaN|%|First to write/.test(text)), [])
  assert.equal(filled(drawing, YOU).length + filled(drawing, MARK).length, 0)
  assert.equal(draw(duo({ balance: null, initiation: undefined })).texts.some((item) => item.text.includes('%')), false)
  assert.equal(draw(anniversary({ balance: Number.NaN })).texts.some((item) => item.text.includes('%')), false)
})

test('a card of two people carries their chart, with its ends and its scale named', () => {
  ;(['story', 'square'] as const).forEach((format) => {
    const drawing = draw(duo({ voices: VOICES, span: 'Mar 2019 – Feb 2025' }), format)
    const texts = said(drawing)

    assert.ok(texts.includes('Mar 2019 – Feb 2025'), format)
    assert.ok(texts.includes('Mar 2019') && texts.includes('Feb 2025') && texts.includes('messages a month'), format)
    // one bar a month for each voice that wrote in it, besides the bar they share
    assert.ok(filled(drawing, YOU).length > 60, `${format}: ${filled(drawing, YOU).length} blue shapes`)
    assert.ok(filled(drawing, MARK).length > 60, `${format}: ${filled(drawing, MARK).length} gray shapes`)
    assert.equal(shareBar(drawing, YOU).length, 1, format)
  })
})

test('the parts of a card come in one order, none over another', () => {
  const drawing = draw(duo({ voices: VOICES }), 'story')
  const order = ['You and Avery', '48,210', 'messages', 'messages a month', 'Feb 2025', 'You 52%', 'First to write', 'Busiest day', 'Recall']
  const bases = order.map((text) => baseOf(drawing, text))

  bases.forEach((base, index) => {
    assert.ok(Number.isFinite(base), `"${order[index]}" is on the card`)
    if (index > 0) assert.ok(base > bases[index - 1], `"${order[index]}" stands under "${order[index - 1]}"`)
  })
  const [bar] = shareBar(drawing, YOU)
  assert.ok(bar.top > baseOf(drawing, 'Feb 2025') && bar.bottom < baseOf(drawing, 'You 52%'))
})

test('a square card leaves off what it has no room for, least important first', () => {
  const full = draw(duo({ voices: VOICES }), 'square')
  const plain = draw(duo(), 'square')

  // with the chart on it there is no room for the named figures; without it there is
  assert.equal(said(full).includes('First to write'), false)
  assert.ok(said(full).includes('You 52%') && said(full).includes('messages a month'))
  assert.ok(said(plain).includes('First to write'))
})

test('a card with no chart to draw says less, and leaves no hole for it', () => {
  const drawing = draw(duo({ voices: null }), 'story')

  assert.equal(said(drawing).includes('messages a month'), false)
  // the parts close up around the middle of the card
  assert.ok(baseOf(drawing, '48,210') > 600, `the figure stands on ${baseOf(drawing, '48,210')}`)
})

test('a day is headed by its date, and stands among the days around it', () => {
  const drawing = draw(onThisDay({ date: 'Saturday, November 16, 2024', count: 259, days: DAYS }), 'story', false)
  const texts = said(drawing)

  assert.ok(texts.includes('Avery Stone') && texts.includes('Saturday, November 16, 2024'))
  assert.ok(texts.includes('259') && texts.includes('messages that day') && texts.includes('3 years ago today'))
  assert.ok(texts.includes('Nov 1') && texts.includes('Dec 1') && texts.includes('Nov 16'))
  assert.equal(filled(drawing, INK).length, 1, 'the day itself is the one bar in ink')
})

test('a day with no date to give is headed by how long ago it was', () => {
  const texts = said(draw(onThisDay({ yearsAgo: 1 }), 'story', false))

  assert.ok(texts.includes('A year ago today'))
  assert.equal(texts.filter((text) => text.includes('ago today')).length, 1)
})

test('a day that stood above most of its year says so, and never claims more than is true', () => {
  const line = (share: number) =>
    said(draw(onThisDay({ busier: { share, year: '2024' } }), 'story', false)).filter((text) => /Busier|busiest|of days in|^2024$/.test(text))

  assert.deepEqual(line(0.964), ['Busier than', '96% of days in 2024'])
  assert.deepEqual(line(0.9999), ['Busier than', '99% of days in 2024'])
  assert.deepEqual(line(1), ['The busiest day of', '2024'])
  assert.deepEqual(line(0.31), [])
  assert.deepEqual(line(Number.NaN), [])
})

test('the two voices share one bar, with a gap between them', () => {
  const drawing = draw(duo({ balance: 0.5 }))
  const [yours] = shareBar(drawing, YOU)
  const [theirs] = shareBar(drawing, MARK)

  assert.equal(yours.left, MARGIN)
  assert.equal(theirs.right, WIDTH - MARGIN)
  assert.ok(theirs.left - yours.right >= 4, `the gap is ${theirs.left - yours.right}`)
  assert.ok(Math.abs(yours.right - yours.left - (theirs.right - theirs.left)) < 0.01, 'equal shares draw equal bars')
})

test('a voice that said nothing has no bar, and the other takes the whole width', () => {
  const allYours = draw(duo({ balance: 1 }))
  const allTheirs = draw(duo({ balance: 0 }))

  assert.equal(filled(allYours, MARK).length, 0)
  assert.deepEqual([filled(allYours, YOU)[0].left, filled(allYours, YOU)[0].right], [MARGIN, WIDTH - MARGIN])
  assert.ok(said(allYours).includes('You 100%') && said(allYours).includes('Avery 0%'))

  assert.equal(filled(allTheirs, YOU).length, 0)
  assert.deepEqual([filled(allTheirs, MARK)[0].left, filled(allTheirs, MARK)[0].right], [MARGIN, WIDTH - MARGIN])
  assert.ok(said(allTheirs).includes('You 0%') && said(allTheirs).includes('Avery 100%'))
})

test('a voice that said very little still shows, without crowding the other', () => {
  ;[0.002, 0.998].forEach((balance) => {
    const drawing = draw(duo({ balance }))
    const [yours] = filled(drawing, YOU)
    const [theirs] = filled(drawing, MARK)

    assert.ok(yours.right - yours.left >= 24, `your bar is ${yours.right - yours.left} wide at ${balance}`)
    assert.ok(theirs.right - theirs.left >= 24, `their bar is ${theirs.right - theirs.left} wide at ${balance}`)
    assert.ok(theirs.left - yours.right >= 4, `the bars are ${theirs.left - yours.right} apart at ${balance}`)
    assert.equal(yours.left, MARGIN)
    assert.equal(theirs.right, WIDTH - MARGIN)
  })
})

test('a share out of range is held to all or nothing', () => {
  assert.ok(said(draw(duo({ balance: 1.4 }))).includes('You 100%'))
  assert.ok(said(draw(duo({ balance: -0.2 }))).includes('Avery 100%'))
})

test('the two shares always add up to a hundred', () => {
  ;[0.495, 0.505, 0.335, 0.004, 0.996].forEach((balance) => {
    const percents = said(draw(duo({ balance })))
      .filter((text) => /^(You|Avery) \d+%$/.test(text))
      .map((text) => Number(/\d+/.exec(text)?.[0]))

    assert.equal(percents.length, 2, `two shares at ${balance}`)
    assert.equal(percents[0] + percents[1], 100, `shares at ${balance}`)
  })
})

test('an anniversary carries the split of the two voices when it is known', () => {
  const texts = said(draw(anniversary({ balance: 0.4, voices: VOICES }), 'story', false))

  assert.ok(texts.includes('You 40%') && texts.includes('Avery 60%'))
  assert.ok(texts.includes('6') && texts.includes('years of messages') && texts.includes('48,210 messages'))
  assert.ok(texts.includes('Since March 4, 2019') && texts.includes('messages a month'))
})

test('whoever writes first more often is named with their share of the days', () => {
  assert.ok(said(draw(duo({ initiation: 0.62 }))).includes('You, on 62% of days'))
  assert.ok(said(draw(duo({ initiation: 0.38 }))).includes('Avery, on 62% of days'))
  assert.ok(said(draw(duo({ initiation: 0.5 }))).includes('You, on 50% of days'))
})

test('the two captions under the bar never meet', () => {
  const drawing = draw(duo({ name: LONG_ADDRESS }))
  const yours = drawing.texts.find((item) => item.text.startsWith('You ') && item.text.endsWith('%'))
  const theirs = drawing.texts.find((item) => item.text.endsWith('48%'))

  assert.ok(yours && theirs, said(drawing).join(' | '))
  assert.ok(theirs.left - yours.right >= 16, `the captions are ${theirs.left - yours.right} apart`)
  assert.ok(theirs.text.includes('…'), theirs.text)
})

test('a fact never runs into its label', () => {
  const drawing = draw(duo({ name: LONG_ADDRESS, initiation: 0.2 }))
  const label = drawing.texts.find((item) => item.text === 'First to write')
  const value = drawing.texts.find((item) => item.text.endsWith('80% of days'))

  assert.ok(label && value, said(drawing).join(' | '))
  assert.ok(value.left - label.right >= 16, `the fact is ${value.left - label.right} from its label`)
  assert.ok(value.text.includes('…'), value.text)
})

test('a canvas with no rounded rectangle of its own still gets its shapes', () => {
  const cards = [
    anniversary({ voices: VOICES, balance: 0.4, reply: { text: 'who is this', fromMe: false } }),
    onThisDay({ days: DAYS }),
    duo({ voices: VOICES }),
    duo({ balance: 1 }),
  ]

  cards.forEach((data) => {
    const native = draw(data, 'story', true, true)
    const byHand = draw(data, 'story', true, false)

    assert.equal(byHand.drawn, true)
    assert.ok(byHand.boxes.length > 0)
    assert.deepEqual(byHand.boxes, native.boxes)
    assert.deepEqual(byHand.texts, native.texts)
  })
})

type Stubbed = { link: { href: string; download: string; clicks: number } }

// stands in for the page long enough to make one picture
function stubPage(context: unknown, blob: Blob | null): Stubbed {
  const link = { href: '', download: '', clicks: 0, click: () => (link.clicks += 1) }
  const canvas = { width: 0, height: 0, getContext: () => context, toBlob: (done: (made: Blob | null) => void) => done(blob) }
  Object.defineProperty(globalThis, 'document', {
    value: { createElement: (tag: string) => (tag === 'canvas' ? canvas : link) },
    configurable: true,
  })
  Object.defineProperty(globalThis, 'window', { value: { setTimeout: () => 0 }, configurable: true })
  URL.createObjectURL = () => 'blob:card'
  URL.revokeObjectURL = () => {}
  return { link }
}

test('saving a card hands the browser a picture under the name asked for', async () => {
  const { link } = stubPage(fakePen(true).pen, new Blob(['png']))

  await exportCardPng(duo(), { format: 'square', includeQuote: false }, 'recall-duo-avery.png')

  assert.deepEqual([link.href, link.download, link.clicks], ['blob:card', 'recall-duo-avery.png', 1])
})

test('saving a card fails out loud when no picture can be made', async () => {
  const options = { format: 'square' as const, includeQuote: false }

  const noPen = stubPage(null, new Blob(['png']))
  await assert.rejects(exportCardPng(duo(), options, 'card.png'), /could not be made/)
  assert.equal(noPen.link.clicks, 0)

  const noBlob = stubPage(fakePen(true).pen, null)
  await assert.rejects(exportCardPng(duo(), options, 'card.png'), /could not be made/)
  assert.equal(noBlob.link.clicks, 0)
})

type Copied = Array<Record<string, Promise<Blob>>>

// stands in for a clipboard that takes pictures; with none, the window has no such clipboard
function stubClipboard(takesPictures: boolean) {
  const copied: Copied = []
  class Item {
    readonly parts: Record<string, Promise<Blob>>
    constructor(parts: Record<string, Promise<Blob>>) {
      this.parts = parts
    }
  }
  Object.defineProperty(globalThis, 'ClipboardItem', { value: takesPictures ? Item : undefined, configurable: true, writable: true })
  Object.defineProperty(globalThis, 'navigator', {
    value: {
      clipboard: takesPictures
        ? {
            write: async (items: Item[]) => {
              // a clipboard reads what it was handed; a picture that could not be made fails the copy
              await Promise.all(items.flatMap((item) => Object.values(item.parts)))
              copied.push(...items.map((item) => item.parts))
            },
          }
        : {},
    },
    configurable: true,
  })
  return copied
}

test('copying a card puts one picture on the clipboard', async () => {
  const picture = new Blob(['png'])
  stubPage(fakePen(true).pen, picture)
  const copied = stubClipboard(true)

  await copyCardPng(duo(), { format: 'square', includeQuote: false })

  assert.equal(copied.length, 1)
  assert.deepEqual(Object.keys(copied[0]), ['image/png'])
  assert.equal(await copied[0]['image/png'], picture)
})

test('copying fails out loud when there is no picture, or nowhere to put one', async () => {
  const options = { format: 'story' as const, includeQuote: false }

  stubPage(null, new Blob(['png']))
  const copied = stubClipboard(true)
  await assert.rejects(copyCardPng(duo(), options), /could not be made/)
  assert.equal(copied.length, 0)

  stubPage(fakePen(true).pen, new Blob(['png']))
  stubClipboard(false)
  await assert.rejects(copyCardPng(duo(), options), /cannot copy a picture/)
})

test('a card stands on the night of its scene, and each kind starts on its own', () => {
  const night = (data: CardData, scene?: 'sunrise' | 'sea' | 'aurora') => {
    const { pen, boxes } = fakePen(true)
    const canvas = { width: 0, height: 0, getContext: () => pen }
    renderCardToCanvas(canvas as unknown as HTMLCanvasElement, data, { format: 'story', includeQuote: false, scene })
    return { ground: boxes.find((box) => box.right - box.left >= WIDTH && box.bottom - box.top >= 1920)?.color, boxes }
  }
  assert.equal(night(anniversary()).ground, GROUND.sunrise)
  assert.equal(night(onThisDay()).ground, GROUND.sea)
  assert.equal(night(duo()).ground, GROUND.aurora)
  assert.equal(night(duo(), 'sunrise').ground, GROUND.sunrise)
  // the picture is drawn across the top, from edge to edge
  const scene = night(duo()).boxes.filter((box) => box.bottom <= 40 * 12)
  assert.ok(scene.length > 40)
  assert.equal(Math.min(...scene.map((box) => box.left)), 0)
  assert.equal(Math.max(...scene.map((box) => box.right)), WIDTH)
})
