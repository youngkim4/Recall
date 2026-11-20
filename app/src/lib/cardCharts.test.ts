import assert from 'node:assert/strict'
import { test } from 'node:test'
import { days, voices } from './cardCharts.ts'
import type { DaysChart, VoicesChart } from './cardData.ts'
import { asPen, fakePen } from './cardPenDouble.test.ts'

const WIDTH = 1080
const MARGIN = 96
const YOU = '#0071e3'
const MARK = '#8c96cc'
const INK = '#ffffff'
const FAINT = '#12425c'
const TOP = 500
const HEIGHT = 400

const chart = (fields: Partial<VoicesChart> = {}): VoicesChart => ({
  unit: 'month',
  points: [
    { sent: 400, received: 900 },
    { sent: 0, received: 0 },
    { sent: 1200, received: 300 },
    { sent: 1, received: 2 },
  ],
  first: 'Mar 2019',
  last: 'Jun 2019',
  ...fields,
})

function drawVoices(data: VoicesChart, them = 'Avery') {
  const double = fakePen()
  voices(data, them, { min: 300, max: 500, rank: 1 }).draw(asPen(double), TOP, HEIGHT)
  return double
}

const said = (double: ReturnType<typeof fakePen>) => double.texts.map((item) => item.text)
const filled = (double: ReturnType<typeof fakePen>, color: string) => double.boxes.filter((box) => box.color === color)
const bars = (double: ReturnType<typeof fakePen>, color: string) =>
  filled(double, color).filter((box) => box.bottom - box.top !== 20 || box.right - box.left !== 20)

test('each voice has a bar for each month it wrote in: yours above the line, theirs below', () => {
  const double = drawVoices(chart())
  const yours = bars(double, YOU)
  const theirs = bars(double, MARK)

  assert.equal(yours.length, 3)
  assert.equal(theirs.length, 3)
  const ground = Math.max(...yours.map((bar) => bar.bottom))
  yours.forEach((bar) => assert.ok(bar.bottom <= ground + 0.01 && bar.top < bar.bottom))
  theirs.forEach((bar) => assert.ok(bar.top >= ground - 0.01, `their bar starts at ${bar.top}, the line is at ${ground}`))
})

test('a month with nothing in it keeps its place', () => {
  const double = drawVoices(chart())
  const [first, third, fourth] = bars(double, YOU).sort((a, b) => a.left - b.left)
  const pitch = fourth.left - third.left

  // the second month is empty, so the first and third bars are two places apart
  assert.ok(Math.abs(third.left - first.left - pitch * 2) < 0.01, `${first.left}, ${third.left}, ${fourth.left}`)
})

test('the tallest bar on each side reaches as far as its side goes, and a small one still shows', () => {
  const double = drawVoices(chart())
  const yours = bars(double, YOU).sort((a, b) => a.left - b.left)
  const heights = yours.map((bar) => bar.bottom - bar.top)

  // 1,200 fills the side that stands for 1,200; 400 is a third of it
  assert.ok(Math.abs(heights[0] / heights[1] - 1 / 3) < 0.02, heights.join(', '))
  assert.ok(heights[2] >= 3, `the month with one message is ${heights[2]} tall`)
})

test('the chart says what its ends and its scale stand for', () => {
  const texts = said(drawVoices(chart()))

  assert.ok(texts.includes('Mar 2019') && texts.includes('Jun 2019'), texts.join(' | '))
  // each side goes up to the round number above its busiest month
  assert.ok(texts.includes('1.2K') && texts.includes('1K'), texts.join(' | '))
  assert.ok(texts.includes('messages a month'))
  assert.ok(texts.includes('You') && texts.includes('Avery'))
  assert.ok(said(drawVoices(chart({ unit: 'day' }))).includes('messages a day'))
})

test('a voice that never wrote has no side of its own', () => {
  const double = drawVoices(chart({ points: [{ sent: 0, received: 40 }, { sent: 0, received: 10 }] }))

  assert.equal(bars(double, YOU).length, 0)
  assert.equal(bars(double, MARK).length, 2)
  assert.deepEqual(said(double).filter((text) => /^\d/.test(text)), ['40'])
})

test('nothing in the chart leaves the margins or the room it was given', () => {
  const long = 'a.remarkably.long.address.for.one.person.to.have@mail.example.com'
  const double = drawVoices(chart({ points: Array.from({ length: 300 }, (_, index) => ({ sent: index, received: 300 - index })) }), long)

  ;[...double.boxes, ...double.texts.map((item) => ({ ...item, top: item.base - item.size, bottom: item.base }))].forEach((item) => {
    assert.ok(item.left >= MARGIN - 0.01 && item.right <= WIDTH - MARGIN + 0.01, `from ${item.left} to ${item.right}`)
    assert.ok(item.top >= TOP - 0.01 && item.bottom <= TOP + HEIGHT + 0.01, `from ${item.top} to ${item.bottom}`)
  })
  const name = double.texts.find((item) => item.text.includes('…'))
  const unit = double.texts.find((item) => item.text === 'messages a month')
  assert.ok(name && unit && unit.left - name.right >= 16, 'the name gives way to what the bars count')
})

const around = (fields: Partial<DaysChart> = {}): DaysChart => ({
  counts: [3, 0, 12, 48, 6],
  at: 3,
  first: 'Nov 13',
  last: 'Nov 17',
  day: 'Nov 16',
  ...fields,
})

function drawDays(data: DaysChart) {
  const double = fakePen()
  days(data, { min: 200, max: 400, rank: 1 }).draw(asPen(double), TOP, HEIGHT)
  return double
}

test('the day the card is about is the one bar in ink', () => {
  const double = drawDays(around())
  const picked = filled(double, INK)
  const others = filled(double, FAINT)

  assert.equal(picked.length, 1)
  assert.equal(others.length, 4)
  // it is the fourth of the five, and the tallest
  assert.equal(others.filter((bar) => bar.left < picked[0].left).length, 3)
  assert.ok(others.every((bar) => bar.bottom - bar.top < picked[0].bottom - picked[0].top))
})

test('a day with nothing on it still has its place marked', () => {
  const quiet = filled(drawDays(around()), FAINT).sort((a, b) => a.left - b.left)[1]

  assert.ok(quiet.bottom - quiet.top >= 2 && quiet.bottom - quiet.top <= 6, `${quiet.bottom - quiet.top}`)
})

test('the days name their ends, their scale and the day itself', () => {
  const double = drawDays(around())
  const texts = said(double)

  assert.ok(texts.includes('Nov 13') && texts.includes('Nov 17') && texts.includes('Nov 16'), texts.join(' | '))
  assert.ok(texts.includes('50'), texts.join(' | '))
  const day = double.texts.find((item) => item.text === 'Nov 16')
  const [bar] = filled(double, INK)
  assert.ok(day && Math.abs((day.left + day.right) / 2 - (bar.left + bar.right) / 2) < 0.01, 'the day is named under its own bar')
})

test('a day at an end is named once, and never runs into the other end', () => {
  const atStart = drawDays(around({ counts: [9, 1, 1, 1, 1], at: 0, first: 'Nov 16', day: 'Nov 16', last: 'Nov 20' }))
  const beside = drawDays(around({ counts: Array.from({ length: 31 }, () => 4), at: 1, first: 'Nov 15', day: 'Nov 16', last: 'Dec 15' }))

  assert.equal(said(atStart).filter((text) => text === 'Nov 16').length, 1)
  assert.equal(said(beside).includes('Nov 15'), false)
  assert.ok(said(beside).includes('Nov 16') && said(beside).includes('Dec 15'))
  beside.texts.forEach((item) => assert.ok(item.left >= MARGIN - 0.01 && item.right <= WIDTH - MARGIN + 0.01))
})

test('the days stay inside the margins and the room they were given', () => {
  const double = drawDays(around({ counts: Array.from({ length: 31 }, (_, index) => index * 7), at: 30 }))

  ;[...double.boxes, ...double.texts.map((item) => ({ ...item, top: item.base - item.size, bottom: item.base }))].forEach((item) => {
    assert.ok(item.left >= MARGIN - 0.01 && item.right <= WIDTH - MARGIN + 0.01, `from ${item.left} to ${item.right}`)
    assert.ok(item.top >= TOP - 0.01 && item.bottom <= TOP + HEIGHT + 0.01, `from ${item.top} to ${item.bottom}`)
  })
})
