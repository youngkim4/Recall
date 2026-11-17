import assert from 'node:assert/strict'
import { test } from 'node:test'
import { leadingMoments, momentsByYear, placeOnSpan, readMoments, readSaid, spanTicks } from './reportMoments.ts'

const event = (date: string, title: string, score: number | null = 0.5) => ({
  date,
  title,
  detail: `${title} in detail`,
  category: 'turning_point',
  score,
  quote: '',
})

test('moments are read oldest first, each with a key of its own', () => {
  const moments = readMoments([event('2024-05-02', 'B'), event('2023-11-20', 'A'), event('2024-05-02', 'C')])

  assert.deepEqual(
    moments.map((moment) => moment.title),
    ['A', 'B', 'C'],
  )
  assert.equal(new Set(moments.map((moment) => moment.key)).size, 3)
})

test('a moment with no day that can be read comes last and has no day', () => {
  const moments = readMoments([event('sometime', 'Undated'), event('2024-02-30', 'Not a day'), event('2024-01-05', 'Dated')])

  assert.deepEqual(
    moments.map((moment) => [moment.title, moment.day]),
    [
      ['Dated', '2024-01-05'],
      ['Undated', ''],
      ['Not a day', ''],
    ],
  )
})

test('a day written with a time keeps only the day', () => {
  assert.equal(readMoments([event('2024-03-09 18:22:01', 'A')])[0].day, '2024-03-09')
})

test('how much a moment matters is held between nothing and everything', () => {
  const moments = readMoments([event('2024-01-01', 'A', 4), event('2024-01-02', 'B', -1), event('2024-01-03', 'C', null)])

  assert.deepEqual(
    moments.map((moment) => moment.weight),
    [1, 0, 0],
  )
})

test('the kind of a moment is given in words', () => {
  assert.equal(readMoments([event('2024-01-01', 'A')])[0].kind, 'Turning point')
})

test('the leading moments are the ones that matter most, the earlier of two equals first', () => {
  const moments = readMoments([
    event('2024-01-01', 'A', 0.2),
    event('2024-02-01', 'B', 0.9),
    event('2024-03-01', 'C', 0.9),
    event('2024-04-01', 'D', 0.6),
  ])
  const leading = leadingMoments(moments, 2)

  assert.deepEqual(
    moments.filter((moment) => leading.has(moment.key)).map((moment) => moment.title),
    ['B', 'C'],
  )
  assert.equal(leadingMoments(moments, 10).size, 4)
})

test('moments are grouped under their years, undated ones under none', () => {
  const groups = momentsByYear(readMoments([event('2023-12-31', 'A'), event('2024-01-01', 'B'), event('', 'C')]))

  assert.deepEqual(
    groups.map((group) => [group.year, group.moments.map((moment) => moment.title)]),
    [
      ['2023', ['A']],
      ['2024', ['B']],
      ['', ['C']],
    ],
  )
})

test('a day has its place along the span from the first day to the last', () => {
  assert.equal(placeOnSpan('2024-01-01', '2024-01-01', '2024-01-11'), 0)
  assert.equal(placeOnSpan('2024-01-06', '2024-01-01', '2024-01-11'), 0.5)
  assert.equal(placeOnSpan('2024-01-11', '2024-01-01', '2024-01-11'), 1)
})

test('a day outside the span is held at its nearer end, and a span of one day is its middle', () => {
  assert.equal(placeOnSpan('2023-06-01', '2024-01-01', '2024-01-11'), 0)
  assert.equal(placeOnSpan('2025-06-01', '2024-01-01', '2024-01-11'), 1)
  assert.equal(placeOnSpan('2024-01-01', '2024-01-01', '2024-01-01'), 0.5)
  assert.equal(placeOnSpan('', '2024-01-01', '2024-01-11'), null)
})

test('a span of a few months is marked at the first of each month', () => {
  const ticks = spanTicks('2024-03-23', '2024-06-17', 'en-US')

  assert.deepEqual(
    ticks.map((tick) => tick.label),
    ['Apr', 'May', 'Jun'],
  )
  assert.ok(ticks.every((tick) => tick.at > 0 && tick.at < 1))
  assert.ok(ticks[0].at < ticks[1].at && ticks[1].at < ticks[2].at)
})

test('a span of years is marked at the start of each year', () => {
  const ticks = spanTicks('2019-08-02', '2024-03-01', 'en-US')

  assert.deepEqual(
    ticks.map((tick) => tick.label),
    ['2020', '2021', '2022', '2023', '2024'],
  )
})

test('a year of months is thinned so the marks do not crowd', () => {
  const ticks = spanTicks('2023-01-15', '2024-06-20', 'en-US')

  assert.ok(ticks.length <= 9)
  // the year is named where it changes
  assert.ok(ticks.some((tick) => tick.label === '2024'))
})

test('a span with no days in it has no marks', () => {
  assert.deepEqual(spanTicks('', '', 'en-US'), [])
  assert.deepEqual(spanTicks('2024-01-01', '2024-01-01', 'en-US'), [])
})

test('a quote that names its speakers is read as the messages it was', () => {
  assert.deepEqual(readSaid('ME: "on my way"\nTHEM: "ok see you soon"'), [
    { fromMe: true, sender: '', text: 'on my way' },
    { fromMe: false, sender: '', text: 'ok see you soon' },
  ])
})

test('a speaker named without quote marks is read the same way', () => {
  assert.deepEqual(readSaid('THEM: running late again'), [{ fromMe: false, sender: '', text: 'running late again' }])
})

test('two quoted pieces set on one line with a slash are two messages', () => {
  assert.deepEqual(readSaid('"did you eat" / "not yet"'), [
    { fromMe: null, sender: '', text: 'did you eat' },
    { fromMe: null, sender: '', text: 'not yet' },
  ])
  assert.deepEqual(readSaid('ME: "did you eat" / THEM: "not yet"'), [
    { fromMe: true, sender: '', text: 'did you eat' },
    { fromMe: false, sender: '', text: 'not yet' },
  ])
})

test('a slash inside one message does not cut it', () => {
  assert.deepEqual(readSaid('"tea and/or coffee, your call / whatever"'), [
    { fromMe: null, sender: '', text: 'tea and/or coffee, your call / whatever' },
  ])
})

test('a line that carries its time and the name of who wrote it gives up both', () => {
  assert.deepEqual(readSaid('[2024-05-12 21:14] Avery Stone: see you at eight\n[2024-05-12 21:15] ME: perfect'), [
    { fromMe: false, sender: 'Avery Stone', text: 'see you at eight' },
    { fromMe: true, sender: '', text: 'perfect' },
  ])
})

test('a colon in what was said is not a speaker', () => {
  assert.deepEqual(readSaid('"note to self: buy milk"'), [{ fromMe: null, sender: '', text: 'note to self: buy milk' }])
  assert.deepEqual(readSaid('Reminder: call at 5:30'), [{ fromMe: null, sender: '', text: 'Reminder: call at 5:30' }])
})

test('a speaker the export could not tell is nobody', () => {
  assert.deepEqual(readSaid('??: "hello?"'), [{ fromMe: null, sender: '', text: 'hello?' }])
})

test('a quote with no words has no lines', () => {
  assert.deepEqual(readSaid(''), [])
  assert.deepEqual(readSaid('  \n ""  '), [])
})

test('a moment carries what it quotes as lines', () => {
  const [moment] = readMoments([{ ...event('2024-01-01', 'A'), quote: 'ME: "hi"' }])

  assert.deepEqual(moment.said, [{ fromMe: true, sender: '', text: 'hi' }])
})
