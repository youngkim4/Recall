import assert from 'node:assert/strict'
import { test } from 'node:test'
import { keyFigures, reactionKinds, sharedKinds, sidesOf } from './reportNumbers.ts'

test('what was shared is listed by kind, the most first, with nothing that was never shared', () => {
  const kinds = sharedKinds({ total: 1734, photos: 1403, videos: 19, audio: 0, gifs: 6, documents: 0, other: 306 })

  assert.deepEqual(kinds, [
    { label: 'Photos', count: 1403 },
    { label: 'Other', count: 306 },
    { label: 'Videos', count: 19 },
    { label: 'GIFs', count: 6 },
  ])
})

test('reactions are listed by kind in the words Messages uses', () => {
  const kinds = reactionKinds({ total: 9, loves: 4, likes: 0, dislikes: 1, laughs: 3, emphasis: 1, questions: 0 })

  // of two kinds with one count, the one Messages lists first comes first
  assert.deepEqual(kinds, [
    { label: 'Loved', count: 4 },
    { label: 'Laughed', count: 3 },
    { label: 'Emphasized', count: 1 },
    { label: 'Disliked', count: 1 },
  ])
})

test('nothing shared and no reactions are empty lists', () => {
  assert.deepEqual(sharedKinds(undefined), [])
  assert.deepEqual(sharedKinds({ total: 0 }), [])
  assert.deepEqual(reactionKinds({}), [])
})

test('a count that is not a number is left out', () => {
  assert.deepEqual(sharedKinds({ photos: Number.NaN, videos: 2 }), [{ label: 'Videos', count: 2 }])
})

test('the two sides of a count are given only when both are known', () => {
  assert.deepEqual(sidesOf({ sent_attachments: 10, received_attachments: 30 }, 'attachments'), { you: 10, them: 30 })
  assert.deepEqual(sidesOf({ sent_reactions: 2, received_reactions: 0 }, 'reactions'), { you: 2, them: 0 })
  assert.equal(sidesOf({ sent_attachments: 0, received_attachments: 0 }, 'attachments'), null)
  assert.equal(sidesOf({ photos: 3 }, 'attachments'), null)
  assert.equal(sidesOf(undefined, 'reactions'), null)
})

test('the figures a report leads with are the ones it has, each with one line about it', () => {
  const figures = keyFigures({
    totalMessages: 420,
    activeDays: 30,
    firstTimestamp: '2024-01-01 09:00:00',
    lastTimestamp: '2024-03-10 21:00:00',
    busiestDay: '2024-02-14',
    busiestDayCount: 61,
    longestGapDays: 12.4,
  })

  assert.deepEqual(
    figures.map((figure) => [figure.label, figure.value]),
    [
      ['Messages', '420'],
      ['Active days', '30'],
      ['Busiest day', '61'],
      ['Longest gap', '12 days'],
    ],
  )
  // 420 messages over the 70 days from the first to the last
  assert.equal(figures[0].note, '6 a day')
  assert.equal(figures[1].note, 'of 70')
  // the busiest day is named, so the figure can be found on the chart
  assert.ok(figures[2].note)
  assert.equal(figures[3].note, undefined)
})

test('a day count is given alone when every day of the span was active', () => {
  const figures = keyFigures({
    totalMessages: 9,
    activeDays: 3,
    firstTimestamp: '2024-01-01',
    lastTimestamp: '2024-01-03',
  })

  assert.deepEqual(
    figures.map((figure) => [figure.label, figure.value, figure.note ?? '']),
    [
      ['Messages', '9', '3 a day'],
      ['Active days', '3', ''],
    ],
  )
})

test('fewer than one message a day is not rounded up to a rate', () => {
  const [messages] = keyFigures({ totalMessages: 4, firstTimestamp: '2024-01-01', lastTimestamp: '2024-01-31' })

  assert.equal(messages.note, undefined)
})

test('a report with no figures leads with none', () => {
  assert.deepEqual(keyFigures({}), [])
})
