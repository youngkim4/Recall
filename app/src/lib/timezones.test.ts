import assert from 'node:assert/strict'
import { test } from 'node:test'
import { axisLabels, pointName } from './activity.ts'
import { addDays, daysSpanned, rangeLabel, weeksOf, type Daily } from './days.ts'
import { lastDayOf } from './months.ts'

// Clocks change at midnight in some of these, and two of them once left a
// whole day out: 30 December 2011 never came in Apia, nor 31 December 1994
// in Kiritimati.
const ZONES = [
  'UTC',
  'America/Los_Angeles',
  'America/Sao_Paulo',
  'America/Havana',
  'Asia/Beirut',
  'Asia/Tehran',
  'Asia/Kolkata',
  'Pacific/Apia',
  'Pacific/Kiritimati',
]

// 1 December 1994 to the end of 2024, one message a day
const YEARS: Daily = { start: '1994-12-01', counts: Array.from({ length: 10_989 }, () => 1) }
const plain = (text: string) => text.replace(/[\u2009\u202f]/g, ' ')
const label = (since: string, until: string) => plain(rangeLabel({ since, until }, YEARS, 'en-US'))
const point = (key: string) => ({ key, total: 1, sent: 1, received: 0 })

function inEveryZone(check: (zone: string) => void) {
  const home = process.env.TZ
  try {
    ZONES.forEach((zone) => {
      process.env.TZ = zone
      check(zone)
    })
  } finally {
    if (home === undefined) delete process.env.TZ
    else process.env.TZ = home
  }
}

test('a day is the same day in every time zone', () => {
  inEveryZone((zone) => {
    assert.equal(addDays('2018-11-03', 1), '2018-11-04', zone)
    assert.equal(addDays('2011-12-29', 1), '2011-12-30', zone)
    assert.equal(daysSpanned('2011-12-29 09:00:00', '2011-12-31 09:00:00'), 3, zone)
    assert.equal(addDays(YEARS.start, YEARS.counts.length - 1), '2024-12-31', zone)
  })
})

test('a month has the same days in every time zone', () => {
  inEveryZone((zone) => {
    assert.equal(lastDayOf('1994-12'), '1994-12-31', zone)
    assert.equal(lastDayOf('2011-12'), '2011-12-31', zone)
    assert.equal(lastDayOf('2024-02'), '2024-02-29', zone)

    // December 2011 starts on a Thursday and ends on a Saturday
    const weeks = weeksOf('2011-12', 0)
    assert.equal(weeks.length, 5, zone)
    assert.deepEqual(weeks[0], [null, null, null, null, '2011-12-01', '2011-12-02', '2011-12-03'], zone)
    assert.deepEqual(
      weeks[4],
      ['2011-12-25', '2011-12-26', '2011-12-27', '2011-12-28', '2011-12-29', '2011-12-30', '2011-12-31'],
      zone,
    )
  })
})

test('a day is named the same in every time zone', () => {
  inEveryZone((zone) => {
    assert.equal(label('2018-11-04', '2018-11-04'), 'Nov 4, 2018', zone)
    assert.equal(label('2011-12-30', '2011-12-30'), 'Dec 30, 2011', zone)
    assert.equal(label('2011-12-29', '2011-12-31'), 'Dec 29 – 31, 2011', zone)
    assert.equal(label('2011-12-01', '2011-12-31'), 'December 2011', zone)
    assert.equal(label('1994-12-31', '1994-12-31'), 'Dec 31, 1994', zone)
    assert.equal(label('1994-12-01', '1994-12-31'), '1994', zone)
    assert.equal(pointName('2011-12-30', 'day', 'en-US'), 'Fri, Dec 30, 2011', zone)
    assert.equal(pointName('1994-12-31', 'day', 'en-US'), 'Sat, Dec 31, 1994', zone)
    assert.equal(pointName('1994-12', 'month', 'en-US'), 'December 1994', zone)
  })
})

test('an axis is labelled the same in every time zone', () => {
  const fiveDays = ['2011-12-28', '2011-12-29', '2011-12-30', '2011-12-31', '2012-01-01'].map(point)
  const threeMonths = ['1994-11', '1994-12', '1995-01'].map(point)

  inEveryZone((zone) => {
    assert.deepEqual(
      axisLabels(fiveDays, 'day', 'en-US').map((entry) => entry.text),
      ['Dec 28', '29', '30', '31', 'Jan 1'],
      zone,
    )
    assert.deepEqual(
      axisLabels(threeMonths, 'month', 'en-US').map((entry) => entry.text),
      ['Nov', 'Dec', 'Jan'],
      zone,
    )
  })
})
