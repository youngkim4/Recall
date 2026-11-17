import assert from 'node:assert/strict'
import { test } from 'node:test'
import { busiestHour, heatLevel, heatSteps, monthMarks, weeksEnding, type HeatSteps } from './heat.ts'

test('the steps of a heat scale split the days that have messages into quarters', () => {
  // zeros are days with nothing; they take no part in the scale
  assert.deepEqual(heatSteps([0, 0, 1, 2, 3, 4, 5, 6, 7, 8]), [2, 4, 6])
  assert.deepEqual(heatSteps([0, 0, 0]), [0, 0, 0])
})

test('a count falls on one of five levels, and nothing is always the lowest', () => {
  const steps: HeatSteps = [2, 4, 6]

  assert.equal(heatLevel(0, steps), 0)
  assert.equal(heatLevel(1, steps), 1)
  assert.equal(heatLevel(2, steps), 1)
  assert.equal(heatLevel(3, steps), 2)
  assert.equal(heatLevel(6, steps), 3)
  assert.equal(heatLevel(900, steps), 4)
})

test('when every busy day is alike they all take the top level', () => {
  assert.equal(heatLevel(5, heatSteps([5, 5, 5, 0])), 4)
})

const levelsOf = (counts: number[]) => {
  const steps = heatSteps(counts)
  return counts.map((count) => heatLevel(count, steps))
}

test('the busiest count takes the top level however many days share it', () => {
  // six days of eight are as busy as it gets
  assert.deepEqual(levelsOf([1, 2, 3, 3, 3, 3, 3, 3]), [1, 1, 4, 4, 4, 4, 4, 4])
  assert.deepEqual(levelsOf([1, 5, 5, 5, 5, 5, 5, 5]), [1, 4, 4, 4, 4, 4, 4, 4])
  // half the days tie at the top: the steps under them keep their own levels
  assert.deepEqual(levelsOf([1, 2, 3, 3]), [1, 2, 4, 4])
})

test('the quietest busy count stays on the lowest level however many days share it', () => {
  assert.deepEqual(levelsOf([1, 1, 1, 1, 1, 1, 1, 2, 5]), [1, 1, 1, 1, 1, 1, 1, 4, 4])
  assert.deepEqual(levelsOf([0, 1, 1, 1, 9]), [0, 1, 1, 1, 4])
})

// what the levels were before ties were handled: right whenever the quarter marks differ
const plainLevel = (count: number, sorted: number[]) => {
  if (count <= 0) return 0
  const at = (share: number) => sorted[Math.ceil(sorted.length * share) - 1]
  return 1 + [at(0.25), at(0.5), at(0.75)].filter((step) => count > step).length
}

test('a heat scale holds its rules for any counts', () => {
  // a fixed sequence, so a failure can be run again
  const sequence = (seed: number) => {
    const next = (state: number) => (state * 1103515245 + 12345) % 2147483648
    return (length: number, most: number) =>
      Array.from({ length }).reduce<{ state: number; values: number[] }>(
        (run) => {
          const state = next(run.state)
          return { state, values: [...run.values, state % (most + 1)] }
        },
        { state: seed, values: [] },
      ).values
  }

  Array.from({ length: 400 }, (_, round) => round).forEach((round) => {
    const counts = sequence(round + 1)(1 + (round % 40), 1 + (round % 7) * (round % 3 === 0 ? 1 : 30))
    const steps = heatSteps(counts)
    const busy = counts.filter((count) => count > 0).sort((a, b) => a - b)
    const about = `counts ${JSON.stringify(counts)}`

    counts.forEach((count) => {
      const level = heatLevel(count, steps)
      assert.equal(level === 0, count === 0, `nothing, and only nothing, is level 0: ${about}`)
      assert.ok(level >= 0 && level <= 4, `a level is 0 to 4: ${about}`)
    })
    busy.slice(1).forEach((count, index) => {
      assert.ok(heatLevel(count, steps) >= heatLevel(busy[index], steps), `a level never falls as the count rises: ${about}`)
    })
    if (!busy.length) return
    const least = busy[0]
    const most = busy[busy.length - 1]
    assert.equal(heatLevel(most, steps), 4, `the busiest count is level 4: ${about}`)
    if (least < most) assert.equal(heatLevel(least, steps), 1, `the quietest busy count is level 1: ${about}`)

    const marks = [0.25, 0.5, 0.75].map((share) => busy[Math.ceil(busy.length * share) - 1])
    if (marks[0] < marks[1] && marks[1] < marks[2] && marks[2] < most) {
      busy.forEach((count) => {
        assert.equal(heatLevel(count, steps), plainLevel(count, busy), `distinct quarters keep their levels: ${about}`)
      })
    }
  })
})

test('weeks run from the day the week starts on and end with the week of the last day', () => {
  // 20 March 2025 is a Thursday
  const weeks = weeksEnding('2025-03-20', 2, 0)

  assert.deepEqual(weeks, [
    ['2025-03-09', '2025-03-10', '2025-03-11', '2025-03-12', '2025-03-13', '2025-03-14', '2025-03-15'],
    ['2025-03-16', '2025-03-17', '2025-03-18', '2025-03-19', '2025-03-20', null, null],
  ])
})

test('a week that starts on Monday puts Sunday last', () => {
  const weeks = weeksEnding('2025-03-20', 1, 1)

  assert.deepEqual(weeks, [['2025-03-17', '2025-03-18', '2025-03-19', '2025-03-20', null, null, null]])
})

test('a month is marked on the first week that holds its first day', () => {
  const weeks = weeksEnding('2025-03-20', 8, 0)

  // weeks start on 26 January, then 2 February … 16 March; 1 February is in the first of them, 1 March in the fifth
  assert.deepEqual(monthMarks(weeks, 'en-US'), [
    { week: 0, text: 'Feb' },
    { week: 4, text: 'Mar' },
  ])
})

test('the busiest hour is one hour of one weekday', () => {
  const rhythm = Array.from({ length: 7 }, () => Array.from({ length: 24 }, () => 0))
  const withPeak = rhythm.map((day, weekday) => day.map((count, hour) => (weekday === 1 && hour === 21 ? 12 : count)))

  // rows run from Monday, so row 1 is Tuesday
  assert.deepEqual(busiestHour(withPeak), { weekday: 1, hour: 21, count: 12 })
  assert.equal(busiestHour(rhythm), null)
})
