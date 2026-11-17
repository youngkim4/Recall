import assert from 'node:assert/strict'
import { test } from 'node:test'
import { place, type Block } from './cardLayout.ts'

const part = (min: number, more: Partial<Block> = {}): Block => ({ min, draw: () => {}, ...more })
const spans = (blocks: Block[], top: number, bottom: number, gap: number) =>
  place(blocks, top, bottom, gap).map(({ top: at, height }) => [at, height])

test('the parts sit a gap apart, together in the middle of the room', () => {
  // 100 + 40 + 60 = 200 of the 400: 100 is left above and 100 below
  assert.deepEqual(spans([part(100), part(60)], 0, 400, 40), [
    [100, 100],
    [240, 60],
  ])
})

test('a part that can grow takes the room that is left, up to its most', () => {
  assert.deepEqual(spans([part(100), part(60, { max: 200 })], 0, 400, 40), [
    [30, 100],
    [170, 200],
  ])
  // all of it, when the room is less than its most
  assert.deepEqual(spans([part(100), part(60, { max: 900 })], 0, 400, 40), [
    [0, 100],
    [140, 260],
  ])
})

test('two parts that can grow are filled in their order', () => {
  assert.deepEqual(spans([part(50, { max: 120 }), part(50, { max: 500 })], 0, 400, 0), [
    [0, 120],
    [120, 280],
  ])
})

test('when the parts do not fit, the least important is left off first', () => {
  const hero = part(200)
  const chart = part(150, { rank: 1 })
  const facts = part(100, { rank: 2 })

  const placed = place([hero, chart, facts], 0, 420, 20)

  assert.deepEqual(
    placed.map((entry) => entry.block),
    [hero, chart],
  )
  assert.deepEqual(place([hero, chart, facts], 0, 300, 20).map((entry) => entry.block), [hero])
})

test('of two parts that matter the same, the later one goes first', () => {
  const first = part(100, { rank: 1 })
  const second = part(100, { rank: 1 })

  assert.deepEqual(place([part(100), first, second], 0, 230, 10).map((entry) => entry.block).includes(second), false)
})

test('a part that must be there stays, even in too little room', () => {
  const hero = part(500)

  assert.deepEqual(place([hero, part(100, { rank: 1 })], 0, 300, 20), [{ block: hero, top: 0, height: 500 }])
})

test('the parts keep their order', () => {
  const blocks = [part(10), part(20, { rank: 3 }), part(30, { rank: 1 }), part(40)]

  assert.deepEqual(
    place(blocks, 0, 1000, 5).map((entry) => entry.block.min),
    [10, 20, 30, 40],
  )
})

test('no parts take no room', () => {
  assert.deepEqual(place([], 0, 400, 40), [])
})
