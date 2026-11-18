import assert from 'node:assert/strict'
import { test } from 'node:test'
import { fetchKept, forgetKept, keptAnswer } from './kept.ts'

// a loader that counts its calls and answers what it is told to
function counting(answer: (key: string) => Promise<string>) {
  const calls: string[] = []
  const loader = (key: string) => {
    calls.push(key)
    return answer(key)
  }
  return { loader, calls }
}

test('nothing is kept before a first answer', () => {
  const { loader } = counting(async (key) => `value of ${key}`)
  assert.equal(keptAnswer(loader, 'a'), undefined)
})

test('an answer is kept for its loader and its key only', async () => {
  const { loader } = counting(async (key) => `value of ${key}`)
  const other = counting(async () => 'other')
  assert.equal(await fetchKept(loader, 'a'), 'value of a')
  assert.deepEqual(keptAnswer(loader, 'a'), { value: 'value of a' })
  assert.equal(keptAnswer(loader, 'b'), undefined)
  assert.equal(keptAnswer(other.loader, 'a'), undefined)
})

test('two asks while one request is out share that request', async () => {
  let release: (value: string) => void = () => {}
  const { loader, calls } = counting(() => new Promise((resolve) => (release = resolve)))
  const first = fetchKept(loader, 'a')
  const second = fetchKept(loader, 'a')
  release('done')
  assert.deepEqual(await Promise.all([first, second]), ['done', 'done'])
  assert.equal(calls.length, 1)
  // once it has answered, the next ask goes to the server again
  release = () => {}
  const third = fetchKept(loader, 'a')
  release('again')
  assert.equal(await third, 'again')
  assert.equal(calls.length, 2)
  assert.deepEqual(keptAnswer(loader, 'a'), { value: 'again' })
})

test('a failure keeps the last good answer and is not kept itself', async () => {
  let fail = false
  const { loader } = counting(async () => {
    if (fail) throw new Error('down')
    return 'good'
  })
  await fetchKept(loader, 'a')
  fail = true
  await assert.rejects(fetchKept(loader, 'a'))
  assert.deepEqual(keptAnswer(loader, 'a'), { value: 'good' })
})

test('a loader that throws before it returns a promise fails like one that rejects', async () => {
  const loader = (): Promise<string> => {
    throw new Error('at once')
  }
  await assert.rejects(fetchKept(loader, 'a'))
  assert.equal(keptAnswer(loader, 'a'), undefined)
})

test('forgetting drops every kept answer, and a request out at that moment keeps nothing', async () => {
  let release: (value: string) => void = () => {}
  const { loader } = counting(async (key) => `value of ${key}`)
  await fetchKept(loader, 'a')
  const slow = counting(() => new Promise((resolve) => (release = resolve)))
  const pending = fetchKept(slow.loader, 'a')
  forgetKept()
  release('from before')
  assert.equal(await pending, 'from before')
  assert.equal(keptAnswer(loader, 'a'), undefined)
  assert.equal(keptAnswer(slow.loader, 'a'), undefined)
})

test('an empty key is never asked for', async () => {
  const { loader, calls } = counting(async () => 'x')
  await assert.rejects(fetchKept(loader, ''))
  assert.equal(calls.length, 0)
})
