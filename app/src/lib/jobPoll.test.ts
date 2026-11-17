import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { Job } from '../types.ts'
import { makePoller } from './jobPoll.ts'

const job = (status: Job['status'], fields: Partial<Job> = {}): Job => ({
  id: 'job-1',
  action: 'bridge',
  status,
  createdAt: '2026-11-17T09:00:00',
  updatedAt: '2026-11-17T09:00:05',
  logs: [],
  ...fields,
})

// a server that gives the answers it was handed, one for each question
function server(answers: Array<Job | Error>) {
  const state = { asked: 0 }
  return {
    asked: () => state.asked,
    ask: async () => {
      const answer = answers[Math.min(state.asked, answers.length - 1)]
      state.asked += 1
      if (answer instanceof Error) throw answer
      return answer
    },
  }
}

// a question that is answered only when the test says so
function heldQuestion() {
  const hold: { answer: (job: Job) => void } = { answer: () => undefined }
  const asked = new Promise<Job>((resolve) => {
    hold.answer = resolve
  })
  return { ask: () => asked, answer: (next: Job) => hold.answer(next) }
}

test('each answer about the job is handed on', async () => {
  const seen: string[] = []
  const from = server([job('running'), job('running'), job('completed')])
  const poller = makePoller({ ask: from.ask, onJob: (next) => void seen.push(next.status), onLost: () => seen.push('lost'), lostAfter: 3 })

  await poller.tick()
  await poller.tick()
  await poller.tick()

  assert.deepEqual(seen, ['running', 'running', 'completed'])
})

test('a job that has ended is not asked about again', async () => {
  const from = server([job('failed', { error: 'The export could not be read.' })])
  const seen: string[] = []
  const poller = makePoller({ ask: from.ask, onJob: (next) => void seen.push(next.status), onLost: () => seen.push('lost'), lostAfter: 3 })

  await poller.tick()
  await poller.tick()

  assert.equal(from.asked(), 1)
  assert.deepEqual(seen, ['failed'])
})

test('one question is asked at a time', async () => {
  const held = heldQuestion()
  const state = { asked: 0 }
  const seen: string[] = []
  const poller = makePoller({
    ask: () => {
      state.asked += 1
      return held.ask()
    },
    onJob: (next) => void seen.push(next.status),
    onLost: () => seen.push('lost'),
    lostAfter: 3,
  })

  const first = poller.tick()
  await poller.tick()
  assert.equal(state.asked, 1)

  held.answer(job('running'))
  await first
  assert.deepEqual(seen, ['running'])
})

test('one unanswered question is not an ending, several in a row are', async () => {
  const from = server([new Error('down'), new Error('down'), new Error('down'), job('running')])
  const seen: string[] = []
  const poller = makePoller({ ask: from.ask, onJob: (next) => void seen.push(next.status), onLost: () => seen.push('lost'), lostAfter: 3 })

  await poller.tick()
  await poller.tick()
  assert.deepEqual(seen, [])

  await poller.tick()
  assert.deepEqual(seen, ['lost'])

  // a job that was lost is not asked about again
  await poller.tick()
  assert.equal(from.asked(), 3)
})

test('an answer starts the count of unanswered questions again', async () => {
  const from = server([new Error('down'), new Error('down'), job('running'), new Error('down'), new Error('down'), job('running')])
  const seen: string[] = []
  const poller = makePoller({ ask: from.ask, onJob: (next) => void seen.push(next.status), onLost: () => seen.push('lost'), lostAfter: 3 })

  // six questions, one after the other
  await Array.from({ length: 6 }).reduce<Promise<void>>((before) => before.then(poller.tick), Promise.resolve())

  assert.deepEqual(seen, ['running', 'running'])
})

test('after the watch is over, an answer on its way changes nothing', async () => {
  const held = heldQuestion()
  const seen: string[] = []
  const poller = makePoller({ ask: held.ask, onJob: (next) => void seen.push(next.status), onLost: () => seen.push('lost'), lostAfter: 1 })

  const asked = poller.tick()
  poller.stop()
  held.answer(job('completed'))
  await asked

  assert.deepEqual(seen, [])
})

test('after the watch is over, a question that fails is not a loss', async () => {
  const seen: string[] = []
  const hold: { fail: () => void } = { fail: () => undefined }
  const asked = new Promise<Job>((_, reject) => {
    hold.fail = () => reject(new Error('down'))
  })
  const poller = makePoller({ ask: () => asked, onJob: (next) => void seen.push(next.status), onLost: () => seen.push('lost'), lostAfter: 1 })

  const ticking = poller.tick()
  poller.stop()
  hold.fail()
  await ticking

  assert.deepEqual(seen, [])
})

test('work on an answer can find out that the watch ended meanwhile', async () => {
  const from = server([job('completed')])
  const seen: boolean[] = []
  const poller = makePoller({
    ask: from.ask,
    onJob: async (_, watched) => {
      seen.push(watched())
      poller.stop()
      seen.push(watched())
    },
    onLost: () => undefined,
    lostAfter: 3,
  })

  await poller.tick()

  assert.deepEqual(seen, [true, false])
})

test('the next question waits for the work on the last answer', async () => {
  const from = server([job('running')])
  const work: { finish: () => void } = { finish: () => undefined }
  const working = new Promise<void>((resolve) => {
    work.finish = resolve
  })
  const poller = makePoller({ ask: from.ask, onJob: () => working, onLost: () => undefined, lostAfter: 3 })

  const first = poller.tick()
  // let the first answer arrive
  await Promise.resolve()
  await Promise.resolve()
  await poller.tick()
  assert.equal(from.asked(), 1)

  work.finish()
  await first
  await poller.tick()
  assert.equal(from.asked(), 2)
})
