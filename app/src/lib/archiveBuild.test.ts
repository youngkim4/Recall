import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { Job } from '../types.ts'
import { ApiError, apiErrorFrom } from './apiError.ts'
import { BUILD_FAILED, buildEnd, buildFailure, buildView, buildingJob, orBuilding } from './archiveBuild.ts'

const job = (fields: Partial<Job> = {}): Job => ({
  id: 'job-9',
  action: 'bridge',
  status: 'running',
  createdAt: '2026-11-17T09:00:00',
  updatedAt: '2026-11-17T09:00:05',
  logs: [],
  ...fields,
})

const building = (body: unknown) => apiErrorFrom(503, 'Service Unavailable', body)
const SENTENCE = 'Recall is getting your archive ready. This takes a moment.'

test('the job that loads the archive is read from the answer that says so', () => {
  const error = building({ error: SENTENCE, code: 'archive_building', job: job() })

  assert.deepEqual(buildingJob(error), job())
})

test('any other refusal names no such job', () => {
  assert.equal(buildingJob(building({ error: 'The load failed.', code: 'archive_failed', job: job() })), null)
  assert.equal(buildingJob(building({ error: 'There is no archive yet.', code: 'no_archive' })), null)
  assert.equal(buildingJob(new Error(SENTENCE)), null)
  assert.equal(buildingJob(new TypeError('Failed to fetch')), null)
  assert.equal(buildingJob(null), null)
})

test('an answer with no job to ask about is not waited on', () => {
  assert.equal(buildingJob(building({ error: SENTENCE, code: 'archive_building' })), null)
  assert.equal(buildingJob(building({ error: SENTENCE, code: 'archive_building', job: 'job-9' })), null)
  assert.equal(buildingJob(building({ error: SENTENCE, code: 'archive_building', job: { id: '' } })), null)
  assert.equal(buildingJob(building({ error: SENTENCE, code: 'archive_building', job: { id: 7, status: 'running' } })), null)
  assert.equal(buildingJob(new ApiError(SENTENCE, { status: 503, code: 'archive_building' })), null)
})

test('a request that failed because the archive is loading gives the job in place of its answer', async () => {
  const refused = Promise.reject(building({ error: SENTENCE, code: 'archive_building', job: job() }))

  assert.deepEqual(await refused.catch(orBuilding), { building: job() })
})

test('a request that failed for another reason still fails', async () => {
  const failed = apiErrorFrom(500, 'Internal Server Error', { error: 'The load failed.', code: 'archive_failed' })

  await assert.rejects(Promise.reject(failed).catch(orBuilding), failed)
  await assert.rejects(Promise.reject(new TypeError('Failed to fetch')).catch(orBuilding), TypeError)
})

test('the wait ends when the job ends, one way or the other', () => {
  assert.equal(buildEnd(job({ status: 'queued' })), 'running')
  assert.equal(buildEnd(job({ status: 'running' })), 'running')
  assert.equal(buildEnd(job({ status: 'completed' })), 'ready')
  assert.equal(buildEnd(job({ status: 'failed' })), 'failed')
})

test('a load that failed says why in the words of the server', () => {
  assert.equal(buildFailure(job({ status: 'failed', error: 'The earlier export could not be read.' })), 'The earlier export could not be read.')
  assert.equal(buildFailure(job({ status: 'failed', error: null })), BUILD_FAILED)
  assert.equal(buildFailure(job({ status: 'failed' })), BUILD_FAILED)
})

test('the window says what the load is doing, and the last line it wrote', () => {
  const running = job({ progress: { step: 2, of: 3, label: 'Writing the archive' }, logs: ['Read 48,200 messages'] })

  assert.deepEqual(buildView(running), { label: 'Writing the archive', said: 'Read 48,200 messages' })
})

test('a load that has not said what it is doing still has a line', () => {
  assert.deepEqual(buildView(job()), { label: 'Loading your messages', said: '' })
  assert.deepEqual(buildView(job({ progress: { step: 1, of: 3, label: '  ' } })), { label: 'Loading your messages', said: '' })
})

test('the same words are not said twice', () => {
  const running = job({ progress: { step: 1, of: 3, label: 'Reading the export' }, logs: ['Reading the export'] })

  assert.deepEqual(buildView(running), { label: 'Reading the export', said: '' })
})
