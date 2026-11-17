import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { Job } from '../types'
import { isActiveJob, jobTitle, lastLog, mergeJobs } from './jobs.ts'

const job = (fields: Partial<Job> = {}): Job => ({
  id: 'job-1',
  action: 'analyze',
  status: 'completed',
  createdAt: '2026-11-10T09:00:00',
  updatedAt: '2026-11-10T09:05:00',
  logs: [],
  ...fields,
})

test('a job is active until it has finished one way or the other', () => {
  assert.equal(isActiveJob(job({ status: 'queued' })), true)
  assert.equal(isActiveJob(job({ status: 'running' })), true)
  assert.equal(isActiveJob(job({ status: 'completed' })), false)
  assert.equal(isActiveJob(job({ status: 'failed' })), false)
  assert.equal(isActiveJob(null), false)
  assert.equal(isActiveJob(undefined), false)
})

test('jobs are listed newest first', () => {
  const older = job({ id: 'older', updatedAt: '2026-11-10T09:05:00' })
  const newer = job({ id: 'newer', updatedAt: '2026-11-12T18:30:00' })

  assert.deepEqual(mergeJobs([older, newer]).map((item) => item.id), ['newer', 'older'])
})

test('a job this session is following stands in for the copy the server listed', () => {
  const listed = job({ id: 'a', status: 'running', updatedAt: '2026-11-10T09:05:00' })
  const followed = job({ id: 'a', status: 'completed', updatedAt: '2026-11-10T09:07:00' })
  const unlisted = job({ id: 'b', status: 'queued', updatedAt: '2026-11-10T09:08:00' })

  const merged = mergeJobs([listed], followed, null, unlisted)

  assert.deepEqual(merged.map((item) => `${item.id} ${item.status}`), ['b queued', 'a completed'])
})

test('a job that is followed but has no id yet is not listed', () => {
  assert.deepEqual(mergeJobs([job({ id: 'a' })], job({ id: '' })).map((item) => item.id), ['a'])
})

test('a job with no time goes after the ones that have one', () => {
  const dated = job({ id: 'dated', updatedAt: '2026-11-10T09:05:00' })
  const started = job({ id: 'started', updatedAt: '', createdAt: '2026-11-11T08:00:00' })
  const timeless = job({ id: 'timeless', updatedAt: undefined as unknown as string, createdAt: '' })

  assert.deepEqual(mergeJobs([timeless, dated, started]).map((item) => item.id), ['started', 'dated', 'timeless'])
})

test('the last thing a job said is its last line a person would read', () => {
  assert.equal(lastLog(['Reading messages', { message: 'Writing the report' }]), 'Writing the report')
  // the server's own record of a request is not for the reader
  assert.equal(lastLog(['Reading messages', '127.0.0.1 - - "GET /api/jobs HTTP/1.1" 200 -']), 'Reading messages')
  assert.equal(lastLog(['', { time: '09:00' }]), '')
  assert.equal(lastLog([]), '')
  assert.equal(lastLog(undefined), '')
})

test('a job is named for what it does', () => {
  const contacts = [{ chat_id: '+15550100199', display_name: 'Avery Stone' }]

  assert.equal(jobTitle(job({ action: 'export' }), contacts), 'Message export')
  assert.equal(jobTitle(job({ action: 'setup_import' }), contacts), 'Archive import')
  assert.equal(jobTitle(job({ action: 'bridge' }), contacts), 'Archive load')
  // a kind of job that is gone has no name of its own
  assert.equal(jobTitle(job({ action: 'semantic' }), contacts), 'semantic')
  assert.equal(jobTitle(job({ action: 'something_new' }), contacts), 'something_new')
})

test('a report job is named for its conversation', () => {
  const contacts = [{ chat_id: '+15550100199', display_name: 'Avery Stone' }]

  assert.equal(jobTitle(job({ contact: '+15550100199' }), contacts), 'Report on Avery Stone')
  assert.equal(jobTitle(job({ contact: '+15550100142' }), contacts), 'Report on Number ending in 0142')
  assert.equal(jobTitle(job({ contact: 'chat4471' }), []), 'Report on Unnamed group')
  assert.equal(jobTitle(job({ contact: '' }), contacts), 'Report')
  assert.equal(jobTitle(job(), contacts), 'Report')
})
