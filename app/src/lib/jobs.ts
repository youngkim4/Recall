import type { Contact, Job } from '../types'
import { conversationName } from './conversations.ts'

export function isActiveJob(job: Job | null | undefined) {
  return job?.status === 'queued' || job?.status === 'running'
}

// when a job last changed, or failing that when it began; '' sorts after every time
const lastChange = (job: Job) => job.updatedAt || job.createdAt || ''

// server list plus the jobs this session is polling, newest first
export function mergeJobs(jobs: Job[], ...polled: Array<Job | null>) {
  const merged = polled.reduce(
    (byId, job) => (job?.id ? new Map(byId).set(job.id, job) : byId),
    new Map<string, Job>(jobs.map((job) => [job.id, job])),
  )
  return [...merged.values()].sort((left, right) => lastChange(right).localeCompare(lastChange(left)))
}

// the last line a job wrote that a person would read; the server's own request log is skipped
export function lastLog(logs: Job['logs'] = []) {
  const lines = logs.map((entry) => (typeof entry === 'string' ? entry : entry?.message || ''))
  return lines.findLast((line) => line && !line.startsWith('127.0.0.1 - -')) ?? ''
}

const KINDS: Record<string, string> = {
  export: 'Message export',
  setup_import: 'Archive import',
  semantic: 'Meaning index',
}

// What a job is, in words: "Report on Sam", "Message export".
export function jobTitle(job: Job, contacts: Contact[]) {
  if (job.action !== 'analyze') return KINDS[job.action] ?? job.action
  if (!job.contact) return 'Report'
  const saved = contacts.find((contact) => contact.chat_id === job.contact)
  return `Report on ${conversationName(saved?.display_name || saved?.displayName, job.contact)}`
}
