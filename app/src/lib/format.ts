import type { Contact, RecentMessage, ReportFile } from '../types'

export function formatNumber(value: number | null | undefined) {
  if (!Number.isFinite(value)) return '--'
  return new Intl.NumberFormat('en-US').format(Number(value))
}

export function formatMoney(value: number | null | undefined) {
  if (!Number.isFinite(value)) return '--'
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 2,
  }).format(Number(value))
}

export function shortDate(value?: string) {
  if (!value) return '--'
  const date = new Date(value)
  if (Number.isNaN(date.valueOf())) return value.slice(0, 10)
  return date.toISOString().slice(0, 10)
}

export function shortDateTime(value?: string) {
  if (!value) return '--'
  const date = new Date(value)
  if (Number.isNaN(date.valueOf())) return value.slice(0, 16)
  return new Intl.DateTimeFormat('en-US', {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(date)
}

export function displayContact(value?: string) {
  if (!value) return 'Unknown'
  if (value.startsWith('chat')) return `Group ${value.replace(/\D/g, '').slice(-4) || value.slice(-4)}`
  if (value.length > 10 && value.startsWith('+')) {
    return 'Unnamed contact'
  }
  return value
}

export function prettyContactId(value?: string) {
  if (!value) return 'Unknown'
  if (value.includes('@')) return value
  if (value.startsWith('chat')) {
    const tail = value.replace(/\D/g, '').slice(-4)
    return tail ? `Group ${tail}` : 'Group chat'
  }
  const digits = value.replace(/\D/g, '')
  const local = digits.length === 11 && digits.startsWith('1') ? digits.slice(1) : digits
  if (local.length === 10) {
    return `(${local.slice(0, 3)}) ${local.slice(3, 6)}-${local.slice(6)}`
  }
  return value
}

// Label for a contact that is NOT saved in Contacts. Never shows a full
// number — keeps last 4 digits so the user can still recognize the thread.
export function contactLabel(displayName?: string, chatId?: string) {
  const name = (displayName || '').trim()
  if (name) return name
  const id = (chatId || '').trim()
  if (!id) return 'Unsaved contact'
  if (id.startsWith('chat')) {
    const tail = id.replace(/\D/g, '').slice(-4)
    return tail ? `Group ${tail}` : 'Group chat'
  }
  if (id.includes('@')) return 'Unsaved contact'
  const digits = id.replace(/\D/g, '')
  return digits ? `Unsaved contact ·${digits.slice(-4)}` : 'Unsaved contact'
}

export function contactTitle(contact?: Contact) {
  if (!contact) return 'Select a conversation'
  return contact.display_name || contact.displayName || displayContact(contact.chat_id)
}

export function reportTitle(report?: ReportFile) {
  if (!report) return 'No report selected'
  return report.displayName || displayContact(report.contact) || report.name
}

export function initials(value: string) {
  const letters = value
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join('')
  return letters || 'R'
}

export function isOutbound(message: RecentMessage) {
  return message.isFromMe === true || message.isFromMe === 1 || message.isFromMe === '1'
}

export function plural(value: number | undefined, singular: string, pluralLabel = `${singular}s`) {
  return `${formatNumber(value || 0)} ${value === 1 ? singular : pluralLabel}`
}
