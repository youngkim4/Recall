import { daysSpanned } from '../../lib/days'
import { formatDate, formatDuration, formatNumber, formatPercent, plural } from '../../lib/format'
import type { Dynamics, PreviewStats } from '../../types'

export type Fact = { label: string; value: string }

const KINDS_NAMED = 3

function shared(attachments: Record<string, number> = {}) {
  const kinds: Array<[number, string, string?]> = [
    [attachments.photos, 'photo'],
    [attachments.videos, 'video'],
    [attachments.gifs, 'GIF'],
    [attachments.audio, 'audio clip'],
    [attachments.documents, 'file'],
  ]
  const present = kinds.filter(([count]) => count > 0).sort((a, b) => b[0] - a[0])
  const named = present.slice(0, KINDS_NAMED).map(([count, word]) => plural(count, word))
  const rest =
    present.slice(KINDS_NAMED).reduce((sum, [count]) => sum + count, 0) + Number(attachments.other || 0)
  return [...named, ...(rest > 0 ? [`${formatNumber(rest)} other`] : [])].join(', ')
}

// The few things about a conversation worth a line each. A fact with nothing
// to say is left out.
export function conversationFacts(stats: PreviewStats, dynamics: Dynamics | undefined, them: string): Fact[] {
  // days on the calendar, not lengths of 24 hours: a late first message and an early last one still count their days
  const span = daysSpanned(stats.firstTimestamp, stats.lastTimestamp)
  const opened = dynamics?.initiationLifetime
  const gap = Math.round(stats.longestGapDays ?? 0)
  const files = shared(stats.attachments)
  const reactions = Number(stats.reactions?.total || 0)

  const facts: Array<Fact | null> = [
    stats.activeDays
      ? {
          label: 'Active days',
          value: span > stats.activeDays ? `${formatNumber(stats.activeDays)} of ${formatNumber(span)}` : formatNumber(stats.activeDays),
        }
      : null,
    stats.busiestDay && stats.busiestDayCount
      ? { label: 'Busiest day', value: `${formatDate(stats.busiestDay)}, ${plural(stats.busiestDayCount, 'message')}` }
      : null,
    gap >= 1 ? { label: 'Longest gap', value: formatDuration(gap) } : null,
    typeof opened === 'number'
      ? {
          label: 'First to write',
          value:
            opened >= 0.5
              ? `You, on ${formatPercent(opened)} of days`
              : `${them}, on ${formatPercent(1 - opened)} of days`,
        }
      : null,
    files ? { label: 'Shared', value: files } : null,
    reactions ? { label: 'Reactions', value: formatNumber(reactions) } : null,
  ]
  return facts.filter((fact): fact is Fact => fact !== null)
}
