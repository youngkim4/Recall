import type { CardData } from '../../lib/cardCanvas'
import { busierThan, daysAround, spanOf, voicesOf } from '../../lib/cardData'
import { formatDate } from '../../lib/format'
import type { PreviewPayload } from '../../types'

// The card with what the conversation's own figures add to it: its chart, the
// split of the two voices, the day among the days around it. A card whose
// figures could not be read is still true; it only says less.
export function withFigures(base: CardData, figures: PreviewPayload | null): CardData {
  if (!figures) return base
  const { stats, dynamics, activity, daily } = figures
  if (base.kind === 'duo') {
    return {
      ...base,
      totalMessages: stats?.totalMessages || base.totalMessages,
      span: spanOf(stats?.firstTimestamp, stats?.lastTimestamp),
      balance: dynamics?.balanceLifetime,
      initiation: dynamics?.initiationLifetime,
      busiestDay: stats?.busiestDay ? formatDate(stats.busiestDay) : undefined,
      busiestDayCount: stats?.busiestDayCount,
      voices: voicesOf(activity),
    }
  }
  if (base.kind === 'anniversary') {
    return { ...base, voices: voicesOf(activity), balance: dynamics?.balanceLifetime }
  }
  if (!base.day || !daily) return base
  return { ...base, days: daysAround(daily, base.day), busier: busierThan(daily, base.day) }
}
