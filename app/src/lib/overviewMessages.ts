// The messages card of the Overview, after Mercury's balance card: how many
// messages over one stretch, and their line. The stretches end on the
// archive's last day (an archive can be behind today). Nothing here talks to
// the server.

import { addDays } from './days.ts'
import type { StatsAsk, StatsBy } from './insights.ts'

export type Stretch = 'month' | 'year' | 'all'

export const STRETCHES: ReadonlyArray<{ value: Stretch; label: string }> = [
  { value: 'month', label: '30 days' },
  { value: 'year', label: 'Year' },
  { value: 'all', label: 'All time' },
]

// how each stretch is counted: a point a day, a week, or a month
const GRAIN: Record<Stretch, Extract<StatsBy, 'day' | 'week' | 'month'>> = { month: 'day', year: 'week', all: 'month' }
const DAYS: Record<Exclude<Stretch, 'all'>, number> = { month: 30, year: 365 }

export function grainOf(stretch: Stretch) {
  return GRAIN[stretch]
}

// What to ask /api/stats for one stretch. All time asks what Insights asks
// first (every message, by month), so the two share one answer.
export function stretchAsk(stretch: Stretch, lastDay: string): StatsAsk {
  if (stretch === 'all' || !lastDay) return { by: GRAIN.all }
  return { by: GRAIN[stretch], since: addDays(lastDay, 1 - DAYS[stretch]), until: lastDay }
}

// the words beside the figure, for the stretch the figure is of
export function stretchWords(stretch: Stretch, all: { conversations: number; since: string }) {
  if (stretch === 'month') return 'in the last 30 days'
  if (stretch === 'year') return 'in the last year'
  const where = `in ${all.conversations.toLocaleString()} conversation${all.conversations === 1 ? '' : 's'}`
  return all.since ? `${where} since ${all.since}` : where
}

// "Good morning" from 5 a.m., "Good afternoon" from noon, "Good evening" from 6 p.m.
export function greetingAt(hour: number) {
  if (hour >= 5 && hour < 12) return 'Good morning'
  if (hour >= 12 && hour < 18) return 'Good afternoon'
  return 'Good evening'
}
