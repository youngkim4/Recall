import type { ReactNode } from 'react'
import { formatNumber, formatPercent, plural } from '../../lib/format'
import { formatWait } from '../../lib/habits'
import type { Habits, StatsRow } from '../../lib/insights'
import { Truncate } from '../ui/Truncate'
import { VoiceBar } from '../ui/VoiceBar'

type InsightsTableProps = {
  // the people or the conversations of the answer, ranked
  rows: StatsRow[]
  heading: string
  // what one row is called in its first column
  column: string
  onOpen: (row: StatsRow) => void
  // a control at the end of the heading, such as what the rows are of
  action?: ReactNode
  // the habits could not be counted: their cells say nothing, not "loading"
  habitsFailed: boolean
}

// the habits of a row, each in a column after its count; a narrow table
// keeps the first ones (`id` names a column for insights.css)
const HABIT_COLUMNS: Array<{ id: string; label: string; show: (habits: Habits) => string }> = [
  { id: 'you-reply', label: 'You reply', show: (habits) => formatWait(habits.youReply) },
  { id: 'you-start', label: 'You start', show: (habits) => formatPercent(habits.youStart) },
  { id: 'they-reply', label: 'They reply', show: (habits) => formatWait(habits.theyReply) },
  {
    id: 'streak',
    label: 'Longest streak',
    show: (habits) => (habits.streak ? plural(habits.streak.days, 'day') : '—'),
  },
  { id: 'late', label: 'Late night', show: (habits) => formatPercent(habits.lateNight) },
]

// The people of the answer as a table, after Mixpanel's breakdown under its
// chart: a heading row, then one row a person with a bar of what you and
// they sent against the busiest, the count, and how each of you writes:
// how fast each side replies, who starts, the longest run of days, and how
// much is late at night. A group has no replies and no starts: a dash.
export function InsightsTable({ rows, heading, column, onOpen, action, habitsFailed }: InsightsTableProps) {
  if (!rows.length) return null
  const largest = Math.max(1, ...rows.map((row) => row.total))
  return (
    <section className="section insights-table" aria-labelledby="insights-table-title">
      <div className="section-head">
        <h2 id="insights-table-title">{heading}</h2>
        {action ? <span className="section-links">{action}</span> : null}
      </div>
      <table className="data-table">
        <thead>
          <tr>
            <th className="num place" scope="col">
              #
            </th>
            <th scope="col">{column}</th>
            <th className="bar" scope="col">
              <span className="sr-only">You and them</span>
            </th>
            <th className="num" scope="col">
              Messages
            </th>
            {HABIT_COLUMNS.map((habit) => (
              <th key={habit.id} className="num habit" data-col={habit.id} scope="col">
                {habit.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={row.key}>
              <td className="num place">{index + 1}</td>
              <td className="name">
                <button type="button" className="link" onClick={() => onOpen(row)}>
                  <Truncate className="row-title">{row.label}</Truncate>
                </button>
              </td>
              <td className="bar">
                <VoiceBar sent={row.sent} received={row.received} scale={row.total / largest} />
              </td>
              <td className="num">{formatNumber(row.total)}</td>
              {HABIT_COLUMNS.map((habit) => (
                <td key={habit.id} className="num habit" data-col={habit.id}>
                  {row.habits ? habit.show(row.habits) : habitsFailed ? '—' : <span className="skeleton">00 min</span>}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}
