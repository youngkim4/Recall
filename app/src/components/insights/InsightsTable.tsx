import { formatNumber, formatPercent } from '../../lib/format'
import { busiest, type StatsRow } from '../../lib/insights'
import { Truncate } from '../ui/Truncate'
import { VoiceBar } from '../ui/VoiceBar'

type InsightsTableProps = {
  rows: StatsRow[]
  // by who: the rows are people or conversations, ranked, and open their
  // conversation; otherwise they are periods, weekdays or hours, and the
  // busiest of them are listed
  who: boolean
  heading: string
  // what one row is called in its first column
  column: string
  nameOf: (row: StatsRow) => string
  total: number
  onOpen?: (row: StatsRow) => void
}

// how many periods the table lists, the busiest first
const BUSIEST = 10

// The rows of the answer as a table, after Amplitude's breakdown under its
// chart: a heading row, then one row a period or a person with the count,
// what you and they sent, and its share of the answer. By who, the table is
// the chart: a bar in each row against the busiest.
export function InsightsTable({ rows, who, heading, column, nameOf, total, onOpen }: InsightsTableProps) {
  const shown = who ? rows : busiest(rows, BUSIEST)
  if (!shown.length) return null
  const largest = Math.max(1, ...shown.map((row) => row.total))
  return (
    <section className="section insights-table" aria-labelledby="insights-table-title">
      <div className="section-head">
        <h2 id="insights-table-title">{heading}</h2>
      </div>
      <table className="data-table">
        <thead>
          <tr>
            {who ? <th className="num place" scope="col">#</th> : null}
            <th scope="col">{column}</th>
            {who ? <th className="bar" scope="col"><span className="sr-only">Share</span></th> : null}
            <th className="num" scope="col">Messages</th>
            <th className="num" scope="col">You</th>
            <th className="num" scope="col">Them</th>
            <th className="num" scope="col">Share</th>
          </tr>
        </thead>
        <tbody>
          {shown.map((row, index) => {
            const name = nameOf(row)
            return (
              <tr key={row.key}>
                {who ? <td className="num place">{index + 1}</td> : null}
                <td className="name">
                  {who && onOpen ? (
                    <button type="button" className="link" onClick={() => onOpen(row)}>
                      <Truncate className="row-title">{name}</Truncate>
                    </button>
                  ) : (
                    <Truncate className="row-title">{name}</Truncate>
                  )}
                </td>
                {who ? (
                  <td className="bar">
                    <VoiceBar sent={row.sent} received={row.received} scale={row.total / largest} />
                  </td>
                ) : null}
                <td className="num">{formatNumber(row.total)}</td>
                <td className="num">{formatNumber(row.sent)}</td>
                <td className="num">{formatNumber(row.received)}</td>
                <td className="num">{formatPercent(row.total / Math.max(1, total))}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </section>
  )
}
