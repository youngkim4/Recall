import { parseDate, relativeTime } from '../../lib/format'
import type { ReportEntry } from '../../lib/reports'
import { rovingKeyDown } from '../../lib/roving'
import { Truncate } from '../ui/Truncate'

type ReportListProps = {
  entries: ReportEntry[]
  selectedKey: string | null
  onSelect: (entry: ReportEntry) => void
}

function written(entry: ReportEntry) {
  const date = parseDate(entry.writtenAt)
  return date ? relativeTime(date.valueOf()) : ''
}

// Every saved report, newest first. One stop for Tab; the arrow keys move through it.
export function ReportList({ entries, selectedKey, onSelect }: ReportListProps) {
  const selectedIsShown = entries.some((entry) => entry.key === selectedKey)
  return (
    <aside className="pane appear" aria-label="Reports">
      <div className="pane-scroll report-list-top">
        <ul className="rows" role="list" onKeyDown={rovingKeyDown}>
          {entries.map((entry, index) => {
            const active = entry.key === selectedKey
            return (
              <li key={entry.key}>
                <button
                  type="button"
                  className={active ? 'row is-active' : 'row'}
                  aria-current={active ? 'true' : undefined}
                  data-roving
                  tabIndex={active || (!selectedIsShown && index === 0) ? 0 : -1}
                  onClick={() => onSelect(entry)}
                >
                  <span className="row-main">
                    <Truncate className="row-title">{entry.title}</Truncate>
                  </span>
                  <span className="row-end">{written(entry)}</span>
                </button>
              </li>
            )
          })}
        </ul>
      </div>
    </aside>
  )
}
