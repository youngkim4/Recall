import type { ReactNode } from 'react'

export type BarListItem = {
  key: string
  label: string
  value: number
  icon?: ReactNode
  onClick?: () => void
}

type BarListProps = {
  items: BarListItem[]
  formatValue?: (value: number) => string
  showRank?: boolean
}

// ranked horizontal bars — label sits on a soft proportional fill (Tremor BarList)
export function BarList({ items, formatValue, showRank = true }: BarListProps) {
  const max = Math.max(1, ...items.map((item) => item.value))

  return (
    <div className="bar-list">
      {items.map((item, index) => {
        const pct = Math.max(2, Math.round((item.value / max) * 100))
        const valueText = formatValue ? formatValue(item.value) : String(item.value)
        const inner = (
          <>
            <span className="bar-fill" style={{ width: `${pct}%` }} aria-hidden />
            <span className="bar-label">
              {showRank ? <span className="bar-rank">{index + 1}</span> : null}
              {item.icon}
              <span>{item.label}</span>
            </span>
            <span className="bar-value">{valueText}</span>
          </>
        )

        if (item.onClick) {
          return (
            <button key={item.key} type="button" className="bar-row" onClick={item.onClick}>
              {inner}
            </button>
          )
        }
        return (
          <div key={item.key} className="bar-row">
            {inner}
          </div>
        )
      })}
    </div>
  )
}
