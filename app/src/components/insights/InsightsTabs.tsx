import { formatNumber } from '../../lib/format'
import type { StatsPayload } from '../../lib/insights'
import type { LineSeries } from './LineChart'

type InsightsTabsProps = {
  stats: StatsPayload | null
  // how many people or conversations the answer is with
  among: number | null
  amongLabel: string
  series: LineSeries
  onSeriesChange: (series: LineSeries) => void
}

const TABS: Array<{ series: LineSeries; label: string; pick: (stats: StatsPayload) => number }> = [
  { series: 'both', label: 'Messages', pick: (stats) => stats.totals.total },
  { series: 'sent', label: 'You sent', pick: (stats) => stats.totals.sent },
  { series: 'received', label: 'They sent', pick: (stats) => stats.totals.received },
]

// The figures of the answer as tabs over its chart, after ElevenLabs' usage
// card: the tab that is chosen says which lines the chart draws and is
// marked by a rule under it. The last figure, how many people the answer is
// with, is a figure and no tab: the chart has no line for it.
export function InsightsTabs({ stats, among, amongLabel, series, onSeriesChange }: InsightsTabsProps) {
  return (
    <div className="insights-tabs">
      <div className="insights-tablist" role="tablist" aria-label="What the chart draws">
        {TABS.map((tab) => (
          <button
            key={tab.series}
            type="button"
            role="tab"
            className="insights-tab"
            aria-selected={tab.series === series}
            onClick={() => onSeriesChange(tab.series)}
          >
            <span className="insights-tab-label">{tab.label}</span>
            <b key={stats ? tab.pick(stats) : 'none'} className="insights-tab-value swap">
              {stats ? formatNumber(tab.pick(stats)) : <span className="skeleton">000,000</span>}
            </b>
          </button>
        ))}
      </div>
      <div className="insights-tab insights-figure">
        <span className="insights-tab-label">{amongLabel}</span>
        <b key={among ?? 'none'} className="insights-tab-value swap">
          {among === null ? <span className="skeleton">0,000</span> : formatNumber(among)}
        </b>
      </div>
    </div>
  )
}
