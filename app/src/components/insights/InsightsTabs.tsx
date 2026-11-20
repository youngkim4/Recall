import { formatNumber } from '../../lib/format'
import type { StatsPayload } from '../../lib/insights'
import { useTweened } from '../../lib/useTweened'
import type { LineSeries } from './LineChart'

type InsightsTabsProps = {
  stats: StatsPayload | null
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
// marked by a rule under it. The last figure, on how many days the answer
// was said, is a figure and no tab: the chart has no line for it. (It was
// once how many people the answer is with: 1 for a direct conversation and
// 0 for a group, which the user called useless.)
export function InsightsTabs({ stats, series, onSeriesChange }: InsightsTabsProps) {
  // a new answer's figures count from the last ones to theirs, as the chart moves
  const figures = useTweened([...TABS.map((tab) => (stats ? tab.pick(stats) : 0)), stats?.days ?? 0])
  const figure = (index: number) => formatNumber(Math.round(figures[index] ?? 0))
  return (
    <div className="insights-tabs">
      <div className="insights-tablist" role="tablist" aria-label="What the chart draws">
        {TABS.map((tab, index) => (
          <button
            key={tab.series}
            type="button"
            role="tab"
            className="insights-tab"
            aria-selected={tab.series === series}
            onClick={() => onSeriesChange(tab.series)}
          >
            <span className="insights-tab-label">{tab.label}</span>
            <b className="insights-tab-value">{stats ? figure(index) : <span className="skeleton">000,000</span>}</b>
          </button>
        ))}
      </div>
      <div className="insights-tab insights-figure" title="Days with at least one of these messages">
        <span className="insights-tab-label">Active days</span>
        <b className="insights-tab-value">{stats ? figure(TABS.length) : <span className="skeleton">0,000</span>}</b>
      </div>
    </div>
  )
}
