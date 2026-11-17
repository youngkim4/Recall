import { formatMoney } from '../../lib/format'
import type { Daily } from '../../lib/days'
import type { Estimate } from '../../types'
import { DateRange } from '../ui/DateRange'
import { QualityControl } from '../ui/QualityControl'

type Scope = { since: string; until: string }

type ReportBarProps = {
  models: string[]
  model: string
  onModelChange: (model: string) => void
  scope: Scope
  onScopeChange: (scope: Scope) => void
  // how many messages each day of the conversation holds; no days while that is not known
  daily: Daily
  // null while the cost is being worked out, and when it cannot be
  estimate: Estimate | null
  // why there is no cost, in one sentence; a report cannot be run then
  estimateError?: string
  canRun: boolean
  running: boolean
  // set when the last report for this conversation could not be written
  failure?: string
  onRun: () => void
}

// What a report will cover, how carefully it is read, what that costs, and
// the button that starts it. The price sits beside the button because running
// a report spends money.
export function ReportBar({
  models,
  model,
  onModelChange,
  scope,
  onScopeChange,
  daily,
  estimate,
  estimateError,
  canRun,
  running,
  failure,
  onRun,
}: ReportBarProps) {
  const cost = estimate?.estimated_cost
  return (
    <div className="report">
      <div className="report-bar">
        <DateRange since={scope.since} until={scope.until} daily={daily} onChange={onScopeChange} />
        <QualityControl models={models} value={model} onChange={onModelChange} />
        {/* the price and the button it belongs to stay together when the row wraps */}
        <span className="report-go">
          <span className="report-cost">
            {estimate ? (
              // a new figure fades in over the old one
              <span key={cost} className="swap">
                {Number.isFinite(cost) ? `About ${formatMoney(cost)}` : ''}
              </span>
            ) : estimateError ? null : (
              <span className="skeleton">About $0.00</span>
            )}
          </span>
          <button type="button" className="btn primary" disabled={!canRun || running || !estimate} onClick={onRun}>
            {running ? <span className="spinner" /> : null}
            {running ? 'Running' : 'Run report'}
          </button>
        </span>
      </div>
      {failure ? (
        <p className="notice bad" role="alert">
          {failure}
        </p>
      ) : null}
      {!estimate && estimateError ? (
        <p className="notice bad" role="alert">
          {estimateError}
        </p>
      ) : null}
    </div>
  )
}
