import { useId, useState, type ReactNode } from 'react'
import { ChevronDownIcon } from '../Icons'

type DisclosureProps = {
  summary: ReactNode
  children: ReactNode
  className?: string
}

// A real button that opens and closes its region. The region stays mounted so
// its height can animate; while closed it is inert, which keeps it out of the
// tab order and away from assistive tech.
export function Disclosure({ summary, children, className = '' }: DisclosureProps) {
  const [open, setOpen] = useState(false)
  const regionId = useId()

  return (
    <div className={`disclosure ${className}`.trim()}>
      <button
        type="button"
        className="disclosure-toggle"
        aria-expanded={open}
        aria-controls={regionId}
        onClick={() => setOpen((current) => !current)}
      >
        {summary}
        <ChevronDownIcon />
      </button>
      <div id={regionId} className={`disclosure-body${open ? ' is-open' : ''}`} inert={!open}>
        <div>{children}</div>
      </div>
    </div>
  )
}
