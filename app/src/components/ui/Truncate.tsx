import { useState } from 'react'

type TruncateProps = {
  className: string
  id?: string
  children: string
}

// One line of text that ends in an ellipsis when it does not fit (the class
// sets that). While it is cut off, the whole text is offered as a tooltip, so
// nothing a row says is ever out of reach.
export function Truncate({ className, id, children }: TruncateProps) {
  const [cut, setCut] = useState(false)
  return (
    <span
      className={className}
      id={id}
      title={cut ? children : undefined}
      onMouseEnter={(event) => setCut(event.currentTarget.scrollWidth > event.currentTarget.clientWidth)}
    >
      {children}
    </span>
  )
}
