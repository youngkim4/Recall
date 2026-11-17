import { useCallback, useEffect, useState, type CSSProperties, type ReactNode, type TransitionEvent } from 'react'

type Props<Id extends string | number> = {
  id: Id
  // -1 and 1 bring the next one in from that side; 0 changes it with no motion
  slide: number
  className?: string
  render: (id: Id) => ReactNode
}

type Shown<Id> = { id: Id; leaving: Id | null; slide: number }

// the way out takes far less than this; it covers a window that never reports the end
const LEFT_BY_MS = 400

// What an arrow moved to. The one in view stays for its way out while the
// next arrives, so the two read as one strip moving under the arrows.
export function Step<Id extends string | number>({ id, slide, className, render }: Props<Id>) {
  const [shown, setShown] = useState<Shown<Id>>({ id, leaving: null, slide: 0 })
  if (shown.id !== id) setShown({ id, leaving: slide === 0 ? null : shown.id, slide })
  const leaving = shown.leaving

  const settle = useCallback(
    () => setShown((current) => (current.leaving === null ? current : { ...current, leaving: null })),
    [],
  )

  useEffect(() => {
    if (leaving === null) return
    const timer = window.setTimeout(settle, LEFT_BY_MS)
    return () => window.clearTimeout(timer)
  }, [leaving, settle])

  function handleLeft(event: TransitionEvent<HTMLDivElement>) {
    // a part inside it ending a transition of its own is not the way out ending
    if (event.target === event.currentTarget && event.propertyName === 'opacity') settle()
  }

  return (
    <div className={className ? `step ${className}` : 'step'} style={{ '--slide': shown.slide } as CSSProperties}>
      {leaving !== null ? (
        <div key={leaving} data-step="out" aria-hidden="true" inert onTransitionEnd={handleLeft}>
          {render(leaving)}
        </div>
      ) : null}
      <div key={shown.id} data-step={shown.slide === 0 ? undefined : 'in'}>
        {render(shown.id)}
      </div>
    </div>
  )
}
