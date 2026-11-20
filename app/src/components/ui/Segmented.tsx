import { useRef, type CSSProperties, type KeyboardEvent } from 'react'

type SegmentedProps<T extends string> = {
  label: string
  value: T
  options: ReadonlyArray<{ value: T; label: string }>
  onChange: (value: T) => void
}

// A few choices side by side, one of them chosen, after the range switch over
// Mercury's balance chart. The options share the width evenly, so the fill
// under the chosen one slides from one to the next by its own width. It is
// one stop for Tab; the arrow keys choose, as in a set of radio buttons.
export function Segmented<T extends string>({ label, value, options, onChange }: SegmentedProps<T>) {
  const group = useRef<HTMLDivElement>(null)
  const at = Math.max(
    0,
    options.findIndex((option) => option.value === value),
  )

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    const steps: Record<string, number> = { ArrowLeft: -1, ArrowUp: -1, ArrowRight: 1, ArrowDown: 1 }
    if (!(event.key in steps)) return
    event.preventDefault()
    const next = (at + steps[event.key] + options.length) % options.length
    onChange(options[next].value)
    group.current?.querySelectorAll<HTMLButtonElement>('[role="radio"]')[next]?.focus()
  }

  return (
    <div
      ref={group}
      className="segmented"
      role="radiogroup"
      aria-label={label}
      style={{ '--at': at, '--count': options.length } as CSSProperties}
      onKeyDown={handleKeyDown}
    >
      {options.map((option) => {
        const chosen = option.value === value
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={chosen}
            tabIndex={chosen ? 0 : -1}
            className="segmented-option"
            onClick={() => onChange(option.value)}
          >
            {option.label}
          </button>
        )
      })}
    </div>
  )
}
