import { useId, useState, type KeyboardEvent } from 'react'
import type { PanelAlign } from '../../lib/anchor'
import { stepIndex } from '../../lib/roving'
import { usePanel } from '../../lib/usePanel'
import { CheckIcon, ChevronDownIcon } from '../Icons'

export type Choice = {
  value: string
  label: string
  // a quiet word beside the label, such as the name of the model behind a level
  meta?: string
  // what choosing it means, in a few words
  note?: string
}

type ChoiceMenuProps = {
  // what is being set: "Quality". Shown in the button ahead of the value.
  label: string
  value: string
  choices: Choice[]
  onChange: (value: string) => void
  // the edge of the button the menu lines up with
  align?: PanelAlign
  // set where the label is already written beside the button; it is then only spoken
  quietLabel?: boolean
}

// A quiet button that names a setting and its value, and opens a short menu
// of the choices: each with a line about what it means, and a check on the
// one in use. The arrow keys move through it, Return chooses, Escape or a
// click outside closes it and focus goes back to the button.
export function ChoiceMenu({ label, value, choices, onChange, align = 'start', quietLabel }: ChoiceMenuProps) {
  const id = useId()
  const [active, setActive] = useState(0)
  const current = choices.find((choice) => choice.value === value)
  const { control, panel, open, close, panelEvents } = usePanel<HTMLUListElement>({
    align,
    onToggle(opening) {
      if (!opening) return
      // each opening starts at the choice in use
      setActive(Math.max(0, choices.indexOf(current ?? choices[0])))
      panel.current?.focus({ preventScroll: true })
    },
  })

  function choose(choice: Choice | undefined) {
    if (!choice) return
    if (choice.value !== value) onChange(choice.value)
    close()
  }

  function handleKeyDown(event: KeyboardEvent<HTMLUListElement>) {
    const next = stepIndex(event.key, active, choices.length)
    if (next !== null) {
      event.preventDefault()
      setActive(next)
    } else if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      choose(choices[active])
    } else if (event.key === 'Tab') {
      // leaving the menu closes it; focus then moves on from the button
      close()
    }
  }

  return (
    <>
      <button
        ref={control}
        type="button"
        className="picker"
        popoverTarget={id}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`${label}: ${current?.label ?? value}`}
      >
        {quietLabel ? null : <span className="picker-label">{label}</span>}
        {/* a new value fades in over the old one */}
        <span key={value} className="picker-value swap">
          {current?.label ?? value}
        </span>
        <ChevronDownIcon />
      </button>
      <ul
        ref={panel}
        id={id}
        popover="auto"
        className="popover choice-menu"
        role="menu"
        tabIndex={-1}
        aria-label={label}
        aria-activedescendant={choices[active] ? `${id}-${active}` : undefined}
        {...panelEvents}
        onKeyDown={handleKeyDown}
      >
        {choices.map((choice, index) => (
          <li
            key={choice.value}
            id={`${id}-${index}`}
            className="choice"
            role="menuitemradio"
            aria-checked={choice.value === value}
            data-active={index === active || undefined}
            // the pointer and the arrow keys move the same highlight
            onPointerMove={() => setActive(index)}
            onClick={() => choose(choice)}
          >
            <span className="choice-main">
              <span className="choice-title">
                {choice.label}
                {choice.meta ? <span className="choice-meta">{choice.meta}</span> : null}
              </span>
              {choice.note ? <span className="choice-note">{choice.note}</span> : null}
            </span>
            {choice.value === value ? <CheckIcon /> : null}
          </li>
        ))}
      </ul>
    </>
  )
}
