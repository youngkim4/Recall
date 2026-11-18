import { useId, useRef, useState, type KeyboardEvent, type ReactNode } from 'react'
import type { PanelAlign } from '../../lib/anchor'
import type { Choice } from '../../lib/choice'
import { stepIndex } from '../../lib/roving'
import { usePanel } from '../../lib/usePanel'
import { CheckIcon, ChevronDownIcon } from '../Icons'

export type { Choice }

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
  // a control of its own under the choices, about the choice in use: it stays
  // in the panel, and Tab goes on to it
  extra?: ReactNode
  // set for a filter: with nothing chosen (the value ''), the button says only
  // what it filters, "From", and the value shows once one is chosen
  bare?: boolean
}

// A quiet button that names a setting and its value, and opens a short menu
// of the choices: each with a line about what it means, and a check on the
// one in use. The arrow keys move through it, Return chooses, Escape or a
// click outside closes it and focus goes back to the button.
export function ChoiceMenu({
  label,
  value,
  choices,
  onChange,
  align = 'start',
  quietLabel,
  extra,
  bare,
}: ChoiceMenuProps) {
  const id = useId()
  const [active, setActive] = useState(0)
  const menu = useRef<HTMLUListElement>(null)
  const current = choices.find((choice) => choice.value === value)
  const { control, panel, open, close, panelEvents } = usePanel<HTMLDivElement>({
    align,
    onToggle(opening) {
      if (!opening) return
      // each opening starts at the choice in use
      setActive(Math.max(0, choices.indexOf(current ?? choices[0])))
      menu.current?.focus({ preventScroll: true })
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
    } else if (event.key === 'Tab' && !(extra && !event.shiftKey)) {
      // leaving the menu closes it; focus then moves on from the button.
      // With a control under the choices, Tab goes on to that control.
      close()
    }
  }

  // Shift+Tab from the control under the choices goes back to the choices,
  // and not to the button behind the open panel
  function backToChoices(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key !== 'Tab' || !event.shiftKey) return
    event.preventDefault()
    menu.current?.focus({ preventScroll: true })
  }

  return (
    <>
      <button
        ref={control}
        type="button"
        className="picker"
        popoverTarget={id}
        aria-haspopup={extra ? 'dialog' : 'menu'}
        aria-expanded={open}
        aria-label={`${label}: ${current?.label ?? value}`}
      >
        {bare && !value ? (
          <span className="picker-value">{label}</span>
        ) : (
          <>
            {quietLabel ? null : <span className="picker-label">{label}</span>}
            {/* a new value fades in over the old one */}
            <span key={value} className="picker-value swap">
              {current?.label ?? value}
            </span>
          </>
        )}
        <ChevronDownIcon />
      </button>
      <div
        ref={panel}
        id={id}
        popover="auto"
        className="popover choice-panel"
        // with a control under the choices the panel is more than a menu
        role={extra ? 'dialog' : undefined}
        aria-label={extra ? label : undefined}
        {...panelEvents}
      >
        <ul
          ref={menu}
          className="choice-menu"
          role="menu"
          tabIndex={-1}
          aria-label={label}
          aria-activedescendant={choices[active] ? `${id}-${active}` : undefined}
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
        {extra ? (
          <div className="choice-extra" onKeyDown={backToChoices}>
            {extra}
          </div>
        ) : null}
      </div>
    </>
  )
}
