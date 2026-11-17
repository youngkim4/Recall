import { useId, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import { stepIndex } from '../../lib/roving'
import type { PickerOption } from '../../lib/scope'
import { usePanel } from '../../lib/usePanel'
import { CheckIcon, ChevronDownIcon, SearchIcon } from '../Icons'

type ListPickerProps = {
  // what is being chosen, for assistive technology: "Conversations to search"
  label: string
  value: string
  options: PickerOption[]
  onChange: (value: string) => void
  searchPlaceholder?: string
  emptyText?: string
}

// A button that names the current choice and opens a searchable list of the
// others, in place of the system menu. Typing filters, the arrow keys move
// through the list, Return chooses, Escape or a click outside closes it and
// focus goes back to the button.
export function ListPicker({
  label,
  value,
  options,
  onChange,
  searchPlaceholder = 'Search',
  emptyText = 'Nothing matches',
}: ListPickerProps) {
  const id = useId()
  const input = useRef<HTMLInputElement>(null)
  const list = useRef<HTMLUListElement>(null)
  const [query, setQuery] = useState('')
  const [active, setActive] = useState(0)
  const { control, panel, open, close, panelEvents } = usePanel<HTMLDivElement>({
    onToggle(opening) {
      // each opening starts from the whole list, at the current choice; the
      // search is cleared as the panel closes, so the next one never opens on old matches
      setQuery('')
      if (!opening) return
      const chosen = Math.max(
        0,
        options.findIndex((option) => option.value === value),
      )
      setActive(chosen)
      input.current?.focus()
      requestAnimationFrame(() => list.current?.children[chosen]?.scrollIntoView({ block: 'center' }))
    },
  })

  const current = options.find((option) => option.value === value)
  const matches = useMemo(() => {
    const needle = query.trim().toLowerCase()
    return needle ? options.filter((option) => option.label.toLowerCase().includes(needle)) : options
  }, [options, query])

  function show(index: number) {
    setActive(index)
    list.current?.children[index]?.scrollIntoView({ block: 'nearest' })
  }

  function choose(option: PickerOption | undefined) {
    if (!option) return
    if (option.value !== value) onChange(option.value)
    close()
  }

  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    const next = stepIndex(event.key, active, matches.length)
    if (next !== null) {
      event.preventDefault()
      show(next)
    } else if (event.key === 'Enter') {
      event.preventDefault()
      choose(matches[active])
    }
  }

  return (
    <>
      <button
        ref={control}
        type="button"
        className="picker"
        popoverTarget={id}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={`${label}: ${current?.label ?? value}`}
      >
        <span className="picker-value">{current?.label ?? value}</span>
        <ChevronDownIcon />
      </button>
      <div
        ref={panel}
        id={id}
        popover="auto"
        className="popover list-picker"
        {...panelEvents}
      >
        <label className="search">
          <SearchIcon />
          <input
            ref={input}
            type="search"
            className="field"
            role="combobox"
            aria-label={label}
            aria-expanded={open}
            aria-controls={`${id}-list`}
            aria-activedescendant={matches[active] ? `${id}-${active}` : undefined}
            aria-autocomplete="list"
            autoComplete="off"
            spellCheck={false}
            placeholder={searchPlaceholder}
            value={query}
            onChange={(event) => {
              setQuery(event.target.value)
              setActive(0)
              list.current?.scrollTo({ top: 0 })
            }}
            onKeyDown={handleKeyDown}
          />
        </label>
        {matches.length ? (
          <ul
            ref={list}
            id={`${id}-list`}
            className="rows"
            role="listbox"
            aria-label={label}
            // a press on a choice leaves the focus in the field, so it goes back to the button when the panel closes
            onMouseDown={(event) => event.preventDefault()}
          >
            {matches.map((option, index) => (
              <li
                key={option.value}
                id={`${id}-${index}`}
                className="row"
                role="option"
                aria-selected={option.value === value}
                data-active={index === active || undefined}
                // the pointer and the arrow keys move the same highlight
                onPointerMove={() => setActive(index)}
                onClick={() => choose(option)}
              >
                <span className="row-main">
                  <span className="row-title">{option.label}</span>
                </span>
                {option.value === value ? <CheckIcon /> : option.meta ? <span className="row-end">{option.meta}</span> : null}
              </li>
            ))}
          </ul>
        ) : (
          <p className="pane-note">{emptyText}</p>
        )}
      </div>
    </>
  )
}
