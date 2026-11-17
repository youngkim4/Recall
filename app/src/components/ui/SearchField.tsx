import type { KeyboardEvent, Ref } from 'react'
import { CloseIcon, SearchIcon } from '../Icons'

type SearchFieldProps = {
  value: string
  onChange: (value: string) => void
  // what is being searched, for assistive technology
  label: string
  placeholder?: string
  // the one thing a page is for is set a size up
  large?: boolean
  autoFocus?: boolean
  disabled?: boolean
  inputRef?: Ref<HTMLInputElement>
  // called for keys the field does not handle itself
  onKeyDown?: (event: KeyboardEvent<HTMLInputElement>) => void
}

// A field to search with. Escape or the button at its end empties it; the
// button is ours, in place of the browser's own.
export function SearchField({
  value,
  onChange,
  label,
  placeholder = 'Search',
  large,
  autoFocus,
  disabled,
  inputRef,
  onKeyDown,
}: SearchFieldProps) {
  function handleKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === 'Escape' && value) {
      event.preventDefault()
      onChange('')
      return
    }
    onKeyDown?.(event)
  }

  return (
    <span className={large ? 'search search-lg' : 'search'}>
      <SearchIcon />
      <input
        ref={inputRef}
        type="search"
        className="field"
        value={value}
        placeholder={placeholder}
        aria-label={label}
        autoFocus={autoFocus}
        autoComplete="off"
        spellCheck={false}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
      />
      {value ? (
        <button
          type="button"
          className="icon-btn sm search-clear"
          aria-label="Clear search"
          // the pointer empties the field without taking the caret out of it
          onMouseDown={(event) => event.preventDefault()}
          onClick={() => onChange('')}
        >
          <CloseIcon />
        </button>
      ) : null}
    </span>
  )
}
