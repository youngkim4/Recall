import { useLayoutEffect, type KeyboardEvent, type RefObject } from 'react'
import { ArrowUpIcon } from '../Icons'
import { QualityControl } from '../ui/QualityControl'
import type { PickerOption } from '../../lib/scope'
import { ListPicker } from '../ui/ListPicker'

type ComposerProps = {
  inputRef: RefObject<HTMLTextAreaElement | null>
  draft: string
  canSend: boolean
  disabled: boolean
  scope: string
  scopeOptions: PickerOption[]
  model: string
  models: string[]
  onDraftChange: (value: string) => void
  onScopeChange: (value: string) => void
  onModelChange: (value: string) => void
  onSubmit: () => void
}

export function Composer({
  inputRef,
  draft,
  canSend,
  disabled,
  scope,
  scopeOptions,
  model,
  models,
  onDraftChange,
  onScopeChange,
  onModelChange,
  onSubmit,
}: ComposerProps) {
  // grow with the text; the stylesheet caps the height and scrolls past it
  useLayoutEffect(() => {
    const input = inputRef.current
    if (!input) return
    input.style.height = 'auto'
    input.style.height = `${input.scrollHeight}px`
  }, [draft, inputRef])

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    // Return confirms a word while an input method is composing; never send then
    if (event.key !== 'Enter' || event.nativeEvent.isComposing) return
    if (event.shiftKey && !event.metaKey && !event.ctrlKey) return
    event.preventDefault()
    onSubmit()
  }

  return (
    <form
      className="composer"
      onSubmit={(event) => {
        event.preventDefault()
        onSubmit()
      }}
    >
      <textarea
        ref={inputRef}
        value={draft}
        rows={1}
        disabled={disabled}
        aria-label="Ask about your messages"
        placeholder="Ask about your messages…"
        onChange={(event) => onDraftChange(event.target.value)}
        onKeyDown={handleKeyDown}
      />
      <div className="composer-bar">
        <ListPicker
          label="Conversations to search"
          value={scope}
          options={scopeOptions}
          onChange={onScopeChange}
          searchPlaceholder="Search conversations"
          emptyText="No conversations match"
        />
        <QualityControl models={models} value={model} onChange={onModelChange} align="end" />
        <button type="submit" className="send" disabled={!canSend} aria-label="Send">
          <ArrowUpIcon />
        </button>
      </div>
    </form>
  )
}
