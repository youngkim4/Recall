import { useId, useState, type FormEvent } from 'react'

type KeyFormProps = {
  busy: boolean
  // why the last save was refused, in the server's sentence
  error: string
  // true when the key was saved
  onSave: (typed: string) => Promise<boolean>
  autoFocus?: boolean
  // the size of the setup guide's controls
  large?: boolean
}

// A field for the API key and its Save button. The field hides what is
// typed, and it is emptied once the key is saved: the key is never shown back.
export function KeyForm({ busy, error, onSave, autoFocus, large }: KeyFormProps) {
  const [typed, setTyped] = useState('')
  const fieldId = useId()
  const errorId = useId()

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (await onSave(typed)) setTyped('')
  }

  return (
    <form className={large ? 'key-form lg' : 'key-form'} onSubmit={(event) => void submit(event)}>
      <label className="sr-only" htmlFor={fieldId}>
        API key
      </label>
      <input
        id={fieldId}
        type="password"
        className="field"
        autoComplete="off"
        spellCheck={false}
        placeholder="Paste your key"
        value={typed}
        // where the step is there to take the key, the field has the focus
        autoFocus={autoFocus}
        aria-invalid={error ? true : undefined}
        aria-describedby={error ? errorId : undefined}
        onChange={(event) => setTyped(event.target.value)}
      />
      <button type="submit" className={large ? 'btn primary lg' : 'btn primary'} disabled={busy || !typed.trim()}>
        {busy ? 'Saving' : 'Save'}
      </button>
      {error ? (
        <p id={errorId} className="key-form-error" role="alert">
          {error}
        </p>
      ) : null}
    </form>
  )
}
