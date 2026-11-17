import { useState } from 'react'
import { recallBridge } from '../../lib/bridge'
import { Disclosure } from '../ui/Disclosure'

type OtherSourceProps = {
  busy: boolean
  error: string
  // opens the system's file chooser
  onPick: () => void
  // checks a path typed by hand, where there is no file chooser
  onPath: (path: string) => void
}

// The way in without granting access: a copy of the Messages database, made
// by hand or brought from another Mac.
export function OtherSource({ busy, error, onPick, onPath }: OtherSourceProps) {
  const [path, setPath] = useState('')
  return (
    <Disclosure className="setup-other" summary="Use a copy of the database instead">
      <p>
        The database is at ~/Library/Messages/chat.db. In Finder, choose Go, then Go to Folder, copy chat.db
        somewhere else, and choose the copy here. Copy chat.db-wal and chat.db-shm with it, so that the newest
        messages come too.
      </p>
      {recallBridge.available ? (
        <button type="button" className="btn" disabled={busy} onClick={onPick}>
          {busy ? <span className="spinner" /> : null}
          {busy ? 'Checking' : 'Choose a file'}
        </button>
      ) : (
        <form
          className="setup-path"
          onSubmit={(event) => {
            event.preventDefault()
            if (path.trim()) onPath(path.trim())
          }}
        >
          <input
            type="text"
            className="field"
            value={path}
            placeholder="/path/to/a/copy/of/chat.db"
            aria-label="Path to a copy of chat.db"
            spellCheck={false}
            autoComplete="off"
            onChange={(event) => setPath(event.target.value)}
          />
          <button type="submit" className="btn" disabled={busy || !path.trim()}>
            {busy ? <span className="spinner" /> : null}
            {busy ? 'Checking' : 'Use this file'}
          </button>
        </form>
      )}
      {error ? (
        <p className="notice bad" role="alert">
          {error}
        </p>
      ) : null}
    </Disclosure>
  )
}
