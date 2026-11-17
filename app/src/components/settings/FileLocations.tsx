import { useState } from 'react'
import type { RuntimePaths } from '../../types'

type FileLocationsProps = {
  // the locations in use
  paths: RuntimePaths
  // the locations the app starts with
  defaultPaths: RuntimePaths
  busy: boolean
  onApply: (paths: RuntimePaths) => void
}

const LIVE_DATABASE = '~/Library/Messages/chat.db'

const FIELDS: Array<{ key: keyof RuntimePaths; label: string }> = [
  { key: 'dbPath', label: 'Messages database' },
  { key: 'messagesPath', label: 'Archive file' },
  { key: 'outDir', label: 'Reports folder' },
]

const trimmed = (paths: RuntimePaths): RuntimePaths => ({
  dbPath: paths.dbPath.trim(),
  messagesPath: paths.messagesPath.trim(),
  outDir: paths.outDir.trim(),
})

// Where the app reads Messages from, where it keeps the archive, and where
// reports are saved. Nothing changes until Apply is pressed.
export function FileLocations({ paths, defaultPaths, busy, onApply }: FileLocationsProps) {
  const [draft, setDraft] = useState(paths)
  const wanted = trimmed(draft)
  const changed = FIELDS.some(({ key }) => wanted[key] !== paths[key])
  const complete = FIELDS.every(({ key }) => wanted[key])
  const atDefaults = FIELDS.every(({ key }) => wanted[key] === defaultPaths[key])

  return (
    <div className="settings">
      {FIELDS.map(({ key, label }) => (
        <label key={key} className="setting">
          <span className="setting-label">{label}</span>
          <input
            type="text"
            className="field setting-field"
            spellCheck={false}
            autoComplete="off"
            value={draft[key]}
            placeholder={defaultPaths[key]}
            onChange={(event) => setDraft((current) => ({ ...current, [key]: event.target.value }))}
          />
        </label>
      ))}
      <div className="setting-actions">
        <button
          type="button"
          className="btn primary"
          disabled={busy || !changed || !complete}
          onClick={() => {
            setDraft(wanted)
            onApply(wanted)
          }}
        >
          Apply
        </button>
        <button type="button" className="btn" disabled={busy || atDefaults} onClick={() => setDraft(defaultPaths)}>
          Reset
        </button>
        {wanted.dbPath === LIVE_DATABASE ? null : (
          <button
            type="button"
            className="link"
            onClick={() => setDraft((current) => ({ ...current, dbPath: LIVE_DATABASE }))}
          >
            Use the live Messages database
          </button>
        )}
      </div>
    </div>
  )
}
