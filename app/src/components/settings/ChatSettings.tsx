import { useState } from 'react'
import { formatMoney } from '../../lib/format'
import { costCeiling, keyLine, progressLine, stepProblems } from '../../lib/firstRun'
import { useFirstRun } from '../../lib/useFirstRun'
import { useKey } from '../../lib/useKey'
import { KeyForm } from '../firstrun/KeyForm'

// Two rows of the Archive part: whether chat is ready, with the one button
// that gets it ready, and the API key that chat uses. A key saved here
// makes the Chat row ask again, so its button can be pressed at once.
export function ChatSettings({ messagesPath }: { messagesPath: string }) {
  const chat = useFirstRun(messagesPath)
  const key = useKey()
  const [editing, setEditing] = useState(false)
  const hasKey = Boolean(key.key?.present ?? chat.status?.key.present)
  const problems = chat.status ? stepProblems(chat.status) : []

  async function save(typed: string) {
    const saved = await key.save(typed)
    if (saved) {
      setEditing(false)
      void chat.reload()
    }
    return saved
  }

  async function remove() {
    if (await key.remove()) {
      setEditing(false)
      void chat.reload()
    }
  }

  return (
    <>
      <div className="setting">
        <span className="setting-label">Chat</span>
        <span className="setting-value">
          <ChatValue {...chat} />
          {problems.map((problem) => (
            <span key={problem} className="setting-note">
              {problem}
            </span>
          ))}
          {chat.phase === 'failed' && chat.job?.error ? (
            <span className="setting-note bad">{chat.job.error}</span>
          ) : null}
          {chat.refusal ? <span className="setting-note bad">{chat.refusal}</span> : null}
          {!hasKey && (chat.phase === 'not_ready' || chat.phase === 'failed') ? (
            <span className="setting-note">Add an API key first</span>
          ) : null}
        </span>
        {chat.phase === 'not_ready' || chat.phase === 'failed' ? (
          <button type="button" className="btn" disabled={chat.starting || !hasKey} onClick={() => void chat.start()}>
            {chat.phase === 'failed' ? 'Try again' : 'Get chat ready'}
          </button>
        ) : null}
      </div>

      <div className="setting">
        <span className="setting-label">API key</span>
        <span className="setting-value">
          {key.key ? keyLine(key.key) : key.failed ? 'Could not be read' : ''}
          {key.key?.source === 'environment' ? (
            <span className="setting-note">A key saved here would not be used</span>
          ) : null}
        </span>
        {key.key && key.key.source !== 'environment' ? (
          <button type="button" className="btn" aria-expanded={editing} onClick={() => setEditing(!editing)}>
            {editing ? 'Cancel' : key.key.present ? 'Change' : 'Add'}
          </button>
        ) : null}
      </div>
      {editing ? (
        <div className="setting setting-key">
          <KeyForm busy={key.busy} error={key.error} onSave={save} autoFocus />
          {key.key?.source === 'file' ? (
            <button type="button" className="btn quiet" disabled={key.busy} onClick={() => void remove()}>
              Remove the key
            </button>
          ) : null}
        </div>
      ) : null}
    </>
  )
}

// what chat is ready for, in a few words
function ChatValue({ phase, status, job, failed }: ReturnType<typeof useFirstRun>) {
  if (phase === 'ready') return 'Ready'
  if (phase === 'running' && job) return progressLine(job) || 'Getting ready'
  if (phase === 'no_messages') return 'No messages yet'
  if (phase === 'unknown') return failed ? 'Could not be checked' : ''
  return status ? `Not ready, up to ${formatMoney(costCeiling(status.estimatedCost))} of usage` : 'Not ready'
}
