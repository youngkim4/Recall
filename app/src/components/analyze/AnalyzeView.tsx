import { useState, type CSSProperties } from 'react'
import { isActiveJob } from '../../lib/jobs'
import { LIST_WIDTH } from '../../lib/layout'
import type { Contact, Defaults, Job, PreviewPayload, ReportFile } from '../../types'
import { ResizeHandle } from '../ResizeHandle'
import { ConversationDetail } from './ConversationDetail'
import { ConversationList } from './ConversationList'

type Scope = { since: string; until: string }

type AnalyzeViewProps = {
  defaults: Defaults | null
  contacts: Contact[]
  selectedContact: Contact | null
  loadingContacts: boolean
  query: string
  onQueryChange: (query: string) => void
  onSelect: (contact: Contact) => void
  // null while the chosen conversation's figures are first worked out
  preview: PreviewPayload | null
  // false while the figures shown are from before a change of dates or quality
  previewSettled: boolean
  // why the figures that were asked for are not here
  previewError: string | null
  onRetryPreview: () => void
  reports: ReportFile[]
  model: string
  onModelChange: (model: string) => void
  scope: Scope
  onScopeChange: (scope: Scope) => void
  runningJob: Job | null
  onRunReport: () => Promise<void>
  onOpenReport: (report: ReportFile) => void
  listWidth: number
  onListWidthChange: (width: number) => void
}

export function AnalyzeView({
  defaults,
  contacts,
  selectedContact,
  loadingContacts,
  query,
  onQueryChange,
  onSelect,
  preview,
  previewSettled,
  previewError,
  onRetryPreview,
  reports,
  model,
  onModelChange,
  scope,
  onScopeChange,
  runningJob,
  onRunReport,
  onOpenReport,
  listWidth,
  onListWidthChange,
}: AnalyzeViewProps) {
  // which conversation the last report was started for, so a failure is shown where it belongs
  const [ranFor, setRanFor] = useState<string | null>(null)
  // in a narrow window only one of the two panes shows
  const [pane, setPane] = useState<'list' | 'detail'>('detail')
  const selectedId = selectedContact?.chat_id ?? null

  function run() {
    setRanFor(selectedId)
    void onRunReport()
  }

  return (
    <div className="split" data-pane={pane} style={{ '--list-w': `${listWidth}px` } as CSSProperties}>
      <ConversationList
        contacts={contacts}
        selectedId={selectedId}
        loading={loadingContacts}
        query={query}
        onQueryChange={onQueryChange}
        onSelect={(contact) => {
          onSelect(contact)
          setPane('detail')
        }}
      />
      <ResizeHandle
        label="Resize conversation list"
        value={listWidth}
        min={LIST_WIDTH.min}
        max={LIST_WIDTH.max}
        onChange={onListWidthChange}
      />
      {selectedContact ? (
        <ConversationDetail
          // a new conversation starts at the top, with nothing picked out
          key={selectedContact.chat_id}
          contact={selectedContact}
          preview={preview}
          settled={previewSettled}
          failed={previewError !== null}
          onRetry={onRetryPreview}
          reports={reports}
          models={defaults?.models ?? []}
          model={model}
          onModelChange={onModelChange}
          scope={scope}
          onScopeChange={onScopeChange}
          canRun={Boolean(defaults?.messagesPath && defaults?.outDir)}
          running={isActiveJob(runningJob)}
          failure={
            runningJob?.status === 'failed' && ranFor === selectedId
              ? runningJob.error || 'The report could not be written.'
              : undefined
          }
          onRunReport={run}
          onOpenReport={onOpenReport}
          onShowList={() => setPane('list')}
        />
      ) : (
        <section className="pane pane-detail" aria-label="Conversation" />
      )}
    </div>
  )
}
