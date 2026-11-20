import { useMemo, useState } from 'react'
import { statsAsk, statsKey } from '../../lib/insights'
import { loadStats } from '../../lib/insightsLoad'
import { latelyRows, latelyWindows, type LatelyRow } from '../../lib/lately'
import { LIST_MOST } from '../../lib/overviewFit'
import { loadMoments } from '../../lib/overviewLoads'
import { NO_FILTERS } from '../../lib/searchFilters'
import { useLoad } from '../../lib/useLoad'
import type { ArchiveSummary, Contact, Defaults, ViewKey } from '../../types'
import { CardStudio } from '../cards/CardStudio'
import { DaySheet } from '../ui/DaySheet'
import { ArchiveHead, ArchiveHeadSkeleton } from './ArchiveHead'
import { GoneQuiet } from './GoneQuiet'
import { Lately } from './Lately'
import { Moments, MomentsSkeleton, type MomentOfDay } from './Moments'

type OverviewViewProps = {
  defaults: Defaults | null
  // null until the archive has been read, or when there is none
  archive: ArchiveSummary | null
  contacts: Contact[]
  loading: boolean
  model: string
  onViewChange: (view: ViewKey) => void
  onSelectContact: (contact: Contact) => void
  onReplayFirstWords: () => void
  // opens the guide that builds the archive, or updates one that exists
  onOpenSetup: () => void
}

// the people of the thirty days before, to set each of today's against
const BEFORE_ROWS = 50

// The archive's own thirty days and the thirty before them, counted by person.
function useLately(messagesPath: string, lastDay: string) {
  const asks = useMemo(() => {
    if (!lastDay) return { now: '', before: '' }
    const { now, before } = latelyWindows(lastDay)
    return {
      now: statsKey(messagesPath, statsAsk({ ...NO_FILTERS, ...now }, '', 'person', LIST_MOST)),
      before: statsKey(messagesPath, statsAsk({ ...NO_FILTERS, ...before }, '', 'person', BEFORE_ROWS)),
    }
  }, [messagesPath, lastDay])
  const { load: now } = useLoad(asks.now, loadStats)
  const { load: before } = useLoad(asks.before, loadStats)
  return useMemo(() => {
    if (now.state !== 'ready' || before.state === 'loading') return null
    return latelyRows(now.value.rows, before.state === 'ready' ? before.value.rows : [], LIST_MOST)
  }, [now, before])
}

// After Spotify's home: what the archive holds, in one line set large; what is
// worth opening today, as a shelf of cards; then two lists side by side, who
// the messages have been with lately and who has gone quiet. Counting the
// archive every other way is Insights' work. The page is open: 32px between
// its blocks, hairlines between the rows of a list, and no card but the
// moments. The two lists end on one line; in a narrow window they follow
// each other.
export function OverviewView({
  defaults,
  archive,
  contacts,
  loading,
  model,
  onViewChange,
  onSelectContact,
  onReplayFirstWords,
  onOpenSetup,
}: OverviewViewProps) {
  const [showCardStudio, setShowCardStudio] = useState(false)
  // the moment whose day is open
  const [openDay, setOpenDay] = useState<MomentOfDay | null>(null)
  const messagesPath = archive ? defaults?.messagesPath || '' : ''
  const lastDay = archive?.lastMessage?.slice(0, 10) ?? ''
  const { load: moments, retry: retryMoments } = useLoad(messagesPath, loadMoments)
  const lately = useLately(messagesPath, lastDay)
  const loaded = useMemo(() => new Map(contacts.map((contact) => [contact.chat_id, contact])), [contacts])

  const openPerson = (row: LatelyRow) => {
    const contact = (row.conversations ?? [row.key]).map((id) => loaded.get(id)).find(Boolean)
    if (contact) onSelectContact(contact)
  }

  if (loading && !archive) {
    return (
      <div className="view">
        <div className="page wide overview" aria-busy="true">
          <p className="sr-only" role="status">
            Reading your archive
          </p>
          <ArchiveHeadSkeleton />
          <MomentsSkeleton />
        </div>
      </div>
    )
  }

  if (!archive || !archive.conversations) {
    return (
      <div className="view">
        <div className="pane-empty arrive-each">
          <h2 className="overview-lead">No messages imported yet</h2>
          <button type="button" className="btn primary" onClick={onOpenSetup}>
            Build your archive
          </button>
        </div>
      </div>
    )
  }

  const memories = moments.state === 'ready' ? moments.value : null

  return (
    <div className="view">
      <div className="page wide overview arrive-each">
        <ArchiveHead archive={archive} />

        {moments.state === 'loading' ? (
          <MomentsSkeleton />
        ) : (
          <Moments
            memories={memories}
            onRetry={retryMoments}
            contacts={contacts}
            onSelectContact={onSelectContact}
            onOpenDay={setOpenDay}
            onReplayFirstWords={onReplayFirstWords}
            onMakeCard={() => setShowCardStudio(true)}
          />
        )}

        <div className="overview-row">
          <Lately
            rows={lately}
            messagesPath={messagesPath}
            lastDay={lastDay}
            onOpen={openPerson}
            onOpenInsights={() => onViewChange('insights')}
          />
          <GoneQuiet
            memories={memories}
            failed={moments.state === 'failed'}
            contacts={contacts}
            onOpenDay={setOpenDay}
            onSelectContact={onSelectContact}
          />
        </div>
      </div>

      {openDay ? (
        <DaySheet
          key={`${openDay.contact.chat_id} ${openDay.day}`}
          messagesPath={messagesPath}
          chatId={openDay.contact.chat_id}
          title={openDay.title}
          day={openDay.day}
          at={openDay.at}
          pickedId={openDay.quotedId}
          onOpenConversation={() => onSelectContact(openDay.contact)}
          onClose={() => setOpenDay(null)}
          from={openDay.from}
        />
      ) : null}

      {showCardStudio ? (
        <CardStudio
          defaults={defaults}
          model={model}
          contacts={contacts}
          memories={memories}
          onClose={() => setShowCardStudio(false)}
        />
      ) : null}
    </div>
  )
}
