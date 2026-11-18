import { useMemo, useState } from 'react'
import { recallApi } from '../../lib/api'
import { messageCount } from '../../lib/conversations'
import { useLoad } from '../../lib/useLoad'
import type { ArchiveSummary, Contact, Defaults, ViewKey } from '../../types'
import { CardStudio } from '../cards/CardStudio'
import { DaySheet } from '../ui/DaySheet'
import { VoicesOverTime, VoicesSkeleton } from '../ui/VoicesOverTime'
import { ArchiveTiles, ArchiveTilesSkeleton } from './ArchiveTiles'
import { Moments, MomentsSkeleton, type MomentOfDay } from './Moments'
import { Rhythm } from './Rhythm'
import { TopConversations, TopConversationsSkeleton } from './TopConversations'
import { YearOfDays } from './YearOfDays'

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

const loadMoments = (messagesPath: string) => recallApi.memories({ messagesPath }).then((response) => response.memories)
const loadPatterns = (messagesPath: string) => recallApi.overview({ messagesPath }).then((response) => response.overview)

// stands in for the chart of months while it loads
function MonthsWaiting() {
  return (
    <div className="section ruled" aria-hidden="true">
      <VoicesSkeleton heading={<h2>By month</h2>} />
    </div>
  )
}

// The archive at a glance, in rows. How much it holds. Then what is worth
// opening: today's moments, with the busiest conversations beside them, as
// tall as the moments are. (Recent reports once stood under the busiest
// conversations and took their rows: the table showed one conversation less
// for each report written. Reports are one click away in the rail.) Then
// when the messages happen: by month, and under that by day and by hour. The page is open: space between the sections,
// hairlines between the rows of a list, and no card but the moments. In a
// wide window the things of a row stand side by side and end on one line; in
// a narrow one they follow each other.
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
  const { load: moments, retry: retryMoments } = useLoad(messagesPath, loadMoments)
  const { load: patterns, retry: retryPatterns } = useLoad(messagesPath, loadPatterns)
  const ranked = useMemo(() => [...contacts].sort((a, b) => messageCount(b) - messageCount(a)), [contacts])
  const voices = useMemo(() => {
    const months = patterns.state === 'ready' ? patterns.value.monthly : []
    return {
      sent: months.reduce((sum, month) => sum + month.sent, 0),
      received: months.reduce((sum, month) => sum + month.received, 0),
    }
  }, [patterns])

  if (loading && !archive) {
    return (
      <div className="view">
        <div className="page wide overview" aria-busy="true">
          <p className="sr-only" role="status">
            Reading your archive
          </p>
          <ArchiveTilesSkeleton />
          <div className="overview-row">
            <MomentsSkeleton />
            <div className="overview-col">
              <TopConversationsSkeleton />
            </div>
          </div>
          <MonthsWaiting />
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

  return (
    <div className="view">
      <div className="page wide overview arrive-each">
        <ArchiveTiles
          archive={archive}
          months={patterns.state === 'ready' ? patterns.value.monthly : null}
          waiting={patterns.state === 'loading'}
          ranked={ranked}
          onOpenSetup={onOpenSetup}
        />

        {/* what is worth opening */}
        <div className="overview-row">
          {moments.state === 'loading' ? (
            <MomentsSkeleton />
          ) : (
            <Moments
              memories={moments.state === 'ready' ? moments.value : null}
              onRetry={retryMoments}
              contacts={contacts}
              onSelectContact={onSelectContact}
              onOpenDay={setOpenDay}
              onReplayFirstWords={onReplayFirstWords}
              onMakeCard={() => setShowCardStudio(true)}
            />
          )}
          <div className="overview-col">
            <TopConversations
              ranked={ranked}
              archive={archive}
              onSelectContact={onSelectContact}
              onViewAll={() => onViewChange('analyze')}
            />
          </div>
        </div>

        {/* when the messages happen */}
        {patterns.state === 'loading' ? (
          <MonthsWaiting />
        ) : patterns.state === 'failed' ? (
          <p className="section section-note">
            The shape of the archive could not be loaded.{' '}
            <button type="button" className="link in-text" onClick={retryPatterns}>
              Try again
            </button>
          </p>
        ) : patterns.value.monthly.length ? (
          <section className="section ruled" aria-labelledby="months-title">
            <VoicesOverTime
              heading={<h2 id="months-title">By month</h2>}
              points={patterns.value.monthly}
              unit="month"
              sent={voices.sent}
              received={voices.received}
              them="Everyone else"
            />
          </section>
        ) : null}

        {patterns.state === 'ready' && patterns.value.daily.counts.length ? (
          <div className="overview-row overview-heat">
            <YearOfDays daily={patterns.value.daily} />
            <Rhythm rhythm={patterns.value.rhythm} />
          </div>
        ) : null}
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
        />
      ) : null}

      {showCardStudio ? (
        <CardStudio
          defaults={defaults}
          model={model}
          contacts={contacts}
          memories={moments.state === 'ready' ? moments.value : null}
          onClose={() => setShowCardStudio(false)}
        />
      ) : null}
    </div>
  )
}
