import { useMemo, useState } from 'react'
import { recallApi } from '../../lib/api'
import { messageCount } from '../../lib/conversations'
import { useLoad } from '../../lib/useLoad'
import type { ArchiveSummary, Contact, Defaults, ReportFile, ViewKey } from '../../types'
import { CardStudio } from '../cards/CardStudio'
import { VoicesOverTime, VoicesSkeleton } from '../ui/VoicesOverTime'
import { ArchiveTiles, ArchiveTilesSkeleton } from './ArchiveTiles'
import { MomentDay } from './MomentDay'
import { Moments, MomentsSkeleton, type MomentOfDay } from './Moments'
import { RecentReports } from './RecentReports'
import { Rhythm } from './Rhythm'
import { TopConversations } from './TopConversations'
import { YearOfDays } from './YearOfDays'

type OverviewViewProps = {
  defaults: Defaults | null
  // null until the archive has been read, or when there is none
  archive: ArchiveSummary | null
  contacts: Contact[]
  reports: ReportFile[]
  loading: boolean
  model: string
  onViewChange: (view: ViewKey) => void
  onSelectContact: (contact: Contact) => void
  onOpenReport: (report: ReportFile) => void
  onReplayFirstWords: () => void
  // opens the guide that builds the archive, or updates one that exists
  onOpenSetup: () => void
}

const loadMoments = (messagesPath: string) => recallApi.memories({ messagesPath }).then((response) => response.memories)
const loadPatterns = (messagesPath: string) => recallApi.overview({ messagesPath }).then((response) => response.overview)

// stands in for the chart of months while it loads
function MonthsWaiting() {
  return (
    <div className="section" aria-hidden="true">
      <div className="section-head">
        <h2>By month</h2>
      </div>
      <div className="card card-pad">
        <VoicesSkeleton />
      </div>
    </div>
  )
}

// The archive at a glance: how much it holds, then two halves. One is what is
// worth opening, today's moments and the busiest conversations. The other is
// when the messages happen, by month, by day and by hour. A wide window sets
// the halves side by side; a narrow one sets them one under the other.
export function OverviewView({
  defaults,
  archive,
  contacts,
  reports,
  loading,
  model,
  onViewChange,
  onSelectContact,
  onOpenReport,
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
        <div className="page wide" aria-busy="true">
          <p className="sr-only" role="status">
            Reading your archive
          </p>
          <ArchiveTilesSkeleton />
          <div className="overview-cols">
            <div className="overview-col">
              <MomentsSkeleton />
            </div>
            <div className="overview-col">
              <MonthsWaiting />
            </div>
          </div>
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

  // the sections below wait for the two above them, so nothing on the page is pushed down as they arrive
  const settled = moments.state !== 'loading' && patterns.state !== 'loading'

  return (
    <div className="view">
      <div className="page wide arrive-each">
        <ArchiveTiles
          archive={archive}
          months={patterns.state === 'ready' ? patterns.value.monthly : null}
          waiting={patterns.state === 'loading'}
          ranked={ranked}
          onOpenSetup={onOpenSetup}
        />

        <div className="overview-cols">
          {/* what is worth opening */}
          <div className="overview-col arrive-each">
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
            {settled ? (
              <TopConversations
                ranked={ranked}
                archive={archive}
                onSelectContact={onSelectContact}
                onViewAll={() => onViewChange('analyze')}
              />
            ) : null}
          </div>

          {/* when the messages happen */}
          <div className="overview-col arrive-each">
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
              <section className="section" aria-labelledby="months-title">
                <div className="section-head">
                  <h2 id="months-title">By month</h2>
                </div>
                <div className="card card-pad">
                  <VoicesOverTime
                    points={patterns.value.monthly}
                    unit="month"
                    sent={voices.sent}
                    received={voices.received}
                    them="Everyone else"
                  />
                </div>
              </section>
            ) : null}
            {settled ? (
              <>
                {patterns.state === 'ready' && patterns.value.daily.counts.length ? (
                  <>
                    <YearOfDays daily={patterns.value.daily} />
                    <Rhythm rhythm={patterns.value.rhythm} />
                  </>
                ) : null}
                <RecentReports
                  reports={reports}
                  onOpenReport={onOpenReport}
                  onViewAll={() => onViewChange('reports')}
                  onChooseConversation={() => onViewChange('analyze')}
                />
              </>
            ) : null}
          </div>
        </div>
      </div>

      {openDay ? (
        <MomentDay
          key={`${openDay.contact.chat_id} ${openDay.day}`}
          messagesPath={messagesPath}
          chatId={openDay.contact.chat_id}
          title={openDay.title}
          day={openDay.day}
          at={openDay.at}
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
