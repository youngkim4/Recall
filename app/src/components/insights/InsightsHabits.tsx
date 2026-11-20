import { formatPercent, plural } from '../../lib/format'
import { formatLength, formatSpan, formatWait } from '../../lib/habits'
import type { Habits } from '../../lib/insights'
import { hourName } from '../../lib/locale'

type InsightsHabitsProps = {
  habits: Habits | null
  // the habits could not be counted
  failed: boolean
  // the words, the sender or the attachments narrow the set: replies and
  // starts still count whole conversations, and their notes say so
  narrowed: boolean
}

// where late at night ends, as the server counts it (recall_services/habits.py)
const LATE_UNTIL_HOUR = 5
const WHOLE = 'all messages, not only these'

type Figure = { label: string; value: string; note: string; title?: string }

function figuresOf(habits: Habits, narrowed: boolean): Figure[] {
  const streak = habits.streak
  return [
    {
      label: 'You reply in',
      value: formatWait(habits.youReply),
      note: narrowed ? WHOLE : `They reply in ${formatWait(habits.theyReply)}`,
      title: 'Half of your replies come sooner. A reply is the first answer after the other side starts a talk.',
    },
    {
      label: 'You start',
      value: formatPercent(habits.youStart),
      note: narrowed ? WHOLE : `of ${plural(habits.talks, 'talk')}`,
      title: 'A talk starts after 6 hours of quiet in a one-to-one conversation.',
    },
    {
      label: 'Longest streak',
      value: streak ? plural(streak.days, 'day') : '—',
      note: streak ? formatSpan(streak.first, streak.last) : 'No day',
    },
    {
      label: 'Late night',
      value: formatPercent(habits.lateNight),
      note: `${hourName(0)} to ${hourName(LATE_UNTIL_HOUR)}`,
    },
    {
      label: 'Message length',
      value: `${formatLength(habits.sentLength)} / ${formatLength(habits.receivedLength)}`,
      note: 'characters, you / them',
    },
  ]
}

const PLACES = ['You reply in', 'You start', 'Longest streak', 'Late night', 'Message length']

// How the messages of the answer are sent, after the figures under a chart
// in Spotify Wrapped and Apple's Screen Time: five figures in one row, a
// hairline between them, each a name, a number set large, and one note. They
// stand on the page, not in a card: the card above is the answer's own.
export function InsightsHabits({ habits, failed, narrowed }: InsightsHabitsProps) {
  return (
    <section className="section insights-habits" aria-labelledby="insights-habits-title">
      <div className="section-head">
        <h2 id="insights-habits-title">Habits</h2>
      </div>
      {failed && !habits ? (
        <p className="insights-failed">Could not be counted</p>
      ) : (
        <HabitFigures habits={habits} narrowed={narrowed} />
      )}
    </section>
  )
}

function HabitFigures({ habits, narrowed }: { habits: Habits | null; narrowed: boolean }) {
  return (
    <dl className="habit-figures">
      {habits
        ? figuresOf(habits, narrowed).map((figure) => (
            <div key={figure.label} className="habit-figure" title={figure.title}>
              <dt>{figure.label}</dt>
              <dd className="habit-value swap" key={figure.value}>
                {figure.value}
              </dd>
              <dd className="habit-note">{figure.note}</dd>
            </div>
          ))
        : PLACES.map((label) => (
            <div key={label} className="habit-figure" aria-hidden="true">
              <dt>{label}</dt>
              <dd className="habit-value">
                <span className="skeleton">00 min</span>
              </dd>
              <dd className="habit-note">
                <span className="skeleton">They reply in 0 min</span>
              </dd>
            </div>
          ))}
    </dl>
  )
}
