import { VoicesSkeleton } from '../ui/VoicesOverTime'

// What the facts are called, and words about as long as what each comes to,
// so that a waiting cell runs on to a second line where the fact will.
const FACTS = [
  ['Active days', '000 of 0,000'],
  ['Busiest day', 'Mon 00, 0000, 0,000 messages'],
  ['Longest gap', '00 days'],
  ['First to write', 'Someone, on 00% of days'],
  ['Shared', '0,000 photos, 00 videos, 0 GIFs, 000 other'],
  ['Reactions', '000'],
]
// the rows a list of people opens with: the busiest five, and your own
const PEOPLE = [0, 1, 2, 3, 4, 5]
// how wide each waiting bubble is, as a share of the thread; every third one is yours
const BUBBLES = [46, 62, 34, 54, 40, 58, 36]

// The shape the figures will have, so nothing moves when they arrive. A
// narrow pane waits with the chart alone, as the figures under it come in
// below everything that is on the page. A wide pane holds every part of its
// two columns: the row is as tall while it waits as it is afterwards.
export function FiguresSkeleton({ group }: { group: boolean }) {
  return (
    <>
      <div className="section">
        <VoicesSkeleton />
      </div>

      <div className="detail-facts is-waiting">
        <dl className="section facts">
          {FACTS.map(([label, value]) => (
            <div key={label}>
              <dt>
                <span className="skeleton">{label}</span>
              </dt>
              <dd>
                <span className="skeleton">{value}</span>
              </dd>
            </div>
          ))}
        </dl>

        {group ? (
          <div className="section people">
            <div className="section-head">
              <span className="skeleton">People</span>
            </div>
            <ol className="rows" role="list">
              {PEOPLE.map((place) => (
                <li key={place} className="row ranked-row">
                  <span className="ranked-place">
                    <span className="skeleton">0</span>
                  </span>
                  <span className="row-title">
                    <span className="skeleton">The name of a person</span>
                  </span>
                  <span className="ranked-bar" />
                  <span className="ranked-count">
                    <span className="skeleton">0,000</span>
                  </span>
                  <span className="ranked-share">
                    <span className="skeleton">00%</span>
                  </span>
                </li>
              ))}
            </ol>
          </div>
        ) : null}
      </div>

      <div className="section latest is-waiting">
        <div className="section-head">
          <span className="skeleton">Latest messages</span>
        </div>
        <div className="latest-scroll">
          <div className="thread">
            {BUBBLES.map((width, index) => (
              <span
                key={index}
                className={`skeleton latest-wait${index % 3 === 2 ? ' you' : ''}`}
                style={{ width: `${width}%` }}
              />
            ))}
          </div>
        </div>
      </div>
    </>
  )
}
