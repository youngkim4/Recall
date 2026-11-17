import { useLayoutEffect, useRef } from 'react'
import { atEnd, type LogLine } from '../../lib/jobFacts'

type JobLogProps = {
  // oldest first
  lines: LogLine[]
  // the job is still writing: the newest line stays in view, unless the reader has gone up to read
  following: boolean
}

const clock = new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit', second: '2-digit' })

function clockTime(time: string) {
  const date = time ? new Date(time) : null
  return date && !Number.isNaN(date.valueOf()) ? clock.format(date) : ''
}

// What a job wrote, oldest first, the time of each line before its words.
// It scrolls inside its own card, which takes the height the pane has left.
export function JobLog({ lines, following }: JobLogProps) {
  const box = useRef<HTMLDivElement>(null)
  // whether the reader was at the end before the newest lines were drawn
  const wasAtEnd = useRef(true)
  const timed = lines.some((line) => line.time)

  useLayoutEffect(() => {
    const element = box.current
    if (element && following && wasAtEnd.current) element.scrollTop = element.scrollHeight
  }, [lines.length, following])

  return (
    <div
      ref={box}
      className="card job-log"
      tabIndex={0}
      role="log"
      aria-label="What this job wrote"
      onScroll={(event) => {
        wasAtEnd.current = atEnd(event.currentTarget)
      }}
    >
      {lines.length ? (
        <ol className={timed ? 'job-lines is-timed' : 'job-lines'} role="list">
          {lines.map((line, index) => (
            <li key={`${index} ${line.time}`}>
              {timed ? (
                <time className="job-line-time" dateTime={line.time || undefined}>
                  {clockTime(line.time)}
                </time>
              ) : null}
              <span className="job-line-words">{line.words}</span>
            </li>
          ))}
        </ol>
      ) : (
        <p className="job-log-empty">{following ? 'Nothing written yet' : 'Nothing was written'}</p>
      )}
    </div>
  )
}
