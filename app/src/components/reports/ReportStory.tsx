import { useId, useMemo } from 'react'
import { plural } from '../../lib/format'
import { readStory } from '../../lib/story'
import { Truncate } from '../ui/Truncate'
import { bringIntoView } from './bringIntoView'

type ReportStoryProps = {
  // the story as the server made it safe: allowed tags only, every piece of text escaped
  html: string
}

// The story that was written, set to be read: its title and opening, its own
// headings as chapters with a list of them to jump by, and a column no wider
// than a line reads well. The pieces put on the page are whole elements of
// the server's safe HTML; readStory only cuts between them.
export function ReportStory({ html }: ReportStoryProps) {
  const baseId = useId()
  const story = useMemo(() => readStory(html), [html])
  const idOf = (chapter: string) => `${baseId}-${chapter}`

  if (!story.opening && !story.chapters.length) {
    return <p className="section section-note">This report has no written story</p>
  }

  function goTo(chapter: string) {
    const found = document.getElementById(idOf(chapter))
    if (found) bringIntoView(found, 'start')
  }

  return (
    <section className="section report-story" aria-labelledby={`${baseId}-title`}>
      <div className="section-head">
        <h3 id={`${baseId}-title`}>The story</h3>
        <span className="meta">{plural(story.minutes, 'minute')} to read</span>
      </div>
      {story.title ? <h4 className="report-chapter-title">{story.title}</h4> : null}
      {story.opening ? (
        <div className="prose report-opening" data-lead dangerouslySetInnerHTML={{ __html: story.opening }} />
      ) : null}
      {story.chapters.length > 1 ? (
        <nav className="report-contents card" aria-label="Chapters">
          <ol className="rows" role="list">
            {story.chapters.map((chapter) => (
              <li key={chapter.id}>
                <button type="button" className="row" onClick={() => goTo(chapter.id)}>
                  <span className="row-main">
                    <Truncate className="row-title">{chapter.title}</Truncate>
                  </span>
                </button>
              </li>
            ))}
          </ol>
        </nav>
      ) : null}
      {story.chapters.map((chapter, index) => (
        <section key={chapter.id} id={idOf(chapter.id)} className="report-chapter" aria-label={chapter.title}>
          <h4 className="report-chapter-title">{chapter.title}</h4>
          {chapter.html ? (
            <div
              className="prose"
              // with no opening, the first paragraph of the first chapter leads the story
              data-lead={(!story.opening && index === 0) || undefined}
              dangerouslySetInnerHTML={{ __html: chapter.html }}
            />
          ) : null}
        </section>
      ))}
    </section>
  )
}
