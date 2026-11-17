import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { Thread, type ThreadMessage } from '../ui/Thread'

// The scroller counts as at its end within this many pixels of it.
const AT_END = 2

// The last messages of a conversation. In a narrow pane they run down the
// page under the figures. In a wide pane they are a card beside the figures
// and as tall as them: the thread scrolls inside the card, and opens at its
// end with the newest message in view, the way a conversation does.
export function LatestMessages({ messages }: { messages: ThreadMessage[] }) {
  const scroller = useRef<HTMLDivElement>(null)
  // whether the newest message is in view, so that it stays there when the card changes height
  const atEnd = useRef(true)
  // only a thread that scrolls is a stop for the keyboard
  const [scrolls, setScrolls] = useState(false)

  // a new run of messages opens at its end
  useLayoutEffect(() => {
    const box = scroller.current
    if (!box) return
    atEnd.current = true
    box.scrollTop = box.scrollHeight
  }, [messages])

  // The card is as tall as the column beside it, which opens and closes.
  useEffect(() => {
    const box = scroller.current
    if (!box) return
    const watcher = new ResizeObserver(() => {
      if (atEnd.current) box.scrollTop = box.scrollHeight
      setScrolls(box.scrollHeight > box.clientHeight + 1)
    })
    watcher.observe(box)
    if (box.firstElementChild) watcher.observe(box.firstElementChild)
    return () => watcher.disconnect()
  }, [])

  function handleScroll() {
    const box = scroller.current
    if (box) atEnd.current = box.scrollHeight - box.scrollTop - box.clientHeight < AT_END
  }

  return (
    <section className="section latest" aria-labelledby="latest-title">
      <div className="section-head">
        <h3 id="latest-title">Latest messages</h3>
      </div>
      <div
        ref={scroller}
        // what scrolls under the heading fades at that edge, as the lists do
        className={scrolls ? 'latest-scroll pane-scroll' : 'latest-scroll'}
        tabIndex={scrolls ? 0 : undefined}
        role={scrolls ? 'group' : undefined}
        aria-labelledby={scrolls ? 'latest-title' : undefined}
        onScroll={handleScroll}
      >
        <Thread messages={messages} />
      </div>
    </section>
  )
}
