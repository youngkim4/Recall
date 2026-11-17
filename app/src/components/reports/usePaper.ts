import { useCallback, useEffect, useRef, useState } from 'react'
import { flushSync } from 'react-dom'

const ATTRIBUTE = 'data-paper'

// Paper: the report set as a document, to print or to save as a PDF. It is a
// state of the page (styles/paper.css). Export opens it on screen, with its
// own bar, so what is saved is what was seen. A print asked for while the
// report is simply open puts it up for the length of that print.
//
// name is what a saved file is offered under.
export function usePaper(name: string) {
  const [paper, setPaper] = useState(false)
  // true while the paper is up only because a print began outside it
  const borrowed = useRef(false)
  const titleBefore = useRef<string | null>(null)

  const lay = useCallback(() => {
    const root = document.documentElement
    if (root.hasAttribute(ATTRIBUTE)) return
    titleBefore.current = document.title
    document.title = name
    root.setAttribute(ATTRIBUTE, 'report')
  }, [name])

  const lift = useCallback(() => {
    const root = document.documentElement
    if (!root.hasAttribute(ATTRIBUTE)) return
    root.removeAttribute(ATTRIBUTE)
    if (titleBefore.current !== null) document.title = titleBefore.current
    titleBefore.current = null
  }, [])

  const open = useCallback(() => {
    borrowed.current = false
    lay()
    setPaper(true)
    // the paper is a page of its own: it starts at its top
    window.scrollTo(0, 0)
  }, [lay])

  const close = useCallback(() => {
    lift()
    setPaper(false)
  }, [lift])

  useEffect(() => {
    // a print from the menu: everything folded away is opened before the page is measured
    const before = () => {
      if (document.documentElement.hasAttribute(ATTRIBUTE)) return
      borrowed.current = true
      flushSync(() => setPaper(true))
      lay()
    }
    const after = () => {
      if (!borrowed.current) return
      borrowed.current = false
      close()
    }
    window.addEventListener('beforeprint', before)
    window.addEventListener('afterprint', after)
    return () => {
      window.removeEventListener('beforeprint', before)
      window.removeEventListener('afterprint', after)
    }
  }, [lay, close])

  // a report taken off the page takes its paper with it
  useEffect(() => lift, [lift])

  return { paper, open, close, print: () => window.print() }
}
