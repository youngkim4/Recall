import { useLayoutEffect, useRef } from 'react'
import { fitSize } from './fitText'

// Sets one line of text across the width of its parent, as a masthead is
// set: the text is measured at `max` on one line, and its size is scaled to
// the room (fitText.ts), between `min` and `max`. Where even `min` is too
// large for one line, it wraps at the size the style sheet gives it.
// The size is written on the element itself, so a resize draws nothing
// again in React. `text` is what the line says: a new text is measured anew.
// It returns the ref for the element.
export function useFitText<Element extends HTMLElement>(text: string, min: number, max: number) {
  const ref = useRef<Element>(null)

  useLayoutEffect(() => {
    const parent = ref.current?.parentElement
    if (!ref.current || !parent) return
    const measure = () => {
      const element = ref.current
      if (!element) return
      element.style.whiteSpace = 'nowrap'
      element.style.fontSize = `${max}px`
      // the width of the words themselves, not of the box they stand in
      const range = document.createRange()
      range.selectNodeContents(element)
      const widthAt = range.getBoundingClientRect().width
      const { size, wraps } = fitSize({ room: parent.clientWidth, widthAt, at: max, min, max })
      // a line that cannot be one line takes the style sheet's size again and wraps
      element.style.fontSize = wraps ? '' : `${size}px`
      element.style.whiteSpace = wraps ? '' : 'nowrap'
    }
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(parent)
    // a face that arrives late changes the width of the words
    let live = true
    void document.fonts?.ready.then(() => {
      if (live) measure()
    })
    return () => {
      live = false
      observer.disconnect()
    }
  }, [text, min, max])

  return ref
}
