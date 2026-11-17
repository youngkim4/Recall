import { useLayoutEffect, useState, type RefObject } from 'react'
import { rowsThatFit } from './fit'

// How many rows a list shows. At rest it is `atRest`. Where the style sheet
// gives the list's box a height from outside (it sets --fit on the box, and
// the box does not grow with its rows), it is as many rows as that height
// holds, and never fewer than `fewest`. rowHeight gives the height of one
// row in pixels for the size of a rem; pass the same function each time.
export function useRowsThatFit(
  box: RefObject<HTMLElement | null>,
  rowHeight: (rem: number) => number,
  atRest: number,
  fewest: number,
) {
  const [rows, setRows] = useState(atRest)

  useLayoutEffect(() => {
    const element = box.current
    if (!element) return
    const measure = () => {
      const fits = getComputedStyle(element).getPropertyValue('--fit').trim() === '1'
      const rem = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16
      setRows(fits ? rowsThatFit(element.clientHeight, rowHeight(rem), fewest) : atRest)
    }
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(element)
    return () => observer.disconnect()
  }, [box, rowHeight, atRest, fewest])

  return rows
}
