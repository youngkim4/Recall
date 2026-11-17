import { useLayoutEffect, useState } from 'react'
import { rowsThatFit } from './fit'

// How many rows a list shows. At rest it is `atRest`. Where the style sheet
// gives the list's box a height from outside (it sets --fit on the box, and
// the box does not grow with its rows), it is the count that leaves a row
// nearest its own height (rowsThatFit), and never fewer than `fewest`. The
// rows of such a box share its height, so one more row makes each a little
// shorter and none is cut off. rowHeight gives the height of one
// row in pixels for the size of a rem; pass the same function each time.
// The first thing it returns is the ref for the box: a box that is put on
// the page later than the list's first drawing is measured from then on.
export function useRowsThatFit(rowHeight: (rem: number) => number, atRest: number, fewest: number) {
  const [box, setBox] = useState<HTMLElement | null>(null)
  const [rows, setRows] = useState(atRest)

  useLayoutEffect(() => {
    if (!box) return
    const measure = () => {
      const fits = getComputedStyle(box).getPropertyValue('--fit').trim() === '1'
      const rem = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16
      setRows(fits ? rowsThatFit(box.clientHeight, rowHeight(rem), fewest) : atRest)
    }
    measure()
    const observer = new ResizeObserver(measure)
    observer.observe(box)
    return () => observer.disconnect()
  }, [box, rowHeight, atRest, fewest])

  return [setBox, rows] as const
}
