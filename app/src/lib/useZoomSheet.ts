import { useLayoutEffect, useRef, type SyntheticEvent } from 'react'
import { zoomIn, zoomOut } from './sheetZoom.ts'

// A modal sheet that grows out of what opened it and goes back into it when
// it closes (sheetZoom.ts). The sheet is open from its first frame, before
// the effects of what is in it run, so they can put the focus in it. Every way
// out (a Close button through `close`, Escape) goes back into the origin first;
// `onClose` runs once the sheet is closed. Spread `dialogProps` on the <dialog>.
export function useZoomSheet(from: Element | null | undefined, onClose: () => void) {
  const dialog = useRef<HTMLDialogElement>(null)
  const closing = useRef(false)

  useLayoutEffect(() => {
    const element = dialog.current
    if (!element) return
    if (!element.open) element.showModal()
    return zoomIn(element, from)
    // the origin is the one the sheet opened from; a sheet opened from another is another sheet (its key)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  async function close() {
    const element = dialog.current
    if (!element || closing.current) return
    closing.current = true
    await zoomOut(element, from)
    element.close()
  }

  const dialogProps = {
    ref: dialog,
    'data-zoom': from ? '' : undefined,
    onCancel: (event: SyntheticEvent<HTMLDialogElement>) => {
      event.preventDefault()
      void close()
    },
    onClose,
  }

  return { close, dialogProps }
}
