import { useId, useLayoutEffect, useRef } from 'react'
import { CloseIcon } from '../Icons'
import { DayReader } from './DayReader'
import { Truncate } from './Truncate'

type DaySheetProps = {
  messagesPath: string
  chatId: string
  title: string
  // the day to open on, "YYYY-MM-DD"
  day: string
  // when the message this was opened for was sent; null to open at the day's start
  at: string | null
  // that message's id, where the server gives one
  pickedId?: string
  onOpenConversation: () => void
  onClose: () => void
}

// One day of one conversation as a sheet, opened from a moment or from a
// message that an answer cites: whose day it is and the ways out of it, over
// the day itself. The sheet is named by the conversation and by the day in view.
export function DaySheet({ messagesPath, chatId, title, day, at, pickedId, onOpenConversation, onClose }: DaySheetProps) {
  const dialog = useRef<HTMLDialogElement>(null)
  const titleId = useId()
  const dayId = useId()

  // Opened before the page is drawn, and before the reader's own effect puts
  // the focus on the messages: a sheet that is not open cannot take it.
  useLayoutEffect(() => {
    const element = dialog.current
    if (element && !element.open) element.showModal()
  }, [])

  return (
    <dialog ref={dialog} className="sheet dayview" aria-labelledby={`${titleId} ${dayId}`} onClose={onClose}>
      <header className="sheet-head">
        <h2 id={titleId}>
          <Truncate className="dayview-title">{title}</Truncate>
        </h2>
        <span className="sheet-actions">
          <button type="button" className="link" onClick={onOpenConversation}>
            Open conversation
          </button>
          <button type="button" className="icon-btn" aria-label="Close" onClick={() => dialog.current?.close()}>
            <CloseIcon />
          </button>
        </span>
      </header>

      <DayReader
        messagesPath={messagesPath}
        chatId={chatId}
        day={day}
        at={at}
        pickedId={pickedId}
        label="Messages of this day"
        dayId={dayId}
        focusOnOpen
      />
    </dialog>
  )
}
