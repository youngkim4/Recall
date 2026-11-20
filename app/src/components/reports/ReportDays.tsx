import { useState, type ReactNode } from 'react'
import { contactTitle } from '../../lib/conversations'
import { DayOpener, type OpenDay } from '../../lib/dayOpener'
import type { MomentDay } from '../../lib/momentDay'
import type { Contact } from '../../types'
import { DaySheet } from '../ui/DaySheet'

type ReportDaysProps = {
  // the archive that the days are read from; '' while there is none
  messagesPath: string
  contacts: Contact[]
  // the name of the report's conversation, for one that is not in the list
  name?: string
  onOpenConversation: (contact: Contact) => void
  children: ReactNode
}

// Holds the day sheet for the Reports view. A moment of a report opens its
// day through DayOpener, wherever in the report it is drawn; the sheet shows
// that day of the conversation over the report.
export function ReportDays({ messagesPath, contacts, name, onOpenConversation, children }: ReportDaysProps) {
  const [open, setOpen] = useState<(MomentDay & { from?: Element }) | null>(null)
  const openDay: OpenDay = (day, from) => setOpen({ ...day, from })
  const contact = open
    ? (contacts.find((one) => one.chat_id === open.chatId) ?? { chat_id: open.chatId, display_name: name })
    : null

  return (
    <DayOpener value={messagesPath ? openDay : null}>
      {children}
      {open && contact ? (
        <DaySheet
          key={`${open.chatId} ${open.day} ${open.at ?? ''}`}
          messagesPath={messagesPath}
          chatId={open.chatId}
          title={contactTitle(contact)}
          day={open.day}
          at={open.at}
          pickedId={open.pickedId}
          onOpenConversation={() => onOpenConversation(contact)}
          onClose={() => setOpen(null)}
          from={open.from}
        />
      ) : null}
    </DayOpener>
  )
}
