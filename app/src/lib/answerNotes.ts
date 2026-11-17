import type { ChatMessage } from './chatMessages.ts'

// What the reader should know of an answer beside its words: a quote that
// the archive does not hold, an answer that was cut short, an answer that
// they stopped. Each note is one short sentence.
export function answerNotes(message: ChatMessage): string[] {
  if (message.streaming) return []
  const response = message.response
  const quotes = Array.isArray(response?.unverifiedQuotes) ? response.unverifiedQuotes.length : 0
  return [
    quotes === 1 ? '1 quote in this answer is not in your messages.' : '',
    quotes > 1 ? `${quotes} quotes in this answer are not in your messages.` : '',
    response?.truncated === true ? 'This answer was cut short.' : '',
    message.stopped === true || response?.cancelled === true ? 'You stopped this answer.' : '',
  ].filter(Boolean)
}
