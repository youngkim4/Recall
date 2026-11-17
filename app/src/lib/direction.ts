// A first message says which way it went in a word: "outgoing" or "incoming".
// Any other word, or none, is a message that does not say who sent it. It
// belongs to neither side, and is never taken for one that was received.
export function sentByMe(direction: string | null | undefined): boolean | null {
  if (direction === 'outgoing') return true
  if (direction === 'incoming') return false
  return null
}
