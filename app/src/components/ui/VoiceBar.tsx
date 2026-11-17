type VoiceBarProps = {
  sent: number
  received: number
  // messages that do not say who sent them
  unsaid?: number
  // the bar's length as a share of its track, 0 to 1; a bar that stands alone is 1
  scale?: number
}

// One length split into the two voices: what you sent in blue, what you
// received in gray, the same colors the bubbles wear. What does not say whose
// it is takes the color of a count that belongs to neither. Decorative on
// its own; the row or figure around it states the numbers.
export function VoiceBar({ sent, received, unsaid = 0, scale = 1 }: VoiceBarProps) {
  if (sent + received + unsaid <= 0) return null
  return (
    <span className="voice-bar" aria-hidden="true" style={{ width: `${Math.min(1, Math.max(0, scale)) * 100}%` }}>
      {sent > 0 ? <span className="you" style={{ flexGrow: sent }} /> : null}
      {received > 0 ? <span className="them" style={{ flexGrow: received }} /> : null}
      {unsaid > 0 ? <span className="unsaid" style={{ flexGrow: unsaid }} /> : null}
    </span>
  )
}

export function VoiceLegend() {
  return (
    <span className="legend" aria-hidden="true">
      <span>
        <i className="you" />
        You
      </span>
      <span>
        <i className="them" />
        Them
      </span>
    </span>
  )
}
