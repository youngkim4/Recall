type VoiceBarProps = {
  sent: number
  received: number
  // the bar's length as a share of its track, 0 to 1; a bar that stands alone is 1
  scale?: number
}

// One length split into the two voices: what you sent in blue, what you
// received in gray, the same colors the bubbles wear. Decorative on its own;
// the row or figure around it states the numbers.
export function VoiceBar({ sent, received, scale = 1 }: VoiceBarProps) {
  if (sent + received <= 0) return null
  return (
    <span className="voice-bar" aria-hidden="true" style={{ width: `${Math.min(1, Math.max(0, scale)) * 100}%` }}>
      {sent > 0 ? <span className="you" style={{ flexGrow: sent }} /> : null}
      {received > 0 ? <span className="them" style={{ flexGrow: received }} /> : null}
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
