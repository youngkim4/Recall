import { greetingAt } from '../../lib/overviewMessages'
import { CardIcon, ChatIcon, PlayIcon, SearchIcon } from '../Icons'

type OverviewHeadProps = {
  onAsk: () => void
  onSearch: () => void
  onReplayFirstWords: () => void
  onMakeCard: () => void
}

// The top of the Overview, after Mercury's home: a short greeting, then the
// things to do from here as one row of buttons. Asking in Chat is the first
// and the only one in ink. (A line of the archive's size set across the whole
// width, under a double rule, was refused as ugly; the size is now in the
// messages card under this.)
export function OverviewHead({ onAsk, onSearch, onReplayFirstWords, onMakeCard }: OverviewHeadProps) {
  return (
    <header className="overview-head">
      <h1 className="overview-greeting">{greetingAt(new Date().getHours())}</h1>
      <div className="overview-actions">
        <button type="button" className="btn primary" onClick={onAsk}>
          <ChatIcon />
          Ask in Chat
        </button>
        <button type="button" className="btn" onClick={onSearch}>
          <SearchIcon />
          Search
        </button>
        <button type="button" className="btn" onClick={onReplayFirstWords}>
          <PlayIcon />
          Replay first words
        </button>
        <button type="button" className="btn" onClick={onMakeCard}>
          <CardIcon />
          Make a card
        </button>
      </div>
    </header>
  )
}

export function OverviewHeadSkeleton() {
  return (
    <div className="overview-head" aria-hidden="true">
      <p className="overview-greeting">
        <span className="skeleton">Good evening</span>
      </p>
      <div className="overview-actions">
        <span className="btn skeleton">Ask in Chat</span>
      </div>
    </div>
  )
}
