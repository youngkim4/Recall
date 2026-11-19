import type { ReactNode } from 'react'

// One icon set, drawn at its render size: a 16px grid with a 1.5px stroke,
// round caps and joins. Line centers sit on quarter pixels so edges land on
// whole or half pixels. Navigation icons also have a filled form, used for
// the selected item; its silhouette matches the outline's outer edge so the
// swap does not shift. Color always comes from the surrounding text.

type IconProps = {
  className?: string
}

type NavIconProps = IconProps & {
  filled?: boolean
}

function Line({
  className = 'icon',
  width = 1.5,
  children,
}: IconProps & { width?: number; children: ReactNode }) {
  return (
    <svg
      className={className}
      viewBox="0 0 16 16"
      fill="none"
      stroke="currentColor"
      strokeWidth={width}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {children}
    </svg>
  )
}

function Solid({ className = 'icon', children }: IconProps & { children: ReactNode }) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      {children}
    </svg>
  )
}

/* ---- navigation ---- */

export function HomeIcon({ className, filled }: NavIconProps) {
  if (filled) {
    return (
      <Solid className={className}>
        <path d="M7.52 2.06a.75.75 0 0 1 .96 0l5.25 4.5a.75.75 0 0 1 .27.57v5.12a2.25 2.25 0 0 1-2.25 2.25H10v-3.75a.75.75 0 0 0-.75-.75h-2.5a.75.75 0 0 0-.75.75v3.75H4.25A2.25 2.25 0 0 1 2 12.25V7.13a.75.75 0 0 1 .27-.57l5.25-4.5Z" />
      </Solid>
    )
  }
  return (
    <Line className={className}>
      <path d="M2.75 7.25 8 2.75l5.25 4.5v5a1.5 1.5 0 0 1-1.5 1.5h-7.5a1.5 1.5 0 0 1-1.5-1.5v-5Z" />
      <path d="M6.75 13.5v-2.75h2.5v2.75" />
    </Line>
  )
}

const CHAT_PATH =
  'M8 2.75c-3.18 0-5.75 2.13-5.75 4.75 0 1.2.54 2.3 1.43 3.13-.1.74-.42 1.48-.93 2.1-.17.2 0 .52.27.5 1.2-.1 2.25-.5 3.06-1.03.6.03 1.25.05 1.92.05 3.18 0 5.75-2.13 5.75-4.75S11.18 2.75 8 2.75Z'

export function ChatIcon({ className, filled }: NavIconProps) {
  return (
    <Line className={className}>
      <path d={CHAT_PATH} fill={filled ? 'currentColor' : 'none'} />
    </Line>
  )
}

export function AnalyzeIcon({ className, filled }: NavIconProps) {
  if (filled) {
    return (
      <Solid className={className}>
        <rect x="2.5" y="8.25" width="3" height="5.75" rx="1" />
        <rect x="6.5" y="2" width="3" height="12" rx="1" />
        <rect x="10.5" y="5.25" width="3" height="8.75" rx="1" />
      </Solid>
    )
  }
  return (
    <Line className={className}>
      <path d="M4 13.25V9M8 13.25V2.75M12 13.25V6" />
    </Line>
  )
}

// a line that rises: counts over time, for Insights; the filled form is the same line, heavier
export function InsightsIcon({ className, filled }: NavIconProps) {
  return (
    <Line className={className} width={filled ? 2 : 1.5}>
      <path d="M2.75 12.25 6.25 8.5l2.5 2.25 4.5-5.5" />
      <path d="M10.5 5.25h2.75V8" />
    </Line>
  )
}

export function SearchIcon({ className, filled }: NavIconProps) {
  return (
    <Line className={className} width={filled ? 2 : 1.5}>
      <circle cx="7.25" cy="7.25" r="4.5" />
      <path d="m10.75 10.75 2.5 2.5" />
    </Line>
  )
}

export function ReportIcon({ className, filled }: NavIconProps) {
  if (filled) {
    return (
      <Solid className={className}>
        <path
          fillRule="evenodd"
          d="M4.75 1.5h4.6a.75.75 0 0 1 .53.22l3.4 3.4a.75.75 0 0 1 .22.53v6.6a2.25 2.25 0 0 1-2.25 2.25h-6.5a2.25 2.25 0 0 1-2.25-2.25v-8.5A2.25 2.25 0 0 1 4.75 1.5Zm.75 6.13h5a.63.63 0 0 1 0 1.25h-5a.63.63 0 0 1 0-1.25Zm0 2.5h3.25a.63.63 0 0 1 0 1.25H5.5a.63.63 0 0 1 0-1.25Z"
        />
      </Solid>
    )
  }
  return (
    <Line className={className}>
      <path d="M4.75 2.25h4.6l3.4 3.4v6.6a1.5 1.5 0 0 1-1.5 1.5h-6.5a1.5 1.5 0 0 1-1.5-1.5v-8.5a1.5 1.5 0 0 1 1.5-1.5Z" />
      <path d="M5.75 8.25h4.5M5.75 10.75h2.75" />
    </Line>
  )
}

export function JobsIcon({ className, filled }: NavIconProps) {
  if (filled) {
    return (
      <Solid className={className}>
        <path
          fillRule="evenodd"
          d="M8 1.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13Zm.63 3.25a.63.63 0 0 0-1.25 0V8c0 .2.09.38.25.5l2 1.5a.63.63 0 0 0 .75-1L8.63 7.69V4.75Z"
        />
      </Solid>
    )
  }
  return (
    <Line className={className}>
      <circle cx="8" cy="8" r="5.75" />
      <path d="M8 4.75V8l2 1.5" />
    </Line>
  )
}

export function SettingsIcon({ className, filled }: NavIconProps) {
  return (
    <Line className={className}>
      <path d="M2.75 5h4.5M10.75 5h2.5M2.75 11h2M8.25 11h5" />
      <circle cx="9" cy="5" r="1.75" fill={filled ? 'currentColor' : 'none'} />
      <circle cx="6.5" cy="11" r="1.75" fill={filled ? 'currentColor' : 'none'} />
    </Line>
  )
}

/* ---- controls ---- */

export function SidebarIcon({ className }: IconProps) {
  return (
    <Line className={className}>
      <rect x="2.25" y="3.25" width="11.5" height="9.5" rx="2" />
      <path d="M6.25 3.5v9" />
    </Line>
  )
}

export function PlusIcon({ className }: IconProps) {
  return (
    <Line className={className}>
      <path d="M8 3.25v9.5M3.25 8h9.5" />
    </Line>
  )
}

export function CloseIcon({ className }: IconProps) {
  return (
    <Line className={className}>
      <path d="m4.25 4.25 7.5 7.5m0-7.5-7.5 7.5" />
    </Line>
  )
}

export function PauseIcon({ className }: IconProps) {
  return (
    <Solid className={className}>
      <rect x="4" y="3.25" width="2.75" height="9.5" rx="0.75" />
      <rect x="9.25" y="3.25" width="2.75" height="9.5" rx="0.75" />
    </Solid>
  )
}

export function PlayIcon({ className }: IconProps) {
  return (
    <Solid className={className}>
      <path d="M5 3.9a.75.75 0 0 1 1.14-.64l6.7 4.1a.75.75 0 0 1 0 1.28l-6.7 4.1A.75.75 0 0 1 5 12.1V3.9Z" />
    </Solid>
  )
}

// standalone and emphasized, so it carries a heavier stroke than inline icons
export function ArrowUpIcon({ className }: IconProps) {
  return (
    <Line className={className} width={2}>
      <path d="M8 12.75v-9.5M4.25 7 8 3.25 11.75 7" />
    </Line>
  )
}

export function ChevronDownIcon({ className }: IconProps) {
  return (
    <Line className={className}>
      <path d="m4.25 6.25 3.75 3.75 3.75-3.75" />
    </Line>
  )
}

export function ChevronLeftIcon({ className }: IconProps) {
  return (
    <Line className={className}>
      <path d="M9.75 4.25 6 8l3.75 3.75" />
    </Line>
  )
}

export function ChevronRightIcon({ className }: IconProps) {
  return (
    <Line className={className}>
      <path d="m6.25 4.25 3.75 3.75-3.75 3.75" />
    </Line>
  )
}

export function CheckIcon({ className }: IconProps) {
  return (
    <Line className={className}>
      <path d="m3.25 8.5 3 3 6.5-7" />
    </Line>
  )
}

// a photo or GIF someone sent
export function ImageIcon({ className }: IconProps) {
  return (
    <Line className={className}>
      <rect x="2.25" y="3.25" width="11.5" height="9.5" rx="2" />
      <circle cx="5.75" cy="6.5" r="0.75" />
      <path d="m2.75 11.5 3-3 2.25 2.25 1.75-1.75 3.5 3.5" />
    </Line>
  )
}

// any other attachment
export function PaperclipIcon({ className }: IconProps) {
  return (
    <Line className={className}>
      <path d="m13.75 7.5-5.4 5.4a3.5 3.5 0 0 1-4.95-4.95l5.4-5.4a2.33 2.33 0 0 1 3.3 3.3l-5.4 5.4a1.17 1.17 0 0 1-1.65-1.65l5-5" />
    </Line>
  )
}

export function BookmarkIcon({ className, filled }: NavIconProps) {
  return (
    <Line className={className}>
      <path
        d="M4.75 2.25h6.5a1 1 0 0 1 1 1v10.5L8 10.75l-4.25 3V3.25a1 1 0 0 1 1-1Z"
        fill={filled ? 'currentColor' : 'none'}
      />
    </Line>
  )
}
