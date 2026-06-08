type IconProps = {
  className?: string
}

// Hand-drawn line icon set — single weight, rounded, softly organic
// (stroke styling comes from .nav-icon / .button-icon in App.css)

export function RecallMark({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 64 64" role="img" aria-label="Recall">
      <rect x="4" y="4" width="56" height="56" rx="16" fill="currentColor" opacity="0.08" />
      <path
        d="M18 16h18.4c5.4 0 9.6 1.4 12.6 4.2 3 2.7 4.5 6.3 4.5 10.8 0 3.8-1.1 7-3.4 9.5-2 2.3-4.8 3.8-8.2 4.6L53 54H42.6L33 46h-5.4v8H18V16Zm9.6 8.4V38h8.2c2.7 0 4.8-.6 6.3-1.8 1.4-1.2 2.2-2.9 2.2-5.1s-.7-3.9-2.2-5c-1.5-1.1-3.6-1.7-6.3-1.7h-8.2Z"
        fill="currentColor"
      />
      <path d="M34.2 44.4h8.7L53 54H43.2l-9-9.6Z" fill="var(--accent)" />
    </svg>
  )
}

export function MessageIcon({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M5 11.5C5 7.9 8.1 5 12 5s7 2.9 7 6.5-3.1 6.5-7 6.5c-1 0-1.9-.2-2.8-.5-1 .7-2.3 1.2-3.2 1.3.5-.8.8-1.7.7-2.4C5.7 14.2 5 12.9 5 11.5Z" />
      <path d="M9 10.8h6" />
      <path d="M9 13.2h3.6" />
    </svg>
  )
}

export function LayersIcon({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 4.6 4.6 8.2 12 11.8l7.4-3.6L12 4.6Z" />
      <path d="M4.8 11.9 12 15.5l7.2-3.6" />
      <path d="M4.8 15.4 12 19l7.2-3.6" />
    </svg>
  )
}

export function HomeIcon({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4.6 11 12 4.8 19.4 11" />
      <path d="M6.6 9.6V19h10.8V9.6" />
      <path d="M10 19v-4.5c0-.6.4-1 1-1h2c.6 0 1 .4 1 1V19" />
    </svg>
  )
}

export function CompassIcon({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 4.6c4.1 0 7.4 3.3 7.4 7.4S16.1 19.4 12 19.4 4.6 16.1 4.6 12 7.9 4.6 12 4.6Z" />
      <path d="M15 9 13.2 13.2 9 15l1.8-4.2L15 9Z" />
    </svg>
  )
}

export function ClockIcon({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 4.6c4.1 0 7.4 3.3 7.4 7.4S16.1 19.4 12 19.4 4.6 16.1 4.6 12 7.9 4.6 12 4.6Z" />
      <path d="M12 7.9V12l2.9 1.7" />
    </svg>
  )
}

export function CalendarIcon({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M5.4 6.6c0-.4.3-.6.6-.6h12c.3 0 .6.2.6.6v11.8c0 .4-.3.6-.6.6H6c-.3 0-.6-.2-.6-.6V6.6Z" />
      <path d="M8.2 3.8v3.4" />
      <path d="M15.8 3.8v3.4" />
      <path d="M5.4 9.8h13.2" />
    </svg>
  )
}

export function FileIcon({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M6.8 4.8c0-.2.2-.4.4-.4h6.1L18 8.7v10c0 .2-.2.4-.4.4H7.2c-.2 0-.4-.2-.4-.4V4.8Z" />
      <path d="M13 4.6v3.7c0 .3.2.5.5.5h3.5" />
      <path d="M9.2 13h5.6" />
      <path d="M9.2 15.6h4" />
    </svg>
  )
}

export function SettingsIcon({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 7.5h7" />
      <path d="M15 7.5h5" />
      <path d="M13 5.5a2 2 0 1 1 0 4 2 2 0 0 1 0-4Z" />
      <path d="M4 12h9" />
      <path d="M17 12h3" />
      <path d="M15 10a2 2 0 1 1 0 4 2 2 0 0 1 0-4Z" />
      <path d="M4 16.5h4" />
      <path d="M12 16.5h8" />
      <path d="M10 14.5a2 2 0 1 1 0 4 2 2 0 0 1 0-4Z" />
    </svg>
  )
}

export function SearchIcon({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M11 4.6a6.4 6.4 0 1 1 0 12.8 6.4 6.4 0 0 1 0-12.8Z" />
      <path d="m15.8 15.8 3.8 3.8" />
    </svg>
  )
}

export function SparkIcon({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 4c.5 4.3 1.7 5.5 6 6-4.3.5-5.5 1.7-6 6-.5-4.3-1.7-5.5-6-6 4.3-.5 5.5-1.7 6-6Z" />
    </svg>
  )
}
