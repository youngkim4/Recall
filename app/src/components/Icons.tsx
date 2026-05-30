type IconProps = {
  className?: string
}

export function RecallMark({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 64 64" role="img" aria-label="Recall">
      <rect x="4" y="4" width="56" height="56" rx="14" fill="currentColor" opacity="0.08" />
      <path
        d="M18 16h18.4c5.4 0 9.6 1.4 12.6 4.2 3 2.7 4.5 6.3 4.5 10.8 0 3.8-1.1 7-3.4 9.5-2 2.3-4.8 3.8-8.2 4.6L53 54H42.6L33 46h-5.4v8H18V16Zm9.6 8.4V38h8.2c2.7 0 4.8-.6 6.3-1.8 1.4-1.2 2.2-2.9 2.2-5.1s-.7-3.9-2.2-5c-1.5-1.1-3.6-1.7-6.3-1.7h-8.2Z"
        fill="currentColor"
      />
      <path d="M34.2 44.4h8.7L53 54H43.2l-9-9.6Z" fill="var(--accent)" />
    </svg>
  )
}

export function LayersIcon({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
      <path d="m12 3 8 4.2-8 4.2-8-4.2L12 3Z" />
      <path d="m4 12 8 4.2 8-4.2" />
      <path d="m4 16.8 8 4.2 8-4.2" />
    </svg>
  )
}

export function HomeIcon({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 10.8 12 4l8 6.8V21h-5.5v-6h-5v6H4V10.8Z" />
    </svg>
  )
}

export function CompassIcon({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="8.5" />
      <path d="m15.5 8.5-2 5-5 2 2-5 5-2Z" />
    </svg>
  )
}

export function MessageIcon({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 5.5h16v10.7H8.7L4 20V5.5Z" />
      <path d="M8 9h8" />
      <path d="M8 12.5h5" />
    </svg>
  )
}

export function ClockIcon({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="12" cy="12" r="8.5" />
      <path d="M12 7.5V12l3.2 2" />
    </svg>
  )
}

export function CalendarIcon({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M5 5.5h14v15H5v-15Z" />
      <path d="M8 3.5v4" />
      <path d="M16 3.5v4" />
      <path d="M5 9h14" />
      <path d="M8.5 12.5h2" />
      <path d="M13.5 12.5h2" />
      <path d="M8.5 16h2" />
      <path d="M13.5 16h2" />
    </svg>
  )
}

export function FileIcon({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M7 3h7l4 4v14H7V3Z" />
      <path d="M14 3v5h5" />
      <path d="M9.5 13h5" />
      <path d="M9.5 16h5" />
    </svg>
  )
}

export function SettingsIcon({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 8a4 4 0 1 1 0 8 4 4 0 0 1 0-8Z" />
      <path d="m4 13.5-.6-3 2.4-.9c.2-.6.4-1 .7-1.5L5.4 5.8l2.5-1.8 2 1.6c.5-.1 1-.2 1.6-.2l1.6-2h3l.7 2.5c.5.2 1 .5 1.4.8l2.4-1 1.8 2.5-1.6 2c.1.5.2 1 .2 1.6l2 1.6-.9 2.9-2.5.2c-.3.5-.6.9-1 1.3l.4 2.6-2.5 1.8-2-1.6c-.5.1-1 .2-1.6.2l-1.6 2h-3l-.7-2.5c-.5-.2-1-.5-1.4-.8l-2.4 1L2 15.8l1.6-2.3Z" />
    </svg>
  )
}

export function SearchIcon({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="11" cy="11" r="7" />
      <path d="m16.5 16.5 4 4" />
    </svg>
  )
}

export function SparkIcon({ className }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3.5 14.3 9l5.7 2.2-5.7 2.3L12 19l-2.3-5.5L4 11.2 9.7 9 12 3.5Z" />
    </svg>
  )
}
