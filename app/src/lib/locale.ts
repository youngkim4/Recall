// What the reader's locale says about the calendar.

type WeekLocale = Intl.Locale & { weekInfo?: { firstDay: number }; getWeekInfo?: () => { firstDay: number } }

function readWeekStart() {
  try {
    const locale = new Intl.Locale(navigator.language) as WeekLocale
    const info = locale.getWeekInfo?.() ?? locale.weekInfo
    // the locale counts Monday as 1 and Sunday as 7
    return info ? info.firstDay % 7 : 0
  } catch {
    return 0
  }
}

// the day the reader's week starts on: 0 is Sunday
export const WEEK_START = readWeekStart()

// the seven days in the reader's order, as the numbers Date gives them (0 is Sunday)
export const WEEK_ORDER = Array.from({ length: 7 }, (_, index) => (WEEK_START + index) % 7)

// 7 January 2024 is a Sunday
const dateOfWeekday = (weekday: number) => new Date(2024, 0, 7 + weekday)

// the name of a weekday, by the number Date gives it (0 is Sunday)
export function weekdayName(weekday: number, width: 'narrow' | 'short' | 'long') {
  return new Intl.DateTimeFormat(undefined, { weekday: width }).format(dateOfWeekday(weekday))
}

// "9 PM", or "21" where the day runs to 24
export function hourName(hour: number) {
  return new Intl.DateTimeFormat(undefined, { hour: 'numeric' }).format(new Date(2024, 0, 7, hour))
}
