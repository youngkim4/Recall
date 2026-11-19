// The loader of Insights for useLoad: one function for the life of the app,
// so kept.ts can keep its answers (a count asked for again shows at once).

import { recallApi } from './api'
import { readStatsKey } from './insights'

export const loadStats = (key: string) => {
  const read = readStatsKey(key)
  if (!read) return Promise.reject(new Error('Nothing to count.'))
  return recallApi.stats(read.messagesPath, read.ask)
}
