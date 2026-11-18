// What the Overview asks the server for, beyond the archive and its
// conversations: the moments of the day and the shape of the archive. The
// app asks for both as soon as it knows the archive (preloadOverview), so the
// Overview has them when it opens.

import { recallApi } from './api'
import { preload } from './kept'

export const loadMoments = (messagesPath: string) =>
  recallApi.memories({ messagesPath }).then((response) => response.memories)

export const loadPatterns = (messagesPath: string) =>
  recallApi.overview({ messagesPath }).then((response) => response.overview)

export function preloadOverview(messagesPath: string) {
  preload(loadMoments, messagesPath)
  preload(loadPatterns, messagesPath)
}
