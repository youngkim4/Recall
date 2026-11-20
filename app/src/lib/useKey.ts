import { useCallback, useEffect, useState } from 'react'
import type { KeyState } from '../types'
import { recallApi } from './api'

// Where the API key is, and the ways to save or remove it. The key that
// is typed goes to the server once and is never kept here or shown back.
export function useKey() {
  const [key, setKey] = useState<KeyState | null>(null)
  // where the key is could not be read: an older server, or none
  const [failed, setFailed] = useState(false)
  const [busy, setBusy] = useState(false)
  // why the last save or removal was refused, in the server's one sentence
  const [error, setError] = useState('')

  useEffect(() => {
    let current = true
    recallApi
      .key()
      .then((found) => current && setKey(found))
      .catch(() => current && setFailed(true))
    return () => {
      current = false
    }
  }, [])

  const change = useCallback(async (asked: () => Promise<KeyState>) => {
    setError('')
    setBusy(true)
    try {
      setKey(await asked())
      setFailed(false)
      return true
    } catch (refused) {
      setError(refused instanceof Error ? refused.message : 'The key could not be saved.')
      return false
    } finally {
      setBusy(false)
    }
  }, [])

  // true when the key was saved
  const save = useCallback((typed: string) => change(() => recallApi.saveKey(typed.trim())), [change])
  const remove = useCallback(() => change(() => recallApi.removeKey()), [change])

  return { key, failed, busy, error, save, remove }
}
