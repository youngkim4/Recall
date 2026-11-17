// Not a test. It is what tests put where the browser's window and its
// localStorage would be, and it carries the test suffix so the app build
// leaves it out.

export type FakeStorage = Storage & {
  // how many times something was stored or removed
  changes: () => number
  // everything it holds
  dump: () => Record<string, string>
}

type FakeOptions = {
  // the most characters one value may have; a longer one is refused the way a full store refuses it
  roomFor?: number
  // every call fails, as in a store that has been closed
  broken?: boolean
}

export function fakeStorage(seed: Record<string, string> = {}, options: FakeOptions = {}): FakeStorage {
  const { roomFor = Infinity, broken = false } = options
  const items = new Map(Object.entries(seed))
  const counter = { changes: 0 }
  const guard = () => {
    if (broken) throw new DOMException('The store is closed', 'InvalidStateError')
  }
  return {
    get length() {
      guard()
      return items.size
    },
    key(index: number) {
      guard()
      return [...items.keys()][index] ?? null
    },
    getItem(key: string) {
      guard()
      return items.get(key) ?? null
    },
    setItem(key: string, value: string) {
      guard()
      if (String(value).length > roomFor) throw new DOMException('The store is full', 'QuotaExceededError')
      counter.changes += 1
      items.set(key, String(value))
    },
    removeItem(key: string) {
      guard()
      counter.changes += 1
      items.delete(key)
    },
    clear() {
      guard()
      counter.changes += 1
      items.clear()
    },
    changes: () => counter.changes,
    dump: () => Object.fromEntries(items),
  }
}

function setWindow(value: unknown) {
  Object.defineProperty(globalThis, 'window', { value, configurable: true, writable: true })
}

// a page whose store is the one given
export function pageWithStorage<Store extends Storage>(storage: Store) {
  setWindow({ localStorage: storage })
  return storage
}

// a page that is not allowed to keep anything: asking for the store is itself refused
export function pageWithNoStorage() {
  setWindow({
    get localStorage(): Storage {
      throw new DOMException('The page may not keep anything', 'SecurityError')
    },
  })
}

// no page at all, as when a module is loaded outside a browser
export function noPage() {
  Reflect.deleteProperty(globalThis, 'window')
}
