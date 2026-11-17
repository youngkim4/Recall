import { buildView } from '../../lib/archiveBuild'
import { useArchiveBuild } from '../../lib/useArchiveBuild'
import type { Job } from '../../types'
import { ImportProgress } from './ImportProgress'
import { SetupFrame } from './SetupFrame'

type ArchiveWaitProps = {
  // the job that loads the archive, as the server named it
  job: Job
  // loads the data of the app again; it fails with the reason when it cannot
  onReload: () => Promise<unknown>
  // leaves the wait with what went wrong: the app has the door to the setup guide
  onLeave: (failure: string) => void
}

// The archive is being loaded from an earlier export, and nothing can be
// read until it is there. The wait takes the whole window, as the import of
// the setup guide does: what the load is doing, or why it stopped, a way to
// try again, and a way into the app, where the setup guide can be opened.
// The app takes the window back once its data has loaded.
export function ArchiveWait({ job: first, onReload, onLeave }: ArchiveWaitProps) {
  const { job, failure, loading, again } = useArchiveBuild(first, onReload)
  const view = buildView(job)

  return (
    <SetupFrame step={null} phase={failure ? 'failed' : 'running'}>
      {failure ? (
        <>
          <h1>Your archive is not ready</h1>
          <p className="notice bad" role="alert">
            {failure}
          </p>
          <div className="setup-actions">
            <button type="button" className="btn primary lg" disabled={loading} onClick={again}>
              {loading ? <span className="spinner" /> : null}
              Try again
            </button>
            <button type="button" className="btn quiet lg" disabled={loading} onClick={() => onLeave(failure)}>
              Open Recall
            </button>
          </div>
        </>
      ) : (
        <>
          <h1>Getting your archive ready</h1>
          <ImportProgress phases={[view.label]} current={1} said={view.said} />
        </>
      )}
    </SetupFrame>
  )
}
