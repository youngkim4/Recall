import { CheckIcon } from '../Icons'

type ImportProgressProps = {
  // the things done in order, each in a few words
  phases: string[]
  // which of them is being done now, from 1
  current: number
  // the line the work last wrote; '' for none
  said: string
}

// How far a long piece of work has got: the things it does in order, with a
// mark for the ones done and for the one being done, and under them the line
// it last wrote, so a long step is seen to be moving.
export function ImportProgress({ phases, current, said }: ImportProgressProps) {
  return (
    <>
      <ol className="setup-phases" role="list" aria-label="Progress">
        {phases.map((phase, index) => {
          const place = index + 1
          const state = place < current ? 'done' : place === current ? 'now' : undefined
          return (
            // by its place: the words of a step can change while it is done
            <li key={place} data-state={state} aria-current={state === 'now' ? 'step' : undefined}>
              <span className="setup-phase-mark" aria-hidden="true">
                {state === 'done' ? <CheckIcon /> : state === 'now' ? <span className="spinner" /> : place}
              </span>
              {phase}
            </li>
          )
        })}
      </ol>
      <p className="setup-note setup-said" role="status">
        {said}
      </p>
    </>
  )
}
