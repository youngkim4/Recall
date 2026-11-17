import { useMemo } from 'react'
import type { PanelAlign } from '../../lib/anchor'
import { qualityOptions } from '../../lib/models'
import { ChoiceMenu } from './ChoiceMenu'

type QualityControlProps = {
  // the models this server can run
  models: string[]
  // the model in use
  value: string
  onChange: (model: string) => void
  // the edge of the button the menu lines up with
  align?: PanelAlign
  // set where "Quality" is already written beside the button
  quietLabel?: boolean
}

// How carefully the AI reads, from Low to High. Each level is a model: the
// menu names it and says in a few words what choosing it means.
export function QualityControl({ models, value, onChange, align, quietLabel }: QualityControlProps) {
  const choices = useMemo(
    () =>
      qualityOptions(models.length ? models : [value]).map((option) => ({
        value: option.value,
        label: option.label,
        meta: option.name,
        note: option.note,
      })),
    [models, value],
  )
  if (choices.length < 2) return null
  return (
    <ChoiceMenu label="Quality" value={value} choices={choices} onChange={onChange} align={align} quietLabel={quietLabel} />
  )
}
