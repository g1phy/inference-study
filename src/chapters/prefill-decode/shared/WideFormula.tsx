import { Formula, type FormulaProps } from '@/ui'

/** Блочная формула для прозы: на узком экране прокручивается внутри себя, а не обрезается. */
export function WideFormula(props: Omit<FormulaProps, 'block'>) {
  return (
    <div className="-mx-1 overflow-x-auto px-1 py-0.5">
      <Formula block {...props} />
    </div>
  )
}
