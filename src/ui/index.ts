/**
 * ui-kit сайта. Документация: src/ui/README.md, витрина: /#kit.
 * Импортируйте отсюда: import { Chapter, Figure, Player, usePlayer, fmt } from '@/ui'
 */

// структура главы
export { Chapter, ChapterHeader, type ChapterProps } from './Chapter'
export { Prose, Block, Panel, type BlockWidth } from './Prose'
export { Figure, FigureLive, useFigureLive, useFigureFocusWithin, type FigureProps } from './Figure'
export { Callout, type CalloutKind, type CalloutProps } from './Callout'
export { Deeper, type DeeperProps } from './Deeper'
export { Takeaways } from './Takeaways'
export { ChapterLink, SimulatorLink, linkClass, type ChapterLinkProps } from './ChapterLink'
export { Term, loadGlossary, type TermProps } from './Term'
export { ErrorBoundary } from './ErrorBoundary'

// пошаговые демо
export { Player, usePlayer, type PlayerProps, type PlayerState, type PlayerBindProps, type UsePlayerOptions } from './Player'

// данные и визуализации
export { TokenChip, TokenStrip, TokenTip, tokenSegments, type TokenVariant, type TokenChipProps, type TokenStripProps } from './TokenChip'
export { MatrixView, type MatrixViewProps, type MatrixCell, type MatrixHighlight } from './MatrixView'
export { Heatmap, type HeatmapProps, type HeatmapCell } from './Heatmap'
export { BarChart, type BarChartProps, type BarDatum } from './BarChart'
export { LineChart, type LineChartProps, type LineSeries, type LineAnnotation } from './LineChart'
export { Formula, renderTex, loadKatex, TEX_MACROS, type FormulaProps } from './Formula'
export { CodeBlock, type CodeBlockProps } from './CodeBlock'
export { ScrollX, ScrollHint, type ScrollXProps } from './ScrollX'
export { highlight, type CodeLang, type CodeToken } from './highlight'
export { Stat, type StatProps } from './Stat'

// контролы
export { Button, ButtonLink, buttonClass, type ButtonProps, type ButtonVariant, type ButtonSize } from './Button'
export { Slider, type SliderProps } from './Slider'
export { Toggle, type ToggleProps } from './Toggle'
export { Segmented, type SegmentedProps, type SegmentedOption } from './Segmented'
export { Select, type SelectProps, type SelectOption } from './Select'
export { Tabs, type TabsProps, type TabItem } from './Tabs'
export { TextArea, type TextAreaProps } from './TextArea'
export { FieldLabel, FieldHint } from './Field'
export { Badge, StatusIcon, type BadgeProps, type BadgeTone, type StatusKind } from './Badge'
export { Tooltip, Popover, type TooltipProps, type PopoverProps } from './Tooltip'
export { Kbd } from './Kbd'
export { Skeleton, TokenStripSkeleton } from './Skeleton'

// утилиты
export { fmt, TOKEN_FORMS, type BytesOptions } from './fmt'
export {
  cssVar,
  tint,
  mix,
  resolveColor,
  seqColor,
  seqInk,
  divColor,
  divInk,
  legendGradient,
  ROLE_LABELS,
  DATA_ROLES,
  type ColorRole,
  type DataRole,
  type StatusRole,
} from './color'
export {
  useInView,
  useMediaQuery,
  useElementSize,
  useScrollEdges,
  useFontsReady,
  useLatest,
  useReducedMotion,
  useTheme,
  measureText,
  truncateText,
  MEDIA,
  type InViewOptions,
  type Size,
  type ScrollEdges,
} from './hooks'
export { FloatingLayer, TipCard, TipRow, useTip, computePosition, pointRect, type Side, type Align, type RectLike, type UseTipOptions } from './floating'
export { renderIcon, type IconLike } from './icon'
export { cn } from './cn'
export { fontCss } from './chart-utils'
export { Z } from './z'
