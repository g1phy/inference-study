import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props {
  children: ReactNode
  /** Что показать вместо упавшего поддерева. */
  fallback?: (error: Error, reset: () => void) => ReactNode
  /** Метка для консоли (например, id главы). */
  label?: string
}

/** Граница ошибок: упавшая глава или фигура не роняет всю страницу. */
export class ErrorBoundary extends Component<Props, { error: Error | null }> {
  state: { error: Error | null } = { error: null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(`[ErrorBoundary${this.props.label ? `: ${this.props.label}` : ''}]`, error, info.componentStack)
  }

  reset = () => this.setState({ error: null })

  render() {
    const { error } = this.state
    if (!error) return this.props.children
    if (this.props.fallback) return this.props.fallback(error, this.reset)
    return (
      <div role="alert" className="rounded-panel border border-line bg-surface-2 px-5 py-4 text-[15px] text-ink-2">
        <p className="font-semibold text-ink">Этот блок не загрузился</p>
        <p className="mt-1">Остальная страница работает. Попробуйте обновить страницу.</p>
        {import.meta.env.DEV && <pre className="mt-3 overflow-x-auto font-mono text-xs whitespace-pre-wrap text-bad">{String(error.stack ?? error)}</pre>}
        <button type="button" onClick={this.reset} className="mt-3 text-sm font-medium text-accent underline underline-offset-2">
          Попробовать ещё раз
        </button>
      </div>
    )
  }
}
