import { Component, type ErrorInfo, type ReactNode } from 'react'

type ErrorBoundaryProps = {
  children: ReactNode
}

type ErrorBoundaryState = {
  error: Error | null
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { error: null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Recall UI crashed', error, info.componentStack)
  }

  render() {
    if (this.state.error) {
      return (
        <main className="app-fallback">
          <h1>Recall hit a UI error</h1>
          <p>{this.state.error.message}</p>
        </main>
      )
    }

    return this.props.children
  }
}
