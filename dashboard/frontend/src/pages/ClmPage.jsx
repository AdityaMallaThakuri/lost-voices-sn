import { useState } from 'react'
import { generateClm } from '../api.js'

function ClmPage() {
  const [prompt, setPrompt] = useState('येसु ख्रीस्‍त आ')
  const [numNewTokens, setNumNewTokens] = useState(15)
  const [continuation, setContinuation] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  async function handleGenerate(e) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const result = await generateClm(prompt, Number(numNewTokens))
      setContinuation(result.continuation)
    } catch (err) {
      setError(err.message)
      setContinuation(null)
    } finally {
      setLoading(false)
    }
  }

  // Strip the model's <s>/</s> markers and split off the prompt so the
  // newly-generated portion can be styled distinctly from what was typed.
  function renderContinuation() {
    if (!continuation) return null
    const cleaned = continuation.replace(/<\/?s>/g, '').trim()
    const generatedPart = cleaned.startsWith(prompt.trim())
      ? cleaned.slice(prompt.trim().length)
      : cleaned

    return (
      <p className="clm-output font-content">
        <span>{prompt.trim()}</span>
        <span className="clm-generated">{generatedPart}</span>
      </p>
    )
  }

  return (
    <section className="page-section">
      <h1 className="page-title">SunuwarCLM-small</h1>
      <p className="page-subtitle">
        Causal (GPT-style) language model · greedy text continuation.
      </p>

      <form className="clm-form" onSubmit={handleGenerate}>
        <label className="field-label" htmlFor="clm-prompt">
          Prompt
        </label>
        <input
          id="clm-prompt"
          type="text"
          className="text-input font-content"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />

        <label className="field-label" htmlFor="clm-tokens">
          New tokens
        </label>
        <input
          id="clm-tokens"
          type="number"
          min="1"
          max="60"
          className="number-input font-mono"
          value={numNewTokens}
          onChange={(e) => setNumNewTokens(e.target.value)}
        />

        <button type="submit" className="btn-solid" disabled={loading}>
          {loading ? 'Generating…' : 'Generate'}
        </button>
      </form>

      {error && <p className="error-text">{error}</p>}

      {continuation && renderContinuation()}
    </section>
  )
}

export default ClmPage
