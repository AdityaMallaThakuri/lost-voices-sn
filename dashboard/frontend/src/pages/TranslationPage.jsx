import { useState } from 'react'
import { translate } from '../api.js'

const DIRECTIONS = [
  { value: 'suz2npi', label: 'Sunuwar → Nepali' },
  { value: 'npi2suz', label: 'Nepali → Sunuwar' },
]

function TranslationPage() {
  const [direction, setDirection] = useState('suz2npi')
  const [text, setText] = useState('येसुमी लोव़ पाबेत।')
  const [output, setOutput] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  async function handleTranslate(e) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const result = await translate(text, direction)
      setOutput(result.translation)
    } catch (err) {
      setError(err.message)
      setOutput(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="page-section">
      <h1 className="page-title">sunuwarNMT-small</h1>
      <p className="page-subtitle">
        Shared bidirectional Sunuwar ↔ Nepali translation · baseline model.
      </p>

      <div className="pill-toggle">
        {DIRECTIONS.map((d) => (
          <button
            key={d.value}
            type="button"
            className={`pill${direction === d.value ? ' pill-active' : ''}`}
            onClick={() => setDirection(d.value)}
          >
            {d.label}
          </button>
        ))}
      </div>

      <form className="input-row" onSubmit={handleTranslate}>
        <input
          type="text"
          className="text-input font-content"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
        <button type="submit" className="btn-solid" disabled={loading}>
          {loading ? 'Translating…' : 'Translate'}
        </button>
      </form>

      <div className="interlinear">
        <p className="interlinear-source">{text}</p>
        {output && <p className="interlinear-output">{output}</p>}
        {error && <p className="error-text">{error}</p>}
      </div>

      <p className="ceiling-note">
        Baseline model, honestly reported: chrF 21.40 / BLEU 2.23 (Sunuwar →
        Nepali), chrF 24.92 / BLEU 3.97 (Nepali → Sunuwar) on held-out
        validation data — fluent, grammatical output, often imprecise
        content. See <code>results/nmt_methodology.md</code>.
      </p>
    </section>
  )
}

export default TranslationPage
