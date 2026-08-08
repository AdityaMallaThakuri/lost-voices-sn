import { useState } from 'react'
import { predictMask, getPerplexity } from '../api.js'

// Same known-good examples as dashboard/model_service.py's DEMO_SENTENCES
const DEMO_SENTENCES = [
  { label: 'Proper name (subject)', masked: 'मिनु [MASK] आ खिं लशा बाक्‍तक।' },
  {
    label: 'Verb in complex clause',
    masked: 'दोपा ग्रानीम देंशा हना, येसु ख्रीस्‍त कली थमा [MASK] मप्रोंइथु ग्रानीम।',
  },
  { label: 'Verb before auxiliary', masked: 'तन्‍न गो इन कली लोव़ का [MASK] माल्‍नुङ।' },
  { label: 'Negated verb in question', masked: 'आफोमी शेंचा [MASK] तौ बाक्‍बा ङा?' },
]

function MlmPage() {
  const [sentence, setSentence] = useState('')
  const [predictions, setPredictions] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  const [perplexity, setPerplexity] = useState(null)
  const [pplLoading, setPplLoading] = useState(false)

  async function handlePredict(e) {
    e.preventDefault()
    setError(null)
    setLoading(true)
    try {
      const result = await predictMask(sentence)
      setPredictions(result.predictions)
    } catch (err) {
      setError(err.message)
      setPredictions(null)
    } finally {
      setLoading(false)
    }
  }

  async function handlePerplexity() {
    setPplLoading(true)
    try {
      const result = await getPerplexity()
      setPerplexity(result)
    } finally {
      setPplLoading(false)
    }
  }

  return (
    <section className="page-section">
      <h1 className="page-title">SunuwarBERT-small</h1>
      <p className="page-subtitle">
        Masked language model · type a Sunuwar sentence with{' '}
        <code>[MASK]</code> where you want the model to guess.
      </p>

      <div className="example-chips">
        {DEMO_SENTENCES.map((ex) => (
          <button
            key={ex.label}
            type="button"
            className="chip"
            onClick={() => setSentence(ex.masked)}
          >
            {ex.label}
          </button>
        ))}
      </div>

      <form className="input-row" onSubmit={handlePredict}>
        <input
          type="text"
          className="text-input font-content"
          placeholder="e.g. मिनु [MASK] आ खिं लशा बाक्‍तक।"
          value={sentence}
          onChange={(e) => setSentence(e.target.value)}
        />
        <button type="submit" className="btn-solid" disabled={loading}>
          {loading ? 'Predicting…' : 'Predict'}
        </button>
      </form>

      {error && <p className="error-text">{error}</p>}

      {predictions && (
        <div className="bars">
          {predictions.map((p) => (
            <div className="bar-row" key={p.piece}>
              <span className="bar-label font-content">{p.piece_clean}</span>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${p.prob}%` }} />
              </div>
              <span className="bar-pct font-mono">{p.prob.toFixed(1)}%</span>
            </div>
          ))}
        </div>
      )}

      <div className="metric-card">
        <button
          type="button"
          className="btn-ghost-small"
          onClick={handlePerplexity}
          disabled={pplLoading}
        >
          {pplLoading ? 'Computing…' : 'Compute test-set perplexity'}
        </button>
        {perplexity && (
          <div className="metric-row">
            <div className="metric">
              <span className="metric-label">Test perplexity</span>
              <span className="metric-value font-mono">
                {perplexity.perplexity}
              </span>
            </div>
            <div className="metric">
              <span className="metric-label">Random baseline</span>
              <span className="metric-value font-mono">
                {perplexity.random_baseline}
              </span>
            </div>
            <div className="metric">
              <span className="metric-label">Improvement</span>
              <span className="metric-value font-mono">
                {perplexity.improvement_factor}×
              </span>
            </div>
          </div>
        )}
      </div>
    </section>
  )
}

export default MlmPage
