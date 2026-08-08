const PROBLEMS = [
  'No cleaned, machine-readable Sunuwar text corpus existed',
  'No word embeddings or language model had been published',
  "The native Kõits Brese script isn't fully supported in Unicode; Devanagari is the practical working script",
  'No Sunuwar text-to-speech system existed; EGIDS status: threatened',
]

const OBJECTIVES = [
  'Assembled and cleaned a Devanagari Sunuwar corpus',
  'Trained a tokenizer, word embeddings, and a transformer from scratch',
  'Aligned scripture audio and fine-tuned a TTS model',
  'Evaluated everything honestly and released it openly',
]

function ProblemAndObjectives() {
  return (
    <section className="problem-objectives" id="objectives">
      <div className="problem-col">
        <h2>The problem</h2>
        <ul className="problem-list">
          {PROBLEMS.map((line) => (
            <li key={line} className="problem-item">
              {line}
            </li>
          ))}
        </ul>
      </div>

      <div className="objectives-col">
        <h2>What we built</h2>
        <ol className="objectives-list">
          {OBJECTIVES.map((line, i) => (
            <li key={line} className="objective-item">
              <span className="objective-num">{i + 1}</span>
              <span>{line}</span>
            </li>
          ))}
        </ol>
      </div>
    </section>
  )
}

export default ProblemAndObjectives
