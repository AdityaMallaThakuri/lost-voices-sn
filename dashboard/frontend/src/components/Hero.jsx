function scrollToObjectives(e) {
  e.preventDefault()
  document.getElementById('objectives')?.scrollIntoView({ behavior: 'smooth' })
}

function Hero() {
  return (
    <section className="hero" id="top">
      <p className="hero-mission">
        The first documented, evaluated, reproducible NLP and TTS pipeline for
        Sunuwar, an endangered Kiranti language of eastern Nepal.
      </p>

      <div className="interlinear">
        <p className="interlinear-source">येसुमी लोव़ पाबेत।</p>
        <p className="interlinear-output">
          येशूले तिनीहरूलाई जवाफ दिनुभयो, र भन्‍नुभयो,
        </p>
      </div>

      <a href="#objectives" className="btn-ghost" onClick={scrollToObjectives}>
        Try the models
        <svg
          className="chevron"
          width="14"
          height="14"
          viewBox="0 0 14 14"
          aria-hidden="true"
        >
          <path
            d="M2 5 L7 10 L12 5"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </a>
    </section>
  )
}

export default Hero
