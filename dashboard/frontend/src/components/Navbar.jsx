import { Link, NavLink } from 'react-router-dom'

const NAV_LINKS = [
  { label: 'MLM', to: '/mlm' },
  { label: 'CLM', to: '/clm' },
  { label: 'TRANSLATION', to: '/translation' },
  { label: 'TTS', to: '/tts' },
]

function Navbar() {
  return (
    <header className="navbar">
      <div className="navbar-inner">
        <Link className="brand" to="/">
          LOST VOICES
        </Link>
        <nav className="nav-links">
          {NAV_LINKS.map((link) => (
            <NavLink
              key={link.label}
              to={link.to}
              className={({ isActive }) => (isActive ? 'active' : undefined)}
            >
              {link.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  )
}

export default Navbar
