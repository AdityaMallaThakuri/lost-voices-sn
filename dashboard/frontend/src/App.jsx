import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './App.css'
import Navbar from './components/Navbar.jsx'
import Footer from './components/Footer.jsx'
import HomePage from './pages/HomePage.jsx'
import MlmPage from './pages/MlmPage.jsx'
import ClmPage from './pages/ClmPage.jsx'
import TranslationPage from './pages/TranslationPage.jsx'
import TtsPage from './pages/TtsPage.jsx'

function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/mlm" element={<MlmPage />} />
        <Route path="/clm" element={<ClmPage />} />
        <Route path="/translation" element={<TranslationPage />} />
        <Route path="/tts" element={<TtsPage />} />
      </Routes>
      <Footer />
    </BrowserRouter>
  )
}

export default App
