// frontend/src/App.jsx
import { useState } from 'react';
import Header from './components/Header';
import Footer from './components/Footer';
import InputForm from './components/InputForm';
import Results from './components/Results';

function App() {
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);

  return (
    <div className="app">
      <Header />
      
      <main className="main-content">
        <div className="grid">
          <div className="card">
            <InputForm 
              setResults={setResults}
              loading={loading}
              setLoading={setLoading}
            />
          </div>

          <div className="card">
            <Results results={results} loading={loading} />
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
}

export default App;
