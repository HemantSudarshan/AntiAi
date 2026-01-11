import React, { useState } from 'react';
import './App.css';
import NewsAnalyzer from './components/NewsAnalyzer';
import ImageAnalyzer from './components/ImageAnalyzer';
import ResultsDisplay from './components/ResultsDisplay';

type TabType = 'news' | 'image';

interface Result {
    type: TabType;
    prediction: string;
    confidence: number;
    explanation: any;
    metadata?: any;
}

function App() {
    const [activeTab, setActiveTab] = useState<TabType>('news');
    const [results, setResults] = useState<Result | null>(null);
    const [loading, setLoading] = useState(false);

    const handleNewsAnalysis = async (text: string) => {
        setLoading(true);
        setResults(null);

        try {
            const formData = new FormData();
            formData.append('text', text);

            const response = await fetch('http://localhost:8000/api/v1/analyze-news', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            setResults({ type: 'news', ...data });
        } catch (error) {
            console.error('Error:', error);
            alert('Error analyzing news. Make sure the API is running on http://localhost:8000');
        } finally {
            setLoading(false);
        }
    };

    const handleImageAnalysis = async (file: File) => {
        setLoading(true);
        setResults(null);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:8000/api/v1/analyze-image', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            setResults({ type: 'image', ...data });
        } catch (error) {
            console.error('Error:', error);
            alert('Error analyzing image. Make sure the API is running on http://localhost:8000');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="App">
            <header className="header">
                <h1>🔍 TruthTracker</h1>
                <p>Production-Grade Misinformation Detection</p>
            </header>

            <div className="tabs">
                <button
                    className={activeTab === 'news' ? 'tab active' : 'tab'}
                    onClick={() => setActiveTab('news')}
                >
                    📰 Analyze News
                </button>
                <button
                    className={activeTab === 'image' ? 'tab active' : 'tab'}
                    onClick={() => setActiveTab('image')}
                >
                    🖼️ Detect Deepfakes
                </button>
            </div>

            <main className="content">
                {activeTab === 'news' ? (
                    <NewsAnalyzer onAnalyze={handleNewsAnalysis} loading={loading} />
                ) : (
                    <ImageAnalyzer onAnalyze={handleImageAnalysis} loading={loading} />
                )}

                {results && <ResultsDisplay result={results} />}
            </main>

            <footer className="footer">
                <p>TruthTracker v2.0 | Powered by FastAPI + React</p>
            </footer>
        </div>
    );
}

export default App;
