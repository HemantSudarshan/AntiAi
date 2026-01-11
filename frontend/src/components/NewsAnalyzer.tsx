import React, { useState } from 'react';

interface NewsAnalyzerProps {
    onAnalyze: (text: string) => void;
    loading: boolean;
}

const NewsAnalyzer: React.FC<NewsAnalyzerProps> = ({ onAnalyze, loading }) => {
    const [text, setText] = useState('');

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (text.trim().length < 20) {
            alert('Please enter at least 20 characters');
            return;
        }
        onAnalyze(text);
    };

    const handleSampleClick = () => {
        const sample = "Breaking news! You won't believe what happened next! This shocking discovery will change everything you know. Experts are speechless!";
        setText(sample);
    };

    return (
        <div className="analyzer-container">
            <h2>📰 Fake News Detection</h2>
            <p className="description">
                Paste a news article or headline to check its authenticity
            </p>

            <form onSubmit={handleSubmit}>
                <textarea
                    className="text-input"
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Enter news article text here... (minimum 20 characters)"
                    rows={8}
                    disabled={loading}
                />

                <div className="button-group">
                    <button
                        type="button"
                        className="btn btn-secondary"
                        onClick={handleSampleClick}
                        disabled={loading}
                    >
                        Try Sample Text
                    </button>
                    <button
                        type="submit"
                        className="btn btn-primary"
                        disabled={loading || text.trim().length < 20}
                    >
                        {loading ? 'Analyzing...' : 'Analyze News'}
                    </button>
                </div>

                <div className="char-count">
                    {text.length} characters {text.length < 20 && '(minimum 20 required)'}
                </div>
            </form>
        </div>
    );
};

export default NewsAnalyzer;
