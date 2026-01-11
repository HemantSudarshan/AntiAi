import React from 'react';

interface Result {
    type: 'news' | 'image';
    prediction: string;
    confidence: number;
    explanation: any;
    metadata?: any;
    filename?: string;
    text_preview?: string;
}

interface ResultsDisplayProps {
    result: Result;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ result }) => {
    const getPredictionClass = () => {
        if (result.type === 'news') {
            return result.prediction === 'FAKE' ? 'prediction-fake' : 'prediction-real';
        } else {
            return result.prediction === 'DEEPFAKE' ? 'prediction-fake' : 'prediction-real';
        }
    };

    const getConfidenceColor = () => {
        if (result.confidence >= 0.8) return '#22c55e';
        if (result.confidence >= 0.6) return '#eab308';
        return '#ef4444';
    };

    return (
        <div className="results-container">
            <h3>Analysis Results</h3>

            <div className={`prediction-card ${getPredictionClass()}`}>
                <div className="prediction-label">Prediction</div>
                <div className="prediction-value">{result.prediction}</div>
                <div className="confidence-container">
                    <div className="confidence-label">Confidence</div>
                    <div className="confidence-bar-wrapper">
                        <div
                            className="confidence-bar"
                            style={{
                                width: `${result.confidence * 100}%`,
                                backgroundColor: getConfidenceColor(),
                            }}
                        />
                    </div>
                    <div className="confidence-value">{(result.confidence * 100).toFixed(1)}%</div>
                </div>
            </div>

            {result.type === 'news' && result.explanation && (
                <div className="explanation-section">
                    <h4>🔍 Explanation</h4>

                    {result.explanation.reasoning && (
                        <div className="reasoning-box">
                            <p><strong>Analysis:</strong> {result.explanation.reasoning}</p>
                        </div>
                    )}

                    {result.explanation.top_features && result.explanation.top_features.length > 0 && (
                        <div className="features-section">
                            <h5>Key Indicators Found:</h5>
                            <ul className="features-list">
                                {result.explanation.top_features.map((feature: any, idx: number) => (
                                    <li key={idx} className="feature-item">
                                        <span className="feature-keyword">"{feature.keyword}"</span>
                                        <span className="feature-impact">{feature.impact}</span>
                                        {feature.reason && <span className="feature-reason">({feature.reason})</span>}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}

                    {result.explanation.patterns && result.explanation.patterns.length > 0 && (
                        <div className="patterns-section">
                            <h5>Suspicious Patterns:</h5>
                            <ul className="patterns-list">
                                {result.explanation.patterns.map((pattern: string, idx: number) => (
                                    <li key={idx}>{pattern}</li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            )}

            {result.type === 'image' && result.explanation && (
                <div className="explanation-section">
                    <h4>🔍 Explanation</h4>

                    {result.explanation.heatmap && (
                        <div className="heatmap-section">
                            <h5>Heatmap Analysis:</h5>
                            <img
                                src={result.explanation.heatmap}
                                alt="Heatmap"
                                className="heatmap-image"
                            />
                            <p className="heatmap-caption">
                                Red areas indicate regions that influenced the deepfake prediction
                            </p>
                        </div>
                    )}

                    {result.explanation.suspicious_regions && result.explanation.suspicious_regions.length > 0 && (
                        <div className="regions-section">
                            <h5>Areas of Concern:</h5>
                            <ul className="regions-list">
                                {result.explanation.suspicious_regions.map((region: string, idx: number) => (
                                    <li key={idx}>{region}</li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            )}

            {result.metadata && (
                <div className="metadata-section">
                    <div className="metadata-item">
                        <span className="metadata-label">Processing Time:</span>
                        <span className="metadata-value">{result.metadata.processing_time_ms}ms</span>
                    </div>
                    <div className="metadata-item">
                        <span className="metadata-label">Model Version:</span>
                        <span className="metadata-value">{result.metadata.model_version}</span>
                    </div>
                </div>
            )}
        </div>
    );
};

export default ResultsDisplay;
