import React, { useState, useRef } from 'react';

interface ImageAnalyzerProps {
    onAnalyze: (file: File) => void;
    loading: boolean;
}

const ImageAnalyzer: React.FC<ImageAnalyzerProps> = ({ onAnalyze, loading }) => {
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [dragActive, setDragActive] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileSelect = (file: File) => {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file');
            return;
        }

        if (file.size > 10 * 1024 * 1024) {
            alert('File size must be less than 10MB');
            return;
        }

        setSelectedFile(file);

        // Create preview
        const reader = new FileReader();
        reader.onloadend = () => {
            setPreview(reader.result as string);
        };
        reader.readAsDataURL(file);
    };

    const handleDrag = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === 'dragenter' || e.type === 'dragover') {
            setDragActive(true);
        } else if (e.type === 'dragleave') {
            setDragActive(false);
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    };

    const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    };

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (selectedFile) {
            onAnalyze(selectedFile);
        }
    };

    return (
        <div className="analyzer-container">
            <h2>🖼️ Deepfake Detection</h2>
            <p className="description">
                Upload an image to check if it contains deepfake manipulation
            </p>

            <form onSubmit={handleSubmit}>
                <div
                    className={`drop-zone ${dragActive ? 'active' : ''}`}
                    onDragEnter={handleDrag}
                    onDragLeave={handleDrag}
                    onDragOver={handleDrag}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                >
                    {preview ? (
                        <div className="preview-container">
                            <img src={preview} alt="Preview" className="preview-image" />
                            <p className="preview-label">{selectedFile?.name}</p>
                        </div>
                    ) : (
                        <div className="drop-zone-content">
                            <div className="upload-icon">📁</div>
                            <p>Drag and drop an image here</p>
                            <p className="or-text">or</p>
                            <button type="button" className="btn btn-secondary">
                                Browse Files
                            </button>
                            <p className="file-info">Supports: JPG, PNG, WebP (max 10MB)</p>
                        </div>
                    )}
                </div>

                <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileInputChange}
                    style={{ display: 'none' }}
                />

                <div className="button-group">
                    {selectedFile && (
                        <button
                            type="button"
                            className="btn btn-secondary"
                            onClick={() => {
                                setSelectedFile(null);
                                setPreview(null);
                            }}
                            disabled={loading}
                        >
                            Clear
                        </button>
                    )}
                    <button
                        type="submit"
                        className="btn btn-primary"
                        disabled={loading || !selectedFile}
                    >
                        {loading ? 'Analyzing...' : 'Analyze Image'}
                    </button>
                </div>
            </form>
        </div>
    );
};

export default ImageAnalyzer;
