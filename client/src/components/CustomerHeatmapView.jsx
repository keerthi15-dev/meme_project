import React, { useState } from 'react';
import { Upload, TrendingUp, MapPin, AlertCircle } from 'lucide-react';

const CustomerHeatmapView = () => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);

    const handleFileChange = (e) => {
        const file = e.target.files[0];
        if (file) {
            setSelectedFile(file);
            setError(null);
        }
    };

    const analyzeHeatmap = async () => {
        if (!selectedFile) {
            setError('Please select a store image first');
            return;
        }

        setLoading(true);
        setError(null);

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await fetch('http://localhost:8000/analytics/customer-heatmap', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Analysis failed');
            }

            const data = await response.json();
            setResult(data.analysis);
        } catch (err) {
            setError(err.message || 'Failed to analyze image');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="customer-heatmap-view">
            <div className="header">
                <h2>🔥 AI Customer Behavior Heatmaps</h2>
                <p>Analyze foot traffic and optimize product placement</p>
            </div>

            {/* Upload Section */}
            <div className="upload-section">
                <div className="upload-box">
                    <input
                        type="file"
                        accept="image/*"
                        onChange={handleFileChange}
                        id="heatmap-upload"
                        style={{ display: 'none' }}
                    />
                    <label htmlFor="heatmap-upload" className="upload-label">
                        <Upload size={48} />
                        <p>{selectedFile ? selectedFile.name : 'Click to upload store image'}</p>
                        <span>Upload image with customers for traffic analysis</span>
                    </label>
                </div>

                <button
                    onClick={analyzeHeatmap}
                    disabled={!selectedFile || loading}
                    className="analyze-btn"
                >
                    {loading ? 'Analyzing...' : 'Generate Heatmap'}
                </button>
            </div>

            {/* Error Message */}
            {error && (
                <div className="error-message">
                    <AlertCircle size={20} />
                    <span>{error}</span>
                </div>
            )}

            {/* Results */}
            {result && (
                <div className="results-section">
                    {/* Heatmap Visualization */}
                    <div className="heatmap-container">
                        <h3>Visual Heatmap</h3>
                        <img
                            src={`data:image/png;base64,${result.heatmap_image}`}
                            alt="Customer Heatmap"
                            className="heatmap-image"
                        />
                    </div>

                    {/* Metrics */}
                    <div className="metrics-grid">
                        <div className="metric-card">
                            <TrendingUp size={24} />
                            <div>
                                <h4>{result.total_customers}</h4>
                                <p>Customers Detected</p>
                            </div>
                        </div>
                        <div className="metric-card">
                            <MapPin size={24} />
                            <div>
                                <h4>{result.metrics.hot_zones}</h4>
                                <p>High Traffic Zones</p>
                            </div>
                        </div>
                        <div className="metric-card">
                            <AlertCircle size={24} />
                            <div>
                                <h4>{result.metrics.coverage_score}%</h4>
                                <p>Coverage Score</p>
                            </div>
                        </div>
                    </div>

                    {/* Zone Analysis */}
                    <div className="zones-section">
                        <h3>Zone Analysis</h3>
                        <div className="zones-list">
                            {result.zones.map((zone, idx) => (
                                <div key={idx} className="zone-item">
                                    <div className="zone-name">{zone.name}</div>
                                    <div className="zone-activity">
                                        <div
                                            className="activity-bar"
                                            style={{ width: `${zone.activity * 100}%` }}
                                        />
                                        <span>{(zone.activity * 100).toFixed(0)}%</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* AI Recommendations */}
                    <div className="recommendations-section">
                        <h3>💡 AI Recommendations</h3>
                        <div className="recommendations-list">
                            {result.recommendations.map((rec, idx) => (
                                <div key={idx} className="recommendation-item">
                                    {rec}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            <style jsx>{`
        .customer-heatmap-view {
          padding: 20px;
          max-width: 1200px;
          margin: 0 auto;
        }

        .header {
          margin-bottom: 30px;
        }

        .header h2 {
          font-size: 28px;
          font-weight: 700;
          margin-bottom: 8px;
          color: #1a1a1a;
        }

        .header p {
          color: #666;
          font-size: 16px;
        }

        .upload-section {
          margin-bottom: 30px;
        }

        .upload-box {
          border: 2px dashed #ddd;
          border-radius: 12px;
          padding: 40px;
          text-align: center;
          margin-bottom: 20px;
          transition: all 0.3s;
        }

        .upload-box:hover {
          border-color: #4F46E5;
          background: #f9fafb;
        }

        .upload-label {
          cursor: pointer;
          display: flex;
          flex-direction: column;
          align-items: center;
          gap: 12px;
          color: #666;
        }

        .upload-label p {
          font-size: 16px;
          font-weight: 600;
          color: #1a1a1a;
        }

        .upload-label span {
          font-size: 14px;
          color: #999;
        }

        .analyze-btn {
          width: 100%;
          padding: 16px;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          border: none;
          border-radius: 12px;
          font-size: 16px;
          font-weight: 600;
          cursor: pointer;
          transition: transform 0.2s;
        }

        .analyze-btn:hover:not(:disabled) {
          transform: translateY(-2px);
        }

        .analyze-btn:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .error-message {
          display: flex;
          align-items: center;
          gap: 12px;
          padding: 16px;
          background: #fee;
          border-left: 4px solid #f44;
          border-radius: 8px;
          color: #c00;
          margin-bottom: 20px;
        }

        .results-section {
          margin-top: 30px;
        }

        .heatmap-container {
          margin-bottom: 30px;
          background: white;
          padding: 20px;
          border-radius: 12px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .heatmap-container h3 {
          margin-bottom: 16px;
          font-size: 20px;
          font-weight: 600;
        }

        .heatmap-image {
          width: 100%;
          border-radius: 8px;
        }

        .metrics-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 20px;
          margin-bottom: 30px;
        }

        .metric-card {
          background: white;
          padding: 24px;
          border-radius: 12px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
          display: flex;
          align-items: center;
          gap: 16px;
        }

        .metric-card h4 {
          font-size: 32px;
          font-weight: 700;
          color: #4F46E5;
          margin: 0;
        }

        .metric-card p {
          font-size: 14px;
          color: #666;
          margin: 0;
        }

        .zones-section, .recommendations-section {
          background: white;
          padding: 24px;
          border-radius: 12px;
          box-shadow: 0 2px 8px rgba(0,0,0,0.1);
          margin-bottom: 20px;
        }

        .zones-section h3, .recommendations-section h3 {
          margin-bottom: 20px;
          font-size: 20px;
          font-weight: 600;
        }

        .zone-item {
          margin-bottom: 16px;
        }

        .zone-name {
          font-weight: 600;
          margin-bottom: 8px;
          color: #1a1a1a;
        }

        .zone-activity {
          display: flex;
          align-items: center;
          gap: 12px;
        }

        .activity-bar {
          height: 8px;
          background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
          border-radius: 4px;
          transition: width 0.3s;
        }

        .recommendation-item {
          padding: 16px;
          background: #f9fafb;
          border-left: 4px solid #4F46E5;
          border-radius: 8px;
          margin-bottom: 12px;
          font-size: 15px;
          line-height: 1.6;
        }
      `}</style>
        </div>
    );
};

export default CustomerHeatmapView;
