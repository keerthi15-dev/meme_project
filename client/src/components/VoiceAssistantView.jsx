import React, { useState, useRef } from 'react';
import { Mic, MicOff, Send, Volume2, Loader, Sparkles, Zap } from 'lucide-react';

const VoiceAssistantView = () => {
  const [recording, setRecording] = useState(false);
  const [audioFile, setAudioFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setAudioFile(file);
      setError(null);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const chunks = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunks.push(e.data);
        }
      };

      recorder.onstop = () => {
        const audioBlob = new Blob(chunks, { type: 'audio/webm' });
        const audioFile = new File([audioBlob], 'recording.webm', { type: 'audio/webm' });
        setAudioFile(audioFile);

        stream.getTracks().forEach(track => track.stop());
      };

      recorder.start();
      setMediaRecorder(recorder);
      setRecording(true);
      setError(null);
    } catch (err) {
      setError('Microphone access denied. Please allow microphone access.');
      console.error('Recording error:', err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
      setRecording(false);
      setMediaRecorder(null);
    }
  };

  const processVoiceQuery = async () => {
    if (!audioFile) {
      setError('Please record or upload an audio file first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', audioFile);

      const response = await fetch('http://localhost:8000/inventory/voice-query', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Voice processing failed');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to process voice query');
    } finally {
      setLoading(false);
    }
  };

  const getIntentIcon = (intent) => {
    const icons = {
      stock_check: '📦',
      product_info: 'ℹ️',
      location: '📍',
      reorder: '🔄'
    };
    return icons[intent] || '💬';
  };

  return (
    <div className="voice-assistant-view">
      {/* Animated Background */}
      <div className="animated-bg">
        <div className="gradient-orb orb-1"></div>
        <div className="gradient-orb orb-2"></div>
        <div className="gradient-orb orb-3"></div>
      </div>

      <div className="content-wrapper">
        {/* Hero Header */}
        <div className="hero-header">
          <div className="icon-badge">
            <Sparkles size={32} />
          </div>
          <h1>Voice Inventory Assistant</h1>
          <p className="subtitle">Speak naturally in 100+ languages • Powered by OpenAI Whisper</p>
          <div className="language-pills">
            <span className="pill">🇮🇳 Hindi</span>
            <span className="pill">🇬🇧 English</span>
            <span className="pill">🇮🇳 Tamil</span>
            <span className="pill">+97 more</span>
          </div>
        </div>

        {/* Main Recording Card */}
        <div className="recording-card">
          <div className="mic-visualizer">
            {recording && (
              <div className="pulse-rings">
                <div className="pulse-ring ring-1"></div>
                <div className="pulse-ring ring-2"></div>
                <div className="pulse-ring ring-3"></div>
              </div>
            )}

            <button
              onClick={recording ? stopRecording : startRecording}
              disabled={loading}
              className={`mic-button ${recording ? 'recording' : ''}`}
            >
              {recording ? <MicOff size={48} /> : <Mic size={48} />}
            </button>
          </div>

          <div className="recording-status">
            {recording ? (
              <>
                <div className="recording-indicator">
                  <span className="red-dot"></span>
                  <span>Recording... Click to stop</span>
                </div>
              </>
            ) : audioFile ? (
              <div className="ready-indicator">
                <Zap size={16} />
                <span>Recording ready • {audioFile.name}</span>
              </div>
            ) : (
              <div className="idle-text">
                Click the microphone to start recording
              </div>
            )}
          </div>

          {audioFile && !recording && (
            <button
              onClick={processVoiceQuery}
              disabled={loading}
              className="process-button"
            >
              {loading ? (
                <>
                  <Loader size={20} className="spinner" />
                  <span>Processing with AI...</span>
                </>
              ) : (
                <>
                  <Send size={20} />
                  <span>Analyze Query</span>
                </>
              )}
            </button>
          )}

          <div className="divider-with-text">
            <span>or upload audio file</span>
          </div>

          <input
            type="file"
            accept="audio/*"
            onChange={handleFileChange}
            ref={fileInputRef}
            style={{ display: 'none' }}
          />

          <button
            onClick={() => fileInputRef.current.click()}
            className="upload-button"
            disabled={recording}
          >
            <Volume2 size={20} />
            <span>Choose Audio File</span>
          </button>
        </div>

        {/* Example Queries */}
        <div className="examples-section">
          <h3>💡 Try asking:</h3>
          <div className="example-grid">
            <div className="example-card">
              <span className="example-icon">📦</span>
              <span>"How many chocolates do we have in stock?"</span>
            </div>
            <div className="example-card">
              <span className="example-icon">💰</span>
              <span>"What is the price of milk?"</span>
            </div>
            <div className="example-card">
              <span className="example-icon">📍</span>
              <span>"Where is the bread located?"</span>
            </div>
            <div className="example-card">
              <span className="example-icon">🔄</span>
              <span>"Do we need to reorder chips?"</span>
            </div>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="error-banner">
            <span>⚠️ {error}</span>
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="results-container">
            {/* Transcription Card */}
            <div className="result-card transcription-card">
              <div className="card-header">
                <h3>🎯 Transcription</h3>
                <div className="badges">
                  <span className="language-badge">
                    {result.transcription.language.toUpperCase()}
                  </span>
                  <span className="confidence-badge">
                    {(result.transcription.confidence * 100).toFixed(0)}% confident
                  </span>
                </div>
              </div>
              <div className="transcription-text">
                "{result.transcription.text}"
              </div>
            </div>

            {/* Analysis Card */}
            <div className="result-card analysis-card">
              <div className="card-header">
                <h3>🧠 AI Analysis</h3>
              </div>
              <div className="analysis-content">
                <div className="analysis-row">
                  <span className="label">Detected Intent:</span>
                  <span className="value">
                    {getIntentIcon(result.analysis.intent)} {result.analysis.intent.replace('_', ' ').toUpperCase()}
                  </span>
                </div>
                <div className="analysis-row">
                  <span className="label">Product:</span>
                  <span className="value">{result.analysis.product}</span>
                </div>
              </div>
            </div>

            {/* AI Response Card */}
            <div className="result-card response-card">
              <div className="card-header">
                <h3>💬 AI Response</h3>
              </div>
              <div className="response-text">
                {result.analysis.response}
              </div>
            </div>
          </div>
        )}
      </div>

      <style jsx>{`
        .voice-assistant-view {
          position: relative;
          min-height: 100vh;
          padding: 40px 20px;
          overflow: hidden;
        }

        .animated-bg {
          position: fixed;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          z-index: 0;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          overflow: hidden;
        }

        .gradient-orb {
          position: absolute;
          border-radius: 50%;
          filter: blur(80px);
          opacity: 0.3;
          animation: float 20s infinite ease-in-out;
        }

        .orb-1 {
          width: 400px;
          height: 400px;
          background: #f093fb;
          top: -200px;
          left: -200px;
          animation-delay: 0s;
        }

        .orb-2 {
          width: 500px;
          height: 500px;
          background: #4facfe;
          bottom: -250px;
          right: -250px;
          animation-delay: 7s;
        }

        .orb-3 {
          width: 300px;
          height: 300px;
          background: #43e97b;
          top: 50%;
          left: 50%;
          animation-delay: 14s;
        }

        @keyframes float {
          0%, 100% { transform: translate(0, 0) scale(1); }
          33% { transform: translate(50px, -50px) scale(1.1); }
          66% { transform: translate(-50px, 50px) scale(0.9); }
        }

        .content-wrapper {
          position: relative;
          z-index: 1;
          max-width: 800px;
          margin: 0 auto;
        }

        .hero-header {
          text-align: center;
          margin-bottom: 40px;
          color: white;
        }

        .icon-badge {
          width: 80px;
          height: 80px;
          background: rgba(255, 255, 255, 0.2);
          backdrop-filter: blur(10px);
          border-radius: 20px;
          display: flex;
          align-items: center;
          justify-content: center;
          margin: 0 auto 20px;
          color: white;
          box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }

        .hero-header h1 {
          font-size: 42px;
          font-weight: 800;
          margin: 0 0 12px 0;
          text-shadow: 0 2px 20px rgba(0, 0, 0, 0.2);
        }

        .subtitle {
          font-size: 18px;
          opacity: 0.95;
          margin-bottom: 20px;
        }

        .language-pills {
          display: flex;
          gap: 10px;
          justify-content: center;
          flex-wrap: wrap;
        }

        .pill {
          background: rgba(255, 255, 255, 0.2);
          backdrop-filter: blur(10px);
          padding: 6px 16px;
          border-radius: 20px;
          font-size: 14px;
          font-weight: 600;
        }

        .recording-card {
          background: rgba(255, 255, 255, 0.95);
          backdrop-filter: blur(20px);
          border-radius: 24px;
          padding: 48px;
          box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
          margin-bottom: 30px;
        }

        .mic-visualizer {
          position: relative;
          display: flex;
          justify-content: center;
          margin-bottom: 24px;
        }

        .pulse-rings {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
        }

        .pulse-ring {
          position: absolute;
          top: 50%;
          left: 50%;
          transform: translate(-50%, -50%);
          border: 3px solid #ef4444;
          border-radius: 50%;
          animation: pulse 2s infinite;
        }

        .ring-1 {
          width: 120px;
          height: 120px;
          animation-delay: 0s;
        }

        .ring-2 {
          width: 160px;
          height: 160px;
          animation-delay: 0.5s;
        }

        .ring-3 {
          width: 200px;
          height: 200px;
          animation-delay: 1s;
        }

        @keyframes pulse {
          0% {
            opacity: 1;
            transform: translate(-50%, -50%) scale(0.8);
          }
          100% {
            opacity: 0;
            transform: translate(-50%, -50%) scale(1.5);
          }
        }

        .mic-button {
          width: 120px;
          height: 120px;
          border-radius: 50%;
          border: none;
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.3s ease;
          box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
          position: relative;
          z-index: 2;
        }

        .mic-button.recording {
          background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
          box-shadow: 0 10px 30px rgba(239, 68, 68, 0.4);
          animation: recordingPulse 1.5s infinite;
        }

        @keyframes recordingPulse {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.05); }
        }

        .mic-button:hover:not(:disabled) {
          transform: scale(1.1);
          box-shadow: 0 15px 40px rgba(102, 126, 234, 0.6);
        }

        .mic-button:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .recording-status {
          text-align: center;
          margin-bottom: 24px;
          min-height: 30px;
        }

        .recording-indicator {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          color: #ef4444;
          font-weight: 600;
          font-size: 16px;
        }

        .red-dot {
          width: 10px;
          height: 10px;
          background: #ef4444;
          border-radius: 50%;
          animation: blink 1s infinite;
        }

        @keyframes blink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }

        .ready-indicator {
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 8px;
          color: #10b981;
          font-weight: 600;
          font-size: 16px;
        }

        .idle-text {
          color: #64748b;
          font-size: 16px;
        }

        .process-button {
          width: 100%;
          padding: 18px;
          background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
          color: white;
          border: none;
          border-radius: 16px;
          font-size: 18px;
          font-weight: 700;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 12px;
          transition: all 0.3s ease;
          box-shadow: 0 8px 24px rgba(240, 147, 251, 0.4);
          margin-bottom: 24px;
        }

        .process-button:hover:not(:disabled) {
          transform: translateY(-2px);
          box-shadow: 0 12px 32px rgba(240, 147, 251, 0.6);
        }

        .process-button:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }

        .spinner {
          animation: spin 1s linear infinite;
        }

        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        .divider-with-text {
          text-align: center;
          margin: 24px 0;
          position: relative;
        }

        .divider-with-text::before,
        .divider-with-text::after {
          content: '';
          position: absolute;
          top: 50%;
          width: 40%;
          height: 1px;
          background: #e2e8f0;
        }

        .divider-with-text::before { left: 0; }
        .divider-with-text::after { right: 0; }

        .divider-with-text span {
          background: white;
          padding: 0 16px;
          color: #94a3b8;
          font-size: 14px;
          font-weight: 600;
        }

        .upload-button {
          width: 100%;
          padding: 14px;
          background: white;
          border: 2px solid #e2e8f0;
          border-radius: 12px;
          font-size: 16px;
          font-weight: 600;
          color: #667eea;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 10px;
          transition: all 0.2s ease;
        }

        .upload-button:hover:not(:disabled) {
          border-color: #667eea;
          background: #f8f9ff;
        }

        .upload-button:disabled {
          opacity: 0.5;
          cursor: not-allowed;
        }

        .examples-section {
          background: rgba(255, 255, 255, 0.95);
          backdrop-filter: blur(20px);
          border-radius: 20px;
          padding: 32px;
          margin-bottom: 30px;
        }

        .examples-section h3 {
          margin: 0 0 20px 0;
          font-size: 20px;
          color: #1e293b;
        }

        .example-grid {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
          gap: 12px;
        }

        .example-card {
          background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
          padding: 16px;
          border-radius: 12px;
          border: 1px solid #e0e7ff;
          display: flex;
          align-items: center;
          gap: 12px;
          font-size: 14px;
          color: #475569;
          transition: all 0.2s ease;
        }

        .example-card:hover {
          transform: translateY(-2px);
          box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
        }

        .example-icon {
          font-size: 24px;
        }

        .error-banner {
          background: #fee;
          border-left: 4px solid #ef4444;
          padding: 16px;
          border-radius: 12px;
          color: #991b1b;
          margin-bottom: 24px;
          font-weight: 600;
        }

        .results-container {
          display: grid;
          gap: 20px;
        }

        .result-card {
          background: rgba(255, 255, 255, 0.95);
          backdrop-filter: blur(20px);
          border-radius: 20px;
          padding: 28px;
          box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        }

        .card-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }

        .card-header h3 {
          margin: 0;
          font-size: 20px;
          color: #1e293b;
        }

        .badges {
          display: flex;
          gap: 10px;
        }

        .language-badge {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
          padding: 6px 14px;
          border-radius: 20px;
          font-size: 12px;
          font-weight: 700;
        }

        .confidence-badge {
          background: #f0fdf4;
          color: #166534;
          padding: 6px 14px;
          border-radius: 20px;
          font-size: 12px;
          font-weight: 700;
          border: 1px solid #bbf7d0;
        }

        .transcription-text {
          font-size: 20px;
          line-height: 1.6;
          color: #1e293b;
          font-style: italic;
          background: linear-gradient(135deg, #f8f9ff 0%, #f0f4ff 100%);
          padding: 24px;
          border-radius: 12px;
          border-left: 4px solid #667eea;
        }

        .analysis-content {
          display: grid;
          gap: 16px;
        }

        .analysis-row {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 16px;
          background: #f8fafc;
          border-radius: 12px;
        }

        .analysis-row .label {
          font-weight: 600;
          color: #64748b;
        }

        .analysis-row .value {
          color: #1e293b;
          font-weight: 700;
        }

        .response-card {
          background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
          color: white;
        }

        .response-card .card-header h3 {
          color: white;
        }

        .response-text {
          font-size: 20px;
          line-height: 1.8;
          background: rgba(255, 255, 255, 0.15);
          padding: 24px;
          border-radius: 12px;
          backdrop-filter: blur(10px);
        }
      `}</style>
    </div>
  );
};

export default VoiceAssistantView;
