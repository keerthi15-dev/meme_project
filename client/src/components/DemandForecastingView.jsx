import React, { useState } from 'react';
import { Line } from 'react-chartjs-2';
import { TrendingUp, Calendar, Brain, AlertCircle, CheckCircle, Loader } from 'lucide-react';

const DemandForecastingView = () => {
    const [selectedProduct, setSelectedProduct] = useState(null);
    const [forecastData, setForecastData] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const userId = localStorage.getItem('userId');

    // Sample products (in real app, fetch from backend)
    const products = [
        { id: 0, name: 'Rice (25kg Bag)', avgSales: 120 },
        { id: 1, name: 'Sunflower Oil (1L)', avgSales: 85 },
        { id: 2, name: 'Sugar (kg)', avgSales: 95 },
        { id: 3, name: 'Toor Dal (kg)', avgSales: 70 },
        { id: 4, name: 'Tea Powder (250g)', avgSales: 110 }
    ];

    const generateHistoricalData = (product) => {
        // Generate 60 days of synthetic historical data
        const data = [];
        const base = product.avgSales;

        for (let i = 0; i < 60; i++) {
            const trend = i * 0.5;
            const weekly = Math.sin(i * 2 * Math.PI / 7) * 10;
            const noise = (Math.random() - 0.5) * 20;
            data.push(Math.max(0, Math.round(base + trend + weekly + noise)));
        }

        return data;
    };

    const generateForecast = async (product) => {
        setLoading(true);
        setError(null);
        setSelectedProduct(product);

        try {
            const historicalSales = generateHistoricalData(product);

            const response = await fetch('http://localhost:8000/api/forecast/forecast', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    product_id: product.id,
                    product_name: product.name,
                    historical_sales: historicalSales
                })
            });

            if (!response.ok) throw new Error('Forecast generation failed');

            const data = await response.json();

            setForecastData({
                ...data,
                historical: historicalSales
            });

        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const createChartData = (horizon = 7) => {
        if (!forecastData) return null;

        const historical = forecastData.historical.slice(-30); // Last 30 days
        const forecast = horizon === 7 ? forecastData.forecast_7day : forecastData.forecast_30day;
        const lower = horizon === 7 ? forecastData.confidence_7day_lower : forecastData.confidence_30day_lower;
        const upper = horizon === 7 ? forecastData.confidence_7day_upper : forecastData.confidence_30day_upper;

        const labels = [
            ...historical.map((_, i) => `Day -${30 - i}`),
            ...forecast.map((_, i) => `Day +${i + 1}`)
        ];

        return {
            labels,
            datasets: [
                {
                    label: 'Historical Sales',
                    data: [...historical, ...Array(forecast.length).fill(null)],
                    borderColor: 'rgb(99, 102, 241)',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    borderWidth: 2,
                    pointRadius: 2,
                    tension: 0.3
                },
                {
                    label: 'Predicted Sales',
                    data: [...Array(historical.length).fill(null), ...forecast],
                    borderColor: 'rgb(16, 185, 129)',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    borderWidth: 3,
                    borderDash: [5, 5],
                    pointRadius: 4,
                    pointBackgroundColor: 'rgb(16, 185, 129)',
                    tension: 0.3
                },
                {
                    label: 'Upper Confidence (±1σ)',
                    data: [...Array(historical.length).fill(null), ...upper],
                    borderColor: 'rgba(16, 185, 129, 0.3)',
                    backgroundColor: 'rgba(16, 185, 129, 0.05)',
                    borderWidth: 1,
                    borderDash: [2, 2],
                    pointRadius: 0,
                    fill: '+1',
                    tension: 0.3
                },
                {
                    label: 'Lower Confidence (±1σ)',
                    data: [...Array(historical.length).fill(null), ...lower],
                    borderColor: 'rgba(16, 185, 129, 0.3)',
                    backgroundColor: 'rgba(16, 185, 129, 0.15)',
                    borderWidth: 1,
                    borderDash: [2, 2],
                    pointRadius: 0,
                    fill: false,
                    tension: 0.3
                }
            ]
        };
    };

    const chartOptions = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    usePointStyle: true,
                    padding: 15,
                    font: { size: 11 }
                }
            },
            tooltip: {
                mode: 'index',
                intersect: false,
                callbacks: {
                    label: (context) => {
                        let label = context.dataset.label || '';
                        if (label) label += ': ';
                        label += Math.round(context.parsed.y);
                        return label;
                    }
                }
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                title: {
                    display: true,
                    text: 'Sales Units',
                    font: { size: 12, weight: 'bold' }
                }
            },
            x: {
                title: {
                    display: true,
                    text: 'Time Period',
                    font: { size: 12, weight: 'bold' }
                }
            }
        }
    };

    return (
        <div style={{ padding: '2rem' }}>
            <header style={{ marginBottom: '2rem' }}>
                <h1 style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '2rem', fontWeight: '800', color: '#1e293b' }}>
                    <Brain size={32} color="#6366f1" />
                    AI Demand Forecasting
                    <span style={{ fontSize: '0.9rem', fontWeight: '600', color: '#10b981', background: '#f0fdf4', padding: '4px 12px', borderRadius: '20px', marginLeft: '12px' }}>
                        Hybrid LSTM-Transformer
                    </span>
                </h1>
                <p style={{ color: '#64748b', marginTop: '8px', fontSize: '1rem' }}>
                    Novel deep learning architecture with 2.2M parameters for multi-horizon demand prediction
                </p>
            </header>

            {/* Product Selection */}
            <div style={{ background: 'white', padding: '1.5rem', borderRadius: '16px', border: '1px solid #e2e8f0', marginBottom: '2rem' }}>
                <h3 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <TrendingUp size={20} color="#6366f1" />
                    Select Product for Forecasting
                </h3>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
                    {products.map(product => (
                        <button
                            key={product.id}
                            onClick={() => generateForecast(product)}
                            disabled={loading}
                            style={{
                                padding: '1rem',
                                background: selectedProduct?.id === product.id ? '#eef2ff' : 'white',
                                border: selectedProduct?.id === product.id ? '2px solid #6366f1' : '1px solid #e2e8f0',
                                borderRadius: '12px',
                                cursor: loading ? 'not-allowed' : 'pointer',
                                transition: '0.2s',
                                textAlign: 'left'
                            }}
                        >
                            <div style={{ fontWeight: '600', color: '#1e293b', marginBottom: '4px' }}>{product.name}</div>
                            <div style={{ fontSize: '0.85rem', color: '#64748b' }}>Avg: {product.avgSales} units/day</div>
                        </button>
                    ))}
                </div>
            </div>

            {/* Loading State */}
            {loading && (
                <div style={{ textAlign: 'center', padding: '3rem', background: 'white', borderRadius: '16px', border: '1px solid #e2e8f0' }}>
                    <Loader size={48} color="#6366f1" style={{ animation: 'spin 1s linear infinite' }} />
                    <p style={{ marginTop: '1rem', color: '#64748b', fontWeight: '600' }}>
                        Running Hybrid LSTM-Transformer model on Mac GPU...
                    </p>
                </div>
            )}

            {/* Error State */}
            {error && (
                <div style={{ padding: '1rem', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: '12px', display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <AlertCircle size={24} color="#ef4444" />
                    <div>
                        <div style={{ fontWeight: '600', color: '#991b1b' }}>Forecasting Error</div>
                        <div style={{ fontSize: '0.9rem', color: '#991b1b' }}>{error}</div>
                    </div>
                </div>
            )}

            {/* Forecast Results */}
            {forecastData && !loading && (
                <div>
                    {/* Model Info Banner */}
                    <div style={{ background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)', color: 'white', padding: '1.5rem', borderRadius: '16px', marginBottom: '2rem' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
                            <div>
                                <div style={{ fontSize: '0.85rem', opacity: 0.9, marginBottom: '4px' }}>MODEL ARCHITECTURE</div>
                                <div style={{ fontSize: '1.3rem', fontWeight: '800' }}>{forecastData.model_info.model_type}</div>
                            </div>
                            <div>
                                <div style={{ fontSize: '0.85rem', opacity: 0.9, marginBottom: '4px' }}>PARAMETERS</div>
                                <div style={{ fontSize: '1.3rem', fontWeight: '800' }}>{(forecastData.model_info.parameters / 1000000).toFixed(1)}M</div>
                            </div>
                            <div>
                                <div style={{ fontSize: '0.85rem', opacity: 0.9, marginBottom: '4px' }}>DEVICE</div>
                                <div style={{ fontSize: '1.3rem', fontWeight: '800', textTransform: 'uppercase' }}>{forecastData.model_info.device}</div>
                            </div>
                            <div>
                                <div style={{ fontSize: '0.85rem', opacity: 0.9, marginBottom: '4px' }}>STATUS</div>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '1.1rem', fontWeight: '700' }}>
                                    <CheckCircle size={20} /> READY
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* 7-Day Forecast */}
                    <div style={{ background: 'white', padding: '1.5rem', borderRadius: '16px', border: '1px solid #e2e8f0', marginBottom: '2rem' }}>
                        <h3 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <Calendar size={20} color="#10b981" />
                            7-Day Forecast with Confidence Intervals
                        </h3>
                        <div style={{ height: '350px' }}>
                            {createChartData(7) && <Line data={createChartData(7)} options={chartOptions} />}
                        </div>
                        <div style={{ marginTop: '1rem', display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: '8px' }}>
                            {forecastData.forecast_7day.map((value, i) => (
                                <div key={i} style={{ textAlign: 'center', padding: '8px', background: '#f8fafc', borderRadius: '8px' }}>
                                    <div style={{ fontSize: '0.75rem', color: '#64748b', marginBottom: '4px' }}>Day {i + 1}</div>
                                    <div style={{ fontSize: '1.1rem', fontWeight: '700', color: '#10b981' }}>{Math.round(value)}</div>
                                    <div style={{ fontSize: '0.7rem', color: '#94a3b8' }}>
                                        ±{Math.round((forecastData.confidence_7day_upper[i] - forecastData.confidence_7day_lower[i]) / 2)}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* 30-Day Forecast */}
                    <div style={{ background: 'white', padding: '1.5rem', borderRadius: '16px', border: '1px solid #e2e8f0' }}>
                        <h3 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <Calendar size={20} color="#6366f1" />
                            30-Day Extended Forecast
                        </h3>
                        <div style={{ height: '400px' }}>
                            {createChartData(30) && <Line data={createChartData(30)} options={chartOptions} />}
                        </div>
                        <div style={{ marginTop: '1.5rem', padding: '1rem', background: '#f0fdf4', borderRadius: '12px', border: '1px solid #bbf7d0' }}>
                            <div style={{ fontWeight: '600', color: '#166534', marginBottom: '8px' }}>📊 Forecast Summary</div>
                            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', fontSize: '0.9rem' }}>
                                <div>
                                    <span style={{ color: '#15803d' }}>Average (30d):</span>
                                    <span style={{ fontWeight: '700', marginLeft: '8px' }}>
                                        {Math.round(forecastData.forecast_30day.reduce((a, b) => a + b, 0) / 30)} units/day
                                    </span>
                                </div>
                                <div>
                                    <span style={{ color: '#15803d' }}>Total (30d):</span>
                                    <span style={{ fontWeight: '700', marginLeft: '8px' }}>
                                        {Math.round(forecastData.forecast_30day.reduce((a, b) => a + b, 0))} units
                                    </span>
                                </div>
                                <div>
                                    <span style={{ color: '#15803d' }}>Trend:</span>
                                    <span style={{ fontWeight: '700', marginLeft: '8px' }}>
                                        {forecastData.forecast_30day[29] > forecastData.forecast_30day[0] ? '📈 Growing' : '📉 Declining'}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Dynamic Pricing Optimizer */}
                    <div style={{ background: 'white', padding: '1.5rem', borderRadius: '16px', border: '1px solid #e2e8f0', marginTop: '2rem' }}>
                        <h3 style={{ marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#6366f1" strokeWidth="2">
                                <circle cx="12" cy="12" r="10" />
                                <path d="M12 6v6l4 2" />
                            </svg>
                            Dynamic Pricing Optimizer
                        </h3>
                        <p style={{ color: '#64748b', marginBottom: '1.5rem' }}>
                            AI-suggested price adjustments for maximum margin:
                        </p>

                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem' }}>
                            {/* Sample pricing suggestions */}
                            <div style={{ padding: '1.5rem', background: '#fafafa', borderRadius: '12px', border: '1px solid #e5e7eb' }}>
                                <div style={{ fontWeight: '700', fontSize: '1.1rem', marginBottom: '0.5rem' }}>
                                    {selectedProduct?.name || 'Product'}
                                </div>
                                <div style={{ display: 'flex', alignItems: 'baseline', gap: '12px', marginBottom: '1rem' }}>
                                    <span style={{ fontSize: '0.9rem', color: '#9ca3af', textDecoration: 'line-through' }}>
                                        ₹{Math.round(selectedProduct?.avgSales * 0.3) || 28}
                                    </span>
                                    <span style={{ fontSize: '1.8rem', fontWeight: '800', color: '#10b981' }}>
                                        ₹{Math.round(selectedProduct?.avgSales * 0.35) || 30}
                                    </span>
                                </div>
                                <div style={{
                                    padding: '8px 16px',
                                    background: '#ede9fe',
                                    borderRadius: '8px',
                                    color: '#6366f1',
                                    fontSize: '0.85rem',
                                    fontWeight: '600',
                                    textAlign: 'center'
                                }}>
                                    High Local Demand
                                </div>
                            </div>

                            <div style={{ padding: '1.5rem', background: '#fafafa', borderRadius: '12px', border: '1px solid #e5e7eb' }}>
                                <div style={{ fontWeight: '700', fontSize: '1.1rem', marginBottom: '0.5rem' }}>
                                    Maggi Noodles 70g
                                </div>
                                <div style={{ display: 'flex', alignItems: 'baseline', gap: '12px', marginBottom: '1rem' }}>
                                    <span style={{ fontSize: '0.9rem', color: '#9ca3af', textDecoration: 'line-through' }}>
                                        ₹14
                                    </span>
                                    <span style={{ fontSize: '1.8rem', fontWeight: '800', color: '#10b981' }}>
                                        ₹12
                                    </span>
                                </div>
                                <div style={{
                                    padding: '8px 16px',
                                    background: '#ede9fe',
                                    borderRadius: '8px',
                                    color: '#6366f1',
                                    fontSize: '0.85rem',
                                    fontWeight: '600',
                                    textAlign: 'center'
                                }}>
                                    Inventory Liquidation
                                </div>
                            </div>

                            <div style={{ padding: '1.5rem', background: '#fafafa', borderRadius: '12px', border: '1px solid #e5e7eb' }}>
                                <div style={{ fontWeight: '700', fontSize: '1.1rem', marginBottom: '0.5rem' }}>
                                    Britannia Biscuits
                                </div>
                                <div style={{ display: 'flex', alignItems: 'baseline', gap: '12px', marginBottom: '1rem' }}>
                                    <span style={{ fontSize: '0.9rem', color: '#9ca3af', textDecoration: 'line-through' }}>
                                        ₹40
                                    </span>
                                    <span style={{ fontSize: '1.8rem', fontWeight: '800', color: '#10b981' }}>
                                        ₹42
                                    </span>
                                </div>
                                <div style={{
                                    padding: '8px 16px',
                                    background: '#ede9fe',
                                    borderRadius: '8px',
                                    color: '#6366f1',
                                    fontSize: '0.85rem',
                                    fontWeight: '600',
                                    textAlign: 'center'
                                }}>
                                    Low Competitor Stock
                                </div>
                            </div>
                        </div>

                        <button style={{
                            width: '100%',
                            marginTop: '1.5rem',
                            padding: '1rem',
                            background: 'white',
                            border: '2px solid #e5e7eb',
                            borderRadius: '12px',
                            fontSize: '1rem',
                            fontWeight: '600',
                            color: '#6366f1',
                            cursor: 'pointer',
                            transition: '0.2s'
                        }}
                            onMouseOver={(e) => e.target.style.background = '#f9fafb'}
                            onMouseOut={(e) => e.target.style.background = 'white'}
                        >
                            Apply All Optimization Suggestions
                        </button>
                    </div>
                </div>
            )}

            <style>{`
                @keyframes spin {
                    from { transform: rotate(0deg); }
                    to { transform: rotate(360deg); }
                }
            `}</style>
        </div >
    );
};

export default DemandForecastingView;
