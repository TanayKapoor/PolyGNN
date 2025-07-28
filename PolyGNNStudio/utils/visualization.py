"""
Visualization utilities for PolyGNN Showcase application.
Creates interactive plots using Plotly for data visualization.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def create_scatter_plot(y_true, y_pred, y_unc, property_name):
    """
    Create scatter plot comparing predicted vs true values with error bars.
    
    Args:
        y_true (array): True values
        y_pred (array): Predicted values  
        y_unc (array): Uncertainty values
        property_name (str): Name of the property being plotted
        
    Returns:
        plotly.graph_objects.Figure: Scatter plot figure
    """
    # Calculate R² for perfect prediction line
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot with error bars
    fig.add_trace(go.Scatter(
        x=y_true,
        y=y_pred,
        error_y=dict(
            type='data',
            array=y_unc,
            visible=True,
            color='rgba(255,0,0,0.3)'
        ),
        mode='markers',
        marker=dict(
            size=8,
            color='blue',
            opacity=0.7
        ),
        name='Predictions',
        hovertemplate='True: %{x}<br>Predicted: %{y}<br>Uncertainty: ±%{error_y.array}<extra></extra>'
    ))
    
    # Add perfect prediction line (y=x)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Perfect Prediction',
        hovertemplate='y = x<extra></extra>'
    ))
    
    # Update layout
    units = {'Tg': '°C', 'Tm': '°C', 'Density': 'g/cm³'}
    unit = units.get(property_name, '')
    
    fig.update_layout(
        title=f'{property_name} Predictions vs True Values (R² = {r2:.3f})',
        xaxis_title=f'True {property_name} ({unit})',
        yaxis_title=f'Predicted {property_name} ({unit})',
        showlegend=True,
        template='plotly_dark',
        width=600,
        height=500
    )
    
    # Make axes equal for better visualization
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    
    return fig

def create_histogram(values, property_name):
    """
    Create histogram of predicted property values.
    
    Args:
        values (array): Property values to plot
        property_name (str): Name of the property
        
    Returns:
        plotly.graph_objects.Figure: Histogram figure
    """
    # Create histogram
    fig = px.histogram(
        x=values,
        nbins=20,
        title=f'Distribution of {property_name} Predictions',
        labels={'x': f'{property_name}', 'count': 'Frequency'}
    )
    
    # Add statistics annotations
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_val:.2f}"
    )
    
    # Update layout
    units = {'Tg': '°C', 'Tm': '°C', 'Density': 'g/cm³'}
    unit = units.get(property_name, '')
    
    fig.update_layout(
        xaxis_title=f'{property_name} ({unit})',
        yaxis_title='Count',
        template='plotly_dark',
        showlegend=False
    )
    
    return fig

def create_shap_plot():
    """
    Create SHAP summary plot (placeholder implementation).
    
    Returns:
        plotly.graph_objects.Figure: SHAP-style plot figure
    """
    # Placeholder SHAP data
    features = [
        'Chain Flexibility',
        'Molecular Weight', 
        'Branching Index',
        'Aromatic Content',
        'Polarity',
        'Crystallinity Index',
        'Backbone Rigidity',
        'Side Chain Length',
        'Thermal Stability',
        'Glass Transition Contributors'
    ]
    
    # Generate placeholder SHAP values
    np.random.seed(42)
    n_samples = 50
    shap_values = []
    
    for i, feature in enumerate(features):
        # Generate SHAP values with different distributions per feature
        values = np.random.normal(0, 0.5 - 0.03*i, n_samples)
        shap_values.extend(values)
    
    # Create feature labels for each SHAP value
    feature_labels = []
    for feature in features:
        feature_labels.extend([feature] * n_samples)
    
    # Create DataFrame for plotting
    shap_df = pd.DataFrame({
        'feature': feature_labels,
        'shap_value': shap_values,
        'feature_value': np.random.uniform(0, 1, len(shap_values))  # Normalized feature values
    })
    
    # Create beeswarm-style plot
    fig = px.scatter(
        shap_df, 
        x='shap_value', 
        y='feature',
        color='feature_value',
        color_continuous_scale='RdYlBu_r',
        title='SHAP Feature Importance Summary',
        size_max=8
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='SHAP Value (Impact on Model Output)',
        yaxis_title='Features',
        template='plotly_dark',
        coloraxis_colorbar_title='Feature Value',
        height=600
    )
    
    # Add vertical line at x=0
    fig.add_vline(x=0, line_dash="dash", line_color="black", opacity=0.5)
    
    return fig

def create_uncertainty_plot(predictions_dict):
    """
    Create visualization for uncertainty analysis.
    
    Args:
        predictions_dict (dict): Dictionary with predictions and uncertainties
        
    Returns:
        plotly.graph_objects.Figure: Uncertainty analysis figure
    """
    # Extract data
    tg_pred = predictions_dict['Tg']
    tm_pred = predictions_dict['Tm']
    density_pred = predictions_dict['Density']
    uncertainty = predictions_dict['unc']
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Tg vs Uncertainty', 'Tm vs Uncertainty', 
                       'Density vs Uncertainty', 'Uncertainty Distribution'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Tg vs Uncertainty
    fig.add_trace(
        go.Scatter(x=tg_pred, y=uncertainty, mode='markers', name='Tg'),
        row=1, col=1
    )
    
    # Tm vs Uncertainty  
    fig.add_trace(
        go.Scatter(x=tm_pred, y=uncertainty, mode='markers', name='Tm'),
        row=1, col=2
    )
    
    # Density vs Uncertainty
    fig.add_trace(
        go.Scatter(x=density_pred, y=uncertainty, mode='markers', name='Density'),
        row=2, col=1
    )
    
    # Uncertainty distribution
    fig.add_trace(
        go.Histogram(x=uncertainty, name='Uncertainty Dist'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title='Prediction Uncertainty Analysis',
        template='plotly_white',
        height=600,
        showlegend=False
    )
    
    return fig

def create_property_correlation_plot(predictions_dict):
    """
    Create correlation plot between predicted properties.
    
    Args:
        predictions_dict (dict): Dictionary with predictions
        
    Returns:
        plotly.graph_objects.Figure: Property correlation figure
    """
    # Create correlation matrix
    properties_df = pd.DataFrame({
        'Tg': predictions_dict['Tg'],
        'Tm': predictions_dict['Tm'], 
        'Density': predictions_dict['Density']
    })
    
    correlation_matrix = properties_df.corr()
    
    # Create heatmap
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title='Property Correlation Matrix',
        color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1
    )
    
    fig.update_layout(template='plotly_white')
    
    return fig

def create_sensitivity_plot(base_predictions, perturbed_predictions, feature_name, perturbation):
    """
    Create sensitivity analysis plot showing effect of feature perturbation.
    
    Args:
        base_predictions (dict): Original predictions
        perturbed_predictions (dict): Predictions with feature perturbation
        feature_name (str): Name of perturbed feature
        perturbation (float): Perturbation percentage
        
    Returns:
        plotly.graph_objects.Figure: Sensitivity analysis plot
    """
    # Calculate changes
    tg_change = perturbed_predictions['Tg'] - base_predictions['Tg']
    tm_change = perturbed_predictions['Tm'] - base_predictions['Tm']
    density_change = perturbed_predictions['Density'] - base_predictions['Density']
    
    # Create subplot
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=['Tg Change', 'Tm Change', 'Density Change'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Add bar plots for changes
    sample_indices = list(range(len(tg_change)))
    
    fig.add_trace(
        go.Bar(x=sample_indices, y=tg_change, name='ΔTg', marker_color='blue'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=sample_indices, y=tm_change, name='ΔTm', marker_color='red'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=sample_indices, y=density_change, name='ΔDensity', marker_color='green'),
        row=1, col=3
    )
    
    # Update layout
    fig.update_layout(
        title=f'Sensitivity Analysis: {feature_name} ({perturbation:+.0%} perturbation)',
        template='plotly_dark',
        height=400,
        showlegend=False
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Sample Index")
    fig.update_yaxes(title_text="Change (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Change (°C)", row=1, col=2)
    fig.update_yaxes(title_text="Change (g/cm³)", row=1, col=3)
    
    return fig

# Model Analysis Visualization Functions

import streamlit as st

@st.cache_data
def create_performance_metrics_table():
    """
    Create performance metrics table for model analysis.
    
    Returns:
        pd.DataFrame: Performance metrics across different splits and properties
    """
    # TODO: Load real performance metrics from your evaluation files
    # Placeholder performance metrics for demonstration
    
    metrics_data = {
        'Split': ['Train', 'Validation', 'External Test'] * 3,
        'Property': ['Tg', 'Tg', 'Tg', 'Tm', 'Tm', 'Tm', 'Density', 'Density', 'Density'],
        'R²': [0.89, 0.78, 0.67, 0.85, 0.72, 0.64, 0.93, 0.86, 0.79],
        'RMSE': [28.5, 35.2, 40.1, 32.8, 45.6, 52.3, 0.045, 0.062, 0.078],
        'MAE': [21.3, 26.8, 31.7, 24.9, 34.1, 39.8, 0.032, 0.048, 0.058],
        'Sample Size': [8450, 1200, 850, 6200, 890, 620, 9100, 1300, 900]
    }
    
    return pd.DataFrame(metrics_data)

@st.cache_data
def create_shap_insights_plots():
    """
    Create SHAP insights visualizations.
    
    Returns:
        tuple: (beeswarm_fig, top_features_fig, feature_importance_data)
    """
    # TODO: Load real SHAP values from your analysis files
    # Placeholder SHAP data for demonstration
    
    # Feature importance data
    features = [
        'chain_flexibility', 'degree_polymerization', 'molecular_weight',
        'aromatic_content', 'branching_index', 'crystallinity_index',
        'backbone_rigidity', 'side_chain_length', 'thermal_stability',
        'glass_transition_contributors', 'polarity', 'cross_linking_density'
    ]
    
    # Generate realistic importance percentages that sum to 100%
    np.random.seed(42)
    raw_importance = np.random.exponential(2, len(features))
    importance_pct = (raw_importance / np.sum(raw_importance)) * 100
    
    # Sort by importance
    sorted_indices = np.argsort(importance_pct)[::-1]
    sorted_features = [features[i] for i in sorted_indices]
    sorted_importance = importance_pct[sorted_indices]
    
    # Top 10 features bar chart
    top_10_fig = px.bar(
        x=sorted_importance[:10],
        y=sorted_features[:10],
        orientation='h',
        title='Top 10 Feature Importance (% Contribution)',
        labels={'x': 'Importance (%)', 'y': 'Features'},
        color=sorted_importance[:10],
        color_continuous_scale='viridis'
    )
    top_10_fig.update_layout(
        template='plotly_dark',
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    # SHAP beeswarm plot (enhanced)
    n_samples = 200
    shap_data = []
    
    for i, feature in enumerate(sorted_features[:10]):
        # Generate SHAP values with different distributions
        base_impact = sorted_importance[i] / 100 * np.random.choice([-1, 1], size=n_samples)
        shap_values = np.random.normal(base_impact, 0.3, n_samples)
        feature_values = np.random.uniform(0, 1, n_samples)
        
        for j in range(n_samples):
            shap_data.append({
                'feature': feature,
                'shap_value': shap_values[j],
                'feature_value': feature_values[j],
                'importance': sorted_importance[i]
            })
    
    shap_df = pd.DataFrame(shap_data)
    
    beeswarm_fig = px.scatter(
        shap_df,
        x='shap_value',
        y='feature',
        color='feature_value',
        size='importance',
        color_continuous_scale='RdYlBu_r',
        title='SHAP Feature Impact Analysis',
        labels={'shap_value': 'SHAP Value (Impact on Prediction)', 'feature': 'Features'},
        size_max=12
    )
    beeswarm_fig.update_layout(
        template='plotly_dark',
        height=600,
        coloraxis_colorbar_title='Feature Value'
    )
    beeswarm_fig.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.7)
    
    return beeswarm_fig, top_10_fig, {'features': sorted_features, 'importance': sorted_importance}

@st.cache_data
def create_uncertainty_robustness_plots():
    """
    Create uncertainty quantification and robustness analysis plots.
    
    Returns:
        tuple: (coverage_pie_fig, noise_robustness_fig, robustness_data)
    """
    # TODO: Load real UQ and robustness metrics from your evaluation
    # Placeholder robustness data for demonstration
    
    # Coverage analysis (pie chart)
    coverage_data = {
        'Coverage Type': ['Within 95% CI', 'Outside 95% CI'],
        'Percentage': [94.7, 5.3],
        'Count': [2135, 119]
    }
    
    coverage_pie_fig = px.pie(
        values=coverage_data['Percentage'],
        names=coverage_data['Coverage Type'],
        title='Uncertainty Quantification Coverage (95% Confidence Intervals)',
        color_discrete_sequence=['#00cc96', '#ff6692']
    )
    coverage_pie_fig.update_layout(template='plotly_dark')
    
    # Noise robustness analysis
    noise_levels = ['0%', '0.5%', '1.0%', '1.5%', '2.0%', '2.5%', '3.0%']
    properties = ['Tg', 'Tm', 'Density']
    
    # Generate realistic performance degradation with noise
    np.random.seed(123)
    base_r2 = {'Tg': 0.67, 'Tm': 0.64, 'Density': 0.79}
    
    robustness_data = []
    for prop in properties:
        base = base_r2[prop]
        for i, noise in enumerate(noise_levels):
            # Performance degrades with noise but not linearly
            degradation = i * 0.02 + np.random.normal(0, 0.01)
            new_r2 = max(0.1, base - degradation)
            robustness_data.append({
                'Noise Level': noise,
                'Property': prop,
                'R² Score': new_r2
            })
    
    robustness_df = pd.DataFrame(robustness_data)
    
    noise_robustness_fig = px.line(
        robustness_df,
        x='Noise Level',
        y='R² Score',
        color='Property',
        title='Model Robustness to Input Noise',
        markers=True,
        line_shape='spline'
    )
    noise_robustness_fig.update_layout(
        template='plotly_dark',
        height=400,
        yaxis_range=[0, 1]
    )
    
    return coverage_pie_fig, noise_robustness_fig, robustness_data

@st.cache_data
def create_model_architecture_diagram():
    """
    Create a simple model architecture diagram placeholder.
    
    Returns:
        plotly.graph_objects.Figure: Architecture diagram
    """
    # Simple flowchart-style architecture diagram
    fig = go.Figure()
    
    # Define layers and connections
    layers = [
        {'name': 'SMILES Input', 'x': 1, 'y': 5, 'color': '#ff7f0e'},
        {'name': 'Molecular Graph\nConstruction', 'x': 2, 'y': 5, 'color': '#2ca02c'},
        {'name': 'GCN Layer 1\n(64 features)', 'x': 3, 'y': 6, 'color': '#1f77b4'},
        {'name': 'GCN Layer 2\n(32 features)', 'x': 4, 'y': 6, 'color': '#1f77b4'},
        {'name': 'GCN Layer 3\n(16 features)', 'x': 5, 'y': 6, 'color': '#1f77b4'},
        {'name': 'Global Pooling', 'x': 6, 'y': 5, 'color': '#d62728'},
        {'name': 'Tg Head', 'x': 7, 'y': 6, 'color': '#9467bd'},
        {'name': 'Tm Head', 'x': 7, 'y': 5, 'color': '#9467bd'},
        {'name': 'Density Head', 'x': 7, 'y': 4, 'color': '#9467bd'}
    ]
    
    # Add nodes
    for layer in layers:
        fig.add_trace(go.Scatter(
            x=[layer['x']],
            y=[layer['y']],
            mode='markers+text',
            marker=dict(size=60, color=layer['color']),
            text=layer['name'],
            textposition='middle center',
            textfont=dict(color='white', size=10),
            showlegend=False,
            hoverinfo='text',
            hovertext=layer['name']
        ))
    
    # Add connections
    connections = [
        (1, 5, 2, 5), (2, 5, 3, 6), (3, 6, 4, 6), (4, 6, 5, 6),
        (5, 6, 6, 5), (6, 5, 7, 6), (6, 5, 7, 5), (6, 5, 7, 4)
    ]
    
    for x1, y1, x2, y2 in connections:
        fig.add_trace(go.Scatter(
            x=[x1, x2],
            y=[y1, y2],
            mode='lines',
            line=dict(color='rgba(255,255,255,0.5)', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig.update_layout(
        title='PolyGNN Architecture Overview',
        template='plotly_dark',
        height=400,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_comparison_plot(df_results):
    """
    Create comparison plot for multiple polymer predictions.
    
    Args:
        df_results (pd.DataFrame): Results dataframe with predictions
        
    Returns:
        plotly.graph_objects.Figure: Comparison plot figure
    """
    if len(df_results) > 10:
        # Show only first 10 for readability
        df_plot = df_results.head(10).copy()
        df_plot['Polymer'] = [f'Polymer {i+1}' for i in range(10)]
    else:
        df_plot = df_results.copy()
        df_plot['Polymer'] = [f'Polymer {i+1}' for i in range(len(df_plot))]
    
    # Create grouped bar chart
    fig = go.Figure()
    
    # Add bars for each property
    fig.add_trace(go.Bar(
        name='Tg (°C)',
        x=df_plot['Polymer'],
        y=df_plot['Tg_pred'],
        yaxis='y1'
    ))
    
    fig.add_trace(go.Bar(
        name='Tm (°C)', 
        x=df_plot['Polymer'],
        y=df_plot['Tm_pred'],
        yaxis='y1'
    ))
    
    # Density on secondary y-axis (different scale)
    fig.add_trace(go.Bar(
        name='Density (g/cm³)',
        x=df_plot['Polymer'],
        y=df_plot['Density_pred'],
        yaxis='y2'
    ))
    
    # Update layout with secondary y-axis
    fig.update_layout(
        title='Property Predictions Comparison',
        xaxis_title='Polymers',
        yaxis=dict(title='Temperature (°C)', side='left'),
        yaxis2=dict(title='Density (g/cm³)', side='right', overlaying='y'),
        template='plotly_dark',
        barmode='group',
        height=500
    )
    
    return fig
