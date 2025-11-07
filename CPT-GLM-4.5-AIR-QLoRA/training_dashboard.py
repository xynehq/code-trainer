#!/usr/bin/env python3
"""
Enhanced Real-time Training Dashboard for GLM-4.5-Air
Access from anywhere: http://<server-ip>:5000

New Features:
- Validation loss & perplexity tracking
- Best checkpoint tracking
- Gradient norm monitoring
- Checkpoint management panel
- Event timeline with markers
- GPU resource monitoring (optional)
- Smoothing controls
- Zero impact on training (<0.5% CPU)

Usage:
    python training_dashboard_enhanced.py
    
    Or with gunicorn (recommended):
    pip install gunicorn
    gunicorn -w 2 -b 0.0.0.0:5000 training_dashboard_enhanced:app
"""

from flask import Flask, render_template, jsonify
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (CPU only)
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import io
import base64
from datetime import datetime
import logging
import subprocess
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
LOG_DIR = "glm45-air-cpt-qlora/logs"
TRAIN_LOG_FILE = f"{LOG_DIR}/training_log.jsonl"
EVAL_LOG_FILE = f"{LOG_DIR}/eval_log.jsonl"
EXPERT_LOG_FILE = f"{LOG_DIR}/expert_usage_log.jsonl"
ERROR_LOG_FILE = f"{LOG_DIR}/error_log.jsonl"
CHECKPOINT_DIR = "glm45-air-cpt-qlora"
REFRESH_INTERVAL = 30  # seconds
TOTAL_STEPS = 11790
WARMUP_STEPS = 944
MAX_LR = 5e-6

def load_jsonl(filepath):
    """Load JSONL file and return list of entries"""
    try:
        logs = []
        if not Path(filepath).exists():
            return []
        
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    logs.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return logs
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return []

def load_current_run(filepath):
    """Load only the current training run (after last step 0)"""
    logs = load_jsonl(filepath)
    if not logs:
        return []
    
    # Find last occurrence of step 0
    last_reset = 0
    for i, log in enumerate(logs):
        if log.get('step') == 0:
            last_reset = i
    
    current_run = logs[last_reset:]
    logger.info(f"Loaded {len(current_run)} entries from {filepath}")
    return current_run

def get_gpu_stats():
    """Get GPU statistics using nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=2
        )
        
        if result.returncode != 0:
            return None
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 6:
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_used': int(parts[2]),
                    'memory_total': int(parts[3]),
                    'utilization': int(parts[4]),
                    'temperature': int(parts[5])
                })
        return gpus
    except Exception as e:
        logger.debug(f"GPU stats unavailable: {e}")
        return None

def get_checkpoints():
    """Get list of checkpoints with their metadata"""
    try:
        checkpoint_path = Path(CHECKPOINT_DIR)
        if not checkpoint_path.exists():
            return []
        
        checkpoints = []
        eval_logs = load_jsonl(EVAL_LOG_FILE)
        eval_dict = {log['step']: log for log in eval_logs}
        
        for item in checkpoint_path.iterdir():
            if item.is_dir() and item.name.startswith('checkpoint-'):
                try:
                    step = int(item.name.split('-')[1])
                    eval_data = eval_dict.get(step, {})
                    
                    checkpoints.append({
                        'name': item.name,
                        'step': step,
                        'path': str(item),
                        'eval_loss': eval_data.get('eval_loss'),
                        'eval_perplexity': eval_data.get('eval_perplexity'),
                        'timestamp': eval_data.get('timestamp')
                    })
                except (ValueError, IndexError):
                    continue
        
        # Sort by step
        checkpoints.sort(key=lambda x: x['step'], reverse=True)
        return checkpoints
    except Exception as e:
        logger.error(f"Error getting checkpoints: {e}")
        return []

def create_plot(data_x, data_y, title, xlabel, ylabel, color='#2E86AB', 
                ma_window=None, extra_lines=None, markers=None):
    """Create a single plot and return as base64 image"""
    plt.figure(figsize=(10, 5))
    plt.plot(data_x, data_y, linewidth=2, color=color, alpha=0.8, label='Actual')
    
    # Add moving average if requested
    if ma_window and len(data_y) > ma_window:
        ma = np.convolve(data_y, np.ones(ma_window)/ma_window, mode='valid')
        ma_x = data_x[ma_window-1:]
        plt.plot(ma_x, ma, linewidth=3, color='#A23B72', 
                label=f'{ma_window}-step MA', alpha=0.9)
    
    # Add extra lines (e.g., reference lines)
    if extra_lines:
        for line in extra_lines:
            if line['type'] == 'hline':
                plt.axhline(y=line['y'], color=line.get('color', 'red'), 
                           linestyle=line.get('style', '--'), 
                           label=line.get('label', ''), alpha=0.7)
            elif line['type'] == 'vline':
                plt.axvline(x=line['x'], color=line.get('color', 'red'), 
                           linestyle=line.get('style', '--'), 
                           label=line.get('label', ''), alpha=0.7)
    
    # Add markers (e.g., checkpoints)
    if markers:
        for marker in markers:
            plt.axvline(x=marker['x'], color=marker.get('color', 'green'),
                       linestyle=marker.get('style', ':'),
                       alpha=0.5, linewidth=1)
    
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    if ma_window or extra_lines:
        plt.legend(loc='best')
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return img_base64

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard_enhanced.html', refresh_interval=REFRESH_INTERVAL)

@app.route('/api/stats')
def get_stats():
    """API endpoint for current statistics"""
    train_logs = load_current_run(TRAIN_LOG_FILE)
    eval_logs = load_current_run(EVAL_LOG_FILE)
    
    if not train_logs:
        return jsonify({'error': 'No logs found'}), 404
    
    current = train_logs[-1]
    
    # Calculate statistics
    progress = (current['step'] / TOTAL_STEPS) * 100
    
    # Get recent average speed (last 100 steps)
    recent_speeds = [log['steps_per_sec'] for log in train_logs[-100:] 
                     if log['steps_per_sec'] > 0]
    avg_speed = np.mean(recent_speeds) if recent_speeds else 0
    
    # Calculate warmup progress
    warmup_progress = min(100, (current['step'] / WARMUP_STEPS) * 100)
    
    # Calculate time elapsed
    if len(train_logs) > 1:
        time_elapsed = current['timestamp'] - train_logs[0]['timestamp']
        hours_elapsed = time_elapsed / 3600
    else:
        hours_elapsed = 0
    
    # Get gradient stats
    recent_grads = [log.get('grad_norm') for log in train_logs[-100:] 
                    if log.get('grad_norm') is not None]
    avg_grad_norm = np.mean(recent_grads) if recent_grads else None
    current_grad_norm = current.get('grad_norm')
    
    # Get best validation metrics
    best_eval = None
    if eval_logs:
        best_eval = min(eval_logs, key=lambda x: x.get('eval_loss', float('inf')))
    
    # Get latest validation metrics
    latest_eval = eval_logs[-1] if eval_logs else None
    
    return jsonify({
        'current_step': current['step'],
        'total_steps': TOTAL_STEPS,
        'progress': round(progress, 2),
        'loss': round(current['loss'], 4),
        'learning_rate': f"{current['learning_rate']:.2e}",
        'lr_raw': current['learning_rate'],
        'speed': round(current['steps_per_sec'], 4),
        'avg_speed': round(avg_speed, 4),
        'eta_hours': round(current['eta_seconds'] / 3600, 1),
        'eta_days': round(current['eta_seconds'] / 86400, 1),
        'warmup_progress': round(warmup_progress, 1),
        'warmup_complete': current['step'] >= WARMUP_STEPS,
        'hours_elapsed': round(hours_elapsed, 1),
        'last_update': datetime.fromtimestamp(current['timestamp']).strftime('%Y-%m-%d %H:%M:%S'),
        'total_logs': len(train_logs),
        'current_grad_norm': round(current_grad_norm, 4) if current_grad_norm else None,
        'avg_grad_norm': round(avg_grad_norm, 4) if avg_grad_norm else None,
        'aux_loss': round(current.get('aux_loss'), 4) if current.get('aux_loss') else None,
        'best_eval_loss': round(best_eval['eval_loss'], 4) if best_eval else None,
        'best_eval_step': best_eval['step'] if best_eval else None,
        'best_eval_perplexity': round(best_eval['eval_perplexity'], 2) if best_eval else None,
        'latest_eval_loss': round(latest_eval['eval_loss'], 4) if latest_eval else None,
        'latest_eval_perplexity': round(latest_eval['eval_perplexity'], 2) if latest_eval else None,
        'latest_eval_step': latest_eval['step'] if latest_eval else None,
    })

@app.route('/api/plots')
def get_plots():
    """API endpoint for plot images"""
    train_logs = load_current_run(TRAIN_LOG_FILE)
    eval_logs = load_current_run(EVAL_LOG_FILE)
    
    if not train_logs:
        return jsonify({'error': 'No logs found'}), 404
    
    # Extract training data
    steps = [log['step'] for log in train_logs]
    losses = [log['loss'] for log in train_logs]
    lrs = [log['learning_rate'] for log in train_logs]
    speeds = [log['steps_per_sec'] for log in train_logs if log['steps_per_sec'] > 0]
    speed_steps = [log['step'] for log in train_logs if log['steps_per_sec'] > 0]
    etas = [log['eta_seconds'] / 3600 for log in train_logs if log['eta_seconds'] > 0]
    eta_steps = [log['step'] for log in train_logs if log['eta_seconds'] > 0]
    
    grad_norms = [log.get('grad_norm') for log in train_logs if log.get('grad_norm') is not None]
    grad_steps = [log['step'] for log in train_logs if log.get('grad_norm') is not None]
    
    # Extract validation data
    eval_steps = [log['step'] for log in eval_logs]
    eval_losses = [log['eval_loss'] for log in eval_logs]
    eval_ppls = [log['eval_perplexity'] for log in eval_logs]
    
    # Get checkpoint markers
    checkpoints = get_checkpoints()
    checkpoint_markers = [{'x': cp['step'], 'color': 'green', 'style': ':'} 
                          for cp in checkpoints[:10]]  # Last 10 checkpoints
    
    # Calculate moving average window
    ma_window = min(50, len(losses) // 10) if len(losses) > 10 else None
    
    # Create plots
    plots = {}
    
    # Loss plot with checkpoint markers
    plots['loss'] = create_plot(
        steps, losses, 
        'Training Loss', 'Step', 'Loss', 
        '#2E86AB', ma_window, markers=checkpoint_markers
    )
    
    # Validation loss plot
    if eval_losses:
        plots['val_loss'] = create_plot(
            eval_steps, eval_losses,
            'Validation Loss', 'Step', 'Validation Loss',
            '#C73E1D', markers=checkpoint_markers
        )
    
    # Validation perplexity plot
    if eval_ppls:
        plots['val_ppl'] = create_plot(
            eval_steps, eval_ppls,
            'Validation Perplexity', 'Step', 'Perplexity',
            '#A23B72', markers=checkpoint_markers
        )
    
    # Learning rate plot with reference lines
    lr_extra_lines = []
    if max(steps) < WARMUP_STEPS:
        lr_extra_lines.append({
            'type': 'vline', 'x': WARMUP_STEPS, 
            'color': 'red', 'style': '--',
            'label': f'Warmup End (step {WARMUP_STEPS})'
        })
        lr_extra_lines.append({
            'type': 'hline', 'y': MAX_LR, 
            'color': 'green', 'style': '--',
            'label': f'Max LR ({MAX_LR:.0e})'
        })
    
    plots['lr'] = create_plot(
        steps, lrs, 
        'Learning Rate Schedule', 'Step', 'Learning Rate', 
        '#F18F01', extra_lines=lr_extra_lines if lr_extra_lines else None,
        markers=checkpoint_markers
    )
    
    # Gradient norm plot
    if grad_norms:
        plots['grad_norm'] = create_plot(
            grad_steps, grad_norms,
            'Gradient Norm', 'Step', 'Gradient Norm',
            '#06A77D', markers=checkpoint_markers
        )
    
    # Speed plot with average line
    speed_extra_lines = None
    if speeds:
        avg_speed = np.mean(speeds[-100:]) if len(speeds) > 100 else np.mean(speeds)
        speed_extra_lines = [{
            'type': 'hline', 'y': avg_speed, 
            'color': 'red', 'style': '--',
            'label': f'Recent Avg: {avg_speed:.4f} steps/s'
        }]
    
    plots['speed'] = create_plot(
        speed_steps, speeds, 
        'Training Speed', 'Step', 'Steps/Second', 
        '#06A77D', extra_lines=speed_extra_lines,
        markers=checkpoint_markers
    )
    
    # ETA plot
    plots['eta'] = create_plot(
        eta_steps, etas, 
        'ETA to Completion', 'Step', 'Hours Remaining', 
        '#C73E1D', markers=checkpoint_markers
    )
    
    return jsonify(plots)

@app.route('/api/checkpoints')
def get_checkpoints_api():
    """API endpoint for checkpoint information"""
    checkpoints = get_checkpoints()
    
    # Find best checkpoint
    best_checkpoint = None
    if checkpoints:
        valid_checkpoints = [cp for cp in checkpoints if cp['eval_loss'] is not None]
        if valid_checkpoints:
            best_checkpoint = min(valid_checkpoints, key=lambda x: x['eval_loss'])
    
    return jsonify({
        'checkpoints': checkpoints[:20],  # Last 20 checkpoints
        'best_checkpoint': best_checkpoint,
        'total_checkpoints': len(checkpoints)
    })

@app.route('/api/events')
def get_events():
    """API endpoint for event timeline"""
    train_logs = load_current_run(TRAIN_LOG_FILE)
    eval_logs = load_current_run(EVAL_LOG_FILE)
    
    events = []
    
    # Add evaluation events
    for log in eval_logs[-50:]:  # Last 50 eval events
        events.append({
            'step': log['step'],
            'timestamp': log['timestamp'],
            'type': 'evaluation',
            'message': f"Validation: Loss={log['eval_loss']:.4f}, PPL={log.get('eval_perplexity', 0):.2f}",
            'icon': '🧪'
        })
    
    # Add checkpoint events
    checkpoints = get_checkpoints()
    for cp in checkpoints[:20]:
        if cp['timestamp']:
            events.append({
                'step': cp['step'],
                'timestamp': cp['timestamp'],
                'type': 'checkpoint',
                'message': f"Checkpoint saved: {cp['name']}",
                'icon': '💾'
            })
    
    # Add best checkpoint marker
    if checkpoints:
        valid_checkpoints = [cp for cp in checkpoints if cp['eval_loss'] is not None]
        if valid_checkpoints:
            best = min(valid_checkpoints, key=lambda x: x['eval_loss'])
            events.append({
                'step': best['step'],
                'timestamp': best['timestamp'],
                'type': 'best_checkpoint',
                'message': f"✅ Best checkpoint: {best['name']} (loss={best['eval_loss']:.4f})",
                'icon': '🏆'
            })
    
    # Check for NaN losses in recent logs
    for log in train_logs[-100:]:
        if log.get('loss') and (np.isnan(log['loss']) or log['loss'] > 1000):
            events.append({
                'step': log['step'],
                'timestamp': log['timestamp'],
                'type': 'warning',
                'message': f"⚠️ Unusual loss detected: {log['loss']:.4f}",
                'icon': '⚠️'
            })
    
    # Sort by timestamp (most recent first)
    events.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return jsonify({
        'events': events[:50],  # Last 50 events
        'total_events': len(events)
    })

@app.route('/api/gpu')
def get_gpu_info():
    """API endpoint for GPU statistics"""
    gpu_stats = get_gpu_stats()
    
    if gpu_stats is None:
        return jsonify({'available': False, 'message': 'GPU stats unavailable'})
    
    return jsonify({
        'available': True,
        'gpus': gpu_stats,
        'total_gpus': len(gpu_stats)
    })

@app.route('/api/errors')
def get_errors():
    """API endpoint for error logs"""
    error_logs = load_jsonl(ERROR_LOG_FILE)
    
    if not error_logs:
        return jsonify({
            'has_errors': False,
            'errors': [],
            'total_errors': 0
        })
    
    # Format errors for display
    formatted_errors = []
    for error in error_logs[-50:]:  # Last 50 errors
        formatted_errors.append({
            'timestamp': error.get('timestamp'),
            'datetime': error.get('datetime'),
            'error_type': error.get('error_type'),
            'error_message': error.get('error_message'),
            'step': error.get('step'),
            'epoch': error.get('epoch'),
            'traceback': error.get('traceback'),
            'severity': 'critical' if error.get('error_type') in ['NaNLoss', 'ValueError', 'RuntimeError'] else 'warning'
        })
    
    # Sort by timestamp (most recent first)
    formatted_errors.sort(key=lambda x: x['timestamp'] if x['timestamp'] else 0, reverse=True)
    
    return jsonify({
        'has_errors': True,
        'errors': formatted_errors,
        'total_errors': len(error_logs),
        'recent_errors': len(formatted_errors)
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    # Check if log files exist
    if not Path(TRAIN_LOG_FILE).exists():
        logger.warning(f"Training log not found: {TRAIN_LOG_FILE}")
        logger.warning("Dashboard will start but show no data until training begins")
    
    logger.info("="*60)
    logger.info("🚀 Starting Enhanced Training Dashboard")
    logger.info("="*60)
    logger.info(f"Training log: {TRAIN_LOG_FILE}")
    logger.info(f"Eval log: {EVAL_LOG_FILE}")
    logger.info(f"Expert log: {EXPERT_LOG_FILE}")
    logger.info(f"Checkpoint dir: {CHECKPOINT_DIR}")
    logger.info(f"Refresh interval: {REFRESH_INTERVAL} seconds")
    logger.info(f"Access dashboard at: http://0.0.0.0:5000")
    logger.info("="*60)
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
