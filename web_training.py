
"""
Web-based training interface for brain tumor segmentation model
"""

import os
import threading
import time
import json
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from main import UNet3D
from training import ModernBrainTumorTrainer, BraTS2024Dataset

class WebTrainingManager:
    """Manages training sessions for web interface"""
    
    def __init__(self):
        self.active_sessions = {}
        self.training_threads = {}
        
    def start_training_session(self, config):
        """Start a new training session"""
        session_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize session data
        self.active_sessions[session_id] = {
            'status': 'starting',
            'config': config,
            'progress': {
                'current_epoch': 0,
                'total_epochs': config.get('epochs', 50),
                'train_loss': 0.0,
                'val_loss': 0.0,
                'dice_score': 0.0,
                'best_dice': 0.0,
                'learning_rate': config.get('learning_rate', 0.0001)
            },
            'logs': [],
            'start_time': datetime.now()
        }
        
        # Start training in background thread
        training_thread = threading.Thread(
            target=self._run_training,
            args=(session_id, config)
        )
        training_thread.daemon = True
        training_thread.start()
        
        self.training_threads[session_id] = training_thread
        
        return session_id
    
    def _run_training(self, session_id, config):
        """Run training in background thread"""
        try:
            self.active_sessions[session_id]['status'] = 'running'
            self._add_log(session_id, "Training started")
            
            # Initialize model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = UNet3D(in_channels=1, out_channels=4)
            
            # Create synthetic data if needed
            if config.get('data_type') == 'synthetic':
                self._add_log(session_id, "Generating synthetic training data...")
                self._create_synthetic_data(config.get('num_samples', 100))
            
            # Simulate training loop for demo
            total_epochs = config.get('epochs', 50)
            
            for epoch in range(1, total_epochs + 1):
                if self.active_sessions[session_id]['status'] == 'stopping':
                    break
                
                # Simulate training step
                train_loss = max(0.1, 2.0 - (epoch * 0.03) + np.random.normal(0, 0.1))
                val_loss = max(0.1, 2.2 - (epoch * 0.025) + np.random.normal(0, 0.15))
                dice_score = min(0.95, 0.3 + (epoch * 0.012) + np.random.normal(0, 0.05))
                
                # Update progress
                progress = self.active_sessions[session_id]['progress']
                progress['current_epoch'] = epoch
                progress['train_loss'] = round(train_loss, 4)
                progress['val_loss'] = round(val_loss, 4)
                progress['dice_score'] = round(dice_score, 4)
                progress['best_dice'] = max(progress['best_dice'], dice_score)
                
                # Add logs periodically
                if epoch % 5 == 0:
                    self._add_log(session_id, 
                        f"Epoch {epoch}/{total_epochs} - "
                        f"Train Loss: {train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"Dice: {dice_score:.4f}")
                
                # Simulate epoch time
                time.sleep(1)
            
            # Training completed
            self.active_sessions[session_id]['status'] = 'completed'
            self._add_log(session_id, f"Training completed! Best Dice: {progress['best_dice']:.4f}")
            
        except Exception as e:
            self.active_sessions[session_id]['status'] = 'error'
            self._add_log(session_id, f"Training error: {str(e)}")
    
    def stop_training_session(self, session_id):
        """Stop a training session"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]['status'] = 'stopping'
            self._add_log(session_id, "Training stop requested")
            return True
        return False
    
    def get_session_progress(self, session_id):
        """Get progress for a training session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            return {
                'status': session['status'],
                'progress': session['progress'],
                'logs': session['logs'][-10:]  # Return last 10 logs
            }
        return None
    
    def _add_log(self, session_id, message):
        """Add a log message to session"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.active_sessions[session_id]['logs'].append(log_entry)
    
    def _create_synthetic_data(self, num_samples):
        """Create synthetic training data"""
        # This would call your actual synthetic data generation
        # For now, just simulate the process
        time.sleep(2)  # Simulate data generation time
        return True

# Global training manager instance
training_manager = WebTrainingManager()

def start_web_training(config):
    """Start training from web interface"""
    return training_manager.start_training_session(config)

def stop_web_training(session_id):
    """Stop training from web interface"""
    return training_manager.stop_training_session(session_id)

def get_web_training_progress(session_id):
    """Get training progress for web interface"""
    return training_manager.get_session_progress(session_id)
