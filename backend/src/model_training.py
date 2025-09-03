import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import time
from config import *
import warnings
warnings.filterwarnings('ignore')

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2  # 20% for testing, 80% for training

# Set GPU memory growth to avoid allocation issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU setup error: {e}")

class CNNRNNStressModelTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.history = None
        self.feature_names = None
        self.training_start_time = None
        self.training_history = {}
        
    def load_and_prepare_data(self, features_path="data/processed/features.csv"):
        """Load and prepare dataset for CNN+RNN training"""
        print("📊 Loading dataset for CNN+RNN training...")
        
        # Check if file exists
        if not Path(features_path).exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        
        df = pd.read_csv(features_path)
        
        # Basic validation
        if df.empty:
            raise ValueError("Dataset is empty")
        
        if 'stress_level' not in df.columns:
            raise ValueError("'stress_level' column not found in dataset")
        
        # Separate features and labels
        X = df.drop('stress_level', axis=1)
        y = df['stress_level']
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   📈 Dataset shape: {df.shape}")
        print(f"   🏷️  Features: {X.shape[1]}")
        print(f"   📋 Samples per class:")
        
        class_counts = y.value_counts()
        for stress_level, count in class_counts.items():
            print(f"      {stress_level}: {count} samples")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        print(f"   🔢 Classes: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # Handle missing values
        if X.isnull().any().any():
            print("⚠️  Found missing values, filling with median...")
            X = X.fillna(X.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # **Key difference for CNN+RNN: Reshape data for 1D convolution**
        # CNN expects 3D input: (samples, timesteps, features)
        # We treat each feature as a timestep with 1 feature channel
        X_reshaped = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))
        
        print(f"  🔄 Reshaped for CNN+RNN: {X_reshaped.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_reshaped, y_encoded,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE ,
            stratify=y_encoded
        )
        
        print(f"   📊 Training set: {X_train.shape}")
        print(f"   📊 Test set: {X_test.shape}")
        print(f"   ✅ Data preparation completed!\n")
        
        return X_train, X_test, y_train, y_test
    
    def build_cnn_rnn_model(self, input_shape, num_classes):
        """Build optimized CNN+RNN architecture for voice stress detection"""
        print(f"🏗️  Building CNN+RNN model...")
        print(f"   📐 Input shape: {input_shape}")
        print(f"   🎯 Number of classes: {num_classes}")
        
        model = Sequential([
            # First Convolutional Block
            Conv1D(filters=64, kernel_size=3, activation='relu', 
                   padding='same', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            # Second Convolutional Block
            Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            
            # Third Convolutional Block
            Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            Dropout(0.4),
            
            # First LSTM Layer
            LSTM(units=128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
            
            # Second LSTM Layer
            LSTM(units=64, return_sequences=False, dropout=0.3, recurrent_dropout=0.3),
            
            # Dense Layers
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            
            # Output Layer
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile with optimized parameters
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Display model architecture
        model.summary()
        
        return model
    
    def create_callbacks(self):
        """Create training callbacks for optimization"""
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Save best model
            ModelCheckpoint(
                MODELS_DIR / 'cnn_rnn_voice_stress_model.h5',
                monitor='val_f1_score',
                save_best_only=True,
                verbose=1,
                mode='max'
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_model(self, X_train, X_test, y_train, y_test):
        """Train the CNN+RNN model"""
        print("🚀 TRAINING CNN+RNN MODEL FOR VOICE STRESS DETECTION")
        print("=" * 60)
        
        self.training_start_time = time.time()
        
        # Build model
        self.model = self.build_cnn_rnn_model(
            input_shape=X_train.shape[1:], 
            num_classes=len(np.unique(y_train))
        )
        
        # Create callbacks
        callbacks = self.create_callbacks()
        
        print(f"\n🎯 Starting training...")
        print(f"   📊 Training samples: {X_train.shape[0]}")
        print(f"   📊 Validation samples: {X_test.shape[0]}")
        print(f"   ⚙️  Optimizer: Adam (lr=0.001)")
        print(f"   📦 Batch size: 32")
        print(f"   🔄 Max epochs: 150")
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=150,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate training time
        training_time = time.time() - self.training_start_time
        
        # Store training history
        self.training_history = {
            'training_time': training_time,
            'total_epochs': len(self.history.history['loss']),
            'best_val_accuracy': max(self.history.history['val_accuracy']),
            'best_val_loss': min(self.history.history['val_loss']),
            'model_params': self.model.count_params()
        }
        
        print(f"\n Training completed!")
        print(f"    Training time: {training_time:.2f} seconds ({training_time/60:.1f} minutes)")
        print(f"    Total epochs: {self.training_history['total_epochs']}")
        print(f"    Best validation accuracy: {self.training_history['best_val_accuracy']:.4f}")
        print(f"    Best validation loss: {self.training_history['best_val_loss']:.4f}")
        print(f"    Model parameters: {self.training_history['model_params']:,}")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """Comprehensive model evaluation"""
        print("\n🔍 MODEL EVALUATION")
        print("=" * 50)
        
        # Make predictions
        print("Making predictions...")
        y_pred_proba = self.model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            X_test, y_test, verbose=0
        )
        
        # Calculate F1 scores
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        # Get class names
        class_names = self.label_encoder.classes_
        
        # Store results
        results = {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_per_class': dict(zip(class_names, f1_per_class)),
            'class_names': class_names
        }
        
        # Print results
        print(f"🎯 PERFORMANCE METRICS:")
        print(f"   📊 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"   📊 Test Loss: {test_loss:.4f}")
        print(f"   📊 Test Precision: {test_precision:.4f}")
        print(f"   📊 Test Recall: {test_recall:.4f}")
        print(f"   📊 F1 Score (Macro): {f1_macro:.4f}")
        print(f"   📊 F1 Score (Weighted): {f1_weighted:.4f}")
        
        print(f"\n📋 F1 SCORES BY CLASS:")
        for class_name, f1 in zip(class_names, f1_per_class):
            print(f"   {class_name:15}: {f1:.4f}")
        
        # Detailed classification report
        report = classification_report(y_test, y_pred, target_names=class_names)
        print(f"\n📄 DETAILED CLASSIFICATION REPORT:")
        print(report)
        
        # Generate visualizations
        self.plot_training_history()
        self.plot_confusion_matrix(y_test, y_pred, class_names)
        self.plot_prediction_confidence(y_pred_proba, class_names)
        
        return results
    
    def plot_training_history(self):
        """Plot training history"""
        if not self.history:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0,0].plot(self.history.history['accuracy'], label='Training Accuracy', color='blue')
        axes[0,0].plot(self.history.history['val_accuracy'], label='Validation Accuracy', color='red')
        axes[0,0].set_title('Model Accuracy Over Time')
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Loss
        axes[0,1].plot(self.history.history['loss'], label='Training Loss', color='blue')
        axes[0,1].plot(self.history.history['val_loss'], label='Validation Loss', color='red')
        axes[0,1].set_title('Model Loss Over Time')
        axes[0,1].set_xlabel('Epoch')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Precision
        axes[1,0].plot(self.history.history['precision'], label='Training Precision', color='green')
        axes[1,0].plot(self.history.history['val_precision'], label='Validation Precision', color='orange')
        axes[1,0].set_title('Model Precision Over Time')
        axes[1,0].set_xlabel('Epoch')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Recall
        axes[1,1].plot(self.history.history['recall'], label='Training Recall', color='purple')
        axes[1,1].plot(self.history.history['val_recall'], label='Validation Recall', color='brown')
        axes[1,1].set_title('Model Recall Over Time')
        axes[1,1].set_xlabel('Epoch')
        axes[1,1].set_ylabel('Recall')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(MODELS_DIR / 'training_history.png', dpi=300, bbox_inches='tight')
        print(f"✅ Training history plot saved: {MODELS_DIR / 'training_history.png'}")
        plt.show()
    
    def plot_confusion_matrix(self, y_test, y_pred, class_names):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[name.replace('_', ' ').title() for name in class_names],
                   yticklabels=[name.replace('_', ' ').title() for name in class_names])
        
        plt.title('Confusion Matrix - CNN+RNN Voice Stress Detection', fontsize=16, fontweight='bold')
        plt.ylabel('True Stress Level', fontsize=12)
        plt.xlabel('Predicted Stress Level', fontsize=12)
        
        # Calculate and display accuracy
        accuracy = np.trace(cm) / np.sum(cm)
        plt.figtext(0.9, 0.02, f'Overall Accuracy: {accuracy:.3f}', 
                   ha='right', fontsize=11, style='italic')
        
        plt.tight_layout()
        plt.savefig(MODELS_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved: {MODELS_DIR / 'confusion_matrix.png'}")
        plt.show()
    
    def plot_prediction_confidence(self, y_pred_proba, class_names):
        """Plot prediction confidence distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, class_name in enumerate(class_names):
            confidences = y_pred_proba[:, i]
            
            axes[i].hist(confidences, bins=30, alpha=0.7, color=f'C{i}', edgecolor='black')
            axes[i].set_title(f'Prediction Confidence: {class_name.replace("_", " ").title()}', 
                             fontweight='bold')
            axes[i].set_xlabel('Confidence Score')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
            
            # Add mean line
            mean_conf = np.mean(confidences)
            axes[i].axvline(mean_conf, color='red', linestyle='--', 
                           label=f'Mean: {mean_conf:.3f}')
            axes[i].legend()
        
        plt.suptitle('Model Confidence Distribution by Class\n(Higher values = more confident predictions)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(MODELS_DIR / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✅ Confidence distribution saved: {MODELS_DIR / 'confidence_distribution.png'}")
        plt.show()
    
    def save_model(self):
        """Save trained model and components"""
        print(f"\n💾 SAVING CNN+RNN MODEL")
        print("=" * 50)
        
        # Ensure models directory exists
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save main model
            model_path = MODELS_DIR / 'cnn_rnn_voice_stress_model.h5'
            self.model.save(model_path)
            print(f"✅ CNN+RNN model saved: {model_path}")
            
            # Save preprocessors
            scaler_path = MODELS_DIR / 'scaler.pkl'
            joblib.dump(self.scaler, scaler_path)
            print(f"✅ Feature scaler saved: {scaler_path}")
            
            label_encoder_path = MODELS_DIR / 'label_encoder.pkl'
            joblib.dump(self.label_encoder, label_encoder_path)
            print(f"✅ Label encoder saved: {label_encoder_path}")
            
            # Save feature names
            features_path = MODELS_DIR / 'feature_names.pkl'
            joblib.dump(self.feature_names, features_path)
            print(f"✅ Feature names saved: {features_path}")
            
            # Save training history
            history_path = MODELS_DIR / 'training_history.pkl'
            joblib.dump(self.training_history, history_path)
            print(f"✅ Training history saved: {history_path}")
            
            # Save model metadata
            metadata = {
                'model_type': 'CNN+RNN (Conv1D + LSTM)',
                'architecture': 'CNN (3 layers) + LSTM (2 layers) + Dense (2 layers)',
                'n_features': len(self.feature_names),
                'n_classes': len(self.label_encoder.classes_),
                'classes': list(self.label_encoder.classes_),
                'input_shape': f"({len(self.feature_names)}, 1)",
                'total_parameters': self.training_history['model_params'],
                'training_time_seconds': self.training_history['training_time'],
                'training_epochs': self.training_history['total_epochs'],
                'best_validation_accuracy': self.training_history['best_val_accuracy'],
                'tensorflow_version': tf.__version__,
                'training_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            metadata_path = MODELS_DIR / 'model_metadata.pkl'
            joblib.dump(metadata, metadata_path)
            print(f"✅ Model metadata saved: {metadata_path}")
            
            # Calculate total size
            total_size = sum(f.stat().st_size for f in MODELS_DIR.glob('*') if f.is_file())
            print(f"📊 Total model size: {total_size / (1024*1024):.2f} MB")
            
            print(f"\n🎉 All CNN+RNN model components saved in: {MODELS_DIR}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            return False

def main():
    """Main training pipeline for CNN+RNN model"""
    print("🚀 VOICE STRESS LEVEL ESTIMATOR - CNN+RNN TRAINING")
    print("=" * 70)
    print("🧠 Training Combined CNN + LSTM Model for Voice Stress Detection")
    print("=" * 70)
    
    # Check for GPU availability
    if tf.config.list_physical_devices('GPU'):
        print("🚀 GPU detected and available for training!")
    else:
        print("⚠️  Using CPU for training (GPU recommended for faster training)")
    
    trainer = CNNRNNStressModelTrainer()
    
    try:
        # Load and prepare data
        X_train, X_test, y_train, y_test = trainer.load_and_prepare_data()
        
        # Train model
        model = trainer.train_model(X_train, X_test, y_train, y_test)
        
        # Evaluate model
        results = trainer.evaluate_model(X_test, y_test)
        
        # Save model
        save_success = trainer.save_model()
        
        # Final summary
        print(f"\n🏁 CNN+RNN TRAINING COMPLETED!")
        print("=" * 70)
        print("🎯 FINAL MODEL PERFORMANCE:")
        print(f"   📊 Test Accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
        print(f"   📊 F1 Score (Macro): {results['f1_macro']:.4f}")
        print(f"   📊 F1 Score (Weighted): {results['f1_weighted']:.4f}")
        print(f"   ⏱️  Training Time: {trainer.training_history['training_time']:.1f} seconds")
        print(f"   🔧 Model Parameters: {trainer.training_history['model_params']:,}")
        print(f"   📁 Model Files: {MODELS_DIR}")
        
        # Performance assessment
        if results['test_accuracy'] >= 0.90:
            print("🌟 EXCELLENT: Outstanding model performance!")
        elif results['test_accuracy'] >= 0.85:
            print("✅ VERY GOOD: Excellent model performance!")
        elif results['test_accuracy'] >= 0.80:
            print("✅ GOOD: Good model performance for production use")
        elif results['test_accuracy'] >= 0.70:
            print("⚠️  FAIR: Acceptable performance, consider more data or tuning")
        else:
            print("❌ POOR: Consider collecting more data or adjusting architecture")
        
        print("\n🚀 CNN+RNN model ready for deployment!")
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        print("\n🔍 Troubleshooting checklist:")
        print("   1. ✓ Features extracted: python src/feature_extraction.py")
        print("   2. ✓ TensorFlow installed: pip install tensorflow")
        print("   3. ✓ Sufficient training data (400+ samples recommended)")
        print("   4. ✓ GPU drivers (optional but recommended)")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Ready for stress detection with CNN+RNN model!")
    else:
        print("\n💡 Check the error messages above and retry")
        