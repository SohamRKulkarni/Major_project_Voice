import librosa
import numpy as np
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import os
import soundfile as sf
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class AudioPreprocessor:
    def __init__(self, sample_rate=22050, duration=5):
        self.sample_rate = sample_rate
        self.duration = duration
        self.scaler = StandardScaler()
        
    def load_audio(self, file_path):
        """Load and preprocess audio file"""
        try:
            # Load audio file
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
            
            # Normalize audio
            audio = librosa.util.normalize(audio)
            
            # Pad or trim to fixed length
            target_length = self.sample_rate * self.duration
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            else:
                audio = audio[:target_length]
                
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def remove_silence(self, audio, threshold=0.01):
        """Remove silence from audio"""
        # Detect non-silent parts
        intervals = librosa.effects.split(audio, top_db=20)
        
        if len(intervals) > 0:
            # Concatenate non-silent parts
            audio_no_silence = []
            for interval in intervals:
                audio_no_silence.extend(audio[interval[0]:interval[1]])
            return np.array(audio_no_silence)
        
        return audio
    
    def apply_noise_reduction(self, audio):
        """Apply basic noise reduction"""
        # Simple spectral gating
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        # Noise gate
        noise_gate = np.percentile(magnitude, 20)
        magnitude = np.where(magnitude < noise_gate, magnitude * 0.1, magnitude)
        
        # Reconstruct audio
        phase = np.angle(stft)
        stft_clean = magnitude * np.exp(1j * phase)
        audio_clean = librosa.istft(stft_clean)
        
        return audio_clean

# Example usage for data preparation
def create_dataset():
    """Create dataset from audio files"""
    # This is a template - you'll need to replace with your actual data paths
    data = []
    
    # Example structure - replace with your data organization
    languages = ['english', 'hindi']
    stress_levels = ['no_stress', 'low_stress', 'medium_stress', 'high_stress']
    
    preprocessor = AudioPreprocessor()
    
    for lang in languages:
        for stress in stress_levels:
            folder_path = f"data/raw/{lang}/{stress}/"
            
            if os.path.exists(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(('.wav', '.mp3')):
                        file_path = os.path.join(folder_path, file_name)
                        audio = preprocessor.load_audio(file_path)
                        
                        if audio is not None:
                            # Apply preprocessing
                            audio_clean = preprocessor.apply_noise_reduction(audio)
                            audio_clean = preprocessor.remove_silence(audio_clean)
                            
                            data.append({
                                'file_path': file_path,
                                'language': lang,
                                'stress_level': stress,
                                'audio_data': audio_clean
                            })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Create and save processed dataset
    dataset = create_dataset()
    dataset.to_pickle("data/processed/dataset.pkl")
    print(f"Processed {len(dataset)} audio files")
