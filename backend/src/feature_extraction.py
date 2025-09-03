import librosa
import numpy as np
import pandas as pd # pyrigtht: ignore[reportMissingModuleSource]
import os
from langdetect import detect
from scipy import stats

class FeatureExtractor:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
    
    def extract_mfcc_features(self, audio, n_mfcc=13):
        """Extract MFCC features"""
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=n_mfcc)
        
        # Statistical features
        mfcc_features = []
        for i in range(n_mfcc):
            mfcc_features.extend([
                np.mean(mfccs[i]),
                np.std(mfccs[i]),
                np.min(mfccs[i]),
                np.max(mfccs[i])
            ])
        
        return mfcc_features
    
    def extract_pitch_features(self, audio):
        """Extract pitch-related features"""
        # Fundamental frequency
        f0, voiced_flag, voiced_probs = librosa.pyin(audio, 
                                                   fmin=librosa.note_to_hz('C2'), 
                                                   fmax=librosa.note_to_hz('C7'))
        
        # Remove NaN values
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) > 0:
            pitch_features = [
                np.mean(f0_clean),
                np.std(f0_clean),
                np.min(f0_clean),
                np.max(f0_clean),
                np.median(f0_clean),
                stats.skew(f0_clean),
                stats.kurtosis(f0_clean)
            ]
        else:
            pitch_features = [0] * 7
        
        return pitch_features
    
    def extract_spectral_features(self, audio):
        """Extract spectral features"""
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        
        spectral_features = [
            np.mean(spectral_centroid),
            np.std(spectral_centroid),
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff),
            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth),
            np.mean(zcr),
            np.std(zcr)
        ]
        
        return spectral_features
    
    def extract_energy_features(self, audio):
        """Extract energy-related features"""
        # RMS energy
        rms = librosa.feature.rms(y=audio)
        
        # Short-time energy
        frame_length = int(0.025 * self.sample_rate)  # 25ms frames
        hop_length = int(0.01 * self.sample_rate)     # 10ms hop
        
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        energy = np.sum(frames ** 2, axis=0)
        
        energy_features = [
            np.mean(rms),
            np.std(rms),
            np.mean(energy),
            np.std(energy),
            np.max(energy),
            np.min(energy)
        ]
        
        return energy_features
    
    def extract_all_features(self, audio, language):
        """Extract all features from audio"""
        features = []
        
        # MFCC features
        mfcc_feats = self.extract_mfcc_features(audio)
        features.extend(mfcc_feats)
        
        # Pitch features
        pitch_feats = self.extract_pitch_features(audio)
        features.extend(pitch_feats)
        
        # Spectral features
        spectral_feats = self.extract_spectral_features(audio)
        features.extend(spectral_feats)
        
        # Energy features
        energy_feats = self.extract_energy_features(audio)
        features.extend(energy_feats)
        
        # Language encoding (one-hot)
        lang_features = [1, 0] if language == 'english' else [0, 1]
        features.extend(lang_features)
        
        return features
    
    def create_feature_names(self):
        """Create feature names for interpretation"""
        names = []
        
        # MFCC feature names
        for i in range(13):
            names.extend([f'mfcc_{i}_mean', f'mfcc_{i}_std', f'mfcc_{i}_min', f'mfcc_{i}_max'])
        
        # Pitch feature names
        names.extend(['pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max', 
                     'pitch_median', 'pitch_skew', 'pitch_kurtosis'])
        
        # Spectral feature names
        names.extend(['spectral_centroid_mean', 'spectral_centroid_std',
                     'spectral_rolloff_mean', 'spectral_rolloff_std',
                     'spectral_bandwidth_mean', 'spectral_bandwidth_std',
                     'zcr_mean', 'zcr_std'])
        
        # Energy feature names
        names.extend(['rms_mean', 'rms_std', 'energy_mean', 'energy_std',
                     'energy_max', 'energy_min'])
        
        # Language features
        names.extend(['lang_english', 'lang_hindi'])
        
        return names

def process_dataset_features():
    """Process entire dataset to extract features"""
    # Load preprocessed dataset
    dataset = pd.read_pickle("data/processed/dataset.pkl")
    
    extractor = FeatureExtractor()
    feature_names = extractor.create_feature_names()
    
    features_list = []
    labels = []
    
    for idx, row in dataset.iterrows():
        print(f"Processing {idx+1}/{len(dataset)}")
        
        audio_data = row['audio_data']
        language = row['language']
        stress_level = row['stress_level']
        
        # Extract features
        features = extractor.extract_all_features(audio_data, language)
        features_list.append(features)
        labels.append(stress_level)
    
    # Create feature DataFrame
    features_df = pd.DataFrame(features_list, columns=feature_names)
    features_df['stress_level'] = labels
    
    # Save features
    features_df.to_csv("data/processed/features.csv", index=False)
    print(f"Features extracted and saved: {features_df.shape}")
    
    return features_df

if __name__ == "__main__":
    features_df = process_dataset_features()
    print(features_df.head())
