# backend/src/dataset_loader.py
from datasets import load_dataset
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import os
from tqdm import tqdm
import json
import sys
sys.path.append('.')
from config import *
import warnings
warnings.filterwarnings('ignore')

class HuggingFaceDatasetLoader:
    """Load and process Hugging Face datasets for voice stress detection"""
    
    def __init__(self):
        self.datasets = {}
        self.processed_stats = {'total': 0, 'english': 0, 'hindi': 0}
        
    def step1_load_datasets(self):
        """Step 1: Load all Hugging Face datasets"""
        print("🚀 STEP 1: Loading Hugging Face Datasets")
        print("=" * 60)
        
        # Load Stress-17K dataset
        try:
            print("📥 Loading slprl/Stress-17K-raw...")
            self.datasets['stress_17k'] = load_dataset("slprl/Stress-17K-raw")
            print(f"✅ Stress-17K loaded: {len(self.datasets['stress_17k']['train'])} samples")
        except Exception as e:
            print(f"❌ Failed to load Stress-17K: {e}")
            print("💡 Try: huggingface-cli login")
            
        # Load StressTest dataset
        try:
            print("📥 Loading slprl/StressTest...")
            self.datasets['stress_test'] = load_dataset("slprl/StressTest")
            print(f"✅ StressTest loaded: {len(self.datasets['stress_test']['train'])} samples")
        except Exception as e:
            print(f"❌ Failed to load StressTest: {e}")
            
        # Load Hindi dataset
        try:
            print("📥 Loading cdactvm/hindi_dataset...")
            self.datasets['hindi'] = load_dataset("cdactvm/hindi_dataset")
            print(f"✅ Hindi dataset loaded: {len(self.datasets['hindi']['train'])} samples")
        except Exception as e:
            print(f"❌ Failed to load Hindi dataset: {e}")
        
        print(f"\n✅ Step 1 Complete: Loaded {len(self.datasets)} datasets")
        return len(self.datasets) > 0
    
    def step2_analyze_datasets(self):
        """Step 2: Analyze dataset structures"""
        print("\n🔍 STEP 2: Analyzing Dataset Structures")
        print("=" * 60)
        
        for name, dataset in self.datasets.items():
            print(f"\n📊 Analyzing {name}:")
            
            # Get sample to understand structure
            train_data = dataset['train']
            sample = train_data[0]
            
            print(f"   Sample keys: {list(sample.keys())}")
            
            # Check audio format
            if 'audio' in sample:
                audio = sample['audio']
                print(f"   Audio sample rate: {audio.get('sampling_rate', 'Unknown')}")
                print(f"   Audio array shape: {np.array(audio['array']).shape}")
            
            # Check labels
            if 'label' in sample:
                print(f"   Label type: {type(sample['label'])}")
                print(f"   Sample label: {sample['label']}")
            
            # Analyze first 10 samples for label distribution
            labels = []
            for i in range(min(10, len(train_data))):
                if 'label' in train_data[i]:
                    labels.append(train_data[i]['label'])
            
            if labels:
                print(f"   Sample labels: {set(labels)}")
        
        print("✅ Step 2 Complete: Dataset analysis done")
        return True
    
    def step3_create_label_mappings(self):
        """Step 3: Create mappings from dataset labels to stress levels"""
        print("\n🏷️  STEP 3: Creating Label Mappings")
        print("=" * 60)
        
        self.label_mappings = {}
        
        # Mapping for Stress-17K (adjust based on actual labels)
        if 'stress_17k' in self.datasets:
            # Analyze label distribution
            train_data = self.datasets['stress_17k']['train']
            labels = []
            for i in range(min(100, len(train_data))):
                if 'label' in train_data[i]:
                    labels.append(train_data[i]['label'])
            
            unique_labels = sorted(set(labels))
            print(f"Stress-17K unique labels: {unique_labels}")
            
            # Create mapping based on number of labels
            if len(unique_labels) == 2:
                self.label_mappings['stress_17k'] = {
                    unique_labels[0]: 'no_stress',
                    unique_labels[1]: 'high_stress'
                }
            elif len(unique_labels) >= 4:
                self.label_mappings['stress_17k'] = {
                    unique_labels[0]: 'no_stress',
                    unique_labels[1]: 'low_stress',
                    unique_labels[2]: 'medium_stress',
                    unique_labels[3]: 'high_stress'
                }
            else:
                # Default binary mapping
                self.label_mappings['stress_17k'] = {0: 'no_stress', 1: 'high_stress'}
        
        # Mapping for StressTest
        if 'stress_test' in self.datasets:
            # Similar analysis for StressTest
            self.label_mappings['stress_test'] = {0: 'no_stress', 1: 'medium_stress'}
        
        # For Hindi dataset (if no stress labels, distribute evenly)
        if 'hindi' in self.datasets:
            self.label_mappings['hindi'] = 'distribute_evenly'
        
        print("Label mappings created:")
        for dataset, mapping in self.label_mappings.items():
            print(f"   {dataset}: {mapping}")
        
        print("✅ Step 3 Complete: Label mappings created")
        return True
    
    def step4_process_stress_17k(self, max_samples_per_class=500):
        """Step 4: Process Stress-17K dataset"""
        if 'stress_17k' not in self.datasets:
            print("⚠️  Stress-17K dataset not available, skipping...")
            return
        
        print("\n🔄 STEP 4: Processing Stress-17K Dataset")
        print("=" * 60)
        
        dataset = self.datasets['stress_17k']['train']
        mapping = self.label_mappings.get('stress_17k', {})
        processed_count = 0
        
        print(f"Processing up to {max_samples_per_class} samples per stress class...")
        
        # Count samples per stress level to ensure balance
        stress_counts = {'no_stress': 0, 'low_stress': 0, 'medium_stress': 0, 'high_stress': 0}
        
        for i, sample in enumerate(tqdm(dataset, desc="Processing Stress-17K")):
            # Stop if we have enough samples
            if all(count >= max_samples_per_class for count in stress_counts.values()):
                break
            
            try:
                # Get audio data
                audio_data = np.array(sample['audio']['array'])
                sample_rate = sample['audio']['sampling_rate']
                
                # Map label to stress level
                label = sample.get('label', 0)
                stress_level = mapping.get(label, 'no_stress')
                
                # Skip if this stress level already has enough samples
                if stress_counts[stress_level] >= max_samples_per_class:
                    continue
                
                # Process audio
                processed_audio = self._process_audio_sample(audio_data, sample_rate)
                
                if processed_audio is not None:
                    # Save audio file
                    self._save_audio_file(processed_audio, 'english', stress_level, 
                                        f"stress17k_{i:06d}")
                    stress_counts[stress_level] += 1
                    processed_count += 1
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
        
        self.processed_stats['english'] += processed_count
        self.processed_stats['total'] += processed_count
        
        print(f"✅ Processed {processed_count} samples from Stress-17K")
        print(f"Distribution: {stress_counts}")
        return processed_count > 0
    
    def step5_process_hindi_dataset(self, max_samples_per_class=400):
        """Step 5: Process Hindi dataset"""
        if 'hindi' not in self.datasets:
            print("⚠️  Hindi dataset not available, skipping...")
            return
        
        print("\n🔄 STEP 5: Processing Hindi Dataset")
        print("=" * 60)
        
        dataset = self.datasets['hindi']['train']
        processed_count = 0
        
        # Since Hindi dataset might not have stress labels, distribute evenly
        stress_levels = ['no_stress', 'low_stress', 'medium_stress', 'high_stress']
        samples_per_level = max_samples_per_class // len(stress_levels)
        
        print(f"Processing {samples_per_level} samples per stress level...")
        
        for i, sample in enumerate(tqdm(dataset, desc="Processing Hindi")):
            if processed_count >= max_samples_per_class:
                break
            
            try:
                # Extract audio data (format varies by dataset)
                if 'audio' in sample:
                    audio_data = np.array(sample['audio']['array'])
                    sample_rate = sample['audio']['sampling_rate']
                elif 'path' in sample and os.path.exists(sample['path']):
                    audio_data, sample_rate = librosa.load(sample['path'], sr=None)
                else:
                    continue
                
                # Assign stress level cyclically for even distribution
                stress_level = stress_levels[i % len(stress_levels)]
                
                # Process audio
                processed_audio = self._process_audio_sample(audio_data, sample_rate)
                
                if processed_audio is not None:
                    # Save audio file
                    self._save_audio_file(processed_audio, 'hindi', stress_level,
                                        f"cdac_{i:06d}")
                    processed_count += 1
                
            except Exception as e:
                print(f"Error processing Hindi sample {i}: {e}")
                continue
        
        self.processed_stats['hindi'] += processed_count
        self.processed_stats['total'] += processed_count
        
        print(f"✅ Processed {processed_count} samples from Hindi dataset")
        return processed_count > 0
    
    def step6_add_synthetic_data(self, samples_per_category=100):
        """Step 6: Add synthetic data to fill gaps"""
        print("\n🎵 STEP 6: Adding Synthetic Data")
        print("=" * 60)
        
        try:
            from sample_data_generator import StressDatasetGenerator
            
            generator = StressDatasetGenerator()
            total_synthetic = generator.generate_sample_dataset(
                samples_per_category=samples_per_category
            )
            
            self.processed_stats['total'] += total_synthetic
            print(f"✅ Added {total_synthetic} synthetic samples")
            return True
            
        except ImportError:
            print("⚠️  Synthetic data generator not available")
            return False
    
    def step7_validate_final_dataset(self):
        """Step 7: Validate final dataset structure"""
        print("\n✅ STEP 7: Validating Final Dataset")
        print("=" * 60)
        
        validation_results = {}
        total_files = 0
        
        for language in LANGUAGES:
            validation_results[language] = {}
            lang_total = 0
            
            print(f"\n📁 {language.title()} files:")
            for stress_level in STRESS_LEVELS:
                folder_path = RAW_DATA_DIR / language / stress_level
                
                if folder_path.exists():
                    files = list(folder_path.glob('*.wav'))
                    count = len(files)
                    validation_results[language][stress_level] = count
                    lang_total += count
                    
                    status = "✅" if count >= 50 else "⚠️" if count > 0 else "❌"
                    print(f"   {status} {stress_level:<15}: {count:3d} files")
                else:
                    validation_results[language][stress_level] = 0
                    print(f"   ❌ {stress_level:<15}:   0 files")
            
            validation_results[language]['total'] = lang_total
            total_files += lang_total
            print(f"   📊 {language} total: {lang_total}")
        
        # Save validation results
        results_file = RAW_DATA_DIR / 'dataset_validation.json'
        with open(results_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        print(f"\n🎯 FINAL DATASET SUMMARY:")
        print(f"   Total files: {total_files}")
        print(f"   Average per category: {total_files/8:.1f}")
        print(f"   Validation saved: {results_file}")
        
        # Recommendations
        if total_files >= 1600:
            print("🌟 EXCELLENT: You have a large, robust dataset!")
        elif total_files >= 800:
            print("✅ VERY GOOD: Dataset size is excellent for training")
        elif total_files >= 400:
            print("✅ GOOD: Dataset size is adequate for training")
        else:
            print("⚠️  Consider adding more samples for better performance")
        
        return validation_results
    
    def _process_audio_sample(self, audio_array, original_sr, target_sr=22050, duration=5):
        """Process individual audio sample to project specifications"""
        try:
            # Resample if needed
            if original_sr != target_sr:
                audio_resampled = librosa.resample(audio_array, orig_sr=original_sr, target_sr=target_sr)
            else:
                audio_resampled = audio_array.copy()
            
            # Normalize
            if np.max(np.abs(audio_resampled)) > 0:
                audio_normalized = librosa.util.normalize(audio_resampled)
            else:
                return None
            
            # Adjust length
            target_length = int(target_sr * duration)
            
            if len(audio_normalized) > target_length:
                # Trim to target length (take middle portion)
                start_idx = (len(audio_normalized) - target_length) // 2
                audio_processed = audio_normalized[start_idx:start_idx + target_length]
            else:
                # Pad with zeros
                pad_length = target_length - len(audio_normalized)
                audio_processed = np.pad(audio_normalized, (0, pad_length), mode='constant')
            
            return audio_processed
            
        except Exception as e:
            print(f"Audio processing error: {e}")
            return None
    
    def _save_audio_file(self, audio_array, language, stress_level, filename_base):
        """Save processed audio to appropriate directory"""
        # Create output directory
        output_dir = RAW_DATA_DIR / language / stress_level
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        filename = f"{filename_base}.wav"
        filepath = output_dir / filename
        
        # Save audio file
        sf.write(filepath, audio_array, SAMPLE_RATE)
        return filepath

def main():
    """Main integration workflow"""
    print("🎯 HUGGING FACE DATASET INTEGRATION WORKFLOW")
    print("=" * 70)
    print("This will download and process your datasets into the project structure")
    print("=" * 70)
    
    loader = HuggingFaceDatasetLoader()
    
    # Execute all steps
    steps = [
        ("Loading Datasets", loader.step1_load_datasets),
        ("Analyzing Structures", loader.step2_analyze_datasets),
        ("Creating Mappings", loader.step3_create_label_mappings),
        ("Processing Stress-17K", lambda: loader.step4_process_stress_17k(max_samples_per_class=400)),
        ("Processing Hindi", lambda: loader.step5_process_hindi_dataset(max_samples_per_class=300)),
        ("Adding Synthetic Data", lambda: loader.step6_add_synthetic_data(samples_per_category=50)),
        ("Final Validation", loader.step7_validate_final_dataset)
    ]
    
    completed_steps = 0
    
    for step_name, step_func in steps:
        try:
            print(f"\n▶️  Executing: {step_name}")
            success = step_func()
            if success:
                completed_steps += 1
                print(f"✅ {step_name} completed successfully")
            else:
                print(f"⚠️  {step_name} completed with warnings")
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
            print("Continuing with next step...")
    
    print(f"\n🏁 INTEGRATION COMPLETE!")
    print(f"✅ Completed {completed_steps}/{len(steps)} steps successfully")
    print(f"📊 Total files processed: {loader.processed_stats['total']}")
    print(f"📁 Data location: {RAW_DATA_DIR}")
    
    print(f"\n📋 NEXT STEPS:")
    print("1. python backend/src/preprocessing.py")
    print("2. python backend/src/feature_extraction.py")
    print("3. python backend/src/model_training.py")

if __name__ == "__main__":
    main()

import numpy as np
import soundfile as sf
from pathlib import Path
from config import *

class StressDatasetGenerator:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE

    def generate_sample_dataset(self, samples_per_category=100):
        total_generated = 0
        for language in LANGUAGES:
            for stress_level in STRESS_LEVELS:
                output_dir = RAW_DATA_DIR / language / stress_level
                output_dir.mkdir(parents=True, exist_ok=True)

                for i in range(samples_per_category):
                    # Generate synthetic audio (simple sine wave with noise)
                    duration = 5
                    t = np.linspace(0, duration, int(self.sample_rate * duration))
                    frequency = 440 + (100 * STRESS_LEVELS.index(stress_level))
                    audio = np.sin(2 * np.pi * frequency * t)
                    noise = np.random.normal(0, 0.1, len(audio))
                    audio = audio + noise
                    audio = audio / np.max(np.abs(audio))

                    # Save the synthetic audio
                    filename = f"synthetic_{language}_{stress_level}_{i:03d}.wav"
                    sf.write(output_dir / filename, audio, self.sample_rate)
                    total_generated += 1

        return total_generated
