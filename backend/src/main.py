import os
import sys
import argparse
from stress_detector import RealTimeStressDetector
from remedy_recommender import RemedyRecommender
import time

class VoiceStressEstimator:
    def __init__(self):
        print("Initializing Voice Stress Level Estimator...")
        self.detector = RealTimeStressDetector()
        self.recommender = RemedyRecommender()
        print("System ready!")
    
    def analyze_file(self, file_path, language=None):
        """Analyze stress from audio file"""
        print(f"Analyzing audio file: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found!")
            return
        
        # Detect stress
        result = self.detector.detect_from_file(file_path, language)
        
        if result:
            print(f"\nStress Detection Result:")
            print(f"Stress Level: {result['stress_level']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Language: {result['language']}")
            print(f"Model F1 Score: {result['model_f1']:.2f}")
            
            # Get recommendations
            recommendation = self.recommender.recommend_remedies(result)
            formatted_rec = self.recommender.format_recommendation(recommendation)
            print(formatted_rec)
            
        else:
            print("Error: Could not analyze the audio file.")
    
    def real_time_monitoring(self):
        """Start real-time stress monitoring"""
        print("Starting real-time stress monitoring...")
        print("Speak into your microphone. The system will analyze your voice in real-time.")
        print("Press Ctrl+C to stop monitoring.\n")
        
        try:
            # Start real-time detection in a separate thread
            import threading
            
            def detection_loop():
                detector = RealTimeStressDetector()
                detector.start_real_time_detection()
            
            detection_thread = threading.Thread(target=detection_loop)
            detection_thread.daemon = True
            detection_thread.start()
            
            # Main loop for handling recommendations
            while True:
                time.sleep(5)  # Check every 5 seconds
                
                # Check if we have a recent result
                if hasattr(self.detector, 'latest_result'):
                    result = self.detector.latest_result
                    
                    # Get and display recommendations
                    recommendation = self.recommender.recommend_remedies(result)
                    formatted_rec = self.recommender.format_recommendation(recommendation)
                    print(formatted_rec)
                    
                    # Clear the result to avoid repeated recommendations
                    delattr(self.detector, 'latest_result')
                
        except KeyboardInterrupt:
            print("\nStopping real-time monitoring...")
    
    def interactive_mode(self):
        """Interactive mode for the application"""
        print("\n" + "="*60)
        print("VOICE STRESS LEVEL ESTIMATOR")
        print("Multi-lingual Stress Detection with F1-based Recommendations")
        print("="*60)
        
        while True:
            print("\nChoose an option:")
            print("1. Analyze audio file")
            print("2. Real-time voice monitoring")
            print("3. View system information")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == '1':
                file_path = input("Enter audio file path: ").strip()
                language = input("Enter language (english/hindi) or press Enter for auto-detection: ").strip()
                if not language:
                    language = None
                self.analyze_file(file_path, language)
                
            elif choice == '2':
                self.real_time_monitoring()
                
            elif choice == '3':
                self.show_system_info()
                
            elif choice == '4':
                print("Thank you for using Voice Stress Level Estimator!")
                break
                
            else:
                print("Invalid choice. Please try again.")
    
    def show_system_info(self):
        """Display system information"""
        print("\n" + "="*50)
        print("SYSTEM INFORMATION")
        print("="*50)
        print("Version: 1.0")
        print("Supported Languages: English, Hindi")
        print("Stress Levels: No Stress, Low Stress, Medium Stress, High Stress")
        print("Model: CNN Classifier")
        print("Features: MFCC, Pitch, Spectral, Energy features")
        print("F1 Score Integration: Yes")
        print("Real-time Processing: Yes")
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description='Voice Stress Level Estimator')
    parser.add_argument('--file', '-f', type=str, help='Audio file to analyze')
    parser.add_argument('--language', '-l', type=str, choices=['english', 'hindi'], 
                       help='Language of the audio')
    parser.add_argument('--realtime', '-r', action='store_true', 
                       help='Start real-time monitoring')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='Start interactive mode')
    
    args = parser.parse_args()
    
    # Create the estimator
    estimator = VoiceStressEstimator()
    
    if args.file:
        estimator.analyze_file(args.file, args.language)
    elif args.realtime:
        estimator.real_time_monitoring()
    elif args.interactive:
        estimator.interactive_mode()
    else:
        # Default to interactive mode
        estimator.interactive_mode()

if __name__ == "__main__":
    main()
