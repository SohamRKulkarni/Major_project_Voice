import random
from datetime import datetime

class RemedyRecommender:
    def __init__(self):
        self.remedies = {
            'no_stress': {
                'english': [
                    "You're doing great! Continue with your current routine.",
                    "Consider light stretching or a short walk to maintain your well-being.",
                    "Keep up the good work with your stress management.",
                    "Take a moment to appreciate your current calm state."
                ],
                'hindi': [
                    "आप बहुत अच्छा कर रहे हैं! अपनी वर्तमान दिनचर्या जारी रखें।",
                    "अपनी भलाई बनाए रखने के लिए हल्की स्ट्रेचिंग या छोटी सैर करें।",
                    "अपने तनाव प्रबंधन के साथ अच्छा काम जारी रखें।",
                    "अपनी वर्तमान शांत अवस्था की सराहना करने के लिए एक पल लें।"
                ]
            },
            'low_stress': {
                'english': [
                    "Try deep breathing: 4 counts in, hold for 4, exhale for 6.",
                    "Take a 5-minute break and listen to calming music.",
                    "Do some light stretching or neck rolls.",
                    "Drink a glass of water and take slow, mindful sips.",
                    "Practice the 5-4-3-2-1 grounding technique."
                ],
                'hindi': [
                    "गहरी सांस लेने की कोशिश करें: 4 गिनती में सांस लें, 4 तक रोकें, 6 में सांस छोड़ें।",
                    "5 मिनट का ब्रेक लें और शांत संगीत सुनें।",
                    "कुछ हल्की स्ट्रेचिंग या गर्दन घुमाएं।",
                    "एक गिलास पानी पिएं और धीरे-धीरे, ध्यानपूर्वक घूंट लें।",
                    "5-4-3-2-1 ग्राउंडिंग तकनीक का अभ्यास करें।"
                ]
            },
            'medium_stress': {
                'english': [
                    "Practice progressive muscle relaxation for 10 minutes.",
                    "Try the 4-7-8 breathing technique: inhale for 4, hold for 7, exhale for 8.",
                    "Take a 10-15 minute walk outside if possible.",
                    "Listen to guided meditation or relaxation audio.",
                    "Write down 3 things you're grateful for today.",
                    "Do some gentle yoga poses or stretches."
                ],
                'hindi': [
                    "10 मिनट के लिए प्रगतिशील मांसपेशी विश्राम का अभ्यास करें।",
                    "4-7-8 श्वास तकनीक करें: 4 में सांस लें, 7 तक रोकें, 8 में छोड़ें।",
                    "यदि संभव हो तो बाहर 10-15 मिनट की सैर करें।",
                    "निर्देशित ध्यान या विश्राम ऑडियो सुनें।",
                    "आज आप जिन 3 चीजों के लिए आभारी हैं उन्हें लिख लें।",
                    "कुछ सौम्य योग आसन या स्ट्रेच करें।"
                ]
            },
            'high_stress': {
                'english': [
                    "IMMEDIATE: Practice box breathing - 4 counts in, hold 4, out 4, hold 4.",
                    "Find a quiet space and do 5 minutes of deep breathing.",
                    "Try cold water on your wrists or splash your face.",
                    "Use the STOP technique: Stop, Take a breath, Observe, Proceed mindfully.",
                    "Consider speaking to a counselor or trusted friend.",
                    "If stress persists, please consult a healthcare professional."
                ],
                'hindi': [
                    "तुरंत: बॉक्स ब्रीदिंग करें - 4 गिनती में सांस लें, 4 रोकें, 4 में छोड़ें, 4 रोकें।",
                    "एक शांत जगह खोजें और 5 मिनट गहरी सांस लें।",
                    "अपनी कलाई पर ठंडा पानी लगाएं या अपने चेहरे पर छींटे मारें।",
                    "STOP तकनीक का प्रयोग करें: रुकें, सांस लें, देखें, सचेत रूप से आगे बढ़ें।",
                    "किसी परामर्शदाता या विश्वसनीय मित्र से बात करने पर विचार करें।",
                    "यदि तनाव बना रहे, तो कृपया स्वास्थ्य पेशेवर से सलाह लें।"
                ]
            }
        }
        
        self.f1_thresholds = {
            'high_confidence': 0.80,
            'medium_confidence': 0.65,
            'low_confidence': 0.50
        }
    
    def get_confidence_level(self, f1_score, prediction_confidence):
        """Determine overall confidence level"""
        combined_confidence = (f1_score + prediction_confidence) / 2
        
        if combined_confidence >= self.f1_thresholds['high_confidence']:
            return 'high_confidence'
        elif combined_confidence >= self.f1_thresholds['medium_confidence']:
            return 'medium_confidence'
        else:
            return 'low_confidence'
    
    def recommend_remedies(self, prediction_result):
        """Recommend remedies based on stress detection result"""
        if not prediction_result:
            return {"error": "No prediction result provided"}
        
        stress_level = prediction_result['stress_level']
        language = prediction_result['language']
        model_f1 = prediction_result['model_f1']
        confidence = prediction_result['confidence']
        
        # Determine confidence level
        confidence_level = self.get_confidence_level(model_f1, confidence)
        
        # Get base remedies
        base_remedies = self.remedies.get(stress_level, {}).get(language, [])
        
        # Select remedies based on confidence
        if confidence_level == 'high_confidence':
            num_remedies = 3
            remedy_prefix = "High confidence recommendations:"
        elif confidence_level == 'medium_confidence':
            num_remedies = 2
            remedy_prefix = "Moderate confidence recommendations:"
        else:
            num_remedies = 1
            remedy_prefix = "Low confidence - general recommendations:"
            # Add generic advice for low confidence
            if language == 'english':
                base_remedies.extend([
                    "Consider taking a few deep breaths and reassessing.",
                    "If you feel stressed, try basic relaxation techniques."
                ])
            else:
                base_remedies.extend([
                    "कुछ गहरी सांसें लेने और पुनर्मूल्यांकन करने पर विचार करें।",
                    "यदि आप तनाव महसूस करते हैं, तो बुनियादी विश्राम तकनीकों का प्रयास करें।"
                ])
        
        # Select random remedies
        selected_remedies = random.sample(base_remedies, min(num_remedies, len(base_remedies)))
        
        # Create recommendation object
        recommendation = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'stress_level': stress_level,
            'language': language,
            'confidence_level': confidence_level,
            'model_f1_score': model_f1,
            'prediction_confidence': confidence,
            'combined_confidence': (model_f1 + confidence) / 2,
            'prefix': remedy_prefix,
            'remedies': selected_remedies,
            'additional_info': self.get_additional_info(stress_level, confidence_level, language)
        }
        
        return recommendation
    
    def get_additional_info(self, stress_level, confidence_level, language):
        """Get additional information based on stress level"""
        additional_info = {}
        
        if stress_level == 'high_stress':
            if language == 'english':
                additional_info['urgent_note'] = "If stress levels remain high, please consider professional help."
                additional_info['emergency_contacts'] = [
                    "Mental Health Helpline: 1800-123-4567",
                    "Crisis Support: 1800-891-4416"
                ]
            else:
                additional_info['urgent_note'] = "यदि तनाव का स्तर उच्च बना रहे, तो कृपया पेशेवर सहायता पर विचार करें।"
                additional_info['emergency_contacts'] = [
                    "मानसिक स्वास्थ्य हेल्पलाइन: 1800-123-4567",
                    "संकट सहायता: 1800-891-4416"
                ]
        
        if confidence_level == 'low_confidence':
            if language == 'english':
                additional_info['note'] = "Low confidence detection. Consider multiple assessments."
            else:
                additional_info['note'] = "कम विश्वास का पता लगाना। कई आकलन पर विचार करें।"
        
        return additional_info
    
    def format_recommendation(self, recommendation):
        """Format recommendation for display"""
        output = f"\n{'='*50}\n"
        output += f"STRESS ASSESSMENT REPORT\n"
        output += f"Time: {recommendation['timestamp']}\n"
        output += f"{'='*50}\n\n"
        
        output += f"Detected Stress Level: {recommendation['stress_level'].upper()}\n"
        output += f"Language: {recommendation['language'].title()}\n"
        output += f"Combined Confidence: {recommendation['combined_confidence']:.2f}\n"
        output += f"Model F1 Score: {recommendation['model_f1_score']:.2f}\n\n"
        
        output += f"{recommendation['prefix']}\n"
        for i, remedy in enumerate(recommendation['remedies'], 1):
            output += f"{i}. {remedy}\n"
        
        if recommendation['additional_info']:
            output += f"\nAdditional Information:\n"
            for key, value in recommendation['additional_info'].items():
                if isinstance(value, list):
                    output += f"{key.title()}:\n"
                    for item in value:
                        output += f"  - {item}\n"
                else:
                    output += f"{key.title()}: {value}\n"
        
        output += f"\n{'='*50}\n"
        return output

if __name__ == "__main__":
    # Test the recommender
    recommender = RemedyRecommender()
    
    # Sample prediction result
    sample_result = {
        'stress_level': 'medium_stress',
        'confidence': 0.75,
        'model_f1': 0.82,
        'language': 'english'
    }
    
    recommendation = recommender.recommend_remedies(sample_result)
    formatted_output = recommender.format_recommendation(recommendation)
    print(formatted_output)
