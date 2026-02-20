from flask import Flask, request, jsonify
import os
import httpx
from dotenv import load_dotenv

load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

app = Flask(__name__)

def compare_texts(expected, recognized):
    """Сравнивает ожидаемый и распознанный текст по словам."""
    if not recognized:
        expected_words = expected.split()
        return [{"expected": w, "recognized": "", "error": True} for w in expected_words]
    
    expected_words = expected.split()
    recognized_words = recognized.split()
    result = []
    for i, exp_word in enumerate(expected_words):
        if i < len(recognized_words):
            rec_word = recognized_words[i]
            is_error = exp_word.lower() != rec_word.lower()
        else:
            rec_word = ""
            is_error = True
        result.append({
            "expected": exp_word,
            "recognized": rec_word,
            "error": is_error
        })
    for i in range(len(expected_words), len(recognized_words)):
        result.append({
            "expected": "",
            "recognized": recognized_words[i],
            "error": True
        })
    return result

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400

    audio_file = request.files['audio']
    expected_text = request.form.get('expected_text', '')
    language = request.form.get('language', 'ru')  # ← ДОБАВИТЬ ЭТУ СТРОКУ

    try:
        audio_data = audio_file.read()
        print(f"Размер аудио: {len(audio_data)} байт")
        print(f"Язык: {language}")  # отладка
        
        if len(audio_data) < 1000:
            print("Аудиофайл слишком маленький")
            return jsonify({
                'success': False,
                'error': 'Audio file too small',
                'recognized': '',
                'expected': expected_text,
                'wordComparison': compare_texts(expected_text, '')
            })
        
        url = "https://api.deepgram.com/v1/listen"
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/m4a"
        }
        
        # ← ИСПОЛЬЗОВАТЬ language В ПАРАМЕТРАХ
        params = {
            "model": "nova-2",
            "language": language,
            "punctuate": "true",
            "smart_format": "true"
        }
        
        client = httpx.Client(verify=False)  # только для теста
        response = client.post(
            url, 
            headers=headers, 
            params=params, 
            content=audio_data,
            timeout=30
        )
        client.close()
        
        print(f"Статус ответа от Deepgram: {response.status_code}")
        result = response.json()
        print(f"Ответ Deepgram: {result}")
        
        if 'results' not in result:
            print("Deepgram не вернул results")
            return jsonify({
                'success': False,
                'error': 'No results from Deepgram',
                'recognized': '',
                'expected': expected_text,
                'wordComparison': compare_texts(expected_text, '')
            })
        
        transcript = result['results']['channels'][0]['alternatives'][0]['transcript']
        print(f"Распознано: '{transcript}'")
        
        word_comparison = compare_texts(expected_text, transcript)
        
        return jsonify({
            'success': True,
            'recognized': transcript,
            'expected': expected_text,
            'wordComparison': word_comparison
        })
        
    except Exception as e:
        print(f"Ошибка: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e),
            'recognized': '',
            'expected': expected_text,
            'wordComparison': compare_texts(expected_text, '')
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
