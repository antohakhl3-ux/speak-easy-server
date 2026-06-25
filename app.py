from flask import Flask, request, jsonify
import os
import httpx
import re
import numpy as np
from dotenv import load_dotenv

load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__)

def normalize_word(word):
    """
    Удаляет знаки препинания.
    Если слово состоит только из цифр, возвращает как есть.
    Иначе приводит к нижнему регистру.
    """
    word = re.sub(r'[^\w\s]', '', word)
    if word.isdigit():
        return word
    return word.lower()

def align_words(expected, recognized):
    """Выравнивание Нидлмана-Вунша для побуквенного сравнения."""
    if not recognized:
        return [True] * len(expected)
    
    n, m = len(expected), len(recognized)
    dp = np.zeros((n+1, m+1), dtype=int)
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if expected[i-1] == recognized[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j-1] + cost,
                dp[i-1][j] + 1,
                dp[i][j-1] + 1
            )
    
    i, j = n, m
    errors = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if expected[i-1] == recognized[j-1] else 1):
            errors.append(expected[i-1] != recognized[j-1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            errors.append(True)
            i -= 1
        else:
            j -= 1
    
    errors.reverse()
    while len(errors) < n:
        errors.append(True)
    return errors

def compare_phonemes(expected_word, recognized_word):
    expected_norm = normalize_word(expected_word)
    recognized_norm = normalize_word(recognized_word) if recognized_word else ""
    return align_words(expected_norm, recognized_norm)

# ============================================================
# ЭНДПОИНТ 1: Анализ аудио через Deepgram
# ============================================================
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400

    audio_file = request.files['audio']
    expected_text = request.form.get('expected_text', '')
    language = request.form.get('language', 'ru')

    try:
        audio_data = audio_file.read()
        print(f"Размер аудио: {len(audio_data)} байт, язык: {language}")

        if len(audio_data) < 1:
            return jsonify({
                'success': False,
                'error': 'Audio file too small',
                'recognized': '',
                'expected': expected_text,
                'wordComparison': [],
                'phonemeDetails': []
            })

        url = "https://api.deepgram.com/v1/listen"
        headers = {
            "Authorization": f"Token {DEEPGRAM_API_KEY}",
            "Content-Type": "audio/m4a"
        }
        params = {
            "model": "nova-2",
            "language": language,
            "punctuate": "true",
            "smart_format": "true",
            "words": "true",
            "phonemes": "true"
        }

        client = httpx.Client(verify=False)
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

        if 'results' not in result:
            return jsonify({
                'success': False,
                'error': 'No results from Deepgram',
                'recognized': '',
                'expected': expected_text,
                'wordComparison': [],
                'phonemeDetails': []
            })

        transcript = result['results']['channels'][0]['alternatives'][0]['transcript']
        words_data = result['results']['channels'][0]['alternatives'][0].get('words', [])

        expected_words = expected_text.split()
        recognized_words = transcript.split()

        print(f"Ожидаемые слова: {expected_words}")
        print(f"Распознанные слова: {recognized_words}")

        # Сравнение слов
        word_comparison = []
        for i, exp_word in enumerate(expected_words):
            if i < len(recognized_words):
                rec_word = recognized_words[i]
                is_error = normalize_word(exp_word) != normalize_word(rec_word)
            else:
                rec_word = ""
                is_error = True
            word_comparison.append({
                "expected": exp_word,
                "recognized": rec_word,
                "error": is_error
            })
        for i in range(len(expected_words), len(recognized_words)):
            word_comparison.append({
                "expected": "",
                "recognized": recognized_words[i],
                "error": True
            })

        # Детали по буквам
        phoneme_details = []
        for i, exp_word in enumerate(expected_words):
            if i < len(recognized_words):
                rec_word = recognized_words[i]
                errors = compare_phonemes(exp_word, rec_word)
                print(f"Слово '{exp_word}' vs '{rec_word}': ошибки {errors}")
            else:
                errors = [True] * len(exp_word)
            phoneme_details.append({
                "word": exp_word,
                "errorMask": errors
            })

        return jsonify({
            'success': True,
            'recognized': transcript,
            'expected': expected_text,
            'wordComparison': word_comparison,
            'phonemeDetails': phoneme_details
        })

    except Exception as e:
        print(f"Ошибка: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ============================================================
# ЭНДПОИНТ 2: Генерация текста через Groq API
# ============================================================
@app.route('/generate_text', methods=['POST'])
def generate_text():
    """Генерирует текст с частым использованием заданной буквы через Groq API"""
    
    if not GROQ_API_KEY:
        return jsonify({'success': False, 'error': 'Groq API key not configured'}), 500
    
    data = request.json
    sound = data.get('sound', 'р')
    language = data.get('language', 'ru')
    
    # Промпты для разных языков
    prompts = {
        'ru': f"Напиши 3 коротких предложения на русском языке, где часто встречается буква '{sound}'. Текст должен быть осмысленным, интересным и естественным. Верни ТОЛЬКО текст, без пояснений.",
        'en': f"Write 3 short sentences in English where the letter '{sound}' appears frequently. The text should be meaningful, interesting and natural. Return ONLY the text, no explanations.",
        'es': f"Escribe 3 oraciones cortas en español donde la letra '{sound}' aparezca con frecuencia. El texto debe ser significativo, interesante y natural. Devuelve SOLO el texto, sin explicaciones.",
        'fr': f"Écris 3 phrases courtes en français où la lettre '{sound}' apparaît fréquemment. Le texte doit être significatif, intéressant et naturel. Retourne SEULEMENT le texte, sans explications.",
        'de': f"Schreibe 3 kurze Sätze auf Deutsch, in denen der Buchstabe '{sound}' häufig vorkommt. Der Text sollte sinnvoll, interessant und natürlich sein. Gib NUR den Text zurück, ohne Erklärungen.",
        'it': f"Scrivi 3 brevi frasi in italiano dove la lettera '{sound}' appare frequentemente. Il testo deve essere significativo, interessante e naturale. Restituisci SOLO il testo, senza spiegazioni.",
        'pt': f"Escreva 3 frases curtas em português onde a letra '{sound}' aparece com frequência. O texto deve ser significativo, interessante e natural. Retorne APENAS o texto, sem explicações.",
        'zh': f"用中文写3个短句，其中经常出现'{sound}'这个字母。文本要有意义、有趣且自然。只返回文本，不要解释。",
        'ja': f"日本語で3つの短い文を書いてください。文字'{sound}'が頻繁に現れるようにしてください。テキストは意味があり、興味深く、自然でなければなりません。説明なしでテキストのみを返してください。",
        'ko': f"한국어로 3개의 짧은 문장을 작성하세요. '{sound}' 글자가 자주 나타나야 합니다. 텍스트는 의미 있고 흥미롭고 자연스러워야 합니다. 설명 없이 텍스트만 반환하세요."
    }
    
    prompt = prompts.get(language, prompts['en'])
    
    try:
        # Запрос к Groq API
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "mixtral-8x7b-32768",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that generates practice texts for pronunciation training."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 300
        }
        
        client = httpx.Client(verify=False)
        response = client.post(url, headers=headers, json=payload, timeout=30)
        client.close()
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result['choices'][0]['message']['content'].strip()
            
            # Очищаем от кавычек, если есть
            generated_text = generated_text.strip('"\'')
            
            return jsonify({
                'success': True,
                'text': generated_text,
                'sound': sound,
                'language': language
            })
        else:
            return jsonify({
                'success': False,
                'error': f"Groq API error: {response.status_code} - {response.text}"
            }), response.status_code
            
    except Exception as e:
        print(f"Ошибка генерации текста: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
