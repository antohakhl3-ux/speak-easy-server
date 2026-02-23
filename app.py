from flask import Flask, request, jsonify
import os
import httpx
import re
import numpy as np
from dotenv import load_dotenv

load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

app = Flask(__name__)

def normalize_word(word):
    """Удаляет знаки препинания и приводит к нижнему регистру"""
    return re.sub(r'[^\w\s]', '', word).lower()

def align_words(expected, recognized):
    """
    Возвращает массив булевых значений для expected:
    True – буква произнесена неверно, False – верно.
    Используется выравнивание Нидлмана-Вунша.
    """
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
    """Сравнивает ожидаемое и распознанное слово побуквенно через выравнивание"""
    expected_word_norm = normalize_word(expected_word)
    recognized_word_norm = normalize_word(recognized_word) if recognized_word else ""
    return align_words(expected_word_norm, recognized_word_norm)

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

        if len(audio_data) < 1000:
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

        # Сравнение на уровне слов с нормализацией
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
        # Лишние распознанные слова
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
