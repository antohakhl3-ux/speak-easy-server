from flask import Flask, request, jsonify
import os
import httpx
from dotenv import load_dotenv

load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

app = Flask(__name__)

def compare_phonemes(expected_word, recognized_word, phonemes):
    """
    Сравнивает ожидаемое слово с распознанными фонемами.
    Возвращает массив булевых значений для каждой буквы expected_word:
    True – буква произнесена неверно, False – верно.
    Если фонем нет или длины не совпадают, все буквы считаются ошибкой.
    """
    expected_letters = list(expected_word.lower())
    # Извлекаем символы фонем (без учёта ударений и т.п.)
    phoneme_symbols = [p['phoneme'] for p in phonemes]
    
    if len(expected_letters) != len(phoneme_symbols):
        return [True] * len(expected_letters)
    
    errors = []
    for exp_letter, rec_phoneme in zip(expected_letters, phoneme_symbols):
        # Упрощённое сравнение: считаем, что фонема должна начинаться с той же буквы
        # Например, для русского "р" фонема может быть "r" или "р" – в Deepgram они используют IPA или свои символы
        # Для точности лучше использовать фонетический словарь, но для MVP сойдёт
        if exp_letter != rec_phoneme.lower():
            errors.append(True)
        else:
            errors.append(False)
    return errors

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
            "phonemes": "true"          # ключевой параметр
        }

        client = httpx.Client(verify=False)  # для теста, в проде лучше с verify=True
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

        # Разбиваем ожидаемый текст на слова
        expected_words = expected_text.split()
        recognized_words = transcript.split()

        # Сравнение на уровне слов (как было)
        word_comparison = []
        for i, exp_word in enumerate(expected_words):
            if i < len(recognized_words):
                rec_word = recognized_words[i]
                is_error = exp_word.lower() != rec_word.lower()
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

        # Подготовка детальной информации по фонемам
        phoneme_details = []
        for i, exp_word in enumerate(expected_words):
            if i < len(words_data):
                phonemes = words_data[i].get('phonemes', [])
                if phonemes:
                    errors = compare_phonemes(exp_word, words_data[i]['word'], phonemes)
                else:
                    errors = [True] * len(exp_word)
            else:
                errors = [True] * len(exp_word)  # слово не распознано – все буквы ошибка
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
