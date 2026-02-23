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
    """
    expected_letters = list(expected_word.lower())
    
    # Если фонем нет – все буквы считаем ошибкой
    if not phonemes:
        print(f"⚠️ Нет фонем для слова {expected_word}")
        return [True] * len(expected_letters)
    
    # Извлекаем символы фонем и приводим к нижнему регистру
    phoneme_symbols = [p['phoneme'].lower() for p in phonemes]
    print(f"🔤 Ожидаемое слово: {expected_word}, буквы: {expected_letters}")
    print(f"🎤 Распознанные фонемы: {phoneme_symbols}")
    
    # Если количество фонем не совпадает с количеством букв – все буквы ошибка
    if len(expected_letters) != len(phoneme_symbols):
        print(f"⚠️ Количество букв ({len(expected_letters)}) не совпадает с количеством фонем ({len(phoneme_symbols)})")
        return [True] * len(expected_letters)
    
    # Соответствия русских букв и фонем IPA (упрощённо)
    # Это приблизительные соответствия, для продакшена нужен полный словарь
    phoneme_map = {
        'а': ['a', 'ɐ', 'ʌ'],
        'б': ['b', 'bʲ'],
        'в': ['v', 'vʲ'],
        'г': ['g', 'ɟ'],
        'д': ['d', 'dʲ'],
        'е': ['je', 'e', 'ɛ'],
        'ё': ['jo'],
        'ж': ['ʐ', 'ʒ'],
        'з': ['z', 'zʲ'],
        'и': ['i', 'ɪ'],
        'й': ['j'],
        'к': ['k', 'c', 'kʲ'],
        'л': ['l', 'lʲ'],
        'м': ['m', 'mʲ'],
        'н': ['n', 'nʲ'],
        'о': ['o', 'ɔ', 'ə'],
        'п': ['p', 'pʲ'],
        'р': ['r', 'ɾ', 'rʲ'],
        'с': ['s', 'sʲ'],
        'т': ['t', 'tʲ'],
        'у': ['u', 'ʊ'],
        'ф': ['f', 'fʲ'],
        'х': ['x', 'ç'],
        'ц': ['ts'],
        'ч': ['tɕ'],
        'ш': ['ʂ', 'ʃ'],
        'щ': ['ɕː'],
        'ъ': [''],
        'ы': ['ɨ'],
        'ь': [''],
        'э': ['ɛ'],
        'ю': ['ju'],
        'я': ['ja']
    }
    
    errors = []
    for i, (exp_letter, rec_phoneme) in enumerate(zip(expected_letters, phoneme_symbols)):
        # Получаем возможные варианты фонем для этой буквы
        possible_phonemes = phoneme_map.get(exp_letter, [exp_letter])
        
        # Проверяем, содержится ли распознанная фонема в возможных вариантах
        if rec_phoneme in possible_phonemes:
            errors.append(False)
            print(f"✅ Буква '{exp_letter}' (фонема '{rec_phoneme}') – верно")
        else:
            errors.append(True)
            print(f"❌ Буква '{exp_letter}' (фонема '{rec_phoneme}') – ошибка")
    
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
            "phonemes": "true"
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
        # print(f"Ответ Deepgram: {result}")  # закомментировано для чистоты

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
                print(f"📊 Слово '{exp_word}', получено фонем: {len(phonemes)}")
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
