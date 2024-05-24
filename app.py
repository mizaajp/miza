from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# 感情分析パイプラインを作成
sentiment_pipeline = pipeline("sentiment-analysis")

@app.route('/')
def home():
    return "Welcome to the Sentiment Analysis API!"

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    text = data.get('text', '')
    if text:
        result = sentiment_pipeline(text)[0]
        return jsonify({
            'label': result['label'],
            'score': round(result['score'], 4)
        })
    return jsonify({'error': 'No text provided'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
