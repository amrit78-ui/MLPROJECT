import app, traceback

try:
    p = app.preprocess_text('Win ₹10,000 cash now!!! Limited time offer!!!')
    v = app.vectorizer.transform([p])
    pred = app.model.predict(v)[0]
    with open('result.txt', 'w', encoding='utf-8') as f:
        f.write(f'Success: {pred}\n')
except Exception as e:
    with open('result.txt', 'w', encoding='utf-8') as f:
        f.write('Exception:\n')
        traceback.print_exc(file=f)
