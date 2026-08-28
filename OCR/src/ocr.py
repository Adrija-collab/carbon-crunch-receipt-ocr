import cv2
import numpy as np
import pytesseract
from pytesseract import Output


def run_ocr(image, psm=6):
    data = pytesseract.image_to_data(image, config=f'--oem 3 --psm {psm}', output_type=Output.DICT)
    words = []
    for i, text in enumerate(data['text']):
        text = text.strip()
        try:
            conf = float(data['conf'][i])
        except Exception:
            conf = -1
        if text and conf >= 0:
            words.append({'text': text, 'confidence': conf / 100.0,
                          'left': int(data['left'][i]), 'top': int(data['top'][i]),
                          'width': int(data['width'][i]), 'height': int(data['height'][i])})
    text = ' '.join(w['text'] for w in words)
    confidence = float(np.mean([w['confidence'] for w in words])) if words else 0.0
    return {'text': text, 'words': words, 'ocr_confidence': confidence}
