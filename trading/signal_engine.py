# Signal generation
# trading/signal_engine.py

def generate_signal(probabilities, threshold=0.6):
    bull = probabilities[0][0]
    bear = probabilities[0][1]

    if bull > threshold:
        return "BUY"
    if bear > threshold:
        return "SELL"
    return "NO_TRADE"
