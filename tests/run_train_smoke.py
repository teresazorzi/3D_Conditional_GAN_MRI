import tempfile
from mrisyngan.train import train_and_evaluate

if __name__ == '__main__':
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            train_and_evaluate({}, tmpdir, device='cpu')
    except Exception as e:
        print('RAISED:', type(e).__name__, str(e))
        raise SystemExit(0)
    print('NO_EXCEPTION')
