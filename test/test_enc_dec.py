import sys
from pathlib import Path
ROOT=Path(__file__).parent.parent
sys.path.append(str(ROOT))

from src import Encoder,Decoder
def main():
    enc=Encoder()
    dec=Decoder()


if __name__=="__main__":
    main()