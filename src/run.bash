python exp.py --k 1 --large 0 --plain_gd 0
python exp.py --k 10 --large 0 --plain_gd 0
python exp.py --k 20 --large 0 --plain_gd 0
python exp.py --k 30 --large 0 --plain_gd 0
python exp.py --k 1 --large 1 --plain_gd 0
python exp.py --k 10 --large 1 --plain_gd 0
python exp.py --k 20 --large 1 --plain_gd 0
python exp.py --k 30 --large 1 --plain_gd 0
python exp.py --k 1 --large 1 --plain_gd 1
python exp.py --k 1 --large 0 --plain_gd 1
python exp-linesearch.py --k 1 --large 0 
python exp-linesearch.py --k 10 --large 0 
python exp-linesearch.py --k 20 --large 0 
python exp-linesearch.py --k 30 --large 0 
python exp-linesearch.py --k 1 --large 1 
python exp-linesearch.py --k 10 --large 1 
python exp-linesearch.py --k 20 --large 1 
python exp-linesearch.py --k 30 --large 1 

python plotting.py
python plotting-linesearch.py
