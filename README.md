# Diacritisation

`extract_articles.py` - Parses articles from vesmir.cz

`create_dictinary.py` - Creates dictionary of all words and their possible diacritizations from 37 pages extracted from vesmir.cz

`diacritization.py` - Performs diacritization 

`run.sh` - Contains commands in order: 

1. Extract 10 pages of articles
2. Create dictionary (uses extracted articles from 37 pages)
3. Train model using 10 pages of articles, as an input data used evaluation set diacritics-dtest.txt. Output and accuracy are printed to std.

Reading text from std:
in `run.sh` change 3. line to:
python3 diacritization.py --train "True" --data "vesmir_articles10.txt" --input "std" --dict "dict.txt" --output "std"

Note: accuracy is printed only for reading from file, for reading from std only diacritized text is printed.

All arguments for `diacritization.py` are explained in --help.
