# -*- coding: utf-8 -*-

''' reads a text file and exports unique tokens separated by space.
'''
import argparse
import codecs
import os
import sys
import spacy

UTF8Writer = codecs.getwriter('utf8')
sys.stdout = UTF8Writer(sys.stdout)

def main(infile, lang):
    ''' main function '''
    # read argument - file with data
    
    parser = spacy.load(lang)
    with codecs.open(os.path.realpath(infile), encoding='utf8') as inp:
        for line in inp.readlines():
            tokens = parser(line)
            tokens = [token.orth_ for token in tokens if not token.orth_.isspace()]
            print u' '.join([t for t in tokens])
            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract vocabulary.')
    parser.add_argument('-i', '--inputfile', required=True, help='The file to tokenise.')
    parser.add_argument('-l', '--language', required=True, help='The language.')

    args = parser.parse_args()

    main(args.inputfile, args.language)
