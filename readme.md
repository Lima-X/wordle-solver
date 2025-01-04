# wordle-solver
A simple expandable wordle solver and CLI tools to solve nytimes wordle v2 using 3b1b's entropy strategy / information theory\
![demo](demo.gif?raw=true)

## How it works's:
Similar to how 3b1b solved it, this uses a word to word matrix of response/info-patterns of every possible combination.\
This matrix can be precomputed (takes a lot of time depending on the word list used), or the neccessary row/vector can be computed on demand if no precomputed table is provided.\
It uses the entirety of wordle v2's accepted wordlist (14855 words), unlike 3b1b's solution the columns of the matrix are inverted through a dictionary, that allows direct look up for a set of words within a possible response pattern.\
It also exposes a way to directly play the wordle of the day through a simple request to nytimes wordle site.

## FAQ:
Yes, this was created purely out of bordem, also to automatically cheat at the game, yes im late.