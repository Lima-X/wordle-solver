"""
Wordle bot, uses 3b1b's entropy pruning strategy that can run realtime as well as
use a precomputed dictionary of patterns for words, to speedup the process for more than 1 game.
Precomputing this dictionary however is slow and will take up a lot of memory + storage so it remains optional.
The bot can be run interactively, or can be used to solve a specific wordle word for a given date.
otherwise this can also be used as a module to be build upon.
"""

# TUI/GUI related imports
from rich.logging import RichHandler
from rich.traceback import install
from rich import console
# temporary cause experimental rich tqdm is alpha
from tqdm.rich import tqdm, trange
from tqdm import TqdmExperimentalWarning
import warnings
warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

# standard library requirements for solver
from collections import defaultdict
import itertools
import argparse
import datetime
import requests
import logging
import pickle
import math
import json
import os

# fun stuff
import random

# script configuration
SKIP_FIRST_ROUND = os.getenv('SKIP_FIRST_ROUND', 'True').lower() in ('true', '1', 't') # if we should skip the first round
USE_PRECOMPUTED = os.getenv('USE_PRECOMPUTED', 'False').lower() in ('true', '1', 't') if not SKIP_FIRST_ROUND else False # use precomputed data but only if we are not skipping the first round
	# loading the precomputed table takes a lot more time than just calculating it on the fly using a precomputed first guess
STARTING_WORD = os.getenv('STARTING_WORD', 'tares') # the starting word for the solver
SKIP_FIRST_ROUND = True if STARTING_WORD != 'tares' else SKIP_FIRST_ROUND # if we have a starting word we can skip the first round
NUMBER_OF_ROUNDS = int(os.getenv('NUMBER_OF_ROUNDS', 6)) # the number of rounds to play, usually 6
# HARD_MODE = os.getenv('HARD_MODE', 'True').lower() in ('true', '1', 't') # if we should use the hard mode, this bot only playes in hard mode
WORDLIST_FILE = os.getenv('WORDLIST_FILE', "wordlist.txt") # the wordlist file to use

def calculate_pattern(canditate: str, solution: str) -> str:
	assert len(canditate) == len(solution), f"Length mismatch: {canditate} vs {solution}"
	pattern = ['B'] * len(canditate)
	solution_count = {}

	# green pass
	for i in range(len(pattern)):
		if canditate[i] == solution[i]:
			pattern[i] = 'G'
		else:
			solution_count[solution[i]] = solution_count.get(solution[i], 0) + 1

	# yellow pass
	for i in range(len(pattern)):
		if pattern[i] == 'B' and solution_count.get(canditate[i], 0) > 0:
			pattern[i] = 'Y'
			solution_count[canditate[i]] -= 1

	return ''.join(pattern)

"""
calculates the entropy of a word, given the possible words still in the list
it uses the pattern_dict (effectively our matrix lookup table) to get the counts of words in the pattern of the word
and uses shannon entropy to calculate the entropy of the word
"""
possible_patterns = [''.join(p) for p in itertools.product('GBY', repeat=5)]
def calculate_entropy(word: str, possible_words: list, pattern_dict: dict) -> float:
	counts = []
	for pattern in possible_patterns:
		matches = pattern_dict[word][pattern].intersection(possible_words)
		counts.append(len(matches))
	# return scipy.stats.entropy(counts)
	return -sum(p * math.log(p) for p in counts if p > 0) # this is non deterministic due to rounding errors

class Solver:
	def __init__(self, tables: dict):
		self.words = tables['words']

		# check if pattern_dict is in tables, if not the solver will build the required vector on demand in __check_lutnode
		self.pattern_dict = tables['pattern_dict'] if 'pattern_dict' in tables else defaultdict(lambda: defaultdict(set))

	"""
	internal, checks that a word is in the LUT, if not the solver will build the required vector on demand.
	this only works as it assumes to never reuse the same guess and the solver has to be destroyed after each game,
	as this function will not properly build the LUT if the simulation is started with a initial guess
	"""
	def _check_lutnode(self, word: str, remaining_words: list):
		if word not in self.pattern_dict:
			LOG.debug(f"Building LUT for {word}...")
			for candidate in remaining_words:
				self.pattern_dict[word][calculate_pattern(word, candidate)].add(candidate)

	# this is a hack to get the description to update in tqdm while preserving uneccessary pattern calculations, could be very unstable
	class __LazyDescription:
		prefix=""
		postfix = ""
		guess = ""
		def __init__(self, feedback: callable):
			self.feedback = feedback

		def __add__(self, other):
			self.postfix += other
			return self
		
		def __str__(self):
			return self.prefix + f": [bright_blue]{self.guess}[/bright_blue] | {self.feedback(self.guess)
				.replace('G', ":green_square:").replace('Y', ":yellow_square:").replace('B', ":black_large_square:") if 
				self.feedback is not None else ' '*15}" + self.postfix

	"""
	calculates the next guess based on the entropy of the remaining words
	"""
	def next_guess(self, remaining_words: list, description: __LazyDescription) -> str:
		entropy_table = {}
		if description.guess == "":
			description.guess=remaining_words[0]
		
		with tqdm(total=len(remaining_words), leave=True, desc=description, options={'console': CON}) as pbar:
			for guess in remaining_words:
				description.guess=guess
				self._check_lutnode(guess, remaining_words)
				entropy_table[guess] = calculate_entropy(guess, remaining_words, self.pattern_dict)
				pbar.update(1)
			
			guess = max(entropy_table, key=lambda x: entropy_table[x])
			description.guess=guess
			pbar.refresh()

		return guess

	"""
	simulates a game of wordle, using the feedback function to get the response/info pattern
	"""
	def solve(self, feedback: callable) -> tuple:
		remaining_words = self.words.copy()
		description = self.__LazyDescription(feedback)
		solve_stack = []

		# we can skip the heavy first round computation by using a precomputed first guess
		if SKIP_FIRST_ROUND:
			guess = STARTING_WORD
			description.prefix = r"\[[bold red]Skipped[/bold red]] Round [cyan]1[/cyan]"
			description.guess=guess
			with tqdm(total=0, leave=True, desc=description, options={'console': CON}) as pbar:
				pass
			self._check_lutnode(guess, remaining_words)
			info = feedback(guess)
			remaining_words = self.pattern_dict[guess][info].intersection(remaining_words)
			solve_stack.append((guess, info))
		
		for i in range(1 if SKIP_FIRST_ROUND else 0, NUMBER_OF_ROUNDS):
			description.prefix = rf"\[[green]Regular[/green]] Round [cyan]{i+1}[/cyan]"
			guess = self.next_guess(remaining_words, description)
			
			# if all of info is G then we have found the solution, we are not fixed to 5  letters
			info = feedback(guess)
			solve_stack.append((guess, info))
			if info == 'G'*len(info):
				return (i + 1, solve_stack)

			# filter the remaining words based on the feedback
			self._check_lutnode(guess, remaining_words)
			remaining_words = self.pattern_dict[guess][info].intersection(remaining_words)

		return (NUMBER_OF_ROUNDS + 1, solve_stack)



def load_wordlist(file: str) -> list:
	with open(file, 'r') as f:
		words = [line.strip() for line in f]
		LOG.info(f"Loaded {len(words)} words from {file}.")
	return words

class WordleDotComSolver(Solver):
	class _Downloader:
		solution = None
		def __init__(self, date: str=None):
			self.date = date
			
		def __call__(self, x):
			# get wordle answer from server
			if self.solution is None:
				today = datetime.datetime.now().strftime("%Y-%m-%d") if self.date in (None, '') else self.date
				solutionjson = requests.get(f"https://www.nytimes.com/svc/wordle/v2/{today}.json")
				jsondata = json.loads(solutionjson.text)
				self.solution = jsondata['solution']
				self.wordle_id = jsondata['days_since_launch']
			return calculate_pattern(x, self.solution)
		def __str__(self):
			return f"Wordle {self.wordle_id:,}"
	
	def __init__(self, solver_config: dict={}):
		solver_config['words'] = load_wordlist(WORDLIST_FILE) if 'words' not in solver_config else solver_config['words']
		super().__init__(solver_config)
		
	def __call__(self, date: str=None) -> tuple:
		feedback = self._Downloader(date)
		return (super().solve(feedback), str(feedback))
		
class InteractiveSolver(Solver):
	def __init__(self, solver_config: dict={}):
		solver_config['words'] = load_wordlist(WORDLIST_FILE) if 'words' not in solver_config else solver_config['words']
		super().__init__(solver_config)
		
	def __call__(self) -> int:
		feedback = lambda x: input(f"Enter pattern for \"{x}\": ")
		super().solve(feedback)



def unpack_pkl(file: str) -> object:
	# check if we have the file already on hand, otherwise filter
	try:
		with open(file, 'rb') as f:
			LOG.info(f'Loading {file}...')
			return pickle.load(f)
	except FileNotFoundError:
		LOG.warning(f'{file} not found, building it...')
	except pickle.UnpicklingError:
		LOG.warning(f'{file} is corrupted, rebuilding it...')
	except Exception as e:
		LOG.error(f'Unknown error loading {file}:', exc_info=True)
		raise e

def main():
	# set global console to a new console
	global CON 
	CON = console.Console()
	global LOG
	logging.basicConfig(level="NOTSET", handlers=[RichHandler(console=CON)])
	LOG = logging.getLogger(__name__)
	LOG.setLevel(logging.INFO)
	install(console=CON)
	
	# cli parser
	parser = argparse.ArgumentParser(
		prog="Wordle-Solver",
		description="A wordle bot using 3b1b\'s entropy pruning strategy.\nIt can also make use of precomputed data and generate optimal first guesses as well.",
		epilog="Cheat the game!")
	verbosity = parser.add_mutually_exclusive_group()
	verbosity.add_argument('-vv', '--verbose', action='store_true', help="Enable verbose logging")
	verbosity.add_argument('-q', '--quiet', action='store_true', help="Disable logging")

	parser.add_argument('-w', '--wordlist', type=str, help="Set the wordlist file to use, by default \"wordlist.txt\" is used", default="wordlist.txt")
	parser.add_argument('-s', '--startingword', type=str, help="Set the starting word for the solver")
	parser.add_argument('--precompute', action='store_true', help="""Precoputes the pattern_dict and entropy_table\n
		this really is only useful when using this as a module for simulations or some shit as loading the file takes more time than just calculating it on the fly\n
		as a bonus this will also output just output the most informative first guess""")

	solving = parser.add_mutually_exclusive_group()
	solving.add_argument('-i', '--interactive', action='store_true', help="Run the interactive solver")
	solving.add_argument('-f', '--solvefor', type=str, help="Solve for a specific word")
	solving.add_argument('-o', '--solvedate', type=str, help="Solve a specific date, automatically downloads wordle site, date supplied as yyyy-mm-dd", nargs='?', const="", default=None)

	parser.add_argument('--version', action='version', version="%(prog)s 1.1 by Lima")
	parser.add_argument('-c', '--clipboard', action='store_true', help="Copy the result to the clipboard, similar to the wordle site")
	
	args = parser.parse_args()
	if args.verbose:
		LOG.setLevel(logging.DEBUG)
	elif args.quiet:
		LOG.setLevel(logging.ERROR)

	# load the wordlist, this is essential
	if args.wordlist:
		global WORDLIST_FILE
		WORDLIST_FILE = args.wordlist
	words = load_wordlist(WORDLIST_FILE)
	# random.shuffle(words)
	solver_config = {}
	solver_config['words'] = words

	if args.precompute:
		pattern_dict = unpack_pkl('pattern_dict.pkl')
		if pattern_dict is None:
			# build the pattern_dict
			solver = Solver(solver_config)
		
			with tqdm(words, desc="Building pattern_dict...", leave=True, options={'console': CON}) as pbar:
				for word in pbar:
					pbar.set_description(f"Building pattern_dict: \\[{word}]")
					solver._check_lutnode(word, words)
			pattern_dict = dict(solver.pattern_dict)

			LOG.debug('Saving pattern_dict.pkl...')
			with open('pattern_dict.pkl', 'wb') as f:
				pickle.dump(pattern_dict, f)
			LOG.info('Saved pattern_dict.pkl.')
		
		solver_config['pattern_dict'] = pattern_dict

		entropy_table = unpack_pkl('entropy_table.pkl')
		if entropy_table is None:
			entropy_table = {}
			with tqdm(words, desc="Building entropy_table...", leave=True, options={'console': CON}) as pbar:
				for word in pbar:
					pbar.set_description(f"Building entropy_table: \\[{word}]")
					entropy_table[word] = calculate_entropy(word, words, pattern_dict)
		
			LOG.debug('Saving entropy_table.pkl...')
			with open('entropy_table.pkl', 'wb') as f:
				pickle.dump(entropy_table, f)
			LOG.info('Saved entropy_table.pkl.')

		solver_config['entropy_table'] = entropy_table
		LOG.info(f"Most informative first guess is \"{max(entropy_table, key=lambda x: entropy_table[x])}\"")

	if args.startingword:
		global STARTING_WORD
		STARTING_WORD = args.startingword.strip()
		LOG.info(f'Starting word set to "{STARTING_WORD}"')
	
	if args.solvedate is not None:
		solver = WordleDotComSolver(solver_config)
		result = solver(args.solvedate.strip())
		if args.clipboard:
			import pyperclip

			# construct the result string
			result_str = result[1]
			result = result[0]
			result_str += f" {result[0] if result[0] <= NUMBER_OF_ROUNDS else 'X'}/{NUMBER_OF_ROUNDS}*\n\n"
			result_str += '\n'.join(f"{x[1]}" for x in result[1]).replace('G', "\U0001f7e9").replace('Y', "\U0001f7e8").replace('B', "\u2b1b")
			pyperclip.copy(result_str)
			LOG.info(f"Result copied to clipboard:\n{result_str}")
	elif args.interactive:
		result = InteractiveSolver()()
	elif args.solvefor:
		solver = Solver(solver_config)
		feedback = lambda x: calculate_pattern(x, args.solvefor.strip())
		result = solver.solve(feedback)
	
if __name__ == '__main__':
	main()
