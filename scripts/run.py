from __future__ import print_function
import sys
import streams

def main():
	# get filename
	Narg = len(sys.argv)
	if (Narg!=2):
		print("Wrong number of arguments. Call with: run.py <fname>")
		return 1
	fname = sys.argv[1]
	
	# fit progenitor
	streams.find_fromfile(fname)

main()
