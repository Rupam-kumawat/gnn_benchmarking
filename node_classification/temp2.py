import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
args = parser.parse_args()

if args.debug:
    print("Debug mode is enabled.")
else:
    print("Debug mode is disabled.")