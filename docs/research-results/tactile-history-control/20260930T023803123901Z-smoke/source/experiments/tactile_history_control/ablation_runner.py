"""All predeclared input-content controls, without architecture search."""
from .evaluate import evaluate
from .ambiguity import run_ambiguity

def run_ablations(run,c):
    evaluate(run,c)
    run_ambiguity(run,c)
