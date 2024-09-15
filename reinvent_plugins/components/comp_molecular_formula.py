"""
Scoring function for closeness to a molecular formula.

The score penalizes deviations from the required number of atoms for each element type, and for the total
number of atoms.

F.i., if the target formula is C2H4, the scoring function is the average of three contributions:
- number of C atoms with a Gaussian modifier with mu=2, sigma=1
- number of H atoms with a Gaussian modifier with mu=4, sigma=1
- total number of atoms with a Gaussian modifier with mu=6, sigma=2
"""

from __future__ import annotations

__all__ = ["MolecularFormula"]

from dataclasses import dataclass
from typing import List, Tuple
import re

from rdkit import Chem
from rdkit.Chem import rdMolHash
import numpy as np

from ..component_results import ComponentResults
from reinvent_plugins.mol_cache import molcache
from ..add_tag import add_tag


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for the scoring component

    Note that all parameters are always lists because components can have
    multiple endpoints and so all the parameters from each endpoint is
    collected into a list.  This is also true in cases where there is only one
    endpoint.
    """

    formula: List[str]

@add_tag("__component")
class MolecularFormula:
    def __init__(self, params: Parameters):
        self.formula = {}

        for f in params.formula:
            self.formula = self.parse_molecular_formula(f)

        if not self.formula:
            raise RuntimeError(f"{__name__}: no valid formulae found")
        
    def generate_molecular_formula(self, mol: Chem.Mol) -> str:
        formula_function = rdMolHash.HashFunction.MolFormula
        return rdMolHash.MolHash(mol, formula_function)

    def parse_molecular_formula(self, formula: str) -> Tuple[List[Tuple[str, int]],int]:
        """
        Parse a molecular formulat to get the element types and counts.

        Args:
            formula: molecular formula, f.i. "C8H3F3Br"

        Returns:
            List of dictionaries containing element types
              and number of occurrences
        """
        matches = re.findall(r'([A-Z][a-z]*)(\d*)', formula)

        # Convert matches to the required format
        results = {}
        for match in matches:
            # convert count to an integer, and set it to 1 if the count is not visible in the molecular formula
            count = 1 if not match[1] else int(match[1])
            results[match[0]] = count

        return results     

    @molcache
    def __call__(self, mols: List[Chem.Mol]) -> np.array:
        scores = []
        formulae = [self.generate_molecular_formula(mol) for mol in mols]
        parsed_formulae = [self.parse_molecular_formula(f) for f in formulae]
        total_counts = [sum(pf.values()) for pf in parsed_formulae]

        ec_scores = [self.score_formula(f) for f in parsed_formulae]
        tc_score = [self.gaussian(tc, sum(self.formula.values()), sigma=2) for tc in total_counts]

        score_list = [es.append(ts) for es, ts in zip(ec_scores, tc_score)]

        scores_final = [self.geometric_mean(score) for score in ec_scores]
        scores.append(np.array(scores_final))

        return ComponentResults(scores)


    def gaussian(self, x, mu, sigma):
        return np.exp(-0.5 * np.power((x - mu) / sigma, 2.))
    

    def geometric_mean(self, scores):
        iterable = np.array(scores)
        return np.exp(np.log(iterable).mean())
    

    def score_formula(self, formula:dict):
        """take the element counts for the formula, appy to every atom"""
        scores = [self.gaussian(formula.get(element, 0), n_atoms, 1) for element, n_atoms in self.formula.items()]
        return scores