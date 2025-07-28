import re
from typing import Dict, List

from rdkit import Chem
from rdkit.Chem import Descriptors


class BigSMILESParser:
    """
    Parser for BigSMILES notation - extended SMILES for polymers

    BigSMILES Format:
    - Repeating units: {[repeat_unit]}
    - Stochastic objects: {[unit1][unit2]...}
    - Bonding descriptors: [$bond1$], [$bond2$]
    - Terminal groups: end_group{[repeat_unit]}end_group
    """

    def __init__(self):
        self.repeat_unit_pattern = r"\{([^}]+)\}"
        self.bond_pattern = r"\[\$([^$]+)\$\]"
        self.stochastic_pattern = r"\{([^}]*\[[^]]+\][^}]*)+\}"

    def parse_bigsmiles(self, bigsmiles: str) -> Dict:
        """Parse BigSMILES string into components"""
        result = {
            "original": bigsmiles,
            "repeat_units": [],
            "bonds": [],
            "terminal_groups": [],
            "stochastic_objects": [],
            "backbone_atoms": 0,
            "side_chain_atoms": 0,
        }

        # Extract repeat units
        repeat_matches = re.findall(self.repeat_unit_pattern, bigsmiles)
        for match in repeat_matches:
            unit_info = self._analyze_repeat_unit(match)
            result["repeat_units"].append(unit_info)

        # Extract bonds
        bond_matches = re.findall(self.bond_pattern, bigsmiles)
        result["bonds"] = bond_matches

        # Extract terminal groups
        terminal_groups = self._extract_terminal_groups(bigsmiles)
        result["terminal_groups"] = terminal_groups

        # Calculate molecular features
        result.update(self._calculate_polymer_features(result))

        return result

    def _analyze_repeat_unit(self, unit_smiles: str) -> Dict:
        """Analyze individual repeat unit"""
        try:
            mol = Chem.MolFromSmiles(unit_smiles)
            if mol is None:
                return {"smiles": unit_smiles, "valid": False}

            return {
                "smiles": unit_smiles,
                "valid": True,
                "atoms": mol.GetNumAtoms(),
                "bonds": mol.GetNumBonds(),
                "molecular_weight": Descriptors.MolWt(mol),
                "rings": Descriptors.RingCount(mol),
                "aromatic_rings": Descriptors.NumAromaticRings(mol),
                "rotatable_bonds": Descriptors.NumRotatableBonds(mol),
                "hbd": Descriptors.NumHDonors(mol),
                "hba": Descriptors.NumHAcceptors(mol),
                "logp": Descriptors.MolLogP(mol),
            }
        except Exception:
            return {"smiles": unit_smiles, "valid": False}

    def _extract_terminal_groups(self, bigsmiles: str) -> List[str]:
        """Extract terminal/end groups from BigSMILES"""
        # Remove repeat units and bonds to find terminal groups
        cleaned = re.sub(self.repeat_unit_pattern, "", bigsmiles)
        cleaned = re.sub(self.bond_pattern, "", cleaned)

        # Split by common separators and filter
        terminals = [t.strip() for t in cleaned.split("{}") if t.strip()]
        return terminals

    def _calculate_polymer_features(self, parsed_data: Dict) -> Dict:
        """Calculate polymer-specific features"""
        features = {
            "num_repeat_units": len(parsed_data["repeat_units"]),
            "total_repeat_mw": 0,
            "avg_repeat_mw": 0,
            "complexity_score": 0,
        }

        if parsed_data["repeat_units"]:
            valid_units = [
                u for u in parsed_data["repeat_units"] if u.get("valid", False)
            ]
            if valid_units:
                mws = [u["molecular_weight"] for u in valid_units]
                features["total_repeat_mw"] = sum(mws)
                features["avg_repeat_mw"] = sum(mws) / len(mws)

                # Complexity based on atoms, rings, and rotatable bonds
                complexity = sum(
                    u["atoms"] + u["rings"] + u["rotatable_bonds"] for u in valid_units
                )
                features["complexity_score"] = complexity

        return features
