from dataclasses import dataclass
from itertools import product
import random
from uuid import uuid4


TERRAIN_MUT_LIMITS = {
    "size": (2, round),
    "scale": (-2, float),
    "octaves": (1, round),
    "persistence": (0.1, float),
    "lacunarity": (0.1, float),
}

# drop the cases of all zeros and ones
PATTERNS = list(product((0, 1), repeat=6))[1:-1]


@dataclass(slots=True)
class TerrainGene:
#    __slots__ = ("id", "size", "scale", "base", "octaves", "persistence", "lacunarity")
    id: str = ""
    size: int = 200
    scale: float = 140
    base: float = 42.0
    octaves: int = 1
    persistence: float = 0.1
    lacunarity: float = 0.5

    def mutate(self, mutation_rate: float) -> None:
        for prop in self.__slots__:
            if prop not in TERRAIN_MUT_LIMITS:
                continue

            will_mutate = random.random() <= mutation_rate
            if will_mutate:
                cur_value = getattr(self, prop)
                limit, smoother_func = TERRAIN_MUT_LIMITS[prop]
                new_value = smoother_func(random.random() * limit + cur_value)
                setattr(self, prop, new_value)

    def crossover(self, other_gene: "TerrainGene") -> "TerrainGene":
        pattern = random.choice(PATTERNS)
        gene_options = (self, other_gene)
        values = []

        for idx, gene_id in enumerate(pattern):
            gene = gene_options[gene_id]
            idx_id_skipped = idx + 1
            prop = self.__slots__[idx_id_skipped] # type: ignore
            values.append(getattr(gene, prop))

        return TerrainGene(*values)

    def __hash__(self) -> int:
        return hash(self.id)
