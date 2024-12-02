"""Uses `pickle` to save and restore populations (and other aspects of the simulation state)."""

from dataclasses import dataclass
import gzip
import pathlib
import pickle
import random
import time

from neat.config import Config

from hive_mind.exploring.mcc import EvolvingEntity, MCCConfig


@dataclass
class SearchState:
    mcc_config: MCCConfig | None
    neat_config: Config
    population: list[EvolvingEntity]
    generation: int
    rand_state: object


class Checkpointer:
    """
    A reporter class that performs checkpointing using `pickle`
    to save and restore populations (and other aspects of the simulation state).
    """

    def __init__(self,
                 folder: pathlib.Path,
                 generation_interval=2,
                 time_interval_seconds=300,
                 filename_prefix='hive_checkpoint_'):
        """
        Saves the current state (at the end of a generation) every ``generation_interval`` generations or
        ``time_interval_seconds``, whichever happens first.

        :param generation_interval: If not None, maximum number of generations between save intervals
        :type generation_interval: int or None
        :param time_interval_seconds: If not None, maximum number of seconds between checkpoint attempts
        :type time_interval_seconds: float or None
        :param str filename_prefix: Prefix for the filename (the end will be the generation number)
        """
        self._folder = folder
        self._generation_interval = generation_interval
        self._time_interval_seconds = time_interval_seconds
        self._filename_prefix = filename_prefix

        self.current_generation: int = 0
        self.last_generation_checkpoint = -1
        self.last_time_checkpoint = time.time()
        self._folder.mkdir(parents=True, exist_ok=True)

    def start_generation(self, generation: int):
        self.current_generation = generation

    def end_generation(self, mcc_config: MCCConfig | None, neat_config: Config, population: list[EvolvingEntity]):
        checkpoint_due = False

        if self._time_interval_seconds is not None:
            dt = time.time() - self.last_time_checkpoint
            if dt >= self._time_interval_seconds:
                checkpoint_due = True

        if (checkpoint_due is False) and (self._generation_interval is not None):
            dg = self.current_generation - self.last_generation_checkpoint
            checkpoint_due = dg >= self._generation_interval

        if checkpoint_due:
            self.save_checkpoint(mcc_config, neat_config, population, self.current_generation)
            self.last_generation_checkpoint = self.current_generation
            self.last_time_checkpoint = time.time()

    def save_checkpoint(self, mcc_config: MCCConfig | None, neat_config: Config, population: list[EvolvingEntity], generation: int):
        """ Save the current simulation state. """
        filename = '{0}{1}'.format(self._filename_prefix, generation)
        file_path = self._folder / filename
        print("Saving checkpoint to {0}".format(file_path))

        with gzip.open(file_path, 'w', compresslevel=5) as f:
            state = SearchState(mcc_config, neat_config, population, generation, random.getstate(),)
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def restore_checkpoint(filename: pathlib.Path) -> SearchState:
        """Resumes the simulation from a previous saved point."""
        with gzip.open(filename) as f:
            state: SearchState = pickle.load(f)
            random.setstate(state.rand_state) # type: ignore
            return state
