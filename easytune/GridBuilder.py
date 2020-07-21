"""
GridBuilder
"""
import random
from typing import Iterable, Dict, Any


class GridBuilder:
    """
    GridBuilder class provides utilities for building grid of all possible params, in a form suitable for hyperparameters search
    """
    def __init__(self, parameters: Dict[Any, Any]):
        if parameters is None or not isinstance(parameters, dict) or len(parameters) == 0:
            raise ValueError("Parameters should be a valid non-empty dict")

        self.__parameters = parameters
        self.__combinations = -1

    def combinations(self) -> int:
        """
        This method calculates number of all possible combinations within grid
        """
        prod = 1
        for key in self.__parameters:
            row = self.__parameters[key]
            prod *= len(row)

        return prod

    def random(self) -> Iterable[dict]:
        """
        This method returns a randomized list of all possible combinations within Grid
        :return:
        """
        # TODO: this can be improved with sliding window
        # pulling all possible values out of sequential grid
        result = list()
        for space in self.sequenial():
            result.append(space)

        # shuffle sequential results and yield them one by one
        random.shuffle(result)
        for v in result:
            yield v

    def sequenial(self) -> Iterable[dict]:
        """
        This method provides a generator with all possible combinations within Grid
        :return:
        """
        if self.__combinations < 0:
            self.__combinations = self.combinations()

        # first of all we build list of counters
        keys = self.__parameters.keys()

        # special case: there's only 1 key, so we just roll through params in this key
        if len(keys) == 1:
            for key in keys:
                for v in self.__parameters[key]:
                    yield v
        else:
            # if we have more than 1 key, we'll need to
            counters = dict()
            for key in keys:
                counters[key] = 0

            # we will be incrementing counter of this primary key
            for primary in keys:
                result = dict()

                filtered_keys = list()
                for key in keys:
                    if key == primary:
                        continue

                    filtered_keys.append(key)

                # now for each primary key we'll iterate over all other keys
                for pv in self.__parameters[primary]:
                    result[primary] = pv

                    # nullify counters
                    counters = dict()
                    for key in keys:
                        counters[key] = 0

                    # now, for every primary key, we'll roll through all possible combinations of other keys
                    combinations_left = 1
                    for incremental in filtered_keys:
                        combinations_left *= len(self.__parameters[incremental])

                    key_index = 0
                    for i in range(combinations_left):
                        # now we increment 1 specific index
                        index = i
                        for r in range(len(filtered_keys) - 1, 0, -1):
                            key = filtered_keys[r]
                            counters[key] = index % len(self.__parameters[key])
                            index //= len(self.__parameters[key])

                        counters[filtered_keys[0]] = index

                        # fix current state
                        for secondary in filtered_keys:
                            result[secondary] = self.__parameters[secondary][counters[secondary]]

                        # yield detached copy of fixed state
                        yield result.copy()

                # we have outer loop, but we use it only to start things
                break
