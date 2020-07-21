"""
This file contains TF.Keras-specific classes/implementations
"""
from typing import Callable, Any, Iterable, Union, Tuple, Dict, List

import tensorflow as tf
import numpy as np


class StatsCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(StatsCallback, self).__init__()
        self.__train_score = list()
        self.__test_score = list()

    def on_epoch_end(self, epoch, logs=None):
        # loss key must exist no matter what
        self.__train_score.append(logs['loss'])

        # val_loss might be absent, i.e. if there was no validation dataset
        if 'val_loss' in logs:
            self.__test_score.append(logs['val_loss'])

    def last_test_score(self):
        return self.__test_score[len(self.__test_score) - 1]

    def last_train_score(self):
        return self.__train_score[len(self.__train_score) - 1]

    def last_score(self):
        return self.last_test_score() if len(self.__test_score) > 0 else self.last_train_score()


class TensorFlowWrapper:
    """
    This class provides a simple wrapper for regular generators to make them endless
    """
    def __init__(self, generator_callable: Callable, epoch_limit: int = 0):
        """
        :param generator_callable: Callable that creates generator
        """
        self.__generator = generator_callable
        self.__limit = epoch_limit

    def __iter__(self):
        """
        Endless iterable here
        :return:
        """
        while True:
            for v in self.__generator():
                yield v

    def callable(self):
        """
        This method acts as wrapper, providing en endless iterable for TF
        :return:
        """
        while True:
            cnt = 0
            for v in self.__generator():
                yield v
                cnt += 1
                if self.__limit == cnt:
                    break


class TuneKeras:
    """
    This class provides methods for hyperparameters search for TF.Keras models
    """
    def __init__(self, parameters: Dict[str, Union[Tuple[Any], List[Any]]],
                 train_generator: Callable[[], Iterable], train_batches: int,
                 class_weight=None,
                 test_generator: Callable[[], Iterable] = None, test_batches: int = 0,
                 workers: int = 1):
        """
        :param parameters: Dictionary with all possible values for parameters
        :param train_generator: void callable, that returns Python generator, yielding either pair of numpy arrays or
        pair of dictionaries, containing string keys and numpy array values
        :param train_batches: number of unique batches to expect from train_gnerator
        :param test_generator: void callable, that returns Python generator, yielding either pair of numpy arrays or
        pair of dictionaries, containing string keys and numpy array values
        :param test_batches: number of unique batches to expect from test_gnerator
        :param workers: Number of workers for the search process
        """
        self.__parameters = parameters
        self.__workers = workers
        self.__train_generator = train_generator
        self.__train_batches = train_batches
        self.__test_generator = test_generator
        self.__test_batches = test_batches
        self.__class_weight = class_weight

    def __convert_dtype(self, dtype: np.dtype) -> tf.dtypes:
        """
        This method converts NumPy dtype to TF dtype
        :param dtype:
        :return:
        """
        return tf.dtypes.as_dtype(dtype)

    def __convert_shape(self, shape: Tuple[int]) -> tf.TensorShape:
        """
        This method converts shape to tf.TensorShape + sets batch dim to None
        :param shape:
        :return:
        """
        result = [None]
        for i in range(1, len(shape)):
            result.append(shape[i])

        return tf.TensorShape(result)

    def __build_tf_dataset(self, generator: Callable[[], Iterable[Union[Tuple[np.ndarray, np.ndarray], Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]]]) -> tf.data.Dataset:
        """
        This method builds TF dataset out of python generator
        :return:
        """
        # instantiating generator and fetching 1 dataset out of it
        gen = generator()
        tpl = next(gen.__iter__())
        # we must have exactly 2 objects in tuple
        assert len(tpl) == 2

        # it's either 2 np.arrays or 2 dictionaries here, but both types must be the same
        assert type(tpl[0]) == type(tpl[1])

        if isinstance(tpl[0], np.ndarray):
            return tf.data.Dataset.from_generator(generator,
                                                  output_types=((self.__convert_dtype(tpl[0].dtype),
                                                                 self.__convert_dtype(tpl[1].dtype))),
                                                  output_shapes=((self.__convert_shape(tpl[0].shape),
                                                                  self.__convert_shape(tpl[1].shape))))
        elif isinstance(tpl[0], dict):
            # now all shapes/types will be pulled into separate dicts
            in_types = dict()
            out_types = dict()
            in_shapes = dict()
            out_shapes = dict()

            in_keys = tpl[0].keys()
            out_keys = tpl[1].keys()

            # filling inputs first
            for key in in_keys:
                array = tpl[0][key]
                assert isinstance(array, np.ndarray)
                in_types[key] = self.__convert_dtype(array.dtype)
                in_shapes[key] = self.__convert_shape(array.shape)

            # now filling outputs
            for key in out_keys:
                array = tpl[1][key]
                assert isinstance(array, np.ndarray)
                out_types[key] = self.__convert_dtype(array.dtype)
                out_shapes[key] = self.__convert_shape(array.shape)

            # and returning the dataset
            return tf.data.Dataset.from_generator(generator,
                                                  output_types=(in_types, out_types),
                                                  output_shapes=(in_shapes, out_shapes))
        else:
            raise ValueError("Generator must yield tuples of NumPy.ndarray or tuples of dict")

    def search(self, model_provider: Callable[[Any], tf.keras.Model],
               epochs: int = 1) -> tf.keras.Model:
        """
        This method iterates over all possible combinations of parameters, in order to find the best possible model
        :param model_provider: callable that accepts some arguments, and returns a TF.Keras model instead.
        Model must be compiled upon return.
        :param epochs: number of epochs per shot
        :return: Model with best validation loss
        """
        from easytune import GridBuilder
        grid = GridBuilder(self.__parameters)

        best_score = float("inf")
        best_params = dict()
        best_model = None
        cnt = 1
        # TODO: make this async
        for p in grid.random():
            print(f"Model {cnt} of  {grid.combinations()}...")
            model = model_provider(**p)
            train_wrapper = TensorFlowWrapper(self.__train_generator)
            train_dataset = self.__build_tf_dataset(train_wrapper.callable)

            # let's make sure we're not passing None here an there
            test_wrapper = None if self.__test_generator is None else TensorFlowWrapper(self.__test_generator)
            test_dataset = None if self.__test_generator is None else self.__build_tf_dataset(test_wrapper.callable)
            test_steps = 0 if self.__test_generator is None else self.__test_batches

            callback = StatsCallback()

            # time to attach callable, and fit the model
            model.fit(train_dataset, verbose=0, workers=1, steps_per_epoch=self.__train_batches, epochs=epochs, callbacks=callback,
                      validation_data=test_dataset, validation_steps=test_steps, class_weight=self.__class_weight)

            # picking the best model based on train or test score
            if callback.last_score() < best_score:
                best_score = callback.last_score()
                best_params = p
                best_model = model

            cnt += 1

        print(f"Best loss: {best_score:.4f}; params: {best_params}")
        return best_model
