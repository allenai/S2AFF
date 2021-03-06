"""
Write integration tests for your model interface code here.

The TestCase class below is supplied a `container`
to each test method. This `container` object is a proxy to the
Dockerized application running your model. It exposes a single method:

```
predict_batch(instances: List[Instance]) -> List[Prediction]
```

To test your code, create `Instance`s and make normal `TestCase`
assertions against the returned `Prediction`s.

e.g.

```
def test_prediction(self, container):
    instances = [Instance(), Instance()]
    predictions = container.predict_batch(instances)

    self.assertEqual(len(instances), len(predictions)

    self.assertEqual(predictions[0].field1, "asdf")
    self.assertGreatEqual(predictions[1].field2, 2.0)
```
"""


import logging
import sys
import unittest

from .interface import Instance, Prediction


try:
    from timo_interface import with_timo_container
except ImportError as e:
    logging.warning(
        """
    This test can only be run by a TIMO test runner. No tests will run. 
    You may need to add this file to your project's pytest exclusions.
    """
    )
    sys.exit(0)


@with_timo_container
class TestInterfaceIntegration(unittest.TestCase):
    def test__predictions(self, container):
        instances = [
            Instance(raw_affiliation="The University of Texas at Austin"),
            Instance(raw_affiliation="Univ of Wash, Seattle, WA, 98115, U.S.A."),
        ]
        predictions = container.predict_batch(instances)

        self.assertEqual(predictions[0].ror_id, "https://ror.org/00hj54h04")
        self.assertEqual(predictions[1].ror_id, "https://ror.org/00cvxb145")
