import numpy as np
import tensorflow as tf
import owen


class SampledOwenTest(tf.test.TestCase):

  def test_simple_linear_model(self):

    class TestModel(tf.keras.Model):
      def call(self, inputs):
        x = inputs * tf.constant([[[1.0], [2.0], [3.0]],
                                  [[4.0], [5.0], [6.0]],
                                  [[7.0], [8.0], [9.0]]])
        x = tf.expand_dims(x, axis=-1)
        return tf.reduce_sum(x, axis=[1, 2, 3])

    model = TestModel()
    so = owen.SampledOwen(model, batch_size=7)
    owen_values = so.explain(
        np.ones((1, 3, 3, 3)),
        np.zeros((1, 3, 3, 3)),
        np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
        20, 5, top_k_group_paths=20)
    expected_owen_values = np.array(
        [[[[1., 1., 1.],
           [2., 2., 2.],
           [3., 3., 3.]],
          [[4., 4., 4.],
           [5., 5., 5.],
           [6., 6., 6.]],
          [[7., 7., 7.],
           [8., 8., 8.],
           [9., 9., 9.]]]])
    self.assertAllClose(expected_owen_values, owen_values)

    owen_values = so.explain(
        np.ones((1, 3, 3, 3)),
        np.zeros((1, 3, 3, 3)),
        np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
        20, 5, top_k_group_paths=0)
    expected_owen_values = np.array(
        [[[[2., 2., 2.],
           [2., 2., 2.],
           [2., 2., 2.]],
          [[5., 5., 5.],
           [5., 5., 5.],
           [5., 5., 5.]],
          [[8., 8., 8.],
           [8., 8., 8.],
           [8., 8., 8.]]]])
    self.assertAllClose(expected_owen_values, owen_values)


if __name__ == '__main__':
  tf.test.main()
