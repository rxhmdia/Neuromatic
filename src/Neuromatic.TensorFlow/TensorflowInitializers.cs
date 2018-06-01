using Neuromatic.Core;
using System;
using System.Collections.Generic;
using System.Text;

namespace Neuromatic.TensorFlow
{
    public class TensorflowInitializers : Initializers
    {
        private TensorFlowModelBackend _backend;

        /// <summary>
        /// Initializes a new instance of <see cref="TensorflowInitializers"/>
        /// </summary>
        /// <param name="backend">Backend to use for implementing the initializer functions</param>
        public TensorflowInitializers(TensorFlowModelBackend backend)
        {
            _backend = backend;
        }

        /// <summary>
        /// Creates an initializer that initializes a set of weights with a normal distribution
        /// </summary>
        /// <param name="mean">Mean of the distribution (default 0.0)</param>
        /// <param name="standardDeviation">Standard distribution (default 0.05)</param>
        /// <param name="seed">The random seed</param>
        /// <returns>Returns the new initializer</returns>
        public override InitializationFunction RandomNormal(float mean = 0, float standardDeviation = 0.05F, int? seed = null)
        {
            return new InitializationFunction((long[] shape) => _backend.RandomNormal(shape, mean, standardDeviation, seed));
        }
    }
}
